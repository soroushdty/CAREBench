"""Track 1 CLI wiring: runner -> RepresentationTrack(adapter, strategy).

Covers:
  - the runner builds the adapter and strategy from config and delegates to
    ``RepresentationTrack.run``
  - ``preprocessing()`` gives the same result on adapter-loaded sheets as on
    sheets it reads itself
  - ``PairedContextRepresentationAdapter`` builds the dataset the old runner
    built (embeddings stubbed)
  - ``ExistingEnsembleTrainingStrategy`` passes fuzzy-fallback inputs only
    when ``fuzzy_threshold`` is set, builds Stage 2 context vectors from
    context records, and runs the statistical analysis when enabled
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from shared.adapters.base import RepresentationDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "main_config.yaml"


@pytest.fixture
def example_cfg(tmp_path: Path) -> dict[str, Any]:
    """main_config.yaml for the synthetic example, with outputs in tmp_path."""
    cfg = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    cfg.update(
        {
            "PROJECT_ROOT": str(REPO_ROOT),
            "ENABLE_SUMMARY": False,
            "DIR_SUMMARY": str(tmp_path / "input_summary"),
            "batch_size": 8,
        }
    )
    return cfg


def _fake_compute_embeddings(model_id, data, *, schema, batch_size=32, cfg=None):
    """Deterministic stand-in for compute_embeddings keyed like the real one."""
    out = {}
    for text in data.dropna().astype(str).unique():
        seed = sum(text.casefold().encode()) % (2**32)
        out[text.casefold()] = np.random.default_rng(seed).random(4).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunnerBuildsComponents:
    def test_defaults(self):
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )
        from tracks.representation.runner import _build_adapter, _build_strategy
        from tracks.representation.strategies.existing_ensemble import (
            ExistingEnsembleTrainingStrategy,
        )

        assert isinstance(_build_adapter({}), PairedContextRepresentationAdapter)
        strategy = _build_strategy({}, resume_from_checkpoint=True)
        assert isinstance(strategy, ExistingEnsembleTrainingStrategy)
        assert strategy._resume_from_checkpoint is True

    def test_unknown_names_raise(self):
        from tracks.representation.runner import _build_adapter, _build_strategy

        with pytest.raises(ValueError, match="Unknown adapter 'nope'"):
            _build_adapter({"adapter": "nope"})
        with pytest.raises(ValueError, match="Unknown strategy 'nope'"):
            _build_strategy({"strategy": "nope"}, resume_from_checkpoint=False)

    def test_main_runs_representation_track(self, monkeypatch, tmp_path):
        """main() delegates to RepresentationTrack with config-built components."""
        import shared.prerequisites.load_prerequisites as lp
        import shared.prerequisites.reproducibility_guards as rg
        import tracks.representation.framework as fw
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )
        from tracks.representation import runner
        from tracks.representation.strategies.existing_ensemble import (
            ExistingEnsembleTrainingStrategy,
        )

        model_dir = tmp_path / "run_x" / "model"
        monkeypatch.setattr(
            lp,
            "load_prerequisites",
            lambda **kw: {"DIR_MODEL": str(model_dir), "RUN_ID": "run_x"},
        )
        monkeypatch.setattr(rg, "enforce_reproducibility", lambda seed: None)

        calls: dict[str, Any] = {}

        def fake_run(self, config):
            calls["adapter"] = self._adapter
            calls["strategy"] = self._strategy
            calls["config"] = config
            return {"test": {"F1": 0.5}}

        monkeypatch.setattr(fw.RepresentationTrack, "run", fake_run)

        rc = runner.main(["--config", str(EXAMPLE_CONFIG), "--dir", str(tmp_path)])

        assert rc == 0
        assert isinstance(calls["adapter"], PairedContextRepresentationAdapter)
        assert isinstance(calls["strategy"], ExistingEnsembleTrainingStrategy)
        assert calls["strategy"]._resume_from_checkpoint is False
        assert calls["config"]["output_dir"] == str(model_dir.parent)
        assert calls["config"]["_config_path"] == str(EXAMPLE_CONFIG.resolve())
        # main_config keys are overlaid onto the prerequisites config
        assert calls["config"]["DIR_DATASET"] == "examples/synthetic/dataset.xlsx"


# ---------------------------------------------------------------------------
# Preprocessing on adapter-loaded sheets
# ---------------------------------------------------------------------------


class TestPreprocessingPreloadedSheets:
    def _sheets(self, cfg):
        from adapters.paired_context.column_map import PairedContextColumnMap
        from adapters.paired_context.dataset_adapter import PairedContextDatasetAdapter

        return PairedContextDatasetAdapter(
            dataset_path=REPO_ROOT / cfg["DIR_DATASET"],
            class_cols=cfg["classes"],
            column_map=PairedContextColumnMap(),
        ).load_sheets()

    def test_same_result_as_reading_the_file(self, example_cfg):
        import pandas as pd

        from shared.preprocessing.preprocessing import preprocessing

        from_file = preprocessing(example_cfg)
        from_sheets = preprocessing(example_cfg, dfs=self._sheets(example_cfg))

        pd.testing.assert_frame_equal(from_file.df, from_sheets.df)
        assert from_file.split_index_maps == from_sheets.split_index_maps

    def test_missing_sheet_raises(self, example_cfg):
        from shared.preprocessing.preprocessing import preprocessing

        sheets = self._sheets(example_cfg)
        del sheets["interview"]
        with pytest.raises(ValueError, match="missing: interview"):
            preprocessing(example_cfg, dfs=sheets)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TestPairedContextRepresentationAdapter:
    @pytest.fixture(autouse=True)
    def _stub_embeddings(self, monkeypatch):
        import shared.embeddings.compute_embeddings as ce

        monkeypatch.setattr(ce, "compute_embeddings", _fake_compute_embeddings)

    def test_builds_dataset_from_preprocessed_frame(self, example_cfg):
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )
        from shared.preprocessing.preprocessing import preprocessing

        ds = PairedContextRepresentationAdapter().load_representation_dataset(example_cfg)
        df = preprocessing(example_cfg).df
        train = df[df["split"] == "train"].reset_index(drop=True)
        test = df[df["split"] == "test"].reset_index(drop=True)
        survey = [f"{c}_survey" for c in example_cfg["classes"]]
        interview = [f"{c}_interview" for c in example_cfg["classes"]]

        assert ds.output_dimensions == example_cfg["classes"]
        assert ds.X_train.shape == (len(train), 4)
        assert ds.X_test.shape == (len(test), 4)
        np.testing.assert_array_equal(ds.Y_train, train[survey].to_numpy(np.float32))
        np.testing.assert_array_equal(ds.Y_test, test[survey].to_numpy(np.float32))
        np.testing.assert_array_equal(ds.Y_test_context_free, ds.Y_test)
        np.testing.assert_array_equal(
            ds.Y_test_correct_context, test[interview].to_numpy(np.float32)
        )
        np.testing.assert_array_equal(ds.patient_ids_train, train["Patient"].to_numpy())
        np.testing.assert_array_equal(ds.item_strings_test, test["Item"].to_numpy())
        np.testing.assert_array_equal(
            ds.unresolved_mask_train, ~train["item_json_resolved"].to_numpy()
        )
        assert ds.embedding_cache is not None
        np.testing.assert_array_equal(
            ds.X_train[0], ds.embedding_cache[train["Item"][0].casefold()]
        )

        assert ds.context_vectors is None
        assert ds.context_records is not None
        assert set(ds.patient_ids_test) <= set(ds.context_records)

        assert ds.metadata["dataset_path"] == str(REPO_ROOT / example_cfg["DIR_DATASET"])
        assert ds.metadata["sheet_names"] == {
            "train": "train",
            "test": "test",
            "interview": "interview",
        }
        assert ds.metadata["label_space"]["behavioral_health"] == "Behavioral health"

    def test_missing_context_file_disables_stage2(self, example_cfg, caplog):
        from adapters.paired_context.representation_adapter import (
            PairedContextRepresentationAdapter,
        )

        example_cfg["DIR_CONTEXT"] = "does/not/exist.json"
        ds = PairedContextRepresentationAdapter().load_representation_dataset(example_cfg)

        assert ds.context_records is None
        assert "Stage 2 fusion disabled" in caplog.text


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


def _dataset(**overrides) -> RepresentationDataset:
    rng = np.random.default_rng(0)
    fields: dict[str, Any] = dict(
        X_train=rng.random((6, 4), dtype=np.float32),
        Y_train=np.zeros((6, 2), dtype=np.float32),
        patient_ids_train=np.array(["1", "1", "2", "2", "3", "3"], dtype=object),
        X_test=rng.random((2, 4), dtype=np.float32),
        Y_test=np.zeros((2, 2), dtype=np.float32),
        patient_ids_test=np.array(["4", "5"], dtype=object),
        output_dimensions=["a", "b"],
        item_strings_train=np.array(list("uvwxyz"), dtype=object),
        item_strings_test=np.array(["u", "q"], dtype=object),
        Y_test_context_free=np.zeros((2, 2), dtype=np.float32),
        Y_test_correct_context=np.ones((2, 2), dtype=np.float32),
        unresolved_mask_train=np.zeros(6, dtype=bool),
        unresolved_mask_test=np.array([False, True]),
        embedding_cache={"u": np.zeros(4)},
        context_records={"4": {"summary": "x"}, "5": {"summary": "y"}},
        metadata={
            "dataset_path": "/data/ds.xlsx",
            "sheet_names": {"train": "tr", "test": "te", "interview": "in"},
        },
    )
    fields.update(overrides)
    return RepresentationDataset(**fields)


class TestExistingEnsembleStrategy:
    @pytest.fixture
    def pipeline_calls(self, monkeypatch) -> list[dict[str, Any]]:
        import tracks.representation.training.orchestrator.train_ensemble_pipeline as tep
        import tracks.representation.training.stage2.stage2_context as s2

        calls: list[dict[str, Any]] = []

        def fake_pipeline(**kwargs):
            calls.append(kwargs)
            return {"test_probs_ca_fold_pure": np.full((2, 2), 0.5)}

        monkeypatch.setattr(tep, "train_ensemble_pipeline", fake_pipeline)
        monkeypatch.setattr(
            s2,
            "build_context_vectors",
            lambda records, model_id, cfg: {int(k): np.ones(3) for k in records},
        )
        return calls

    def _strategy(self, **kw):
        from tracks.representation.strategies.existing_ensemble import (
            ExistingEnsembleTrainingStrategy,
        )

        return ExistingEnsembleTrainingStrategy(**kw)

    def test_fuzzy_inputs_passed_when_enabled(self, pipeline_calls):
        ds = _dataset()
        self._strategy().fit_and_evaluate(ds, {"fuzzy_threshold": 85, "llm": "m"})

        call = pipeline_calls[0]
        assert call["item_strings_train"] is ds.item_strings_train
        assert call["item_strings_test"] is ds.item_strings_test
        assert call["unresolved_mask_train"] is ds.unresolved_mask_train
        assert call["unresolved_mask_test"] is ds.unresolved_mask_test
        assert call["embedding_cache"] is ds.embedding_cache
        assert call["Y_test_survey"] is ds.Y_test_context_free
        assert call["Y_test_interview"] is ds.Y_test_correct_context
        assert call["resume_from_checkpoint"] is False

    def test_fuzzy_inputs_withheld_when_disabled(self, pipeline_calls):
        self._strategy(resume_from_checkpoint=True).fit_and_evaluate(
            _dataset(), {"llm": "m"}
        )

        call = pipeline_calls[0]
        for key in (
            "item_strings_train",
            "item_strings_test",
            "unresolved_mask_train",
            "unresolved_mask_test",
            "embedding_cache",
        ):
            assert call[key] is None, key
        assert call["resume_from_checkpoint"] is True

    def test_context_vectors_built_from_records(self, pipeline_calls):
        self._strategy().fit_and_evaluate(_dataset(), {"llm": "m"})
        assert set(pipeline_calls[0]["context_vectors"]) == {4, 5}

    def test_precomputed_context_vectors_used_as_is(self, pipeline_calls):
        vectors = {4: np.zeros(3)}
        self._strategy().fit_and_evaluate(_dataset(context_vectors=vectors), {"llm": "m"})
        assert pipeline_calls[0]["context_vectors"] is vectors

    def test_no_context_records_means_no_context_vectors(self, pipeline_calls):
        self._strategy().fit_and_evaluate(_dataset(context_records=None), {"llm": "m"})
        assert pipeline_calls[0]["context_vectors"] is None

    def test_statistical_analysis_runs_when_enabled(
        self, pipeline_calls, monkeypatch, tmp_path
    ):
        import tracks.representation.statistical.orchestrator.run_analysis as ra

        stat_calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            ra, "run_statistical_analysis", lambda **kw: stat_calls.append(kw)
        )

        ds = _dataset()
        cfg = {
            "llm": "m",
            "DIR_MODEL": str(tmp_path / "model"),
            "statistical_analysis": {"enabled": True, "n_resamples": 7},
        }
        self._strategy().fit_and_evaluate(ds, cfg)

        (kw,) = stat_calls
        assert kw["y_survey"] is ds.Y_test_context_free
        assert kw["y_interview"] is ds.Y_test_correct_context
        assert kw["item_texts"] is ds.item_strings_test
        assert kw["item_texts_train"] is None  # fuzzy_threshold not set
        assert kw["context_json"] is ds.context_records
        assert kw["dataset_path"] == Path("/data/ds.xlsx")
        assert kw["sheet_names"] == {"train": "tr", "test": "te", "interview": "in"}
        assert kw["output_dir"] == tmp_path / "statistical_analysis"
        assert kw["ensemble_bundle_path"] == tmp_path / "model" / "ensemble_bundle.joblib"
        assert kw["n_resamples"] == 7
        np.testing.assert_array_equal(kw["avg_thresh_f1opt"], [0.5, 0.5])
        # No context-free predictions: falls back to the fold-pure Stage 2 ones.
        assert kw["y_hat_cf"] is kw["y_hat_ca"]

    def test_statistical_analysis_skipped_when_disabled(
        self, pipeline_calls, monkeypatch
    ):
        import tracks.representation.statistical.orchestrator.run_analysis as ra

        monkeypatch.setattr(
            ra,
            "run_statistical_analysis",
            lambda **kw: pytest.fail("statistical analysis should not run"),
        )
        self._strategy().fit_and_evaluate(_dataset(), {"llm": "m"})
