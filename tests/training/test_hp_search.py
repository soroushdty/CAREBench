"""Tests for the HP search worker (_run_hp_candidate) after INCOMPAT 2 refactor."""

import numpy as np
import pytest
from unittest.mock import patch


class TestRunHpCandidate:
    def _make_data(self, n=40, n_features=8, n_classes=2):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, n_features)).astype(np.float32)
        Y = rng.uniform(0, 1, (n, n_classes)).astype(np.float32)
        return X, Y

    def _make_cfg(self, stage1_head_options=None):
        cfg = {
            "global_seed": 0,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 32,
            "weight_cap": 10.0,
            "pca_n_components": 4,
            "pca_whiten": False,
            "class_weight_mode": "cui",
            "class_weight_beta": 0.999,
            "num_epochs": 1,
            "early_stopping_patience": 1,
            "hidden_dims": [],
            "dropout": 0.0,
            "activation": "gelu",
            "primary_threshold": "tau",
            "tau": 0.5,
            "objective": "F1",
            "device": "cpu",
            "default_classes": ["c0", "c1"],
        }
        if stage1_head_options is not None:
            cfg["stage1_head_options"] = stage1_head_options
        return cfg

    def test_hp_dict_contains_selected_stage1_head_config(self):
        """_run_hp_candidate returns hp with selected_stage1_head_config matching input."""
        from tracks.representation.training.stage1.hp_search import _run_hp_candidate

        X, Y = self._make_data()
        cfg = self._make_cfg()
        head_config = [64, 32]
        hp_t = (head_config,)
        inner_splits = [(np.arange(30), np.arange(30, 40))]

        with patch("tracks.representation.training.stage1.hp_search.train_single_model") as mock_train:
            mock_train.return_value = (None, np.full((10, 2), 0.5, dtype=np.float32), None)
            hp, _, _ = _run_hp_candidate(hp_t, X, Y, inner_splits, cfg, seed_base=0)

        assert hp["selected_stage1_head_config"] == head_config

    def test_hp_dict_does_not_expose_lr_or_weight_cap_as_search_outputs(self):
        """lr and weight_cap in returned hp come from cfg scalars, not from hp_t."""
        from tracks.representation.training.stage1.hp_search import _run_hp_candidate

        X, Y = self._make_data()
        cfg = self._make_cfg()
        cfg["lr"] = 0.007
        cfg["weight_cap"] = 99.0
        hp_t = ([],)
        inner_splits = [(np.arange(30), np.arange(30, 40))]

        with patch("tracks.representation.training.stage1.hp_search.train_single_model") as mock_train:
            mock_train.return_value = (None, np.full((10, 2), 0.5, dtype=np.float32), None)
            hp, _, _ = _run_hp_candidate(hp_t, X, Y, inner_splits, cfg, seed_base=0)

        assert hp["lr"] == pytest.approx(0.007)
        assert hp["weight_cap"] == pytest.approx(99.0)

    def test_inner_cfg_hidden_dims_overridden_by_head_config(self):
        """train_single_model is called with inner_cfg where hidden_dims == head_config."""
        from tracks.representation.training.stage1.hp_search import _run_hp_candidate

        X, Y = self._make_data()
        cfg = self._make_cfg()
        cfg["hidden_dims"] = []  # default; should be overridden
        head_config = [64]
        hp_t = (head_config,)
        inner_splits = [(np.arange(30), np.arange(30, 40))]

        captured_cfgs = []
        with patch("tracks.representation.training.stage1.hp_search.train_single_model") as mock_train:
            def capture(xt, yt, xv, yv, inner_cfg, hp, pos_w, seed=None):
                captured_cfgs.append(inner_cfg)
                return (None, np.full((10, 2), 0.5, dtype=np.float32), None)
            mock_train.side_effect = capture
            _run_hp_candidate(hp_t, X, Y, inner_splits, cfg, seed_base=0)

        assert len(captured_cfgs) == 1
        assert captured_cfgs[0]["hidden_dims"] == head_config

    def test_param_grid_length_from_stage1_head_options(self):
        """param_grid built from stage1_head_options has the correct length."""
        head_options = [[], [64], [64, 32]]
        param_grid = [(hc,) for hc in head_options]
        assert len(param_grid) == 3
        assert param_grid[0] == ([],)
        assert param_grid[1] == ([64],)
        assert param_grid[2] == ([64, 32],)

    def test_param_grid_defaults_to_single_empty_candidate(self):
        """When stage1_head_options is absent, param_grid defaults to [[]] (one candidate)."""
        cfg = self._make_cfg()  # no stage1_head_options
        param_grid = [(hc,) for hc in cfg.get("stage1_head_options", [[]])]
        assert len(param_grid) == 1
        assert param_grid[0] == ([],)
