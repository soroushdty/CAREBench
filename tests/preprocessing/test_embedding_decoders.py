"""Decoder models as embedding models (#29).

Builds tiny random GPT-2 and Llama checkpoints with a tokenizer that has no
padding token, saved to a temporary directory, so no download is needed.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from shared.embeddings.compute_embeddings import (  # noqa: E402
    _build_pooling_mask,
    _last_token_pooling,
    embedding,
)

_WORDS = "patient has diabetes takes insulin lives alone with daughter no known allergies".split()
_TEXTS = [
    "patient has diabetes",
    "takes insulin",
    "lives alone with daughter no known allergies",
    "diabetes",
]


def _tokenizer(*, eos: bool = True):
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["<unk>", "</s>"] if eos else ["<unk>"]
    vocab = {tok: i for i, tok in enumerate(specials + _WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    kwargs = {"eos_token": "</s>"} if eos else {}
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", model_max_length=64, **kwargs
    )


def _save_model(path, arch: str, *, eos: bool = True):
    tokenizer = _tokenizer(eos=eos)
    torch.manual_seed(0)
    vocab_size = len(tokenizer)
    if arch == "gpt2":
        config = transformers.GPT2Config(
            vocab_size=vocab_size, n_embd=16, n_layer=2, n_head=2, n_positions=64,
        )
        model = transformers.GPT2LMHeadModel(config)
    else:
        config = transformers.LlamaConfig(
            vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    assert transformers.AutoTokenizer.from_pretrained(path).pad_token is None
    return str(path)


@pytest.fixture(params=["gpt2", "llama"])
def decoder_dir(request, tmp_path):
    return _save_model(tmp_path / request.param, request.param)


def _cfg(**overrides):
    return {
        "embedding_backend": "transformers",
        "pooling": "last_token",
        "hf_local_files_only": True,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

class TestLastTokenPooling:

    def test_right_and_left_padding(self):
        hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
        right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        assert _last_token_pooling(hidden, right).squeeze(-1).tolist() == [1.0, 7.0]
        assert _last_token_pooling(hidden, left).squeeze(-1).tolist() == [3.0, 7.0]

    def test_pooling_mask_without_sep_token(self):
        ids = torch.tensor([[5, 6, 1, 1]])
        mask = torch.tensor([[1, 1, 0, 0]])
        out = _build_pooling_mask(ids, mask, sep_id=None, pad_id=1)
        assert out.tolist() == [[True, True, False, False]]


# ---------------------------------------------------------------------------
# Decoder models end to end
# ---------------------------------------------------------------------------

class TestDecoderEmbeddings:

    def test_embeds_without_a_padding_token(self, decoder_dir):
        out = embedding(_TEXTS, decoder_dir, batch_size=4, cfg=_cfg())
        assert set(out) == set(_TEXTS)
        for vec in out.values():
            assert vec.shape == (16,)
            assert vec.dtype == np.float32
            assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_last_token_is_the_last_real_token(self, decoder_dir):
        """Each vector is the last real token's state from the text run alone."""
        out = embedding(_TEXTS, decoder_dir, batch_size=4, cfg=_cfg())
        tokenizer = transformers.AutoTokenizer.from_pretrained(decoder_dir)
        model = transformers.AutoModel.from_pretrained(decoder_dir).eval()
        for text in _TEXTS:
            ids = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                h = model(**ids).last_hidden_state[0, -1].numpy()
            np.testing.assert_allclose(out[text], h / np.linalg.norm(h), atol=1e-5)

    def test_batching_does_not_change_vectors(self, decoder_dir):
        batched = embedding(_TEXTS, decoder_dir, batch_size=4, cfg=_cfg())
        single = embedding(_TEXTS, decoder_dir, batch_size=1, cfg=_cfg())
        for text in _TEXTS:
            np.testing.assert_allclose(batched[text], single[text], atol=1e-5)

    @pytest.mark.parametrize("pooling", ["mean", "max"])
    def test_other_poolings_run_on_decoders(self, decoder_dir, pooling):
        out = embedding(_TEXTS, decoder_dir, batch_size=4, cfg=_cfg(pooling=pooling))
        assert len(out) == len(_TEXTS)

    def test_bfloat16_returns_float32_close_to_float32(self, decoder_dir):
        full = embedding(_TEXTS, decoder_dir, cfg=_cfg())
        half = embedding(_TEXTS, decoder_dir, cfg=_cfg(embedding_dtype="bfloat16"))
        for text in _TEXTS:
            assert half[text].dtype == np.float32
            assert np.dot(full[text], half[text]) > 0.99

    def test_device_map_loads_and_embeds(self, decoder_dir):
        pytest.importorskip("accelerate")
        out = embedding(_TEXTS, decoder_dir, cfg=_cfg(embedding_device_map="cpu"))
        assert len(out) == len(_TEXTS)

    def test_first_token_pooling_on_decoder_warns(self, decoder_dir, caplog):
        with caplog.at_level(logging.WARNING, logger="shared.embeddings.compute_embeddings"):
            embedding(_TEXTS, decoder_dir, cfg=_cfg(pooling="none"))
        assert "pooling='last_token'" in caplog.text

    def test_tokenizer_without_pad_or_eos_raises(self, tmp_path):
        path = _save_model(tmp_path / "noeos", "gpt2", eos=False)
        with pytest.raises(ValueError, match="neither a padding nor an EOS token"):
            embedding(_TEXTS, path, cfg=_cfg())


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class TestBackendSelection:

    @pytest.fixture
    def no_sentence_transformers(self, monkeypatch):
        import sentence_transformers

        def fail(*args, **kwargs):
            raise AssertionError("SentenceTransformer should not be used")

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)

    def test_transformers_backend_skips_sentence_transformers(self, decoder_dir, no_sentence_transformers):
        embedding(_TEXTS, decoder_dir, cfg=_cfg(pooling="mean"))

    def test_auto_with_non_mean_pooling_skips_sentence_transformers(
        self, decoder_dir, no_sentence_transformers,
    ):
        embedding(_TEXTS, decoder_dir, cfg=_cfg(embedding_backend="auto"))

    def test_auto_with_non_mean_pooling_warns_that_results_changed(self, decoder_dir, caplog):
        with caplog.at_level(logging.WARNING, logger="shared.embeddings.compute_embeddings"):
            embedding(_TEXTS, decoder_dir, cfg=_cfg(embedding_backend="auto"))
        assert "differ from runs of the same config with earlier versions" in caplog.text

    def test_explicit_transformers_backend_does_not_warn(self, decoder_dir, caplog):
        with caplog.at_level(logging.WARNING, logger="shared.embeddings.compute_embeddings"):
            embedding(_TEXTS, decoder_dir, cfg=_cfg())
        assert "earlier versions" not in caplog.text

    def test_auto_with_mean_pooling_tries_sentence_transformers_first(self, decoder_dir, monkeypatch):
        import sentence_transformers

        calls = []

        def fail(*args, **kwargs):
            calls.append(args)
            raise RuntimeError("not a sentence-transformers model")

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)
        cfg = _cfg(embedding_backend="auto", pooling="mean", hf_local_files_only=False)
        out = embedding(_TEXTS, decoder_dir, cfg=cfg)
        assert calls and len(out) == len(_TEXTS)

    def test_sentence_transformers_backend_does_not_fall_back(self, decoder_dir, monkeypatch):
        import sentence_transformers

        def fail(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail)
        cfg = _cfg(embedding_backend="sentence_transformers", pooling="mean")
        with pytest.raises(RuntimeError, match="embedding_backend is 'sentence_transformers'"):
            embedding(_TEXTS, decoder_dir, cfg=cfg)

    @pytest.mark.parametrize("cfg, match", [
        ({"embedding_backend": "vllm"}, "embedding_backend"),
        ({"embedding_backend": "sentence_transformers", "pooling": "last_token"}, "own pooling"),
        ({"embedding_dtype": "int8"}, "embedding_dtype"),
        ({"pooling": "cls"}, "pooling"),
    ])
    def test_invalid_settings_raise(self, cfg, match):
        with pytest.raises(ValueError, match=match):
            embedding(_TEXTS, "unused/model", cfg=cfg)
