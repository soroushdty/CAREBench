from __future__ import annotations

import logging

from shared.embeddings.compute_embeddings import apply_to_json


def test_apply_to_json_self_key_string_becomes_none_and_warns(monkeypatch, caplog):
    def should_not_embed(*args, **kwargs):
        raise AssertionError("embedding() should not be called for self-key placeholder-only input")

    monkeypatch.setattr("shared.embeddings.compute_embeddings.embedding", should_not_embed)

    payload = {"3": {"procedures": "procedures"}}

    with caplog.at_level(logging.WARNING):
        output = apply_to_json(payload, "dummy/model", cfg={})

    assert output == {3: {"procedures": {"procedures": None}}}
    assert any("Key 'procedures' has no corresponding value" in msg for msg in caplog.messages)


def test_apply_to_json_self_key_list_becomes_none_and_warns(monkeypatch, caplog):
    def should_not_embed(*args, **kwargs):
        raise AssertionError("embedding() should not be called for self-key placeholder-only input")

    monkeypatch.setattr("shared.embeddings.compute_embeddings.embedding", should_not_embed)

    payload = {"3": {"procedures": ["procedures"]}}

    with caplog.at_level(logging.WARNING):
        output = apply_to_json(payload, "dummy/model", cfg={})

    assert output == {3: {"procedures": {"procedures": None}}}
    assert any("Key 'procedures' has no corresponding value" in msg for msg in caplog.messages)


def test_apply_to_json_empty_list_becomes_null_dict(monkeypatch):
    def should_not_embed(*args, **kwargs):
        raise AssertionError("embedding() should not be called for empty-list input")

    monkeypatch.setattr("shared.embeddings.compute_embeddings.embedding", should_not_embed)

    payload = {"2": {"radiology": []}}
    output = apply_to_json(payload, "dummy/model", cfg={})

    assert output == {2: {"radiology": {"radiology": None}}}


def test_apply_to_json_regular_value_still_embeds(monkeypatch):
    def fake_embedding(texts, model_id, batch_size, cfg=None):
        return {text: [float(idx)] for idx, text in enumerate(texts)}

    monkeypatch.setattr("shared.embeddings.compute_embeddings.embedding", fake_embedding)

    payload = {"1": {"procedures": "Endoscopy"}}
    output = apply_to_json(payload, "dummy/model", cfg={})

    assert output == {1: {"procedures": {"Endoscopy": [0.0]}}}
