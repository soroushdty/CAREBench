from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Mapping


import numpy as np
import pandas as pd

from tqdm import TqdmWarning

try:
    import torch
    from google.colab import _ as _colab_marker  # type: ignore
    from tqdm.auto import tqdm
except ImportError:
    try:
        import torch
    except ImportError:
        torch = None
    from tqdm import tqdm

warnings.filterwarnings("ignore", message="IProgress not found.*", category=TqdmWarning)

logger = logging.getLogger(__name__)

_VALID_POOLING = {"mean", "max", "none", "attention-weighted", "simcse", "last_token"}
_VALID_BACKENDS = {"auto", "sentence_transformers", "transformers"}
_VALID_DTYPES = {"float32", "float16", "bfloat16"}


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _build_pooling_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sep_id: int | None,
    pad_id: int | None,
) -> torch.Tensor:
    token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if sep_id is not None:
        token_mask &= input_ids != sep_id
    if pad_id is not None:
        token_mask &= input_ids != pad_id
    return attention_mask.bool() & token_mask


def _mean_pooling(last_hidden_state: torch.Tensor, pooling_mask: torch.Tensor) -> torch.Tensor:
    mask = pooling_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _max_pooling(last_hidden_state: torch.Tensor, pooling_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~pooling_mask.unsqueeze(-1), float("-inf"))
    pooled, _ = torch.max(masked, dim=1)
    pooled[torch.isinf(pooled)] = 0.0
    return pooled


def _last_token_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden state of each sequence's last non-padding token.

    Under causal attention only the last token has seen the whole input, so
    this is the pooling for decoder models. Works for left or right padding.
    """
    seq_len = attention_mask.size(1)
    last = seq_len - 1 - attention_mask.flip(dims=[1]).int().argmax(dim=1)
    rows = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[rows, last]


def _is_causal_lm(model: Any) -> bool:
    """True when the checkpoint was saved as a causal (decoder) language model."""
    architectures = getattr(model.config, "architectures", None) or []
    return any("CausalLM" in a or "LMHead" in a for a in architectures)


def _attention_weighted_pooling(
    last_hidden_state: torch.Tensor,
    last_layer_attentions: torch.Tensor,
    pooling_mask: torch.Tensor,
) -> torch.Tensor:
    token_weights = last_layer_attentions.mean(dim=1).mean(dim=1)
    token_weights = token_weights.masked_fill(~pooling_mask, 0.0)
    token_weights = token_weights / token_weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
    return torch.sum(last_hidden_state * token_weights.unsqueeze(-1), dim=1)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-9, None)


def _is_valid_patient_id(key: str) -> bool:
    """True if key is an integer or a float that represents a whole number."""
    try:
        val = float(key)
        return val == int(val)
    except (ValueError, TypeError):
        return False


def _to_patient_id(key: str) -> int:
    return int(float(key))


def _is_self_key_placeholder(category: Any, text: str) -> bool:
    """True when a value is just repeating its own category key."""
    category_text = str(category).strip()
    return bool(category_text) and text.casefold() == category_text.casefold()


def _warn_missing_context_value(category: Any) -> None:
    logger.warning("Key '%s' has no corresponding value", category)


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def embedding(
    texts: list[str],
    model_id: str,
    batch_size: int = 32,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """
    Embed a flat list of pre-normalized raw strings.

    Handles its own deduplication. Returns {text: embedding_vector}.
    Texts are assumed to be pre-normalized — no transformation is applied.

    Args:
        texts:      Flat list of raw strings to embed.
        model_id:   HuggingFace model ID or local path.
        batch_size: Batch size for encoding.
        cfg:        Optional config with keys:
                        llm_revision            (str)
                        hf_local_files_only     (bool)
                        tokenizer_model_max_length (int, fallback default: 512)
                        pooling                 (str; default: mean)
                            last_token: hidden state of the last non-padding
                            token; use it for decoder models.
                        embedding_backend       (str; default: auto)
                            auto: SentenceTransformer when pooling is mean,
                            falling back to transformers if it fails;
                            transformers otherwise.
                            sentence_transformers: SentenceTransformer only;
                            the model's own pooling applies.
                            transformers: AutoModel with cfg['pooling'].
                        embedding_dtype         (str; float32 | float16 | bfloat16;
                                                 default: the checkpoint's loading default)
                        embedding_device_map    (str or dict, passed to from_pretrained;
                                                 default: load on one device)
                        # SimCSE: Gao, Tianyu, Xingcheng Yao, and Danqi Chen.
                        # "SimCSE: Simple Contrastive Learning of Sentence Embeddings."
                        # Version 4. Preprint, arXiv, 2021.
                        # https://doi.org/10.48550/ARXIV.2104.08821

    Returns:
        dict mapping each unique input string to its L2-normalized embedding vector.
    """
    if not texts:
        logger.warning("embedding() received an empty text list. Returning empty dict.")
        return {}

    if cfg is None:
        cfg = {}

    pooling = str(cfg.get("pooling", "mean"))
    if pooling not in _VALID_POOLING:
        valid = ", ".join(sorted(_VALID_POOLING))
        raise ValueError(f"Invalid cfg['pooling']={pooling!r}. Valid options: {valid}.")

    backend = str(cfg.get("embedding_backend") or "auto")
    if backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"Invalid cfg['embedding_backend']={backend!r}. Valid options: {valid}.")
    if backend == "sentence_transformers" and pooling != "mean":
        raise ValueError(
            f"cfg['pooling']={pooling!r} needs embedding_backend 'transformers' or 'auto': "
            "the sentence_transformers backend uses the model's own pooling."
        )

    dtype_name = cfg.get("embedding_dtype")
    if dtype_name is not None and dtype_name not in _VALID_DTYPES:
        valid = ", ".join(sorted(_VALID_DTYPES))
        raise ValueError(f"Invalid cfg['embedding_dtype']={dtype_name!r}. Valid options: {valid}.")
    device_map = cfg.get("embedding_device_map")

    unique_texts = sorted({t for t in texts if isinstance(t, str) and t.strip()})
    if not unique_texts:
        logger.warning("embedding(): no valid strings remain after deduplication.")
        return {}


    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_revision = cfg.get("llm_revision")
    local_files_only = bool(cfg.get("hf_local_files_only", False))

    # Loading options only passed when configured, so the default load is unchanged.
    model_kwargs: dict[str, Any] = {}
    if dtype_name is not None:
        model_kwargs["dtype"] = getattr(torch, dtype_name)
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    logger.info("Computing embeddings — model: %s | unique texts: %d", model_id, len(unique_texts))

    use_sentence_transformers = backend == "sentence_transformers" or (
        backend == "auto" and pooling == "mean"
    )

    # --- primary path: SentenceTransformer ---
    if not use_sentence_transformers:
        logger.info(
            "Loading %s with transformers AutoModel (embedding_backend=%s, pooling=%s).",
            model_id, backend, pooling,
        )
    else:
        try:
            logger.info("Loading %s as SentenceTransformer...", model_id)
            st_kwargs: dict[str, Any] = {}
            if model_kwargs:
                st_kwargs["model_kwargs"] = model_kwargs
            model = SentenceTransformer(
                model_id,
                device=None if device_map is not None else str(device),
                revision=llm_revision,
                local_files_only=local_files_only,
                **st_kwargs,
            )
            embeddings = model.encode(
                unique_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return {text: embeddings[i] for i, text in enumerate(unique_texts)}

        except Exception as e:
            if backend == "sentence_transformers":
                raise RuntimeError(
                    f"SentenceTransformer failed to load or encode '{model_id}' and "
                    f"embedding_backend is 'sentence_transformers'. Original error: {e}"
                ) from e
            if local_files_only:
                raise RuntimeError(
                    f"Model load failed in local-only mode for '{model_id}' "
                    f"(revision='{llm_revision}'). Ensure artifacts are pre-cached, "
                    f"or set embedding_backend: transformers. Original error: {e}"
                ) from e

            logger.warning("SentenceTransformer failed (%s). Falling back to AutoModel...", e)

    # --- fallback path: AutoModel + manual pooling ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        revision=llm_revision,
        local_files_only=local_files_only,
    )
    safe_max = min(tokenizer.model_max_length, 512)
    if tokenizer.model_max_length != safe_max:
        logger.warning(
            f"Tokenizer for '{model_id}' reported model_max_length="
            f"{tokenizer.model_max_length}, which exceeds the safe threshold. "
            f"Capping at {safe_max}. Override via cfg['tokenizer_model_max_length']."
        )
    tokenizer.model_max_length = cfg.get("tokenizer_model_max_length", safe_max)

    # Many decoder tokenizers have no padding token; pad with EOS. Pad on the
    # right so real tokens keep positions 0..n-1 whatever the batch.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer for '{model_id}' has neither a padding nor an EOS token; "
                "batched embedding needs one of them."
            )
        logger.info("Tokenizer for '%s' has no padding token; using EOS (%r).", model_id, tokenizer.eos_token)
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    model = AutoModel.from_pretrained(
        model_id,
        revision=llm_revision,
        local_files_only=local_files_only,
        **model_kwargs,
    )
    if device_map is None:
        model.to(device)
    input_device = model.get_input_embeddings().weight.device
    if pooling == "none" and _is_causal_lm(model):
        logger.warning(
            "pooling='none' takes the first token, which in a decoder model has seen "
            "nothing else in the input. Use pooling='last_token' for '%s'.", model_id,
        )
    if pooling == "simcse":
        model.train()
    else:
        model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts), batch_size), desc=f"AutoModel {model_id}", leave=False):
            batch = unique_texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(input_device)
            attention_mask = enc["attention_mask"].to(input_device)
            pooling_mask = _build_pooling_mask(input_ids, attention_mask, sep_id, pad_id)

            if pooling == "none":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = outputs.last_hidden_state[:, 0, :]

            elif pooling == "last_token":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = _last_token_pooling(outputs.last_hidden_state, attention_mask)

            elif pooling == "max":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = _max_pooling(outputs.last_hidden_state, pooling_mask)

            elif pooling == "attention-weighted":
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_attentions=True,
                )
                pooled = _attention_weighted_pooling(
                    outputs.last_hidden_state,
                    outputs.attentions[-1],
                    pooling_mask,
                )

            elif pooling == "simcse":
                outputs_1 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                outputs_2 = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled_1 = _mean_pooling(outputs_1.last_hidden_state, pooling_mask)
                pooled_2 = _mean_pooling(outputs_2.last_hidden_state, pooling_mask)
                pooled = (pooled_1 + pooled_2) / 2.0

            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                pooled = _mean_pooling(outputs.last_hidden_state, pooling_mask)

            all_embeddings.extend(pooled.float().cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    stacked = _l2_normalize(np.vstack(all_embeddings))
    return {text: stacked[i] for i, text in enumerate(unique_texts)}


# ---------------------------------------------------------------------------
# Schema-specific apply functions
# ---------------------------------------------------------------------------

def apply_to_df(
    series: pd.Series,
    model_id: str,
    batch_size: int = 32,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """
    Patient-agnostic embedding of a flat pandas Series column.

    Extracts unique non-null strings, embeds them, and returns a flat mapping.
    The output is deliberately blind to patient identity and category context.

    Args:
        series:     A pandas Series of raw pre-normalized strings.
        model_id:   HuggingFace model ID or local path.
        batch_size: Batch size for encoding.
        cfg:        Optional config forwarded to embedding().

    Returns:
        {unique_raw_string: embedding_array}
    """
    texts = (
        series.dropna()
              .astype(str)
              .str.strip()
              .loc[lambda s: s != ""]
              .unique()
              .tolist()
    )

    if not texts:
        logger.warning("apply_to_df: no valid strings found in the provided Series.")
        return {}

    emb_map = embedding(texts, model_id, batch_size, cfg=cfg)
    return emb_map


def apply_to_json(
    data: dict | str | Path,
    model_id: str,
    batch_size: int = 32,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Patient-aware and context-aware embedding of a nested patient JSON.

    Expected input schema:
        {
            "<patient_id>": {
                "<category>": <str> | <list[str]>,
                ...
            },
            ...
        }

    Rules:
        - Top-level keys must be numeric patient IDs (int or float-as-int string).
          Violating keys are logged and raise ValueError — partial output is never returned.
        - Patient IDs are cast to int in the output.
        - Category keys (second level) are preserved as-is.
        - Bare string values (e.g. "summary") → {text: embedding}.
        - List values → {item: embedding, ...} for each non-empty item.
        - Empty list or blank string → null (None) in output.

    All unique strings across the entire document are embedded in a single model
    pass for efficiency — the structure is reconstructed afterward from the cache.

    Args:
        data:       Patient JSON as a dict, or a file path to a .json file.
        model_id:   HuggingFace model ID or local path.
        batch_size: Batch size for encoding.
        cfg:        Optional config forwarded to embedding().

    Returns:
        Nested dict with the same structure as input, embeddings at leaf level.
    """
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)

    # --- validate all top-level keys before doing any work ---
    invalid_keys = [k for k in data if not _is_valid_patient_id(str(k))]
    if invalid_keys:
        msg = (
            f"apply_to_json: top-level keys must be numeric patient IDs "
            f"(int or float-as-int). Invalid keys: {invalid_keys}"
        )
        logger.error(msg)
        raise ValueError(msg)

    # --- single-pass collection of all unique embeddable strings ---
    all_texts: set[str] = set()
    for record in data.values():
        for category, value in record.items():
            if isinstance(value, str) and value.strip():
                text = value.strip()
                if not _is_self_key_placeholder(category, text):
                    all_texts.add(text)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        item_text = item.strip()
                        if not _is_self_key_placeholder(category, item_text):
                            all_texts.add(item_text)

    if all_texts:
        # --- embed all unique strings in one model pass ---
        emb_cache = embedding(list(all_texts), model_id, batch_size, cfg=cfg)
    else:
        logger.warning(
            "apply_to_json: no embeddable strings found in the JSON; returning null-only structure."
        )
        emb_cache = {}

    # --- reconstruct output structure ---
    output: dict[int, dict[str, Any]] = {}

    for raw_key, record in data.items():
        patient_id = _to_patient_id(str(raw_key))
        output[patient_id] = {}

        for category, value in record.items():
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    output[patient_id][category] = {str(category): None}
                elif _is_self_key_placeholder(category, text):
                    _warn_missing_context_value(category)
                    output[patient_id][category] = {str(category): None}
                else:
                    output[patient_id][category] = (
                        {text: emb_cache[text]}
                    )

            elif isinstance(value, list):
                valid_items: list[str] = []
                saw_placeholder = False
                for item in value:
                    if not isinstance(item, str):
                        continue
                    item_text = item.strip()
                    if not item_text:
                        continue
                    if _is_self_key_placeholder(category, item_text):
                        saw_placeholder = True
                        continue
                    valid_items.append(item_text)

                if saw_placeholder and not valid_items:
                    _warn_missing_context_value(category)

                output[patient_id][category] = (
                    {str(category): None} if not valid_items
                    else {item: emb_cache[item] for item in valid_items}
                )

            else:
                output[patient_id][category] = {str(category): None}

    return output


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

def compute_embeddings(
    model_id: str,
    data: dict | pd.Series | str | Path,
    *,
    schema: str,
    batch_size: int = 32,
    cfg: Mapping[str, Any] | None = None,
) -> dict:
    """
    Dispatch embedding computation based on the declared data schema.

    Args:
        model_id:   HuggingFace model ID or local path.
        data:       pd.Series for schema='pandas'; dict or JSON path for schema='json'.
        schema:     REQUIRED. Either 'pandas' or 'json'. No default.
        batch_size: Batch size forwarded to embedding().
        cfg:        Optional config dict forwarded to embedding().

    Returns:
        For 'pandas': {unique_string: embedding_array}  (patient/context agnostic)
        For 'json':   {patient_id_int: {category: {string: embedding_array} | null}}

    Raises:
        ValueError:  If schema is not 'pandas' or 'json', or JSON has invalid patient IDs.
        TypeError:   If data type is incompatible with the declared schema.
    """
    if schema == "pandas":
        if not isinstance(data, pd.Series):
            raise TypeError(
                f"schema='pandas' requires a pd.Series, got {type(data).__name__}."
            )
        return apply_to_df(data, model_id, batch_size, cfg=cfg)

    elif schema == "json":
        if isinstance(data, pd.Series):
            raise TypeError(
                "schema='json' does not accept a pd.Series. Pass a dict or a file path."
            )
        return apply_to_json(data, model_id, batch_size, cfg=cfg)

    else:
        raise ValueError(
            f"Invalid schema='{schema}'. Must be one of: 'json', 'pandas'."
        )
