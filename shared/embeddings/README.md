# shared/embeddings/

**Canonical path:** `shared/embeddings/`

Embedding computation utilities shared across all evaluation tracks.

## Contents

- `compute_embeddings.py` — sentence/text embedding computation. `embedding()` embeds a list of strings; `compute_embeddings()` dispatches a pandas Series (items) or a patient JSON (context records) to it. Track 1 uses it for both item and context embeddings.

## Choosing the representation model

Track 1 takes its model from `llm` in `configs/main_config.yaml`. Both encoders (BERT-style) and decoder LLMs work. The keys that control embedding are:

| Key | Values | Default |
|-----|--------|---------|
| `llm` | Hugging Face model ID or local path | `emilyalsentzer/Bio_ClinicalBERT` |
| `llm_revision` | branch, tag or commit | `main` |
| `pooling` | `mean`, `max`, `none`, `attention-weighted`, `simcse`, `last_token` | `mean` |
| `embedding_backend` | `auto`, `transformers`, `sentence_transformers` | `auto` |
| `embedding_dtype` | `float32`, `float16`, `bfloat16`, or null | null (checkpoint default) |
| `embedding_device_map` | anything `from_pretrained` accepts as `device_map` (e.g. `auto`), or null | null (one device) |
| `tokenizer_model_max_length` | maximum tokens per text (longer texts are truncated) | 512 |
| `hf_local_files_only` | load only from the local cache | `false` |

### Backends

- **`auto`**: with `pooling: mean`, loads the model as a SentenceTransformer, and if that fails falls back to `transformers`. With any other pooling, uses `transformers` directly. The default; this is how Bio_ClinicalBERT has always been embedded.
- **`transformers`**: loads `AutoModel` and applies `pooling` to the last hidden state. Use this for decoder models. Its `mean` pooling leaves out `[SEP]`, while SentenceTransformer's includes it, so switching an encoder from `auto` to `transformers` changes its embeddings (for Bio_ClinicalBERT, cosine similarity about 0.94 on short items).
- **`sentence_transformers`**: SentenceTransformer only, no fallback. The model's own pooling configuration applies, so `pooling` must be `mean`.

### Pooling for encoders and decoders

| Pooling | What it takes | Suits |
|---------|---------------|-------|
| `mean` | Mean over real tokens (padding and `[SEP]` excluded) | Encoders |
| `max` | Elementwise max over real tokens | Encoders |
| `none` | The first token (`[CLS]` in BERT) | Encoders only. In a decoder the first token has seen nothing else in the input, so a warning is logged. |
| `attention-weighted` | Tokens weighted by last-layer attention | Encoders |
| `simcse` | Mean of two dropout-perturbed passes | Encoders |
| `last_token` | The last non-padding token | Decoders: under causal attention it is the only token that has seen the whole input |

When the tokenizer has no padding token (most decoder tokenizers), the `transformers` backend pads with the EOS token. It always pads on the right, so each text's tokens keep the same positions whatever else is in the batch, and a text gets the same vector alone or batched.

### Large models

A 7B-scale decoder does not fit in float32 on most single GPUs. Set `embedding_dtype: bfloat16` (or `float16`), and if needed `embedding_device_map: auto` to spread the model over devices. `device_map` needs the `accelerate` package, which is not in `requirements.txt`. Embeddings are returned as float32 whatever the loading dtype. Quantized loading is not supported.

`tokenizer_model_max_length` caps the tokens per text. 512 is ample for item strings but may truncate long context snapshots; raise it if the model allows.

### Example: a decoder model

```yaml
llm: meta-llama/Llama-3.1-8B-Instruct
pooling: last_token
embedding_backend: transformers
embedding_dtype: bfloat16
```

## Tested models

Track 1 on `examples/synthetic`, end to end (embedding, training, Stage 2, statistical analysis):

| Model | Type | Settings | Status |
|-------|------|----------|--------|
| `emilyalsentzer/Bio_ClinicalBERT` | encoder, d = 768 | defaults | Default; used for all reported results |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | decoder (random weights), d = 16 | `pooling: last_token`, `embedding_backend: transformers` | Runs end to end (pipeline check only) |
| `sshleifer/tiny-gpt2` | decoder (random weights), d = 2 | `pooling: last_token`, `embedding_backend: transformers` | Runs end to end (pipeline check only) |

The two decoders have random weights; they show that the pipeline handles decoder checkpoints, not that their embeddings mean anything. No full-size decoder has been run end to end yet, so Bio_ClinicalBERT stays the default.

The unit tests in `tests/preprocessing/test_embedding_decoders.py` build tiny GPT-2 and Llama checkpoints offline, with no padding token, and check last-token pooling, batch invariance, bfloat16 loading, and backend selection.
