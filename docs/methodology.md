# Methodology

This document states what CAREBench measures, the study design it assumes, how each endpoint is computed and tested, and which design decisions protect the results from known pitfalls. It describes the code as it is in this repository. Where the code falls short of the intended design, the gap is listed under [Known limitations](#known-limitations).

## The question

When a clinician sees an EHR item (a diagnosis, a lab result, a medication) on its own, they may file it under one sensitive-data category. With the patient's chart in view, they may file it differently. CAREBench asks whether a language model shows **the same context-induced shifts**:

1. Does the model's judgment change when patient context is added?
2. Does it change in the same direction as the human judgment?
3. Do the categories that shift most for humans also shift most for the model?
4. Does the alignment depend on the context belonging to *this* patient, or would any plausible clinical text produce it?

The last question is what separates *appropriate* context sensitivity from mere sensitivity. A model whose outputs move whenever a chart is pasted into the prompt is context-sensitive. It is only appropriately sensitive if the movement tracks the human shift and disappears when the chart belongs to someone else.

CAREBench does not measure classification accuracy. A model can be accurate and still ignore context, or shift correctly while being poorly calibrated.

## Study design the framework assumes

The built-in `paired_context` adapter ([`data_format.md`](data_format.md)) expects a **paired counterfactual design**:

- A set of **patients** (context entities), each with a structured clinical snapshot.
- For each patient, a set of **items** (task instances): EHR strings drawn from that patient's record.
- **Reference observers** (for example physicians) who label each item into the output categories twice: first **context-free** (item text only), then **correct-context** (item text plus the patient's snapshot). Each patient is assigned a fixed group of observers.

Other designs can be supported by writing another adapter ([`adapters.md`](adapters.md)); the endpoints below only need the canonical arrays the adapter produces.

### Unit of analysis and clustering

The **unit of analysis is the (patient, item) pair** after observer aggregation. Pairs from the same patient share a context snapshot and an observer group, so they are not independent. **The patient is the cluster**: confidence intervals resample whole patients, and the effective sample size for any between-patient claim is the number of patients, not the number of pairs.

### Reference labels and the reference delta

For each (patient, item) pair and category *c*, the reference label is the **mean of the observers' binary labels** (`shared/reference/aggregation/paired_reference_mean.py`). With two observers per patient, this takes values in {0, 0.5, 1}. **0.5 means the two observers disagreed.** It is not a probability of category membership. The aggregation fails loudly or deduplicates (with a warning) when a group's observer count differs from the configured count.

The reference delta is

```
Δ_ref(i, c) = y_correct_context(i, c) − y_context_free(i, c)      ∈ {−1, −0.5, 0, 0.5, 1}
```

In typical data most Δ_ref cells are zero. Endpoints that compare directions are therefore computed only on cells where Δ_ref ≠ 0.

### Model conditions and the model delta

Each candidate model scores each (patient, item) pair under up to three conditions:

| Condition | Input | Role |
|-----------|-------|------|
| `context_free` | Item text only | Baseline |
| `correct_context` | Item text + this patient's snapshot | Treatment |
| `shuffled_context` | Item text + a different patient's snapshot | Negative control |

The shuffled patient is drawn uniformly from the other patients, with a fixed seed (`shuffled_context_seed`); the mapping is saved for audit. Scores are per-category values in [0, 1]. The model delta is

```
Δ_model(i, c) = score_correct_context(i, c) − score_context_free(i, c)
```

and `Δ_shuffled` is defined the same way with the shuffled-context score.

## Tracks

Each track is a functional analogue of one way a physician could reach a context-dependent judgment. The analogy organizes the comparison; it is not a claim about how physicians think.

| Track | Physician analogue | How the model sees context | Status |
|-------|--------------------|----------------------------|--------|
| 1 – Representation | General clinical knowledge | A frozen clinical encoder embeds item and context separately; a small trained head combines them | Available |
| 2 – Adaptation | Learning from supervised cases | Fine-tuning on labelled (context, item) cases from training patients | Planned ([#13](https://github.com/soroushdty/CAREBench/issues/13)) |
| 3 – Reasoning | Deliberating over the chart | An LLM reads item and context in one prompt and returns category scores | Available |

**Track 1** (`tracks/representation/`) trains a context-free multilabel classifier on frozen encoder embeddings (Stage 1), then a regularized linear fusion head over the item embedding *e* and context embedding *c* (Stage 2). The fusion is configurable (`fusion_strategy`): `2d` = `[e, c]` (default), `3d` adds `e⊙c`, `4_vector` adds `|e−c|`, plus a low-rank bilinear option. The head is fitted to training patients' correct-context labels. All fitting uses nested leave-one-patient-out cross-validation, so every prediction is for a patient the model never saw. Stage 2 is trained only to fit the labels; it has no loss term that rewards agreement with the reference *delta*, so the delta endpoints are not optimized directly.

**Track 3** (`tracks/reasoning/`) prompts an LLM under all three conditions with deterministic prompts (default `temperature: 0`) and parses a JSON object of per-category scores. Responses are cached per model, condition, patient and item. There is no training.

## Endpoints

The two tracks currently number their hypotheses differently. The table maps them to the underlying questions. A shared, named set of endpoints is planned ([#15](https://github.com/soroushdty/CAREBench/issues/15)).

| Question | Track 3 | Track 1 |
|----------|---------|---------|
| Does context change the model's scores? | H1 | — |
| Does it change in the reference direction? | H2 | H1 |
| Do class-level shifts correspond? | H3 | — |
| Is the alignment specific to the correct patient? | H4 | — (needs a shuffled condition, [#11](https://github.com/soroushdty/CAREBench/issues/11)) |
| Does context bring scores closer to the correct-context labels? | — | H2 |

### Track 3 (`shared/evaluation/hypothesis_analyzer.py`)

| | Statistic | Uncertainty / test |
|---|---|---|
| **H1 – Context sensitivity** | Mean \|Δ_model\| over all cells, overall and per category | Patient-cluster bootstrap 95% CI |
| **H2 – Directional alignment** | On cells with Δ_ref ≠ 0: mean alignment `sign(Δ_ref)·Δ_model`, and sign-agreement rate `sign(Δ_model) = sign(Δ_ref)` | Patient-cluster bootstrap 95% CIs |
| **H3 – Class-level correspondence** | Pearson *r* between the per-category mean Δ_ref and mean Δ_model | One-sided permutation test over category labels |
| **H4 – Correct vs shuffled context** | On cells with Δ_ref ≠ 0: mean of `sign(Δ_ref)·(Δ_model − Δ_shuffled)` | Patient-cluster bootstrap 95% CI; one-sided paired permutation test |

A zero model delta counts as disagreement in the sign-agreement rate.

### Track 1 (`tracks/representation/statistical/`)

Confirmatory analyses are restricted to categories with at least `confirmatory_min_nonzero` (default 15) non-zero reference deltas. Other categories are reported descriptively.

| | Statistic | Uncertainty / test |
|---|---|---|
| **H1 – Directional alignment, per class** | Sign-agreement rate on cells with Δ_ref ≠ 0 | Exact one-sided binomial test against 0.5; Benjamini–Hochberg across eligible classes; patient-cluster bootstrap CI |
| **H1 – Directional alignment, pooled** | Sign-agreement rate pooled over eligible classes | Permutation test that shuffles model deltas among the items of the same patient (tests item-level specificity within patients); patient-cluster bootstrap CI |
| **H1 – Cross-class** | Mantel–Haenszel common odds ratio, stratified by class | CMH test |
| **H2 – Contextual alignment** | Per-class Brier improvement `Brier(context_free) − Brier(correct_context)` against correct-context labels | One-sided Wilcoxon signed-rank; Benjamini–Hochberg; patient-cluster bootstrap CI; macro summary |

Secondary and descriptive outputs: Wasserstein distance to the label distribution, ICC(2,1) and Lin's CCC of the model against each observer compared with observer–observer agreement, expected calibration error, context-induced entropy change, repeated vs novel item strata, per-field context ablation, and fusion-architecture comparison.

## Inference

- **Bootstrap.** All confidence intervals are percentile intervals from a patient-cluster bootstrap (`shared/statistical/bootstrap.py`): whole patients are resampled with replacement, default 1,000 resamples. With fewer than two patients the CI is reported as NaN.
- **Permutation tests.** Default 10,000 permutations. p-values are one-sided, in the direction of the hypothesis.
- **Multiplicity.** Track 1 applies Benjamini–Hochberg (q = 0.05) across classes. Track 3 reports per-category results descriptively and does not correct them.
- **Seeds.** Bootstrap, permutation, shuffled-context and generation seeds are set in config and recorded in the run manifest. `PYTHONHASHSEED` must be set before Python starts (see the README).

## Design decisions and what they guard against

Most of the machinery in CAREBench exists because a simpler version gave a misleading answer during development. Each row names the pitfall and the safeguard.

| Pitfall | Safeguard |
|---------|-----------|
| Treating (patient, item) pairs as independent understates uncertainty, because pairs from one patient share context and observers | Patient-cluster bootstrap for every CI; patient-grouped cross-validation in Track 1 |
| Training a model on the quantity you then test it on (for example, a loss that rewards matching the reference delta direction) makes the endpoint circular | Track 1's Stage 2 fits labels only; Track 3 involves no training |
| Mistaking any response to extra text for a response to *this patient's* context | Shuffled-context control and H4 |
| The item's own text appearing in the context snapshot, so the "context" effect is really the item being repeated | `LabelLeakageError`: a context build fails if the item text appears in any context field, unless the check is explicitly disabled |
| Text-standardization rules learned from evaluation items leak information into training | Fuzzy item matching is built from training items only (`build_fuzzy_matcher_from_train`) |
| An item string seen during training (under a different patient) is easier than a novel one | Track 1 reports results separately for repeated and novel items |
| Reading an aggregated 0.5 label as a calibrated probability | 0.5 is documented and handled as observer disagreement; delta endpoints compare directions, not magnitudes |
| Mismatched or duplicated observer rows silently distorting the reference | The adapter checks that context-free and correct-context (patient, observer, item) triplets correspond and that each pair has the configured observer count. With `mismatch_error: true` a mismatch stops the run; otherwise the rows are reported and skipped |
| Claiming a model is "indistinguishable from a physician" because its agreement falls inside the human range | ICC is reported descriptively next to the observer–observer distribution; no equivalence claim is made |
| Over-reading categories with very few shifts | Track 1's confirmatory eligibility threshold |

## Scope of claims

Results are claims about **the patients and reference observers in the evaluated dataset**. With few patients, between-patient generalization is weak by construction, and the cluster bootstrap makes that visible in the interval widths. A null result is informative: the assay is designed to show the absence of appropriate context sensitivity as readily as its presence.

## Known limitations

These are known gaps between the intended design and the current code.

- **H4's permutation test is not cluster-aware.** It flips each cell's correct/shuffled pair independently, which is equivalent to an unclustered sign-flip test and can understate the p-value when cells within a patient are correlated. Its bootstrap CI is clustered. ([#16](https://github.com/soroushdty/CAREBench/issues/16))
- **Some Track 1 p-values treat cells as independent.** The per-class binomial test, the CMH test, and the per-class Wilcoxon test do not account for clustering by patient; their bootstrap CIs do. ([#16](https://github.com/soroushdty/CAREBench/issues/16))
- **Track 3's H1 and H2 have intervals but no reference point.** Mean \|Δ_model\| is above zero for almost any model that reads the context at all, and the chance level of the sign-agreement rate is not 0.5 when zero deltas count as disagreement. The shuffled-context condition provides the natural reference for both. ([#17](https://github.com/soroushdty/CAREBench/issues/17))
- **H3 is a correlation over the number of categories** (ten by default), so it has little power and is best read descriptively.
- **No simulation-based validation yet.** The false-positive rate and power of the endpoints under known effects have not been measured. ([#18](https://github.com/soroushdty/CAREBench/issues/18))
- **One label taxonomy** ([#5](https://github.com/soroushdty/CAREBench/issues/5)) and **one dataset format** ([#6](https://github.com/soroushdty/CAREBench/issues/6)) so far.
