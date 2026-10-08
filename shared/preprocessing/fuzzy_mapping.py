"""Item standardization via graph-based fuzzy clustering with multi-key blocking."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable

import pandas as pd
from shared.utils.text_utils import normalize_for_matching

logger = logging.getLogger(__name__)

try:
    from rapidfuzz.fuzz import ratio as rf_ratio
except ImportError:
    rf_ratio = None


def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = normalize_for_matching(s, strip=False)
    s = re.sub(r"[\/\\,;•·]", " ", s)
    s = re.sub(r"[^\w\s\(\)\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", s)


def token_sort_key(s: str) -> str:
    tokens = tokenize(s)
    tokens.sort()
    return " ".join(tokens)


def fuzzy_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    a_tok = token_sort_key(a)
    b_tok = token_sort_key(b)

    if rf_ratio is not None:
        score_tok = rf_ratio(a_tok, b_tok) / 100.0
        score_raw = rf_ratio(a, b) / 100.0
    else:
        from difflib import SequenceMatcher

        score_tok = SequenceMatcher(None, a_tok, b_tok).ratio()
        score_raw = SequenceMatcher(None, a, b).ratio()

    return 0.65 * score_tok + 0.35 * score_raw


def title_keep_paren(text: str) -> str:
    m = re.match(r"^(.*?)(\s*\(.*\))?$", text.strip())
    if not m:
        return text.title()
    left, paren = m.group(1), m.group(2) or ""
    return left.title().strip() + (f" {paren.strip()}" if paren else "")


def choose_representative_original(originals: list[str]) -> str:
    """
    Deterministic representative selection:
    1) highest frequency
    2) longest string
    3) lexicographically smallest normalized form
    4) lexicographically smallest original string
    """
    counter = Counter(originals)

    best = min(
        counter.keys(),
        key=lambda s: (
            -counter[s],
            -len(s),
            normalize_text(s),
            s,
        ),
    )
    return title_keep_paren(best)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def length_bucket_key(s: str, width: int = 5) -> int:
    """
    Bucket normalized strings by length to avoid exact-length fragmentation.
    """
    if width < 1:
        raise ValueError("width must be at least 1.")
    return len(s) // width


def prefixes_for_blocking(s: str) -> set[str]:
    """
    Generate short prefix keys from the normalized string.
    Includes the whole string if shorter than 3 chars.
    """
    if not s:
        return set()
    if len(s) <= 3:
        return {s}
    return {s[:3]}


def shared_token_block_keys(
    s: str,
    *,
    min_token_len: int = 3,
    stop_tokens: set[str] | None = None,
) -> set[str]:
    """
    Tokens used for inverted-index blocking.
    """
    toks = set(tokenize(s))
    keys: set[str] = set()
    for tok in toks:
        if len(tok) < min_token_len:
            continue
        if stop_tokens and tok in stop_tokens:
            continue
        keys.add(tok)
    return keys


def sorted_token_first_key(s: str) -> str | None:
    toks = tokenize(s)
    if not toks:
        return None
    toks.sort()
    return toks[0]


def first_token_key(s: str) -> str | None:
    toks = tokenize(s)
    return toks[0] if toks else None


def char_trigram_keys(s: str) -> set[str]:
    """
    Optional extra blocking key to improve recall for near-matches
    with weak token overlap.
    """
    compact = re.sub(r"\s+", "", s)
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def compute_stop_tokens(normalized: list[str], max_df_ratio: float = 0.25) -> set[str]:
    """
    Drop overly common tokens from shared-token blocking to prevent giant blocks.
    """
    if not normalized:
        return set()

    doc_freq: Counter[str] = Counter()
    for s in normalized:
        doc_freq.update(set(tokenize(s)))

    max_df = max(2, math.ceil(len(normalized) * max_df_ratio))
    return {tok for tok, df in doc_freq.items() if df > max_df}


def _yield_pairs_from_block(indices: list[int]) -> Iterable[tuple[int, int]]:
    m = len(indices)
    if m < 2:
        return
    sorted_indices = sorted(indices)
    for i in range(m):
        a = sorted_indices[i]
        for j in range(i + 1, m):
            b = sorted_indices[j]
            yield (a, b)


def candidate_pairs(
    normalized: list[str],
    *,
    length_bucket_width: int = 5,
    max_block_size: int = 200,
    use_char_trigrams: bool = True,
) -> Iterable[tuple[int, int]]:
    """
    Generate plausible candidate pairs using multiple blocking keys.

    Blocking strategies:
    - shared informative token
    - same first 3 normalized characters
    - same length bucket
    - same sorted-token first token
    - same first token
    - optional shared character trigram

    Notes
    -----
    - Large blocks are skipped to avoid recreating an O(N^2) hotspot inside a block.
    - Pairs are deduplicated globally across all blocking strategies.
    """
    stop_tokens = compute_stop_tokens(normalized)

    blocks: dict[tuple[str, Any], list[int]] = defaultdict(list)

    for idx, s in enumerate(normalized):
        if not s:
            continue

        # 1) shared informative token
        for tok in shared_token_block_keys(s, stop_tokens=stop_tokens):
            blocks[("tok", tok)].append(idx)

        # 2) first 3 normalized characters
        for prefix in prefixes_for_blocking(s):
            blocks[("pfx3", prefix)].append(idx)

        # 3) length bucket
        blocks[("len", length_bucket_key(s, width=length_bucket_width))].append(idx)

        # 4) same sorted-token first token
        stf = sorted_token_first_key(s)
        if stf:
            blocks[("stf", stf)].append(idx)

        # Extra helpful key: first token as-written after normalization
        ft = first_token_key(s)
        if ft:
            blocks[("ft", ft)].append(idx)

        # Optional extra recall key
        if use_char_trigrams:
            for tri in char_trigram_keys(s):
                blocks[("tri", tri)].append(idx)

    seen_pairs: set[tuple[int, int]] = set()

    for block_key, indices in blocks.items():
        unique_indices = sorted(set(indices))
        if len(unique_indices) < 2:
            continue

        # Skip pathological blocks that are too large, especially common trigram/prefix groups.
        if len(unique_indices) > max_block_size:
            logger.debug(
                "Skipping oversized block %s with %d items.",
                block_key,
                len(unique_indices),
            )
            continue

        for pair in _yield_pairs_from_block(unique_indices):
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                yield pair


def make_items_auto_std(
    df: pd.DataFrame,
    item_col: str,
    similarity_threshold: float,
    min_cluster_size: int = 1,
    include_unique_items: bool = False,
) -> dict[str, list[str]]:
    """
    Cluster similar item strings using graph connected components and return:

        standardized_string -> [string variation 1, string variation 2, ...]

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    item_col : str
        Column containing item strings.
    similarity_threshold : float
        Similarity threshold. May be given as 0-1 or 0-100.
    min_cluster_size : int, default=1
        Minimum connected-component size to keep before applying the
        include_unique_items filter.
    include_unique_items : bool, default=False
        If False, only clusters with more than one unique variation are returned.
        If True, singleton clusters are also included.

    Returns
    -------
    dict[str, list[str]]
        Mapping of representative standardized string to list of observed variants.

    Notes
    -----
    - Only unique non-empty values from `item_col` are clustered.
    - Candidate comparisons are generated by multiple blocking keys rather than
      all-pairs comparison.
    - Two strings are linked if their fuzzy similarity exceeds the threshold,
      or if one contains the other and their similarity exceeds 75% of threshold.
    - Final clusters are connected components of that similarity graph.
    """
    if item_col not in df.columns:
        raise ValueError(f"Column '{item_col}' not found.")

    if similarity_threshold > 1:
        similarity_threshold = similarity_threshold / 100.0

    if not (0 <= similarity_threshold <= 1):
        raise ValueError(
            "similarity_threshold must be between 0 and 1, or between 0 and 100."
        )

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1.")

    # Collect unique non-empty originals in first-seen order
    seen: set[str] = set()
    unique_originals: list[str] = []

    for val in df[item_col]:
        if pd.isna(val):
            continue
        orig = str(val).strip()
        if not orig:
            continue
        if orig not in seen:
            seen.add(orig)
            unique_originals.append(orig)

    if not unique_originals:
        logger.info("No non-empty unique values found in column '%s'.", item_col)
        return {}

    originals = unique_originals
    normalized = [normalize_text(orig) for orig in originals]
    n = len(originals)

    uf = UnionFind(n)

    comparisons = 0
    linked = 0

    for i, j in candidate_pairs(
        normalized,
        length_bucket_width=5,
        max_block_size=200,
        use_char_trigrams=True,
    ):
        norm_i = normalized[i]
        norm_j = normalized[j]

        comparisons += 1
        sim = fuzzy_similarity(norm_i, norm_j)
        contains = bool(norm_i and norm_j and ((norm_i in norm_j) or (norm_j in norm_i)))

        if sim >= similarity_threshold or (
            contains and sim >= similarity_threshold * 0.75
        ):
            uf.union(i, j)
            linked += 1

    # Collect connected components from union-find
    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[uf.find(idx)].append(idx)

    components = [
        sorted(component)
        for component in groups.values()
        if len(component) >= min_cluster_size
    ]
    components.sort(key=lambda comp: comp[0])  # deterministic ordering

    result: dict[str, list[str]] = {}
    skipped_unique = 0

    for component in components:
        if not include_unique_items and len(component) == 1:
            skipped_unique += 1
            continue

        variations = [originals[idx] for idx in component]
        standard = choose_representative_original(variations)

        key = standard
        suffix = 1
        while key in result:
            key = f"{standard} ({suffix})"
            suffix += 1

        result[key] = variations

    logger.info(
        "Auto-standardization complete. Unique strings: %d | candidate comparisons: %d | linked pairs: %d | components: %d | returned: %d | unique skipped: %d",
        n,
        comparisons,
        linked,
        len(components),
        len(result),
        skipped_unique,
    )

    return result


def generate_fuzzy_mapping(
    df: pd.DataFrame,
    *,
    item_col: str,
    fuzzy_threshold: float,
) -> dict[str, list[str]]:
    """
    Generate grouped mapping for item standardization using fuzzy clustering.

    Returns
    -------
    dict[str, list[str]]
        Grouped mapping in the format:
        {
            "standardized item": ["variant 1", "variant 2", ...]
        }
    """
    return make_items_auto_std(
        df=df,
        item_col=item_col,
        similarity_threshold=fuzzy_threshold,
        include_unique_items=False,
    )


# ---------------------------------------------------------------------------
# Train-only runtime fuzzy matcher
# ---------------------------------------------------------------------------


class FuzzyMatcher:
    """
    Train-only fuzzy matcher built from a clustering of training item strings.

    Build phase  : cluster training strings via generate_fuzzy_mapping to get
                   {canonical: [variant, ...]} mapping.
    Query phase  : compare any query string against the normalized variants and
                   return the casefolded canonical when the best score exceeds
                   threshold.

    Held-out strings are NEVER added to the candidate variant space; they are
    only ever used as queries.
    """

    def __init__(
        self,
        canonical_to_variants: dict[str, list[str]],
        threshold: float,
    ) -> None:
        self._threshold = threshold if threshold <= 1.0 else threshold / 100.0
        # norm_variant → casefolded canonical (for consistency with JSON path)
        self._norm_variant_to_norm_canonical: dict[str, str] = {}
        for canonical, variants in canonical_to_variants.items():
            norm_c = normalize_for_matching(canonical)
            for v in variants:
                norm_v = normalize_text(v)
                self._norm_variant_to_norm_canonical[norm_v] = norm_c
        self._norm_variants: list[str] = list(self._norm_variant_to_norm_canonical.keys())

    def match(self, query: str) -> tuple[str | None, float]:
        """Return (casefolded_canonical, score) or (None, score) if below threshold."""
        if not self._norm_variants:
            return None, 0.0
        norm_q = normalize_text(query)
        best_score = 0.0
        best_norm_var: str | None = None
        for nv in self._norm_variants:
            s = fuzzy_similarity(norm_q, nv)
            if s > best_score:
                best_score = s
                best_norm_var = nv
        if best_score >= self._threshold and best_norm_var is not None:
            return self._norm_variant_to_norm_canonical[best_norm_var], best_score
        return None, best_score


def build_fuzzy_matcher_from_train(
    df: pd.DataFrame,
    item_col: str,
    threshold: float,
    unresolved_mask: "pd.Series | None" = None,
) -> FuzzyMatcher:
    """
    Build a FuzzyMatcher from training-only data.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame (post-JSON mapping).  Only training rows should
        be passed; held-out rows must never be included.
    item_col : str
        Item column name.
    threshold : float
        Fuzzy similarity threshold (0–1 or 0–100).
    unresolved_mask : pd.Series[bool] or None
        When provided only rows where the mask is True (unresolved by JSON)
        participate in fuzzy clustering.  When None all rows are used.
    """
    target_df = df[unresolved_mask] if unresolved_mask is not None else df
    canonical_to_variants = generate_fuzzy_mapping(
        df=target_df,
        item_col=item_col,
        fuzzy_threshold=threshold,
    )
    return FuzzyMatcher(canonical_to_variants, threshold)


def apply_fuzzy_fallback(
    df: pd.DataFrame,
    item_col: str,
    matcher: FuzzyMatcher,
    unresolved_mask: "pd.Series | None" = None,
) -> tuple["pd.DataFrame", list[str]]:
    """
    Apply fuzzy fallback to unresolved items in *df*.

    Only rows where *unresolved_mask* is True are processed.  JSON-resolved
    rows (mask=False) are never touched.  For each unresolved row:
    - If the matcher finds a canonical above threshold, replace the item value.
    - Otherwise emit a WARNING log and record the item in the returned list.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose item column may be partially unresolved.
    item_col : str
        Item column name.
    matcher : FuzzyMatcher
        Matcher built from training-only data.
    unresolved_mask : pd.Series[bool] or None
        Rows to process.  When None every row is processed.

    Returns
    -------
    (updated_df, unmapped_warnings)
        updated_df  : copy of df with fuzzy-mapped values applied.
        unmapped_warnings : list of warning strings for items that could not
                            be mapped (score < threshold).
    """
    out_df = df.copy()
    warnings_list: list[str] = []

    if unresolved_mask is not None:
        rows_to_process = df.index[unresolved_mask]
    else:
        rows_to_process = df.index

    for idx in rows_to_process:
        raw_val = df.at[idx, item_col]
        if not isinstance(raw_val, str) or not raw_val.strip():
            continue
        canonical, score = matcher.match(raw_val)
        if canonical is not None:
            out_df.at[idx, item_col] = canonical
        else:
            msg = (
                f"Fuzzy fallback: item at index {idx} ('{raw_val}') could not be mapped "
                f"(best score={score:.3f} < threshold={matcher._threshold:.3f})"
            )
            logger.warning(msg)
            warnings_list.append(msg)

    return out_df, warnings_list
