"""Tests for shared.statistical.cluster_tests.cluster_sign_flip_test (#16)."""
from __future__ import annotations

import numpy as np
import pytest

from shared.statistical.cluster_tests import cluster_sign_flip_test


def _cell_level_sign_flip(values, n_permutations, rng):
    """The old, unclustered test: every cell gets its own random sign."""
    observed = values.sum()
    signs = rng.choice(np.array([1.0, -1.0]), size=(n_permutations, values.size))
    null = signs @ values
    return (1 + np.sum(null >= observed)) / (n_permutations + 1)


def test_exact_p_matches_hand_enumeration():
    # Cluster sums are 2, 1, -1; observed total 2. Of the 8 sign patterns,
    # totals are 2, 4, 0, 2, -2, 0, -4, -2, so 3 of 8 are >= 2.
    values = np.array([1.0, 1.0, 1.0, -1.0])
    ids = np.array(["a", "a", "b", "c"])
    assert cluster_sign_flip_test(values, ids) == pytest.approx(3 / 8)


def test_floor_is_one_over_two_to_the_clusters():
    # Three patients, each with many perfectly positive cells: a cell-level
    # test would give p ~ 0, but with three clusters the exact p is 1/8.
    ids = np.repeat(["p1", "p2", "p3"], 50)
    values = np.ones(ids.size)
    assert cluster_sign_flip_test(values, ids) == pytest.approx(1 / 8)


def test_fewer_than_two_clusters_is_nan():
    assert np.isnan(cluster_sign_flip_test(np.ones(5), np.zeros(5)))
    assert np.isnan(cluster_sign_flip_test(np.array([]), np.array([])))


def test_all_zero_is_one():
    assert cluster_sign_flip_test(np.zeros(6), np.repeat([1, 2, 3], 2)) == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cluster_sign_flip_test(np.ones(3), np.ones(4))


def test_monte_carlo_is_seeded_and_close_to_exact():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(12), 5)
    values = rng.normal(0.1, 1.0, ids.size)
    exact = cluster_sign_flip_test(values, ids, n_permutations=10_000)  # 4096 patterns
    mc1 = cluster_sign_flip_test(values, ids, n_permutations=2_000, rng=np.random.default_rng(3))
    mc2 = cluster_sign_flip_test(values, ids, n_permutations=2_000, rng=np.random.default_rng(3))
    assert mc1 == mc2
    assert mc1 == pytest.approx(exact, abs=0.03)


def test_detects_a_real_effect():
    rng = np.random.default_rng(1)
    ids = np.repeat(np.arange(20), 10)
    values = rng.normal(0.8, 1.0, ids.size)
    assert cluster_sign_flip_test(values, ids, rng=np.random.default_rng(2)) < 0.01


def test_false_positive_rate_under_within_patient_correlation():
    """With a shared patient effect and no true effect, the cluster test keeps
    its nominal level while the cell-level test does not."""
    rng = np.random.default_rng(16)
    n_sims, n_patients, n_cells = 300, 20, 10
    ids = np.repeat(np.arange(n_patients), n_cells)
    cluster_hits = cell_hits = 0
    for _ in range(n_sims):
        patient_effect = rng.normal(0.0, 1.0, n_patients)[ids]
        values = patient_effect + rng.normal(0.0, 0.3, ids.size)
        cluster_hits += cluster_sign_flip_test(values, ids, n_permutations=499, rng=rng) < 0.05
        cell_hits += _cell_level_sign_flip(values, 499, rng) < 0.05
    assert cluster_hits / n_sims <= 0.09
    # The cell-level null has variance ~sum(values**2) ~ 218, while the
    # observed sum has variance ~n_cells**2 * n_patients = 2000, so its
    # rejection rate is about P(Z > 1.645 / 3.03) ~ 0.29 instead of 0.05.
    assert cell_hits / n_sims >= 0.2


# ---------------------------------------------------------------------------
# Callers
# ---------------------------------------------------------------------------

def test_h4_permutation_flips_whole_patients():
    """Regression for #16: the old test flipped each cell independently and
    returned p = 0.0 here; with three patients the exact p is 1/8."""
    from shared.evaluation.hypothesis_analyzer import permutation_test_h4

    ids = np.repeat(["p1", "p2", "p3"], 50)
    p = permutation_test_h4(np.ones(ids.size), np.zeros(ids.size), ids,
                            n_permutations=10_000, rng=np.random.default_rng(0))
    assert p == pytest.approx(1 / 8)


def _track1_inputs():
    rng = np.random.default_rng(5)
    n_patients, per_patient, classes = 8, 12, ["A", "B"]
    pids = np.repeat(np.arange(n_patients), per_patient)
    dp = rng.choice([-0.5, 0.0, 0.5], size=(pids.size, 2))
    dm = np.sign(dp) * 0.2 + rng.normal(0, 0.1, dp.shape)
    return dp, dm, pids, classes


def test_h1_per_class_bh_uses_perm_p():
    from tracks.representation.statistical.hypotheses.h1 import (
        _bh_correct,
        h1_binomial_per_class,
    )

    dp, dm, pids, classes = _track1_inputs()
    df = h1_binomial_per_class(dp, dm, classes, classes, pids, n_resamples=50,
                               rng=np.random.default_rng(1),
                               n_permutations=999, null_rng=np.random.default_rng(2))
    assert {"null_rate", "perm_p", "binom_p", "cluster_p", "bh_adj_p"} <= set(df.columns)
    # Reference sign-flip test: 2**8 = 256 patterns -> exact; all 8 patients
    # agree, so p = 1/256.
    assert df["cluster_p"].tolist() == pytest.approx([1 / 256, 1 / 256])
    # Item-level agreement: no permutation reaches it.
    assert df["perm_p"].tolist() == pytest.approx([1 / 1000, 1 / 1000])
    assert df["bh_adj_p"].to_numpy() == pytest.approx(_bh_correct(df["perm_p"].to_numpy()))


def _drift_inputs():
    """Most reference shifts go up, and the model always moves up a little."""
    rng = np.random.default_rng(6)
    n_patients, per_patient, classes = 8, 12, ["A", "B"]
    pids = np.repeat(np.arange(n_patients), per_patient)
    dp = rng.choice([-0.5, 0.5, 0.5, 0.5], size=(pids.size, 2))
    dm = np.full(dp.shape, 0.1)
    dm[::7] = -0.1  # some movement, so the odds-ratio tables are not degenerate
    return dp, dm, pids, classes


def test_h1_per_class_drift_is_chance_not_evidence():
    from tracks.representation.statistical.hypotheses.h1 import h1_binomial_per_class

    dp, dm, pids, classes = _drift_inputs()
    df = h1_binomial_per_class(dp, dm, classes, classes, pids, n_resamples=50,
                               rng=np.random.default_rng(1),
                               n_permutations=999, null_rng=np.random.default_rng(2))
    # Agreement is well above 0.5, and the old test against 0.5 calls it significant ...
    assert (df["sign_agree_rate"] > 0.6).all()
    assert (df["cluster_p"] < 0.05).all()
    # ... but the chance rate is just as high, so the confirmatory test does not.
    assert df["null_rate"].to_numpy() == pytest.approx(df["sign_agree_rate"].to_numpy(), abs=0.05)
    assert (df["perm_p"] > 0.05).all()


def test_h1_cmh_permutation_p_ignores_drift():
    from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

    dp, dm, pids, classes = _drift_inputs()
    res = h1_cmh_test(dp, dm, classes, classes, patient_ids=pids,
                      n_permutations=999, null_rng=np.random.default_rng(2))
    assert res["p_cluster"] < 0.05  # reference test against 0.5 is fooled
    assert res["p_permutation"] > 0.05

    dp, dm, pids, classes = _track1_inputs()
    res = h1_cmh_test(dp, dm, classes, classes, patient_ids=pids,
                      n_permutations=999, null_rng=np.random.default_rng(2))
    assert res["p_permutation"] == pytest.approx(1 / 1000)


def test_h1_cmh_reports_cluster_p_only_with_patient_ids():
    from tracks.representation.statistical.hypotheses.h1 import h1_cmh_test

    dp, dm, pids, classes = _track1_inputs()
    no_ids = h1_cmh_test(dp, dm, classes, classes)
    assert np.isnan(no_ids["p_cluster"]) and np.isnan(no_ids["p_permutation"])
    res = h1_cmh_test(dp, dm, classes, classes, patient_ids=pids)
    assert res["p_cluster"] == pytest.approx(1 / 256)


def test_h2_per_class_has_cluster_p():
    from tracks.representation.statistical.hypotheses.h2 import h2_wilcoxon_per_class

    rng = np.random.default_rng(2)
    pids = np.repeat(np.arange(8), 10)
    y = rng.choice([0.0, 0.5, 1.0], size=(pids.size, 2))
    cf = np.clip(y + rng.normal(0, 0.4, y.shape), 0, 1)
    ca = np.clip(y + rng.normal(0, 0.1, y.shape), 0, 1)
    df = h2_wilcoxon_per_class(cf, ca, y, ["A", "B"], pids, n_resamples=50,
                               rng=np.random.default_rng(3))
    assert {"wilcoxon_p", "cluster_p", "bh_adj_p"} <= set(df.columns)
    assert (df["cluster_p"] < 0.05).all()
