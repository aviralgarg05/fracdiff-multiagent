"""The dwell tail index, and the cut-off rule it depends on.

These tests exist because the project's own results were computed with a cut-off
fixed at 2 or 3, and that choice turned out to move the estimate by up to 0.25
depending on the dwell scale -- in opposite directions at either end, so it was
confounded with the effects being measured.  What is pinned here is therefore
not only that the estimator works, but the specific comparison that justifies
its default, and the regime where the default is still biased.
"""
import numpy as np
import pytest

from fracdiff import dwell_tail_index, pos_stable_sub

SCALES = (2, 5, 20)
DRAWS = 8
N = 2000


def make_dwells(beta, median, n, rng):
    """Dwells of an inverse beta-stable subordinator, scaled to a target median
    and rounded to whole events, which is how a conversation records them."""
    draws = pos_stable_sub(beta, n, rng)
    s = 1.0
    for _ in range(60):
        med = np.median(np.maximum(1, np.round(s * draws)))
        if med <= 0:
            break
        s *= (median / med) ** 0.4
    return np.maximum(1, np.round(s * draws)).astype(int)


def mean_estimate(beta, median, d_min=None):
    out = []
    for t in range(DRAWS):
        d = make_dwells(beta, median, N, np.random.default_rng(100 * t + median))
        out.append(dwell_tail_index(d, d_min=d_min)["beta"])
    return float(np.mean(out))


@pytest.mark.parametrize("beta", [0.4, 0.6])
@pytest.mark.parametrize("median", SCALES)
def test_ks_rule_recovers_beta_without_scale_drift(beta, median):
    """At the betas these experiments inject, the KS cut-off is unbiased across
    an order of magnitude of dwell scale."""
    assert abs(mean_estimate(beta, median) - beta) < 0.05


@pytest.mark.parametrize("beta", [0.4, 0.6, 0.8])
def test_ks_rule_beats_a_fixed_cutoff_at_long_dwells(beta):
    """The defect this estimator was written for: a cut-off fixed at 3 fits the
    body once the dwells are long, and is further from the truth than the KS
    rule at every beta tested."""
    ks = abs(mean_estimate(beta, 20) - beta)
    fixed = abs(mean_estimate(beta, 20, d_min=3) - beta)
    assert ks < fixed


def test_fixed_cutoff_drifts_with_scale_and_ks_does_not():
    """Stated as a difference, which is the form the claim takes: across median
    dwells of 2 and 20 the fixed rule moves far more than the KS rule."""
    fixed = [mean_estimate(0.6, m, d_min=3) for m in (2, 20)]
    ks = [mean_estimate(0.6, m) for m in (2, 20)]
    assert abs(fixed[0] - fixed[1]) > 0.15
    assert abs(ks[0] - ks[1]) < 0.05


def test_ks_rule_is_biased_high_at_beta_point_eight():
    """Recorded rather than hidden: the default is not a cure.  At beta 0.8 and
    long dwells it reads high, so an absolute estimate there is an upper bound."""
    assert mean_estimate(0.8, 20) > 0.85


def test_bootstrap_interval_brackets_the_truth():
    d = make_dwells(0.6, 5, N, np.random.default_rng(7))
    r = dwell_tail_index(d, n_boot=400)
    assert r["ci"] is not None
    lo, hi = r["ci"]
    assert lo < 0.6 < hi
    assert lo <= r["beta"] <= hi


def test_explicit_cutoff_is_honoured_and_reported():
    d = make_dwells(0.6, 5, N, np.random.default_rng(11))
    r = dwell_tail_index(d, d_min=7)
    assert r["d_min"] == 7
    assert r["n_tail"] == int((d >= 7).sum())


def test_small_and_degenerate_samples_return_nan_not_a_number():
    assert not np.isfinite(dwell_tail_index(np.ones(10, int))["beta"])
    assert not np.isfinite(dwell_tail_index(np.ones(500, int))["beta"])
    assert dwell_tail_index(np.array([], int))["ci"] is None
