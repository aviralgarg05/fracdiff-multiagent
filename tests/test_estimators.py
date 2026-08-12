"""End-to-end estimator checks on CTRW trajectories.

These are Monte Carlo estimators, so the assertions are on the mean over several
seeds rather than on a single draw.  Measured per-seed spreads at the settings
used here (N = 800, 12k jumps, L = 40, 8 seeds):

    normal   alpha 1.987 +- 0.021   beta 0.983 +- 0.018
    neutral  alpha 0.500 +- 0.011   beta 0.500 +- 0.020
    space    alpha 0.501 +- 0.007   beta 0.999 +- 0.015
    time     alpha 1.964 +- 0.047   beta 0.484 +- 0.042
    mixed    alpha 1.505 +- 0.039   beta 0.756 +- 0.028

The "time" regime (alpha = 2, beta = 0.5) is the hardest, matching the paper's
own observation that alpha = 2 sits on the boundary of the admissible range.
"""
import numpy as np
import pytest

from fracdiff import SCENARIOS, algorithm1, algorithm2, ctrw_paths

SEEDS = 5


def _paths(name, N=800, seed=0, n_jumps=12_000, L=40, **kw):
    alpha, beta, theta = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    X, ts = ctrw_paths(alpha, beta, theta, 1.0, N, rng, n_jumps=n_jumps, L=L, **kw)
    return X, ts, alpha, beta, theta


def _mean_fit(name, N=800, seeds=SEEDS, estimator=algorithm2, transform=None, **kw):
    """Mean estimate over ``seeds`` independent replications."""
    out = []
    for s in range(seeds):
        X, ts, *_ = _paths(name, N=N, seed=s, **kw)
        if transform is not None:
            X = transform(X, ts)
        out.append(estimator(X, ts))
    return {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_algorithm2_recovers_parameters(name):
    alpha, beta, theta = SCENARIOS[name]
    r = _mean_fit(name)
    assert abs(r["alpha"] - alpha) / alpha < 0.10
    assert abs(r["beta"] - beta) / beta < 0.10
    assert abs(r["theta"] - theta) < 0.10


@pytest.mark.parametrize("name", ["neutral", "time", "mixed"])
def test_as_printed_variant_is_biased_when_beta_lt_1(name):
    """Implementing Algorithm 2 line 8 verbatim gives a visibly wrong alpha."""
    alpha, beta, theta = SCENARIOS[name]
    good = _mean_fit(name, estimator=lambda X, ts: algorithm2(X, ts, variant="corrected"))
    bad = _mean_fit(name, estimator=lambda X, ts: algorithm2(X, ts, variant="as_printed"))
    assert abs(good["alpha"] - alpha) / alpha < 0.10
    assert abs(bad["alpha"] - alpha) / alpha > 0.15


def test_algorithm1_agrees_with_algorithm2():
    X, ts, *_ = _paths("space")
    r1, r2 = algorithm1(X, ts), algorithm2(X, ts)
    assert abs(r1["alpha"] - r2["alpha"]) < 0.10
    assert abs(r1["beta"] - r2["beta"]) < 0.10


def test_exact_zeros_are_tolerated():
    """An agent that repeats itself yields X = 0 exactly; masking must absorb it."""
    def zap(X, ts):
        return np.where(np.random.default_rng(9).random(X.shape) < 0.20, 0.0, X)
    clean = _mean_fit("mixed")["alpha"]
    zeros = _mean_fit("mixed", transform=zap)["alpha"]
    assert abs(zeros - clean) / clean < 0.05


def test_few_time_points_are_enough():
    """Accuracy is driven by the number of trajectories, not the number of times."""
    alpha, beta, theta = SCENARIOS["mixed"]
    r = _mean_fit("mixed", L=5)
    assert abs(r["alpha"] - alpha) / alpha < 0.10
    assert abs(r["beta"] - beta) / beta < 0.10


def test_identifiability_at_fixed_self_similarity_exponent():
    """Same beta/alpha, different (alpha, beta): the ratio cannot separate them, alpha can."""
    fits = {}
    for (alpha, beta) in [(2.0, 1.0), (1.2, 0.6)]:
        vals = []
        for s in range(SEEDS):
            rng = np.random.default_rng(s)
            X, ts = ctrw_paths(alpha, beta, 0.0, 1.0, 800, rng, n_jumps=12_000, L=40)
            vals.append(algorithm2(X, ts))
        fits[alpha] = {k: float(np.nanmean([v[k] for v in vals])) for k in vals[0]}
    assert abs(fits[2.0]["b_over_a"] - fits[1.2]["b_over_a"]) < 0.05   # indistinguishable
    assert fits[2.0]["alpha"] - fits[1.2]["alpha"] > 0.5               # clearly separated


def test_bounded_observable_pushes_alpha_to_two():
    """A saturating observable manufactures normal diffusion; documented failure mode."""
    def squash(X, ts):
        R = 0.5 * np.subtract(*np.percentile(X[:, -1], [75, 25]))
        return R * np.tanh(X / R)
    assert _mean_fit("mixed", transform=squash)["alpha"] > 1.9


def test_detrending_removes_drift():
    """Detrending restores theta, and the result no longer depends on the drift size.

    It costs roughly 18% on theta even when there is no drift (0.205 against a
    true 0.25), which is the price of the correction.
    """
    _, _, _, _, theta = _paths("mixed")

    def add_drift(k):
        def f(X, ts):
            iqr = np.subtract(*np.percentile(X[:, -1], [75, 25]))
            return X + k * iqr * (ts / ts[-1])[None, :]
        return f

    def add_drift_then_detrend(k):
        inner = add_drift(k)

        def f(X, ts):
            Xd = inner(X, ts)
            return Xd - Xd.mean(axis=0, keepdims=True)
        return f

    assert _mean_fit("mixed", transform=add_drift(4.0))["theta"] < 0.0   # badly wrong

    d0 = _mean_fit("mixed", transform=add_drift_then_detrend(0.0))["theta"]
    d4 = _mean_fit("mixed", transform=add_drift_then_detrend(4.0))["theta"]
    assert abs(d4 - theta) < 0.12          # recovered, with the known ~18% cost
    assert abs(d4 - d0) < 0.02             # and independent of the drift magnitude
