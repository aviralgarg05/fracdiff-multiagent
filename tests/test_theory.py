"""Monte Carlo checks of Propositions 1-5, including the Proposition 4 discrepancy."""
import numpy as np
import pytest

from fracdiff import (SCENARIOS, feller_stable, marginal_X, prop1_abs, prop2_signed,
                      prop3_logmean, prop4_logvar_as_printed, prop4_logvar_corrected,
                      prop5_logsq)
from scipy.special import gamma as G

N_MC = 1_000_000


@pytest.mark.parametrize("alpha,theta", [(2.0, 0.0), (1.5, 0.25), (0.5, 0.25),
                                         (0.5, 0.5), (1.8, -0.2)])
def test_feller_absolute_moment(alpha, theta):
    """The Feller->Nolan map reproduces the known fractional absolute moment."""
    rng = np.random.default_rng(0)
    S = feller_stable(alpha, theta, N_MC, rng)
    d = 0.1
    got = np.mean(np.abs(S) ** d)
    want = (G(1 - d / alpha) * np.cos(d * np.pi * theta / (2 * alpha))
            / (G(1 - d) * np.cos(d * np.pi / 2)))
    assert abs(got / want - 1) < 0.01


@pytest.mark.parametrize("name", list(SCENARIOS))
@pytest.mark.parametrize("t", [0.5, 5.0])
def test_props_1_2_3(name, t):
    alpha, beta, theta = SCENARIOS[name]
    rng = np.random.default_rng(1)
    X = marginal_X(alpha, beta, theta, 1.0, t, N_MC, rng)
    d = 0.1

    assert abs(np.mean(np.abs(X) ** d) / prop1_abs(alpha, beta, theta, 1.0, t, d) - 1) < 0.01
    assert abs(np.mean(np.abs(X) ** d * np.sign(X))
               - prop2_signed(alpha, beta, theta, 1.0, t, d)) < 0.02
    assert abs(np.mean(np.log(np.abs(X)))
               - prop3_logmean(alpha, beta, theta, 1.0, t)) < 0.02


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_prop4_corrected_matches_and_as_printed_fails_when_beta_lt_1(name):
    """Eq. (10) is right at beta = 1 and wrong otherwise; the corrected form is right."""
    alpha, beta, theta = SCENARIOS[name]
    rng = np.random.default_rng(2)
    X = marginal_X(alpha, beta, theta, 1.0, 1.0, N_MC, rng)
    mc = np.var(np.log(np.abs(X)), ddof=1)

    corrected = prop4_logvar_corrected(alpha, beta, theta)
    assert abs(mc / corrected - 1) < 0.02, "corrected Prop. 4 should match Monte Carlo"

    as_printed = prop4_logvar_as_printed(alpha, theta)
    if beta >= 1.0:
        assert abs(mc / as_printed - 1) < 0.02
    else:
        assert abs(mc / as_printed - 1) > 0.15, "Eq. (10) should visibly fail for beta < 1"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_prop4_corrected_equals_prop5_minus_prop3_squared(name):
    """The correction is not an ad-hoc patch: it follows from the paper's Prop. 5."""
    alpha, beta, theta = SCENARIOS[name]
    for t in (0.3, 1.0, 7.0):
        var_from_5 = (prop5_logsq(alpha, beta, theta, 1.0, t)
                      - prop3_logmean(alpha, beta, theta, 1.0, t) ** 2)
        assert abs(var_from_5 - prop4_logvar_corrected(alpha, beta, theta)) < 1e-9


def test_log_variance_is_time_independent():
    alpha, beta, theta = SCENARIOS["mixed"]
    rng = np.random.default_rng(3)
    vals = [np.var(np.log(np.abs(
        marginal_X(alpha, beta, theta, 1.0, t, 400_000, rng))), ddof=1)
        for t in (0.01, 1.0, 100.0)]
    assert max(vals) / min(vals) - 1 < 0.05


def test_prop5_is_correct_only_at_unit_diffusivity():
    """Eq. (11) loses a log(D) cross term, so it is exact at D = 1 and wrong otherwise.

    Squaring Proposition 3 gives 2 (beta/alpha) log(t) [log(D)/alpha + gamma(beta/alpha - 1)];
    the printed equation keeps only the gamma half. Nothing published depends on it: Algorithm 2
    never evaluates Eq. (11), and the paper's numerical checks do not cover it.
    """
    from fracdiff import prop5_logsq_as_printed, prop5_logsq_corrected
    alpha, beta, theta = SCENARIOS["mixed"]
    for t in (0.2, 5.0):
        assert prop5_logsq_as_printed(alpha, beta, theta, 1.0, t) == pytest.approx(
            prop5_logsq_corrected(alpha, beta, theta, 1.0, t))

    N, D, t = 400_000, 3.0, 5.0
    x = marginal_X(alpha, beta, theta, D, t, N, np.random.default_rng(11))
    q = np.log(np.abs(x)) ** 2
    mc, se = q.mean(), q.std(ddof=1) / np.sqrt(N)
    assert abs(mc - prop5_logsq_corrected(alpha, beta, theta, D, t)) < 4 * se
    assert abs(mc - prop5_logsq_as_printed(alpha, beta, theta, D, t)) > 50 * se


def test_prop4_and_prop5_as_printed_contradict_each_other():
    """Eq. (11)'s constant already contains the term Eq. (10) omits, so the paper's own two
    propositions imply different variances. That is the cleanest evidence for the misprint."""
    from fracdiff import prop5_logsq_as_printed
    alpha, beta, theta, D, t = 1.5, 0.75, 0.25, 1.0, 1.0   # log t = 0 isolates the constant
    implied = prop5_logsq_as_printed(alpha, beta, theta, D, t) - prop3_logmean(
        alpha, beta, theta, D, t) ** 2
    assert implied == pytest.approx(prop4_logvar_corrected(alpha, beta, theta))
    assert abs(implied - prop4_logvar_as_printed(alpha, theta)) > 0.1
