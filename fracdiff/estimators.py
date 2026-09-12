"""Parameter estimators: Algorithm 1 (absolute moments) and Algorithm 2 (log moments).

Both follow the authors' released MATLAB (``fract_diff_est_absm.m`` and
``fract_diff_est_logm.m``): tiny +/- delta for the ratio channels, and for
Algorithm 1 a 36-point delta grid with a multi-start non-linear fit.

Two deviations from the published text, both deliberate:

1. ``algorithm2(..., variant="corrected")`` inverts the corrected
   ``var(log|X|)`` (see ``fracdiff.theory``).  ``variant="as_printed"`` inverts
   Eq. (10) verbatim and is kept so the discrepancy is reproducible.
2. Exact zeros are masked rather than clamped.  ``X(t) = 0`` sends ``log|X|``
   to ``-inf`` and destroys both estimators; clamping to a tiny floor is worse,
   since it injects a large finite outlier.  With masking the estimators
   tolerate ~20% zeros with under 2% bias.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gamma as G

from .theory import EULER

__all__ = ["w_L", "theta_over_alpha", "algorithm1", "algorithm2", "is_admissible"]


def w_L(x):
    """The clipping function of Eq. (6); enforces |theta / alpha| <= 1."""
    return np.clip(x, -1.0, 1.0)


def theta_over_alpha(M, S, delta):
    """Eq. (6): invert the signed/absolute moment ratio for ``theta / alpha``.

    ``M``, ``S`` are the empirical absolute and signed moments of order ``delta``
    at each observation time; the ratio is time-independent in theory, so it is
    averaged over time.
    """
    r = np.mean(S / M)
    return w_L(-(2.0 / (np.pi * delta)) * np.arctan(np.tan(np.pi * delta / 2) * r))


def is_admissible(alpha, theta, tol=1e-9):
    """Does a fitted pair satisfy the model's own constraint |theta| <= min(alpha, 2 - alpha)?

    A fit that violates it is not a parameter estimate: the PDE the estimator is inverting has
    no such member. In practice a violation means the data are outside the family, or the
    observable is wrong. Callers should refuse to report an inadmissible fit rather than
    quoting the numbers.

    Calibrate before using it as evidence. On in-model data at a realistic design (50
    trajectories, 32 times) the flag fires for 78% of fits at (alpha, beta) = (2, 1), 60% at
    (1.9, 0.9) and 50% at (1.8, 0.7): near alpha = 2 the admissible set collapses to {theta = 0}
    and almost any output violates it. A high violation rate is therefore not evidence that data
    are outside the family unless it exceeds the in-model rate at the same design.

    This check is necessary, not sufficient. A non-negative observable forces
    theta/alpha == -1 by construction, and |theta| = alpha breaches the bound only when
    alpha > 1; below that the tautological fit sits exactly on the admissible boundary and
    passes. Check the observable as well: use a signed projection, never a distance.
    """
    if not (np.isfinite(alpha) and np.isfinite(theta)):
        return False
    return abs(theta) <= min(alpha, 2.0 - alpha) + tol


def _prepare(X):
    Xa = np.asarray(X, float)
    absX = np.where(Xa == 0.0, np.nan, np.abs(Xa))   # exact zeros are undefined
    return Xa, absX, np.sign(Xa)


def algorithm2(X, ts, delta=1e-3, variant="corrected"):
    """Log-absolute-moments estimator (paper Algorithm 2).

    Closed form throughout: two linear regressions plus one algebraic inversion.

    Parameters
    ----------
    X : ``(N_trajectories, L_times)`` displacements.
    ts : ``(L,)`` observation times, strictly positive.
    variant : ``"corrected"`` (default) or ``"as_printed"``.

    Returns
    -------
    dict with ``alpha``, ``beta``, ``theta``, ``D`` and the intermediate ratios
    ``b_over_a``, ``t_over_a``.
    """
    Xa, absX, sgn = _prepare(X)
    ts = np.asarray(ts, float)

    L1 = np.nanmean(np.log(absX), axis=0)
    ok = np.isfinite(L1)
    m, c = np.polyfit(np.log(ts[ok]), L1[ok], 1)          # slope = beta / alpha

    toas = []
    for d in (-delta, +delta):
        M = np.nanmean(absX ** d, axis=0)
        S = np.nanmean(sgn * absX ** d, axis=0)
        toas.append(theta_over_alpha(M[ok], S[ok], d))
    toa = float(np.mean(toas))

    s2 = float(np.nanmean(np.nanvar(np.log(absX), axis=0, ddof=1)))
    q = (s2 + (np.pi * toa / 2) ** 2) * (6 / np.pi ** 2) - 0.5
    if variant == "corrected":
        q = (q + m ** 2) / 2.0
    elif variant != "as_printed":
        raise ValueError("variant must be 'corrected' or 'as_printed'")

    if q <= 0:
        return dict(alpha=np.nan, beta=np.nan, theta=np.nan, D=np.nan,
                    b_over_a=m, t_over_a=toa, admissible=False, alpha_at_cap=False)

    alpha = min(2.0, q ** -0.5)
    D = np.exp(alpha * (c - EULER * (m - 1)))
    theta = toa * alpha
    return dict(alpha=alpha, beta=m * alpha, theta=theta, D=D,
                b_over_a=m, t_over_a=toa, admissible=is_admissible(alpha, theta),
                alpha_at_cap=bool(alpha >= 2.0 - 1e-12))


def algorithm1(X, ts, deltas=None, delta_theta=1e-3):
    """Absolute-moments estimator with sinc inversion (paper Algorithm 1).

    Slower than Algorithm 2 and solves a non-convex problem, so a multi-start is
    used.  Kept for comparison; Algorithm 2 matches or beats it everywhere in
    ``experiments/02_reproduce_paper_tables.py``.
    """
    if deltas is None:
        deltas = np.linspace(0.02, -0.02, 36)
        deltas = deltas[np.abs(deltas) > 1e-3]
    dd = np.asarray(deltas, float)

    Xa, absX, sgn = _prepare(X)
    ts = np.asarray(ts, float)
    ok = np.isfinite(np.nanmean(np.log(absX), axis=0))

    boa, toa_l = [], []
    for d in (-delta_theta, +delta_theta):
        Md = np.nanmean(absX ** d, axis=0)
        boa.append(np.polyfit(np.log(ts[ok]), np.log(Md[ok]), 1)[0] / d)
        Sd = np.nanmean(sgn * absX ** d, axis=0)
        toa_l.append(theta_over_alpha(Md[ok], Sd[ok], d))
    boa = float(np.mean(boa))
    toa = float(np.mean(toa_l))

    Cs = []
    for d in dd:
        Md = np.nanmean(absX ** d, axis=0)[ok]
        basis = (ts ** (d * boa))[ok]
        m2 = np.sum(basis * Md) / np.sum(basis * basis)      # zero-intercept slope
        Cs.append(m2 * (G(1 + d * boa) * G(1 - d) * np.cos(np.pi * d / 2)
                        / np.cos(np.pi * d * toa / 2)))
    Cs = np.asarray(Cs)

    def resid(p):
        a, D = p
        z = dd * np.pi / np.clip(a, 1e-3, 2.0)
        return D ** (dd / a) * z / np.sin(z) - Cs

    best, best_cost = None, np.inf
    for a0 in (0.4, 0.8, 1.2, 1.6, 1.95):
        for D0 in (0.5, 1.0, 2.0):
            try:
                r = least_squares(resid, [a0, D0], bounds=([0.05, 1e-3], [2.0, 1e3]))
                if r.cost < best_cost:
                    best_cost, best = r.cost, r.x
            except Exception:
                continue
    if best is None:
        return dict(alpha=np.nan, beta=np.nan, theta=np.nan, D=np.nan,
                    b_over_a=boa, t_over_a=toa, admissible=False, alpha_at_cap=False)
    alpha, D = best
    theta = toa * alpha
    return dict(alpha=alpha, beta=boa * alpha, theta=theta, D=D,
                b_over_a=boa, t_over_a=toa, admissible=is_admissible(alpha, theta),
                alpha_at_cap=bool(alpha >= 2.0 - 1e-12))
