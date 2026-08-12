"""Closed-form moment expressions from Znaidi et al. (2020), Propositions 1-5.

Model
-----
    t D_*^beta u(x, t) = D * x D_theta^alpha u(x, t)

with 0 < alpha <= 2, 0 < beta <= 1, |theta| <= min(alpha, 2 - alpha), D > 0.

Note on Proposition 4
---------------------
Eq. (10) of the published paper reads

    var(log|X(t)|) = pi^2/6 * (1/alpha^2 + 1/2) - (pi*theta / (2*alpha))^2

which holds only at beta = 1.  It omits the variance contributed by the random
subordinator, because

    log|X| = (beta/alpha) log t + (1/alpha) log E_beta + log|S_alpha|

and Eq. (10) accounts only for ``var(log|S_alpha|)``.  The missing term is
``(pi^2 / (6 alpha^2)) (1 - beta^2)``.

The corrected expression also follows from the paper's own Proposition 5 minus
Proposition 3 squared (Prop. 5's constant already carries the term), and it is
what the authors' released MATLAB computes in
``fractDiffusion/matlab/fract_diff_est_logm.m`` line 37.  Both forms are exposed
here so the discrepancy can be reproduced; see ``tests/test_theory.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as G

EULER = 0.5772156649015329

__all__ = ["EULER", "prop1_abs", "prop2_signed", "prop3_logmean",
           "prop4_logvar_as_printed", "prop4_logvar_corrected", "prop5_logsq"]


def prop1_abs(alpha, beta, theta, D, t, delta):
    """Proposition 1: ``E[|X(t)|^delta]`` for ``0 < delta < alpha``."""
    return (t ** (delta * beta / alpha) * D ** (delta / alpha)
            * G(1 - delta / alpha) * G(1 + delta / alpha)
            * np.cos(delta * np.pi * theta / (2 * alpha))
            / (G(1 - delta) * G(1 + delta * beta / alpha)
               * np.cos(delta * np.pi / 2)))


def prop2_signed(alpha, beta, theta, D, t, delta):
    """Proposition 2: ``E[|X(t)|^delta sign(X(t))]``."""
    return (-t ** (delta * beta / alpha) * D ** (delta / alpha)
            * G(1 - delta / alpha) * G(1 + delta / alpha)
            * np.sin(delta * np.pi * theta / (2 * alpha))
            / (G(1 + delta * beta / alpha) * G(1 - delta)
               * np.sin(delta * np.pi / 2)))


def prop3_logmean(alpha, beta, theta, D, t):
    """Proposition 3: ``E[log|X(t)|]``.  Linear in ``log t`` with slope beta/alpha."""
    return (beta / alpha) * np.log(t) + np.log(D) / alpha + EULER * (beta / alpha - 1)


def prop4_logvar_as_printed(alpha, theta):
    """Proposition 4 exactly as printed (Eq. 10).  Correct only at beta = 1."""
    return (np.pi ** 2 / 6) * (1 / alpha ** 2 + 0.5) - (np.pi * theta / (2 * alpha)) ** 2


def prop4_logvar_corrected(alpha, beta, theta):
    """Corrected ``var(log|X(t)|)``; still independent of ``t``.

    Equivalently ``pi^2/6 * [(2 - beta^2)/alpha^2 + 1/2] - (pi theta / 2 alpha)^2``.
    """
    return (prop4_logvar_as_printed(alpha, theta)
            + (np.pi ** 2 / (6 * alpha ** 2)) * (1 - beta ** 2))


def prop5_logsq(alpha, beta, theta, D, t):
    """Proposition 5: ``E[(log|X(t)|)^2]``."""
    m = beta / alpha
    c = ((np.pi ** 2 / 6) * (1 / alpha ** 2 + 0.5)
         - (np.pi * theta / (2 * alpha)) ** 2
         + (np.log(D) / alpha + EULER * (m - 1)) ** 2
         + (np.pi ** 2 / (6 * alpha ** 2)) * (1 - beta ** 2))
    return m ** 2 * np.log(t) ** 2 + 2 * (beta * EULER / alpha) * (m - 1) * np.log(t) + c
