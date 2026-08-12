"""Alpha-stable random variates and the Riesz-Feller <-> Nolan reparameterisation.

The paper states its results in the Riesz-Feller convention, in which the space
operator has characteristic exponent

    -|k|^alpha * exp(i * sign(k) * theta * pi / 2)

while samplers are conventionally written in Nolan's S1 convention,

    exp(-|k|^alpha * [1 - i * beta_N * sign(k) * tan(pi * alpha / 2)]).

`feller_to_nolan` is the map between them.  It is verified numerically in
`tests/test_theory.py` against the closed-form fractional absolute moment.
"""
from __future__ import annotations

import numpy as np

__all__ = ["stable_rvs", "feller_to_nolan", "feller_stable", "pos_stable_sub"]


def stable_rvs(alpha: float, beta_N: float, size, rng) -> np.ndarray:
    """Chambers-Mallows-Stuck sampler, Nolan S1 parameterisation.

    Parameters
    ----------
    alpha : stability index in (0, 2], excluding 1.
    beta_N : Nolan skewness in [-1, 1].
    size : shape passed to the generator.
    rng : ``numpy.random.Generator``.
    """
    if alpha == 1.0:
        raise ValueError("alpha == 1 is a separate CMS branch and is not used here")
    U = rng.uniform(-np.pi / 2, np.pi / 2, size)
    W = rng.exponential(1.0, size)
    zeta = -beta_N * np.tan(np.pi * alpha / 2)
    xi = np.arctan(-zeta) / alpha
    return ((1 + zeta ** 2) ** (1 / (2 * alpha))
            * np.sin(alpha * (U + xi)) / np.cos(U) ** (1 / alpha)
            * (np.cos(U - alpha * (U + xi)) / W) ** ((1 - alpha) / alpha))


def feller_to_nolan(alpha: float, theta: float) -> tuple[float, float]:
    """Riesz-Feller ``(alpha, theta)`` -> Nolan ``(beta_N, scale)``."""
    beta_N = -np.tan(theta * np.pi / 2) / np.tan(np.pi * alpha / 2)
    scale = np.cos(theta * np.pi / 2) ** (1 / alpha)
    return beta_N, scale


def feller_stable(alpha: float, theta: float, size, rng) -> np.ndarray:
    """Standard Riesz-Feller alpha-stable variate with skewness ``theta``."""
    beta_N, scale = feller_to_nolan(alpha, theta)
    return scale * stable_rvs(alpha, beta_N, size, rng)


def pos_stable_sub(beta: float, size, rng) -> np.ndarray:
    """Standard positive beta-stable subordinator at time 1.

    Normalised so that ``E[exp(-lam * S)] = exp(-lam ** beta)``, equivalently
    ``E[S ** -p] = Gamma(1 + p / beta) / Gamma(1 + p)`` for ``p > 0``.
    """
    scale = np.cos(np.pi * beta / 2) ** (1 / beta)
    return scale * stable_rvs(beta, 1.0, size, rng)
