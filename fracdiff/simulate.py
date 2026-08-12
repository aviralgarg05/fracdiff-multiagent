"""Synthetic trajectory generation for the space-time fractional diffusion.

Two routes are provided.

``marginal_X`` returns the exact one-point law at a single time,

    X(t) =d D^(1/alpha) * t^(beta/alpha) * S_beta(1)^(-beta/alpha) * S_(alpha,theta),

i.e. alpha-stable Levy motion subordinated by the inverse beta-stable
subordinator.  This is cheap and is what the Proposition checks use.

``ctrw_paths`` returns full trajectories with the correct joint law, as a
continuous-time random walk with beta-stable waiting times and alpha-stable
jumps, following the calibration in the authors' ``gen_sim_data.m``:
``c_jump = (D * ta)^(1/alpha)``, ``c_wait = ta^(1/beta)``, so that
``c_jump / c_wait^(beta/alpha) = D^(1/alpha)`` and the scaling limit is the
intended equation.
"""
from __future__ import annotations

import numpy as np

from .stable import feller_stable, pos_stable_sub

__all__ = ["marginal_X", "ctrw_paths", "ctrw_paths_highdim"]


def marginal_X(alpha, beta, theta, D, t, N, rng):
    """Exact one-point sample of ``X(t)``; shape ``(N,)``."""
    S_space = feller_stable(alpha, theta, N, rng)
    E = np.ones(N) if beta >= 1.0 else pos_stable_sub(beta, N, rng) ** (-beta)
    return D ** (1 / alpha) * (t ** (beta / alpha)) * (E ** (1 / alpha)) * S_space


def ctrw_paths(alpha, beta, theta, D, N, rng, n_jumps=20_000, L=50,
               ta=1e-5, log_spaced=False, coupling=0.0, drift=0.0):
    """CTRW trajectories whose scaling limit is the space-time fractional diffusion.

    Parameters
    ----------
    N : number of independent trajectories.
    n_jumps : jumps simulated per trajectory.
    L : number of observation times.
    ta : CTRW time scale (1e-5 matches the authors' generator).
    log_spaced : log-spaced observation grid instead of the authors' linear one.
    coupling : in [0, 1); fraction of every jump shared across all trajectories,
        which breaks the i.i.d.-trajectory assumption on purpose.
    drift : adds ``drift * t``, which the model has no term for.

    Returns
    -------
    X : ``(N, L)`` displacements from the origin.
    ts : ``(L,)`` observation times.
    """
    c_jump = (D * ta) ** (1 / alpha)
    c_wait = ta ** (1 / beta) if beta < 1 else ta

    W = c_wait * (pos_stable_sub(beta, (N, n_jumps), rng) if beta < 1
                  else rng.exponential(1.0, (N, n_jumps)))
    J = feller_stable(alpha, theta, (N, n_jumps), rng)
    if coupling > 0:
        shared = feller_stable(alpha, theta, (1, n_jumps), rng)
        J = (1 - coupling) ** (1 / alpha) * J + coupling ** (1 / alpha) * shared
    J = c_jump * J

    T = np.cumsum(W, axis=1)
    S = np.concatenate([np.zeros((N, 1)), np.cumsum(J, axis=1)], axis=1)

    # every trajectory must cover the observation window
    tmax = T[:, -1].min()
    ts = (np.logspace(np.log10(tmax / 1e3), np.log10(tmax), L) if log_spaced
          else np.linspace(0, tmax, L + 1)[1:])

    idx = np.empty((N, L), dtype=np.int64)
    for n in range(N):
        idx[n] = np.searchsorted(T[n], ts, side="right")
    X = np.take_along_axis(S, idx, axis=1)
    if drift:
        X = X + drift * ts[None, :]
    return X, ts


def ctrw_paths_highdim(alpha, beta, D, N, d, rng, n_jumps=4_000, L=30,
                       ta=1e-5, spectrum=None):
    """Same CTRW in ``R^d`` with isotropic alpha-stable jumps.

    Isotropy uses the sub-Gaussian representation ``J = sqrt(2A) * G`` with
    ``A`` positive ``(alpha/2)``-stable and ``G`` standard normal, so every 1-D
    projection is alpha-stable with the same ``alpha``.

    ``spectrum`` : optional per-coordinate scaling of length ``d``, used to make
    the increments anisotropic (real embedding increments are).

    Returns ``X`` of shape ``(N, L, d)`` and ``ts`` of shape ``(L,)``.
    """
    c_jump = (D * ta) ** (1 / alpha)
    c_wait = ta ** (1 / beta) if beta < 1 else ta

    W = c_wait * (pos_stable_sub(beta, (N, n_jumps), rng) if beta < 1
                  else rng.exponential(1.0, (N, n_jumps)))
    amp = (np.sqrt(2 * pos_stable_sub(alpha / 2, (N, n_jumps), rng)) if alpha < 2
           else np.full((N, n_jumps), np.sqrt(2.0)))
    Gv = rng.normal(size=(N, n_jumps, d)).astype(np.float32)
    if spectrum is not None:
        Gv *= np.asarray(spectrum, np.float32)[None, None, :]
    J = (c_jump * amp[:, :, None].astype(np.float32)) * Gv

    T = np.cumsum(W, axis=1)
    S = np.concatenate([np.zeros((N, 1, d), np.float32), np.cumsum(J, axis=1)], axis=1)
    tmax = T[:, -1].min()
    ts = np.linspace(0, tmax, L + 1)[1:]
    idx = np.empty((N, L), dtype=np.int64)
    for n in range(N):
        idx[n] = np.searchsorted(T[n], ts, side="right")
    return np.take_along_axis(S, idx[:, :, None], axis=1), ts
