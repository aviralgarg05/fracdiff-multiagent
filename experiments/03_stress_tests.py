"""Stress-test the estimator against assumptions a multi-agent setting violates.

Perturbation sizes are expressed relative to the interquartile range of X at the
final observation time, so they are scale free.

    python experiments/03_stress_tests.py
"""
import warnings

import numpy as np

from fracdiff import SCENARIOS, algorithm2, ctrw_paths

warnings.filterwarnings("ignore")

A, B, TH, D = SCENARIOS["mixed"] + (1.0,)
SEEDS = 12


def run(post=None, N=500, L=50, **kw):
    out = []
    for s in range(SEEDS):
        rng = np.random.default_rng(4000 + s)
        X, ts = ctrw_paths(A, B, TH, D, N, rng, n_jumps=20_000, L=L, **kw)
        if post is not None:
            X = post(X, ts, rng)
        try:
            out.append(algorithm2(X, ts))
        except Exception:
            out.append(dict(alpha=np.nan, beta=np.nan, theta=np.nan))
    return {k: (np.nanmean([o[k] for o in out]), np.nanstd([o[k] for o in out]))
            for k in ("alpha", "beta", "theta")}


def row(tag, r):
    (am, asd), (bm, bsd), (tm, tsd) = r["alpha"], r["beta"], r["theta"]
    print(f"{tag:<34}| a {am:6.3f}+-{asd:5.3f} ({100*abs(am/A-1):5.1f}%) "
          f"| b {bm:6.3f}+-{bsd:5.3f} ({100*abs(bm/B-1):5.1f}%) "
          f"| th {tm:6.3f}+-{tsd:5.3f} ({100*abs(tm/TH-1):5.1f}%)")


def iqr_of(X):
    return np.subtract(*np.nanpercentile(X[:, -1], [75, 25]))


print(f"true alpha={A} beta={B} theta={TH};  N=500, L=50, {SEEDS} seeds")
print("=" * 108)
row("baseline", run())

print("\nFEW TIME POINTS  (conversations have tens of rounds, not hundreds)")
print("-" * 108)
for L in (5, 10, 20, 50, 100):
    row(f"L = {L} (linear grid)", run(L=L))
for L in (10, 50):
    row(f"L = {L} (log-spaced grid)", run(L=L, log_spaced=True))

print("\nBOUNDED DOMAIN  (unit-normalised embeddings cap displacement at 2)")
print("-" * 108)
for mult in (0.5, 1, 3, 10, 100):
    def squash(X, ts, rng, mult=mult):
        R = mult * iqr_of(X)
        return R * np.tanh(X / R)
    row(f"soft bound R = {mult:>5} x IQR", run(post=squash))

print("\nCOUPLED TRAJECTORIES  (agents inside one run are not independent)")
print("-" * 108)
for c in (0.0, 0.05, 0.2, 0.5, 0.8):
    row(f"coupling c = {c:.2f}", run(coupling=c))

print("\nDRIFT  (the model has no advection term)")
print("-" * 108)
for k in (0.0, 0.25, 1.0, 4.0):
    def drift(X, ts, rng, k=k):
        return X + k * iqr_of(X) * (ts / ts[-1])[None, :]
    row(f"drift = {k:.2f} x IQR over window", run(post=drift))

print("\nOBSERVATION NOISE")
print("-" * 108)
for k in (0.0, 0.01, 0.1, 0.5):
    def noise(X, ts, rng, k=k):
        return X + rng.normal(0, k * iqr_of(X), X.shape)
    row(f"gaussian noise sd = {k:.2f} x IQR", run(post=noise))

print("\nREPETITION  (an agent repeating itself gives exactly zero displacement)")
print("-" * 108)
for p in (0.0, 0.01, 0.05, 0.2):
    def zeros(X, ts, rng, p=p):
        return np.where(rng.random(X.shape) < p, 0.0, X)
    row(f"zero fraction p = {p:.2f}", run(post=zeros))
