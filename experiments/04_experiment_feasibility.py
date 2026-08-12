"""Feasibility checks for the proposed multi-agent experiments.

Covers: which observable to use, whether alpha and beta are separately
identifiable, whether detrending rescues theta, whether a prefix of the rounds
is enough for early warning, and whether a topology contrast is detectable.

    python experiments/04_experiment_feasibility.py
"""
import warnings

import numpy as np

from fracdiff import SCENARIOS, algorithm2, ctrw_paths, ctrw_paths_highdim

warnings.filterwarnings("ignore")
A, B, TH = SCENARIOS["mixed"]


def mean_sd(vals):
    return np.nanmean(vals), np.nanstd(vals)


print("=" * 100)
print("OBSERVABLE:  1-D projection vs radial norm of a d-dimensional trajectory")
print("             true alpha = 1.5, beta = 0.75")
print("=" * 100)
print(f"{'setting':<44}{'alpha':>10}{'err%':>8}{'beta':>10}{'err%':>8}")
for d, N, nj, kind in [(16, 300, 4000, "isotropic"),
                       (128, 150, 2500, "isotropic"),
                       (16, 300, 4000, "anisotropic k^-1")]:
    spec = None if kind == "isotropic" else 1.0 / np.arange(1, d + 1)
    proj, radial = [], []
    for s in range(6):
        rng = np.random.default_rng(700 + s)
        X, ts = ctrw_paths_highdim(A, B, 1.0, N, d, rng, n_jumps=nj, L=30, spectrum=spec)
        u = np.random.default_rng(11).normal(size=d)
        u /= np.linalg.norm(u)
        proj.append(algorithm2(X @ u, ts))
        radial.append(algorithm2(np.linalg.norm(X, axis=2), ts))
    for label, res in (("1-D projection", proj), ("radial norm ||X||", radial)):
        am, _ = mean_sd([r["alpha"] for r in res])
        bm, _ = mean_sd([r["beta"] for r in res])
        print(f"d={d:<4} {kind:<18}{label:<19}{am:10.3f}{100*abs(am/A-1):8.1f}"
              f"{bm:10.3f}{100*abs(bm/B-1):8.1f}")

print()
print("=" * 100)
print("IDENTIFIABILITY:  processes with identical beta/alpha = 0.5")
print("=" * 100)
print(f"{'true (a, b)':<16}{'H':>6}{'b/a est':>10}{'alpha':>10}{'sd':>7}{'beta':>9}{'sd':>7}")
store = {}
for a, b in [(2.0, 1.0), (1.6, 0.8), (1.2, 0.6)]:
    res = []
    for s in range(8):
        rng = np.random.default_rng(200 + s)
        X, ts = ctrw_paths(a, b, 0.0, 1.0, 400, rng, n_jumps=8000, L=30)
        res.append(algorithm2(X, ts))
    am, asd = mean_sd([r["alpha"] for r in res])
    bm, bsd = mean_sd([r["beta"] for r in res])
    boa, _ = mean_sd([r["b_over_a"] for r in res])
    store[a] = [r["alpha"] for r in res]
    print(f"({a:.1f}, {b:.2f}){'':<7}{b/a:6.2f}{boa:10.3f}{am:10.3f}{asd:7.3f}{bm:9.3f}{bsd:7.3f}")
d_ = (np.mean(store[2.0]) - np.mean(store[1.6])) / np.sqrt(
    (np.std(store[2.0]) ** 2 + np.std(store[1.6]) ** 2) / 2)
print(f"\n  beta/alpha is the same for all three: that is all MSD or Hurst can see.")
print(f"  alpha separation (2.0 vs 1.6): Cohen's d = {d_:.1f}")

print()
print("=" * 100)
print("DETRENDING:  does removing the cross-sectional mean restore theta?")
print("=" * 100)
print(f"{'condition':<40}{'alpha':>10}{'beta':>9}{'theta':>10}{'theta err%':>12}")
for k in (0.0, 1.0, 4.0):
    for detrend in (False, True):
        res = []
        for s in range(8):
            rng = np.random.default_rng(4000 + s)
            X, ts = ctrw_paths(A, B, TH, 1.0, 400, rng, n_jumps=8000, L=30)
            iqr = np.subtract(*np.percentile(X[:, -1], [75, 25]))
            X = X + k * iqr * (ts / ts[-1])[None, :]
            if detrend:
                X = X - X.mean(axis=0, keepdims=True)
            res.append(algorithm2(X, ts))
        am, _ = mean_sd([r["alpha"] for r in res])
        bm, _ = mean_sd([r["beta"] for r in res])
        tm, _ = mean_sd([r["theta"] for r in res])
        tag = f"drift {k:.1f} x IQR, detrend {'yes' if detrend else 'no '}"
        print(f"{tag:<40}{am:10.3f}{bm:9.3f}{tm:10.3f}{100*abs(tm/TH-1):12.1f}")

print()
print("=" * 100)
print("EARLY WARNING:  estimate on the first k of 30 rounds")
print("=" * 100)
print(f"{'rounds used':<16}{'alpha':>10}{'sd':>8}{'err%':>8}{'beta':>10}{'sd':>8}{'err%':>8}")
paths = []
for s in range(8):
    rng = np.random.default_rng(600 + s)
    paths.append(ctrw_paths(A, B, TH, 1.0, 400, rng, n_jumps=8000, L=30))
for k in (5, 10, 20, 30):
    res = [algorithm2(X[:, :k], ts[:k]) for X, ts in paths]
    am, asd = mean_sd([r["alpha"] for r in res])
    bm, bsd = mean_sd([r["beta"] for r in res])
    print(f"first {k:<10}{am:10.3f}{asd:8.3f}{100*abs(am/A-1):8.1f}"
          f"{bm:10.3f}{bsd:8.3f}{100*abs(bm/B-1):8.1f}")

print()
print("=" * 100)
print("TOPOLOGY CONTRAST:  low vs high coupling as a proxy for decentralised vs central")
print("=" * 100)
print(f"{'N runs':<10}{'a | c=0.05':>12}{'sd':>7}{'a | c=0.5':>11}{'sd':>7}{'Cohen d':>10}")
for N in (50, 100, 300):
    g = {}
    for c in (0.05, 0.5):
        vals = []
        for s in range(8):
            rng = np.random.default_rng(800 + s)
            X, ts = ctrw_paths(A, B, TH, 1.0, N, rng, n_jumps=8000, L=30, coupling=c)
            vals.append(algorithm2(X, ts)["alpha"])
        g[c] = np.array(vals, float)
    d_ = (np.nanmean(g[0.05]) - np.nanmean(g[0.5])) / np.sqrt(
        (np.nanstd(g[0.05]) ** 2 + np.nanstd(g[0.5]) ** 2) / 2 + 1e-12)
    print(f"{N:<10}{np.nanmean(g[0.05]):12.3f}{np.nanstd(g[0.05]):7.3f}"
          f"{np.nanmean(g[0.5]):11.3f}{np.nanstd(g[0.5]):7.3f}{d_:10.2f}")
print("\n  Non-monotonic in N: the contrast is smaller than the estimator's own")
print("  seed-to-seed spread at this number of seeds. Treat as exploratory.")
