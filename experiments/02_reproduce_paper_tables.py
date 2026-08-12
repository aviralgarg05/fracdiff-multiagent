"""Reproduce the paper's parameter-recovery results (Tables 1-3) and N_90.

    python experiments/02_reproduce_paper_tables.py
"""
import warnings

import numpy as np

from fracdiff import SCENARIOS, algorithm1, algorithm2, ctrw_paths

warnings.filterwarnings("ignore")
FOUR = ["normal", "neutral", "space", "time"]

print("=" * 100)
print("2a  Algorithm 2 as printed vs corrected   (D=1, N=2000, L=100)")
print("=" * 100)
print(f"{'scenario':>9}{'true a':>8}{'true b':>8} | {'a printed':>10}{'e%':>7}"
      f"{'b printed':>10}{'e%':>7} | {'a corr':>8}{'e%':>7}{'b corr':>8}{'e%':>7}")
for name in FOUR:
    a, b, th = SCENARIOS[name]
    rng = np.random.default_rng(7)
    X, ts = ctrw_paths(a, b, th, 1.0, 2000, rng, n_jumps=30_000, L=100)
    p = algorithm2(X, ts, variant="as_printed")
    c = algorithm2(X, ts, variant="corrected")
    print(f"{name:>9}{a:8.2f}{b:8.2f} | {p['alpha']:10.3f}{100*abs(p['alpha']/a-1):7.1f}"
          f"{p['beta']:10.3f}{100*abs(p['beta']/b-1):7.1f} | "
          f"{c['alpha']:8.3f}{100*abs(c['alpha']/a-1):7.1f}"
          f"{c['beta']:8.3f}{100*abs(c['beta']/b-1):7.1f}")

print()
print("=" * 100)
print("2b  Full recovery, Algorithm 1 vs Algorithm 2   (D=1, N=1000, L=60, 20 seeds)")
print("=" * 100)
print(f"{'scenario':>9}{'param':>7}{'true':>8} | {'Alg1 mean':>10}{'sd':>7}{'err%':>7} |"
      f" {'Alg2 mean':>10}{'sd':>7}{'err%':>7}")
for name in FOUR:
    a, b, th = SCENARIOS[name]
    r1s, r2s = [], []
    for s in range(20):
        rng = np.random.default_rng(1000 + s)
        X, ts = ctrw_paths(a, b, th, 1.0, 1000, rng, n_jumps=20_000, L=60)
        r1s.append(algorithm1(X, ts))
        r2s.append(algorithm2(X, ts))
    for key, true in (("alpha", a), ("beta", b), ("theta", th), ("D", 1.0)):
        v1 = np.array([r[key] for r in r1s], float)
        v2 = np.array([r[key] for r in r2s], float)
        den = abs(true) if true else 1.0
        print(f"{name:>9}{key:>7}{true:8.2f} | {np.nanmean(v1):10.3f}{np.nanstd(v1):7.3f}"
              f"{100*abs(np.nanmean(v1)-true)/den:7.1f} | {np.nanmean(v2):10.3f}"
              f"{np.nanstd(v2):7.3f}{100*abs(np.nanmean(v2)-true)/den:7.1f}")

print()
print("=" * 100)
print("2c  N_90: fewest trajectories for median error <= 10%   (Algorithm 2, 15 seeds)")
print("=" * 100)
for name in FOUR:
    a, b, th = SCENARIOS[name]
    line = f"{name:>9}: "
    for key, true in (("alpha", a), ("beta", b)):
        n90 = None
        for N in (10, 20, 50, 100, 300, 1000):
            errs = []
            for s in range(15):
                rng = np.random.default_rng(50_000 + s)
                X, ts = ctrw_paths(a, b, th, 1.0, N, rng, n_jumps=20_000, L=60)
                errs.append(abs(algorithm2(X, ts)[key] - true) / (abs(true) or 1.0))
            if np.nanmedian(errs) <= 0.10:
                n90 = N
                break
        line += f"{key} N90 = {n90 if n90 else '>1000':>5}   "
    print(line)
