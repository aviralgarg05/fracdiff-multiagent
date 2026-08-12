"""Monte Carlo verification of Propositions 1-4, and the Prop. 4 discrepancy.

    python experiments/01_verify_propositions.py
"""
import numpy as np
from scipy.special import gamma as G

from fracdiff import (SCENARIOS, feller_stable, marginal_X, prop1_abs, prop2_signed,
                      prop3_logmean, prop4_logvar_as_printed, prop4_logvar_corrected,
                      prop5_logsq)

N_MC = 4_000_000
rng = np.random.default_rng(0)

print("=" * 96)
print("1a  Feller alpha-stable absolute moment vs closed form")
print("=" * 96)
print(f"{'alpha':>7}{'theta':>7}{'delta':>7}{'Monte Carlo':>14}{'theory':>12}{'rel err %':>11}")
for alpha, theta in [(2.0, 0.0), (1.5, 0.25), (0.5, 0.25), (0.5, 0.5), (1.8, -0.2)]:
    S = feller_stable(alpha, theta, N_MC, rng)
    for d in (0.1, 0.3):
        if d >= alpha:
            continue
        mc = np.mean(np.abs(S) ** d)
        want = (G(1 - d / alpha) * np.cos(d * np.pi * theta / (2 * alpha))
                / (G(1 - d) * np.cos(d * np.pi / 2)))
        print(f"{alpha:7.2f}{theta:7.2f}{d:7.2f}{mc:14.5f}{want:12.5f}"
              f"{100 * abs(mc / want - 1):11.3f}")

print()
print("=" * 96)
print("1b  Propositions 1-4 on the exact space-time-fractional marginal (delta = 0.1)")
print("=" * 96)
print(f"{'scenario':>9}{'t':>6} | {'E|X|^d MC':>11}{'Prop1':>11}{'e%':>6} |"
      f" {'E log|X| MC':>12}{'Prop3':>10}{'err':>7} |"
      f" {'var MC':>9}{'Prop4 corr':>11}{'e%':>6}{'Eq.(10)':>10}{'e%':>7}")
for name, (alpha, beta, theta) in SCENARIOS.items():
    for t in (0.5, 5.0):
        X = marginal_X(alpha, beta, theta, 1.0, t, N_MC // 2, rng)
        d = 0.1
        m1, p1 = np.mean(np.abs(X) ** d), prop1_abs(alpha, beta, theta, 1.0, t, d)
        ml, p3 = np.mean(np.log(np.abs(X))), prop3_logmean(alpha, beta, theta, 1.0, t)
        mv = np.var(np.log(np.abs(X)), ddof=1)
        p4c = prop4_logvar_corrected(alpha, beta, theta)
        p4p = prop4_logvar_as_printed(alpha, theta)
        print(f"{name:>9}{t:6.1f} | {m1:11.5f}{p1:11.5f}{100 * abs(m1 / p1 - 1):6.2f} |"
              f" {ml:12.5f}{p3:10.5f}{abs(ml - p3):7.4f} |"
              f" {mv:9.4f}{p4c:11.4f}{100 * abs(mv / p4c - 1):6.2f}"
              f"{p4p:10.4f}{100 * abs(mv / p4p - 1):7.2f}")

print()
print("Eq. (10) is exact at beta = 1 and wrong otherwise. The missing term is")
print("(pi^2 / 6 alpha^2)(1 - beta^2). Analytic cross-check, Prop.5 - Prop.3^2:")
print(f"{'scenario':>9}{'Prop5-Prop3^2':>16}{'Prop4 corrected':>18}{'Eq.(10)':>12}")
for name, (alpha, beta, theta) in SCENARIOS.items():
    v5 = prop5_logsq(alpha, beta, theta, 1.0, 1.0) - prop3_logmean(alpha, beta, theta, 1.0, 1.0) ** 2
    print(f"{name:>9}{v5:16.4f}{prop4_logvar_corrected(alpha, beta, theta):18.4f}"
          f"{prop4_logvar_as_printed(alpha, theta):12.4f}")

print()
print("=" * 96)
print("1c  var(log|X|) does not depend on t   (alpha=1.5, beta=0.75, theta=0.25)")
print("=" * 96)
alpha, beta, theta = SCENARIOS["mixed"]
for t in (0.01, 0.1, 1.0, 10.0, 100.0):
    X = marginal_X(alpha, beta, theta, 1.0, t, 2_000_000, rng)
    print(f"   t = {t:8.2f}   var = {np.var(np.log(np.abs(X)), ddof=1):.4f}"
          f"   (theory {prop4_logvar_corrected(alpha, beta, theta):.4f})")
