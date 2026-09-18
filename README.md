# fracdiff-multiagent

[![tests](https://github.com/aviralgarg05/fracdiff-multiagent/actions/workflows/tests.yml/badge.svg)](https://github.com/aviralgarg05/fracdiff-multiagent/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An independent re-implementation of

> Znaidi M. R., Gupta G., Asgari K. & Bogdan P. (2020),
> *Identifying Arguments of Space-Time Fractional Diffusion: Data-Driven Approach*,
> Frontiers in Applied Mathematics and Statistics **6**:14.
> [doi:10.3389/fams.2020.00014](https://doi.org/10.3389/fams.2020.00014)

written to check the paper before building on it, and to test whether the method
transfers to multi-agent language-model systems.

Nothing here reuses the authors' code. The generator and both estimators were
written from the paper and then compared against the released MATLAB, so
agreement is evidence rather than a shared assumption.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

`numpy` and `scipy` are the only requirements. The embedding pilot additionally
needs `torch` and `transformers` (`pip install -e ".[embeddings]"`).

## Run

```bash
python -m pytest tests/ -q                          # 40 tests, about 70 s
python experiments/01_verify_propositions.py        # Monte Carlo vs Props 1-4
python experiments/02_reproduce_paper_tables.py     # recovery and N_90
python experiments/03_stress_tests.py               # assumption violations
python experiments/04_experiment_feasibility.py     # observable, identifiability, power
python experiments/05_embedding_pilot.py            # real sentence embeddings
```

## Layout

```
fracdiff/
  stable.py       alpha-stable sampling, Riesz-Feller <-> Nolan map
  theory.py       Propositions 1-5 in closed form
  simulate.py     exact one-point marginal; CTRW paths in 1-D and R^d
  estimators.py   Algorithm 1 (absolute moments), Algorithm 2 (log moments)
experiments/      the five scripts above; each prints a table
tests/            assertions for every claim below
```

## Synthetic data

Trajectories are a continuous-time random walk, following the scheme in the
authors' `gen_sim_data.m` but implemented independently (`fracdiff/simulate.py`).

* **Waiting times** are totally skewed positive `beta`-stable with scale
  `c_wait = ta^(1/beta)`; **jumps** are Riesz-Feller `alpha`-stable with skewness
  `theta` and scale `c_jump = (D * ta)^(1/alpha)`, with `ta = 1e-5`. That
  calibration gives `c_jump / c_wait^(beta/alpha) = D^(1/alpha)`, so the scaling
  limit is the intended equation.
* **Sampling** is Chambers-Mallows-Stuck. Riesz-Feller maps to Nolan's S1 by
  `beta_N = -tan(theta pi/2) / tan(alpha pi/2)`, scale `cos(theta pi/2)^(1/alpha)`
  (`fracdiff/stable.py`), verified against the closed-form fractional absolute
  moment in `tests/test_theory.py`.
* **Observation** takes cumulative sums of waiting times and of jumps and reads
  `X` off by search-sort, on a linear grid over `[0, t_max]` with `t_max` the
  smallest final renewal time across trajectories, so every trajectory covers the
  window.
* **Sweeps** span 2,500-30,000 jumps per trajectory, 10-2,000 trajectories, 5-100
  observation times and 6-20 seeds. The recovery tables use 1,000 trajectories,
  60 observation times and 20 seeds.
* **Scenarios** are the paper's four, `(2, 1, 0)`, `(0.5, 0.5, 0.5)`,
  `(0.5, 1, 0.25)`, `(2, 0.5, 0)`, plus a mixed case `(1.5, 0.75, 0.25)` used for
  the stress tests because it exercises `alpha`, `beta` and `theta` at once.
* **Stress perturbations** are applied to those same trajectories: bounding as
  `R tanh(X/R)`, drift as an added linear term, noise as additive Gaussian,
  coupling by mixing in a jump component shared across trajectories with weight
  `c^(1/alpha)`, and repetition by forcing a fraction of cells to zero. All
  magnitudes are relative to the interquartile range of `X` at the final time, so
  they are scale free.
* **High-dimensional** checks use the sub-Gaussian representation
  `J = sqrt(2A) G` with `A` positive `(alpha/2)`-stable and `G` standard normal,
  made anisotropic by scaling coordinate `k` by `1/k`.

The generator is checked rather than trusted: `marginal_X` gives the exact
one-point law

```
X(t) =d D^(1/alpha) * t^(beta/alpha) * S_beta(1)^(-beta/alpha) * S_(alpha,theta)
```

(alpha-stable Levy motion subordinated by the inverse beta-stable subordinator),
and the CTRW output is compared against Propositions 1-3 with four million
samples. Proposition 1 agrees to within 0.01 %; Propositions 2 and 3 to within
1e-3 absolute, their true values passing through zero so a relative error is not
meaningful. The generator reproducing Props. 1-3 but not Prop. 4 is what located
the discrepancy below.

## What the checks found

### Proposition 4 as printed is wrong for beta < 1

Eq. (10) omits `(pi^2 / 6 alpha^2)(1 - beta^2)`, the variance contributed by the
random subordinator. It is exact at `beta = 1` and wrong otherwise:

Printed by `experiments/01_verify_propositions.py` (4M samples, `t = 0.5`); the
Monte Carlo column is stable to about three significant figures.

| scenario | Monte Carlo | Eq. (10) | error |
|---|---|---|---|
| normal (beta=1) | 1.2340 | 1.2337 | 0.02 % |
| space (beta=1) | 6.7888 | 6.7854 | 0.05 % |
| time (beta=0.5) | 1.5424 | 1.2337 | 25.0 % |
| mixed (beta=0.75) | 1.8053 | 1.4850 | 21.6 % |
| neutral (beta=0.5) | 9.8821 | 4.9348 | 100.2 % |

The corrected expression is

```
var(log|X(t)|) = pi^2/6 * [ (2 - beta^2) / alpha^2 + 1/2 ] - (pi theta / 2 alpha)^2
```

Three independent confirmations, all in `tests/test_theory.py`:

1. Monte Carlo matches it to under 2 % in every scenario.
2. It equals the paper's own Proposition 5 minus Proposition 3 squared. Prop. 5
   already carries the term, so Props. 4 and 5 contradict each other and Prop. 5
   is the correct one.
3. The authors' MATLAB (`fract_diff_est_logm.m` line 37) already computes the
   corrected inversion.

So the paper's results are unaffected, but Algorithm 2 line 8 carries the same
omission. Implementing from the printed text costs a mean of about 23 % on alpha
across the two `beta < 1` scenarios, 13-28 % depending on the seed (10 seeds,
N = 1000). Both forms are available as
`algorithm2(..., variant="corrected" | "as_printed")`.

Line 8 is printed as `alpha_hat <- (sigma^2 * 6/pi^2 - 1/2)^(-1/2)`, which is
more reduced still: it drops the `(pi theta / 2 alpha)^2` term that Eq. (10)
does carry. Implementing that line verbatim gives 13.3 %, 22.3 % and 15.5 % on
the neutral, time and mixed scenarios - not uniformly worse than inverting
Eq. (10), because dropping theta partly cancels the beta error.

### Proposition 5 is correct only at D = 1

Eq. (11) prints the `log(t)` coefficient as `2 (beta gamma / alpha)(beta/alpha - 1)`.
Expanding `(E log|X(t)|)^2` from Eq. (9) gives that term plus
`2 beta log(D) log(t) / alpha^2`, which is absent. It vanishes exactly when
`D = 1`, and every experiment in the paper uses `D = 1`. Monte Carlo at `D = 3`,
mixed scenario, 300k samples, as `(MC - formula)/SE`: `-148` and `+165` at
`t = 0.2` and `t = 5`, against `0.5` and `0.7` once the term is restored.
Eq. (11) is not used by Algorithm 2, so nothing downstream is affected.

### The paper reproduces

Corrected Algorithm 2, N = 1000, L = 60, 20 seeds, D = 1:

| scenario | alpha | beta | theta | D |
|---|---|---|---|---|
| normal | 0.8 % | 1.4 % | 0.8 % | 1.6 % |
| neutral | 0.1 % | 0.1 % | 0.1 % | 3.0 % |
| space | 0.5 % | 0.3 % | 0.1 % | 1.1 % |
| time | 0.9 % | 0.1 % | 0.9 % | 6.5 % |

matching Tables 1-3, including that D is the hardest parameter and that
`alpha = 2` is the hardest regime.

### An unreported fragility

`X(t) = 0` exactly sends `log|X|` to `-inf` and destroys both estimators. Masking
the zeros fixes it; the estimators then tolerate 20 % zeros with under 2 % bias.
Clamping to a small floor is worse than masking, since it injects a large finite
outlier. This matters for agent logs, where a repeated message gives exactly zero
displacement.

### What transfers to a multi-agent setting

Ground truth `alpha = 1.5, beta = 0.75, theta = 0.25`; perturbations relative to
the IQR of X at the final time.

| violation | effect | verdict |
|---|---|---|
| few time points, L = 5 to 100 | alpha error 1.3-1.8 % at every L | fine; precision comes from the number of trajectories |
| 0-20 % repeated states | alpha error under 1.5 % | fine, with masking |
| coupling 0.05 / 0.2 / 0.5 / 0.8 | alpha error 0.4 / 3.6 / 9.6 / 12.2 %; theta sd 0.04 to 0.64 | usable to about 0.2; theta fails first |
| noise 0.01 / 0.1 / 0.5 x IQR | alpha 1.52 / 1.65 / 2.00 | noise biases alpha towards 2 |
| bounded observable, R = 0.5 to 10 x IQR | alpha 2.00 / 1.89 / 1.61 / 1.54 | a bounded space manufactures alpha = 2 |
| drift 0.25 / 1 / 4 x IQR | theta -0.09 / -0.66 / -0.90 | theta unusable undetrended |

Two consequences. Trajectories must be **independent runs**, not the agents
inside one run, since agents in a conversation are coupled by construction. And
the observable must be a **signed 1-D projection**: the radial norm `||X||`
biases alpha and beta by about 27 %, at every dimension tested.

### Why use this method at all

Three processes with an identical self-similarity exponent `beta/alpha = 0.5`:

| true (alpha, beta) | beta/alpha estimated | alpha estimated |
|---|---|---|
| (2.0, 1.00) | 0.501 | 1.976 |
| (1.6, 0.80) | 0.484 | 1.611 |
| (1.2, 0.60) | 0.481 | 1.210 |

The ratio is one number for all three, which is everything MSD or Hurst analysis
can see. The log-variance channel separates them, Cohen's d = 6.1. Separating
memory from jumpiness is the reason to use this estimator.

### A caution about the obvious observable

`experiments/05_embedding_pilot.py`, on `all-MiniLM-L6-v2` over 40-turn
conversations:

| conversation | headroom | growth exponent |
|---|---|---|
| 40 distinct turns across 20 topics | 26.3 | 0.064 |
| same, unnormalised | 24.8 | 0.066 |
| control, 40 near-identical turns | 34.1 | 0.078 |

Free diffusion would give roughly 0.25-0.5. A conversation deliberately
traversing twenty topics has the same displacement growth as one repeating a
single sentence. The headroom check passes, so this is not a normalisation
artifact: the observable is simply not diffusive.

This was one conversation with templated turns and no error bars, so it was a
warning rather than a result. It has since been confirmed at scale, and the
follow-up it originally suggested has been ruled out.

**Update.** On a 200-run corpus and on 100 runs generated for the purpose, the
obstruction was traced to the encoder rather than to the conversations: changing
a single word of a 16-word sentence moves the embedding 26% of the attainable
diameter, and the distance saturates once the window is replaced, so no text
process gets near the regime where free diffusion is visible. A latent process
with known alpha and beta, rendered as text and passed through the same encoder,
is recovered 0% of the time against a no-encoder ceiling of 0.62.

The CTRW route this section used to recommend does not apply. There are no
waiting times to measure, and the increments are near-Gaussian rather than
stable, so neither leg of a renewal model exists. The model family is rejected
rather than unfitted: fits violate the admissibility bound
|theta| <= min(alpha, 2 - alpha) in 20 of 20 projection directions, and a
parametric bootstrap over an (alpha, beta) grid at the same design cannot reach
the data on either the TA-MSD slope or the ergodicity-breaking parameter. An
Ornstein-Uhlenbeck process with a relaxation time of three to four events
reproduces both. `algorithm2` now returns an `admissible` flag so an
out-of-family fit cannot be quoted by accident.

## Caveats

* All synthetic results are CTRW approximations to the scaling limit, not exact
  samples of the PDE solution; the Proposition checks use the exact marginal.
* The embedding pilot is a single conversation with one small encoder.
* The topology contrast in `04` is not reliably powered and is marked as such.
* No advection term here. It has since been worked out, and neither half of it
  is new. The forward theory is published: the space-time fractional
  advection-dispersion equation with a Riesz-Feller operator is solved in Huang
  and Liu (2005), and the Laplace-Fourier form `s^(b-1) / (s^b + psi(k))` for a
  drift-plus-skewed generator is in Meerschaert, Benson, Scheffler and Baeumer,
  Phys. Rev. E **65**, 041103 (2002). The inverse problem is published too —
  including by the paper re-implemented here, which recovers `alpha`, `beta`,
  `theta` and `D` from an ensemble of trajectories, and by
  Chakraborty, Meerschaert and Lim (2009) and the FracFit package
  (Kelly et al. 2017), which fit `alpha`, skewness, velocity and dispersivity
  to plume and breakthrough data. What is left is narrower: the drift as a
  fifth jointly-estimated parameter, which Znaidi et al. name as future work in
  their own conclusion.
* The failure mode is the part worth knowing. Unmodelled drift pulls `alpha`
  down by a third and drags `theta` from 0 to about -0.83 while `beta` survives,
  and the estimator reports every one of those fits as admissible. The drift is
  being absorbed into the skew, so the fix is to fit it rather than to detrend.

## License

MIT, see [LICENSE](LICENSE).
