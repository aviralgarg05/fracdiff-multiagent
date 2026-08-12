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

## What the checks found

### Proposition 4 as printed is wrong for beta < 1

Eq. (10) omits `(pi^2 / 6 alpha^2)(1 - beta^2)`, the variance contributed by the
random subordinator. It is exact at `beta = 1` and wrong otherwise:

| scenario | Monte Carlo | Eq. (10) | error |
|---|---|---|---|
| normal (beta=1) | 1.2345 | 1.2337 | 0.07 % |
| space (beta=1) | 6.7766 | 6.7854 | 0.13 % |
| time (beta=0.5) | 1.5421 | 1.2337 | 25.0 % |
| mixed (beta=0.75) | 1.8067 | 1.4850 | 21.6 % |
| neutral (beta=0.5) | 9.8804 | 4.9348 | 100.2 % |

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
omission, and implementing from the printed text gives 22-25 % error on alpha.
Both forms are available as `algorithm2(..., variant="corrected" | "as_printed")`.

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

This is one conversation with templated turns and no error bars, so it is a
warning rather than a result. It is the reason the plan starts with a validity
gate rather than with a model fit, and why the first alternative to try is
modelling the conversation as an explicit CTRW, taking beta from the tail of the
dwell-time distribution and alpha from the tail of the jump-size distribution.

## Caveats

* All synthetic results are CTRW approximations to the scaling limit, not exact
  samples of the PDE solution; the Proposition checks use the exact marginal.
* The embedding pilot is a single conversation with one small encoder.
* The topology contrast in `04` is not reliably powered and is marked as such.
* No advection term. Adding one, and deriving the corresponding moment
  expressions, is the natural next piece of theory.

## License

MIT, see [LICENSE](LICENSE).
