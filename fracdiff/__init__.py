"""Independent re-implementation and validation of Znaidi et al. (2020).

Znaidi M. R., Gupta G., Asgari K. & Bogdan P., "Identifying Arguments of
Space-Time Fractional Diffusion: Data-Driven Approach",
Front. Appl. Math. Stat. 6:14 (2020), doi:10.3389/fams.2020.00014.
"""
from .estimators import algorithm1, algorithm2, is_admissible, theta_over_alpha, w_L
from .simulate import ctrw_paths, ctrw_paths_highdim, marginal_X
from .stable import feller_stable, feller_to_nolan, pos_stable_sub, stable_rvs
from .theory import (EULER, prop1_abs, prop2_signed, prop3_logmean,
                     prop4_logvar_as_printed, prop4_logvar_corrected, prop5_logsq,
                     prop5_logsq_as_printed, prop5_logsq_corrected)

__version__ = "0.1.0"

SCENARIOS = {
    # name:      (alpha, beta, theta) -- the paper's four regimes
    "normal":    (2.0, 1.0, 0.0),
    "neutral":   (0.5, 0.5, 0.5),
    "space":     (0.5, 1.0, 0.25),
    "time":      (2.0, 0.5, 0.0),
    # a mixed case that exercises alpha, beta and theta at once (stress tests)
    "mixed":     (1.5, 0.75, 0.25),
}

__all__ = [
    "algorithm1", "algorithm2", "is_admissible", "theta_over_alpha", "w_L",
    "ctrw_paths", "ctrw_paths_highdim", "marginal_X",
    "stable_rvs", "feller_to_nolan", "feller_stable", "pos_stable_sub",
    "prop1_abs", "prop2_signed", "prop3_logmean",
    "prop4_logvar_as_printed", "prop4_logvar_corrected", "prop5_logsq",
    "prop5_logsq_as_printed", "prop5_logsq_corrected",
    "EULER", "SCENARIOS", "__version__",
]
