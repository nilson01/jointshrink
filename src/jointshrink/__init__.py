"""Multivariate adaptive shrinkage for effects across conditions."""

from .data import MashData, set_data
from .patterns import (
    canonical_covariances,
    validate_covariances,
    select_strong,
    cov_pca,
    cov_ed,
    CovarianceFit,
    make_scale_grid,
    expand_covariances,
    estimate_null_correlation,
    CorrelationEstimate,
)
from .results import PosteriorResult
from .model import Mash, MashPrior, MashResult, fit, fit_prior, apply_prior, score_samples
from .io import save_data, load_data, save_prior, load_prior, save_result, load_result
from .plotting import (
    plot_covariances,
    plot_weights,
    plot_sharing,
    plot_intervals,
    plot_diagnostics,
    plot_effects,
)
from .student import fit_student_t, apply_student_prior, StudentTResult, load_student_result
from .likelihoods import make_likelihood, Likelihood, FAMILIES
from .regression import (
    fit_regression,
    RegressionResult,
    DrawResult,
    fit_gaussian_regression,
    fit_logistic_regression,
    fit_poisson_regression,
    fit_negative_binomial_regression,
    fit_survival,
    fit_multinomial,
    fit_ordinal,
    fit_zero_inflated_poisson,
    fit_zero_inflated_negative_binomial,
    fit_hurdle_poisson,
    fit_hurdle_negative_binomial,
    save_regression_result,
    load_regression_result,
)
from .regression_plots import (
    plot_coefficients,
    plot_chains,
    plot_regression_weights,
    plot_random_effects,
)


__version__ = "0.1.0.dev0"
__all__ = [
    "MashData",
    "set_data",
    "canonical_covariances",
    "validate_covariances",
    "select_strong",
    "cov_pca",
    "cov_ed",
    "CovarianceFit",
    "make_scale_grid",
    "expand_covariances",
    "estimate_null_correlation",
    "CorrelationEstimate",
    "PosteriorResult",
    "Mash",
    "MashPrior",
    "MashResult",
    "fit",
    "fit_prior",
    "apply_prior",
    "score_samples",
    "save_data",
    "load_data",
    "save_prior",
    "load_prior",
    "save_result",
    "load_result",
    "plot_covariances",
    "plot_weights",
    "plot_sharing",
    "plot_intervals",
    "plot_diagnostics",
    "plot_effects",
]

__all__ += [
    "fit_student_t",
    "apply_student_prior",
    "StudentTResult",
    "load_student_result",
    "make_likelihood",
    "Likelihood",
    "FAMILIES",
    "fit_regression",
    "RegressionResult",
    "DrawResult",
    "fit_gaussian_regression",
    "fit_logistic_regression",
    "fit_poisson_regression",
    "fit_negative_binomial_regression",
    "fit_survival",
    "fit_multinomial",
    "fit_ordinal",
    "fit_zero_inflated_poisson",
    "fit_zero_inflated_negative_binomial",
    "fit_hurdle_poisson",
    "fit_hurdle_negative_binomial",
    "save_regression_result",
    "load_regression_result",
]

__all__ += ["plot_coefficients", "plot_chains", "plot_regression_weights", "plot_random_effects"]
