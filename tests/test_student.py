"""Independent numerical and distribution checks for native t summary models."""

import json
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import multivariate_t, norm, t

from jointshrink import (
    MashPrior,
    StudentTResult,
    set_data,
    fit_student_t,
    apply_student_prior,
    load_student_result,
    save_result,
    load_result,
)
from jointshrink.student import _Integral


def scalar_data(y, scale=1.0):
    return set_data([[y, np.nan]], standard_errors=[[scale, 1.0]])


def fixed_prior(matrix):
    return MashPrior([matrix], ["fixed"], [1.0])


@pytest.mark.parametrize("df", [0.1, 1.0, 2.0, 3.0, 20.0, 1000.0])
def test_null_prior_is_exact_multivariate_t_density(df):
    y = np.array([[1.2, -0.4], [3.1, np.nan]])
    v = np.array([[0.8, 0.3], [0.3, 1.5]])
    data = set_data(y, error_covariance=v)
    result = apply_student_prior(data, fixed_prior(np.zeros((2, 2))), df)
    expected = []
    for row, mask in zip(y, data.mask):
        expected.append(multivariate_t.logpdf(row[mask], shape=v[np.ix_(mask, mask)], df=df))
    assert_allclose(result.log_likelihood, expected, atol=3e-7, rtol=1e-8)
    assert_array_equal(result.posterior_mean, np.zeros((2, 2)))
    assert_array_equal(result.posterior_covariance, np.zeros((2, 2, 2)))
    assert_allclose(result.joint_null_probability, 1)
    assert_allclose(result.lfsr, 1)
    assert result.diagnostics["quadrature_converged"]


@pytest.mark.parametrize(
    "df,y,scale,prior_sd", [(1.0, 2.3, 0.7, 1.2), (3.0, -0.8, 1.5, 0.6), (0.2, 4.0, 1.0, 2.0)]
)
def test_convolution_against_direct_effect_space_integrals(df, y, scale, prior_sd):
    # This integrates over the effect, not the implementation's log precision.
    def integrand(effect):
        return norm.pdf(effect, scale=prior_sd) * t.pdf((y - effect) / scale, df=df) / scale

    marginal = quad(integrand, -np.inf, np.inf, epsabs=2e-12, epsrel=2e-11)[0]
    mean = quad(lambda b: b * integrand(b), -np.inf, np.inf, epsabs=2e-12)[0] / marginal
    second = quad(lambda b: b * b * integrand(b), -np.inf, np.inf, epsabs=2e-12)[0] / marginal
    cdf = quad(integrand, -np.inf, 0.4, epsabs=2e-12)[0] / marginal
    result = apply_student_prior(
        scalar_data(y, scale),
        fixed_prior(np.diag([prior_sd**2, 0.0])),
        df,
        quadrature_order=128,
        max_quadrature_order=512,
        quadrature_tol=1e-7,
    )
    assert_allclose(result.log_likelihood, [np.log(marginal)], atol=3e-8)
    assert_allclose(result.posterior_mean[0, 0], mean, atol=3e-8)
    assert_allclose(result.posterior_covariance[0, 0, 0], second - mean**2, atol=3e-8)
    assert_allclose(result.cdf(0.4)[0, 0], cdf, atol=3e-8)
    assert_allclose(result.lfdr[0], [0, 1])
    assert_array_equal(result.interval()[0, 1], [0, 0])


def test_gaussian_limit_preserves_complete_and_observed_blocks():
    data = set_data([[1.2, -0.3], [2.0, np.nan]], standard_errors=[[0.8, 1.1], [0.5, 2.0]])
    prior = fixed_prior(np.array([[1.0, 0.4], [0.4, 0.7]]))
    result = apply_student_prior(data, prior, 1e8)
    gaussian = prior.apply(data)
    assert_allclose(result.posterior_mean, gaussian.posterior_mean, atol=3e-8)
    assert_allclose(result.posterior_covariance, gaussian.posterior_covariance, atol=3e-8)
    assert_allclose(result.log_likelihood, gaussian.log_likelihood, atol=3e-8)


def test_ee_ez_change_of_variables_with_correlated_missing_data():
    effects = [[1.0, -0.3], [2.0, np.nan]]
    se = np.array([[0.8, 1.2], [0.8, 1.2]])
    corr = np.array([[1.0, 0.25], [0.25, 1.0]])
    prior_cov = np.array([[1.0, 0.4], [0.4, 0.7]])
    ee = set_data(effects, se, correlation=corr)
    ez = set_data(effects, se, correlation=corr, alpha=1)
    pz = MashPrior([prior_cov / se[0, :, None] / se[0, None, :]], ["fixed"], [1], alpha=1)
    a = apply_student_prior(ee, fixed_prior(prior_cov), [1.0, 7.0])
    b = apply_student_prior(ez, pz, [1.0, 7.0])
    assert_allclose(a.posterior_mean, b.posterior_mean, atol=2e-12)
    assert_allclose(a.posterior_covariance, b.posterior_covariance, atol=2e-12)
    assert_allclose(a.log_likelihood, b.log_likelihood, atol=2e-12)
    assert_allclose(a.cdf(0.2), b.cdf(0.2), atol=2e-12)


def test_missing_unobserved_error_covariance_has_no_effect():
    a = set_data([[1.0, np.nan]], error_covariance=[[1.0, 0.1], [0.1, 4.0]])
    b = set_data([[1.0, np.nan]], error_covariance=[[1.0, -0.2], [-0.2, 2.0]])
    prior = fixed_prior(np.array([[1.0, 0.5], [0.5, 1.0]]))
    x = apply_student_prior(a, prior, 3)
    y = apply_student_prior(b, prior, 3)
    assert_allclose(x.posterior_mean, y.posterior_mean)
    assert_allclose(x.posterior_covariance, y.posterior_covariance)
    assert_allclose(x.log_likelihood, y.log_likelihood)
    assert x.posterior_mean[0, 1] > 0  # Missing effect is inferred through the prior.


@pytest.mark.parametrize("df", [1e-3, 1.0, 2.0, 1e200, 1e308])
def test_finite_df_has_no_arbitrary_upper_cutoff(df):
    result = apply_student_prior(scalar_data(0.5), fixed_prior(np.eye(2)), df)
    assert np.all(np.isfinite(result.posterior_mean))
    assert np.all(np.isfinite(result.posterior_covariance))
    assert_allclose(result.component_weights.sum(1), 1)


@pytest.mark.parametrize("df", [0, -1, np.inf, np.nan, True, [3, 4], [True]])
def test_invalid_degrees_rejected(df):
    with pytest.raises((ValueError, TypeError)):
        apply_student_prior(scalar_data(0.5), fixed_prior(np.eye(2)), df)


@pytest.mark.parametrize(
    "options",
    [
        dict(quadrature_order=3),
        dict(quadrature_order=True),
        dict(quadrature_order=8.0, max_quadrature_order=16),
        dict(quadrature_order=64, max_quadrature_order=32),
        dict(quadrature_tol=0),
        dict(quadrature_tol=np.nan),
    ],
)
def test_invalid_quadrature_options_rejected(options):
    with pytest.raises(ValueError):
        apply_student_prior(scalar_data(0.5), fixed_prior(np.eye(2)), 3, **options)


def test_unchecked_single_order_warns_and_does_not_claim_convergence():
    with pytest.warns(RuntimeWarning, match="did not meet"):
        result = apply_student_prior(
            scalar_data(4.0), fixed_prior(np.eye(2)), 1, quadrature_order=4, max_quadrature_order=4
        )
    assert result.diagnostics["quadrature_converged"] is False
    assert result.diagnostics["quadrature_relative_change"] is None


def test_tail_bound_guard_rejects_nondecaying_integrand(monkeypatch):
    # Exercise the failure guard with an intentionally nonintegrable controlled
    # integrand. This is not a statistical claim about valid t likelihoods.
    monkeypatch.setattr(
        _Integral, "evaluate", lambda self, x, moments=False: np.ones_like(np.atleast_1d(x))
    )
    with pytest.raises(FloatingPointError, match="tails could not be bounded"):
        apply_student_prior(scalar_data(0.5), fixed_prior(np.eye(2)), 3)


def test_memory_guard_checks_before_allocating_nodes():
    with pytest.raises(MemoryError, match="256 MiB"):
        apply_student_prior(
            scalar_data(0.5),
            fixed_prior(np.eye(2)),
            3,
            quadrature_order=10**9,
            max_quadrature_order=10**9,
        )


def test_tiny_positive_prior_variance_stays_continuous():
    result = apply_student_prior(scalar_data(0.0), fixed_prior(np.diag([1e-200, 0.0])), 3)
    assert result.posterior_covariance[0, 0, 0] > 0
    assert_allclose(result.lfdr, [[0, 1]])
    assert_allclose(result.lfsr, [[0.5, 1]])
    assert result.quantile(0.975)[0, 0] > 0


def test_one_prior_component_fit_and_reuse():
    data = set_data([[0.5, -0.3], [1.0, np.nan]])
    fit = fit_student_t(data, 3, {"identity": np.eye(2)}, scales=[1], point_mass=False)
    assert_allclose(fit.prior.weights, [1])
    assert fit.diagnostics["converged"]
    new = scalar_data(0.2)
    assert_allclose(
        fit.shrink(new, df=4).posterior_mean, apply_student_prior(new, fit.prior, 4).posterior_mean
    )
    assert_allclose(fit.score_samples(new, df=4), fit.shrink(new, df=4).log_likelihood)


def test_fitted_weights_obey_simplex_and_improve_uniform_objective():
    data = set_data([[0.0, 0.1], [2.0, 1.4], [-1.0, -0.8]])
    fit = fit_student_t(data, [3, 4, 5], {"identity": np.eye(2)}, scales=[1])
    assert_allclose(fit.prior.weights.sum(), 1)
    assert np.all(fit.prior.weights >= 0)
    uniform = MashPrior(
        fit.prior.covariances,
        fit.prior.names,
        np.full(len(fit.prior.weights), 1 / len(fit.prior.weights)),
    )
    initial = apply_student_prior(data, uniform, [3, 4, 5])
    assert fit.log_likelihood.sum() >= initial.log_likelihood.sum() - 1e-10
    assert fit.diagnostics["converged"]


def test_persistence_both_entrypoints_keep_t_likelihood(tmp_path):
    result = fit_student_t(scalar_data(1.3), 1.0, {"identity": np.eye(2)}, scales=[1])
    for name, saver, loader in [
        ("method", result.save, load_student_result),
        ("generic", lambda p: save_result(result, p), load_result),
    ]:
        path = tmp_path / (name + ".json")
        saver(path)
        restored = loader(path)
        assert isinstance(restored, StudentTResult)
        assert_allclose(restored.posterior_mean, result.posterior_mean, atol=0, rtol=0)
        assert_allclose(restored.log_likelihood, result.log_likelihood, atol=0, rtol=0)
        assert_allclose(restored.df, result.df)
        with pytest.raises(FileExistsError):
            saver(path)
    payload = json.loads(path.read_text())
    payload["payload"]["df"] = [True]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_result(path)


def test_null_slab_responsibilities_follow_independent_bayes_calculation():
    y, df, scale, slab_sd = 1.5, 2.5, 0.8, 1.2
    slab_marginal = quad(
        lambda b: norm.pdf(b, scale=slab_sd) * t.pdf((y - b) / scale, df) / scale,
        -np.inf,
        np.inf,
        epsabs=1e-12,
    )[0]
    null_marginal = t.pdf(y / scale, df) / scale
    null_posterior = 0.3 * null_marginal / (0.3 * null_marginal + 0.7 * slab_marginal)
    prior = MashPrior([np.zeros((2, 2)), np.diag([slab_sd**2, 0.0])], ["null", "slab"], [0.3, 0.7])
    result = apply_student_prior(scalar_data(y, scale), prior, df)
    assert_allclose(result.joint_null_probability, [null_posterior], atol=2e-8)
    assert_allclose(result.lfdr, [[null_posterior, 1]], atol=2e-8)
    contrast = result.contrast([[1, -1]])
    assert_allclose(contrast.posterior_mean[:, 0], result.posterior_mean[:, 0])
    assert_allclose(contrast.cdf(0.1)[:, 0], result.cdf(0.1)[:, 0])
