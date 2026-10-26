import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize_scalar
from jointshrink import set_data, MashPrior, fit


def test_gaussian_conditioning_matches_independent_linear_algebra_oracle():
    observed = np.array([1.2, -0.7])
    noise = np.array([[0.4, 0.1], [0.1, 0.9]])
    prior_covariance = np.array([[2, 0.8], [0.8, 1]])
    data = set_data([observed], error_covariance=noise)
    result = MashPrior([prior_covariance], ["a"], [1]).apply(data)
    expected_mean = prior_covariance @ np.linalg.solve(prior_covariance + noise, observed)
    expected_cov = np.linalg.inv(np.linalg.inv(prior_covariance) + np.linalg.inv(noise))
    assert_allclose(result.posterior_mean[0], expected_mean)
    assert_allclose(result.posterior_covariance[0], expected_cov)
    assert_allclose(
        result.log_likelihood, [multivariate_normal.logpdf(observed, cov=prior_covariance + noise)]
    )


def test_missing_block_conditions_unobserved_coordinate_exactly():
    covariance = np.array([[2, 1], [1, 3]])
    data = set_data([[1.5, np.nan]], standard_errors=[[0.5, 2]])
    result = MashPrior([covariance], ["shared"], [1]).apply(data)
    expected_mean = covariance[:, 0] * 1.5 / 2.25
    expected_cov = covariance - np.outer(covariance[:, 0], covariance[0]) / 2.25
    assert_allclose(result.posterior_mean[0], expected_mean)
    assert_allclose(result.posterior_covariance[0], expected_cov)
    assert_allclose(result.log_likelihood, [norm.logpdf(1.5, scale=1.5)])
    assert result.posterior_mean[0, 1] != 0


def test_ez_backtransformation_and_observed_jacobian():
    se = np.array([[2, 3], [4, 5]])
    effects = np.array([[1, 2], [3, np.nan]])
    data = set_data(effects, se, alpha=1)
    prior = MashPrior([np.eye(2)], ["a"], [1], alpha=1)
    result = prior.apply(data)
    assert_allclose(result.posterior_mean[0], effects[0] / 2)
    assert_allclose(result.posterior_covariance[0], np.diag(se[0] ** 2 / 2))
    expected = [
        multivariate_normal.logpdf(effects[0] / se[0], cov=2 * np.eye(2)) - np.log(se[0]).sum(),
        norm.logpdf(3 / 4, scale=np.sqrt(2)) - np.log(4),
    ]
    assert_allclose(result.log_likelihood, expected)
    assert_allclose(prior.score_samples(data), expected)
    subset = data.subset([1])
    assert subset.alpha == 1
    assert_allclose(subset.scale, se[1:])


def test_simplex_fit_matches_independent_scalar_optimization():
    values = np.array([[0, 0], [0.1, -0.1], [0.3, 0.2], [3, 2], [-2, -3], [0.2, 0.1]])
    data = set_data(values)
    result = fit(data, {"alternative": np.eye(2)}, scales=[2], normalize=False, tol=1e-10)
    null_density = multivariate_normal.pdf(values, cov=np.eye(2))
    alternative_density = multivariate_normal.pdf(values, cov=5 * np.eye(2))

    def objective(w):
        return -np.log(w * null_density + (1 - w) * alternative_density).sum()

    oracle = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-13})
    assert_allclose(result.prior.weights[0], oracle.x, atol=2e-7)
    assert_allclose(result.log_likelihood.sum(), -oracle.fun, atol=1e-9)
    assert result.diagnostics["converged"]


def test_prior_reuse_does_not_refit_and_scale_mismatch_fails():
    prior = MashPrior([np.zeros((2, 2)), np.eye(2)], ["null", "a"], [0.4, 0.6])
    data = set_data([[20, 10]])
    result = prior.apply(data)
    assert_allclose(result.prior.weights, [0.4, 0.6])
    assert_allclose(result.shrink(data).posterior_mean, result.posterior_mean)
    with pytest.raises(ValueError):
        prior.apply(set_data([[1, 2]], alpha=1))
    with pytest.raises(ValueError):
        prior.apply(set_data([[1, 2, 3]]))


@pytest.mark.parametrize(
    "effects,kwargs",
    [
        ([], {}),
        ([1, 2], {}),
        ([[1]], {}),
        ([[np.nan, np.nan]], {}),
        ([[np.inf, 1]], {}),
        ([[1, 2]], {"standard_errors": 0}),
        ([[1, 2]], {"standard_errors": [-1, 1]}),
        ([[1, 2]], {"standard_errors": [np.nan, 1]}),
        ([[1, 2]], {"standard_errors": [1, 2, 3]}),
        ([[1, 2]], {"alpha": 0.5}),
        ([[1, 2]], {"row_names": []}),
        ([[1, 2]], {"condition_names": ["a", "a"]}),
        ([[1, 2]], {"correlation": [[1, 2], [2, 1]]}),
        ([[1, 2]], {"correlation": [[2, 0], [0, 2]]}),
        ([[1, 2]], {"error_covariance": np.eye(2), "standard_errors": 2}),
        ([[1, 2]], {"error_covariance": np.eye(2), "correlation": np.eye(2)}),
    ],
)
def test_malformed_data_fails(effects, kwargs):
    with pytest.raises(ValueError):
        set_data(effects, **kwargs)


def test_data_copies_inputs_and_keeps_missing_scale_information():
    raw = np.array([[1.0, np.nan], [2, 3]])
    data = set_data(raw, [[2, 3], [4, 5]], alpha=1)
    raw[0, 0] = 999
    assert data.effects[0, 0] == 1
    assert data.mask.tolist() == [[True, False], [True, True]]
    assert data.scale[0, 1] == 3
    with pytest.raises(ValueError):
        data.effects[0, 0] = 5


@pytest.mark.parametrize(
    "covariances,names,weights",
    [
        ([np.eye(2)], ["a"], [0.3]),
        ([np.eye(2)], ["a"], [-1]),
        ([np.eye(2)], ["a"], [np.nan]),
        ([np.eye(2), np.eye(2)], ["a", "a"], [0.5, 0.5]),
        ([np.array([[1, 2], [2, 1]])], ["a"], [1]),
    ],
)
def test_malformed_priors_fail(covariances, names, weights):
    with pytest.raises(ValueError):
        MashPrior(covariances, names, weights)


def test_all_null_boundary_fit_and_null_penalty_match_scalar_oracle():
    values = np.array([[0.0, 0.0], [0.1, -0.1], [-0.2, 0.1]])
    result = fit(set_data(values), {"a": np.eye(2)}, scales=[2], tol=1e-10)
    assert result.prior.weights[0] > 1 - 1e-7
    values = np.array([[0, 0], [0.1, -0.1], [3, 2], [-2, -3]])
    null_density = multivariate_normal.pdf(values, cov=np.eye(2))
    alt_density = multivariate_normal.pdf(values, cov=5 * np.eye(2))

    def objective(w):
        return -np.log(w * null_density + (1 - w) * alt_density).sum() - 2 * np.log(w)

    oracle = minimize_scalar(
        objective, bounds=(1e-12, 1), method="bounded", options={"xatol": 1e-13}
    )
    penalized = fit(set_data(values), {"a": np.eye(2)}, scales=[2], null_weight=3, tol=1e-10)
    assert_allclose(penalized.prior.weights[0], oracle.x, atol=2e-7)
    with pytest.raises(ValueError):
        fit(set_data(values), {"a": np.eye(2)}, scales=[2], point_mass=False, null_weight=3)


def test_large_prior_preserves_finite_observation_uncertainty():
    prior = MashPrior([1e20 * np.eye(2)], ["diffuse"], [1])
    result = prior.apply(set_data([[1, 2]]))
    assert_allclose(result.posterior_covariance[0], np.eye(2), rtol=1e-12)
    assert_allclose(result.posterior_mean[0], [1, 2])


def test_extreme_finite_loglikelihood_recovers_from_boundary_initialization():
    from jointshrink._numeric import weight_fit

    weights, diagnostics = weight_fit(
        np.array([[-1000.0, 0.0], [-1000.0, 0.0]]), initial_weights=[1, 0], maxiter=100
    )
    assert_allclose(weights, [0, 1], atol=1e-7)
    assert diagnostics["converged"]
    assert np.isfinite(diagnostics["objective"])


@pytest.mark.parametrize("standard_error", [1e-200, 1e200])
def test_standard_error_square_underflow_and_overflow_fail_at_input(standard_error):
    with pytest.raises(ValueError):
        set_data([[1, 2]], standard_error)


def test_likelihood_only_kernel_matches_full_without_moment_arrays():
    from jointshrink._numeric import component_posterior

    data = set_data([[1, 2], [0.2, np.nan]])
    covariance = np.stack([np.zeros((2, 2)), np.eye(2)])
    full = component_posterior(data, covariance)
    density_only = component_posterior(data, covariance, compute_moments=False)
    assert_allclose(density_only[0], full[0])
    assert density_only[1] is None and density_only[2] is None


def test_fit_prior_uses_likelihood_only_and_batches_preserve_row_order(monkeypatch):
    import jointshrink.model as model

    original = model.component_posterior
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs.get("compute_moments", True))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "component_posterior", checked)
    data = set_data(
        [[0, 1], [2, 3], [0.1, 0.2], [-1, -2], [0.3, 0.5]], row_names=["a", "b", "c", "d", "e"]
    )
    prior = model.fit_prior(data, {"a": np.eye(2)}, scales=[1, 2])
    assert calls and not any(calls)
    full = prior.apply(data)
    batches = list(prior.iter_apply(data, batch_size=2))
    assert [batch.data.n_effects for batch in batches] == [2, 2, 1]
    assert tuple(name for batch in batches for name in batch.row_names) == data.row_names
    assert_allclose(np.concatenate([b.posterior_mean for b in batches]), full.posterior_mean)
    assert_allclose(
        np.concatenate([b.posterior_covariance for b in batches]), full.posterior_covariance
    )
    assert_allclose(np.concatenate([b.log_likelihood for b in batches]), full.log_likelihood)
    for bad in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            list(prior.iter_apply(data, batch_size=bad))


def test_posterior_memory_guard_prevents_component_tensor_allocation(monkeypatch):
    from types import SimpleNamespace
    import jointshrink._numeric as numeric

    data = SimpleNamespace(working_effects=SimpleNamespace(shape=(100000, 100)))
    original_empty = numeric.np.empty
    requested_shapes = []

    def checked_empty(shape, *args, **kwargs):
        requested_shapes.append(shape)
        assert len(shape) < 3, "large posterior tensor must not be allocated"
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(numeric.np, "empty", checked_empty)
    with pytest.raises(MemoryError, match="256 MiB"):
        numeric.component_posterior(data, np.eye(100)[None])
    assert all(len(shape) < 3 for shape in requested_shapes)
