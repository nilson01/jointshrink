"""Independent posterior oracles and public regression workflow checks.

Fixed-data posterior checks allow estimated Monte Carlo error. Tiny family
checks establish executability, not interval calibration or general mixing.
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit
from scipy.stats import multivariate_normal

from jointshrink import (
    DrawResult,
    FAMILIES,
    fit_regression,
    load_regression_result,
)
from jointshrink._sampling import chain_diagnostics, elliptical_slice, gaussian_factor


def _mc_check(draws, expected, *, floor=0.012):
    """Use independent chain/batch means to estimate a conservative MC tolerance."""
    x = np.asarray(draws)
    batches = x.reshape(x.shape[0], 20, x.shape[1] // 20, *x.shape[2:]).mean(axis=2)
    means = batches.reshape(-1, *x.shape[2:])
    se = means.std(axis=0, ddof=1) / np.sqrt(len(means))
    error = np.abs(x.mean(axis=(0, 1)) - expected)
    assert np.all(error < np.maximum(6 * se, floor)), (error, se, expected)


def test_one_component_gaussian_posterior_matches_conjugate_normal():
    X = np.array([[-1.0, 0.2], [-0.5, -0.5], [0.0, 0.7], [0.4, -0.3], [1.0, 0.5]])
    y = np.array([-0.5, -0.1, 0.3, 0.5, 0.8])
    sigma = 0.7
    prior_variance = 1.3**2
    covariance = np.linalg.inv(X.T @ X / sigma**2 + np.eye(2) / prior_variance)
    mean = covariance @ X.T @ y / sigma**2
    fit = fit_regression(
        X,
        y,
        sigma=sigma,
        scales=[1.3],
        point_mass=False,
        fit_intercept=False,
        chains=4,
        draws=1000,
        warmup=500,
        seed=40,
    )
    samples = fit.draws[..., 0]
    _mc_check(samples, mean)
    centered = samples - mean
    _mc_check(centered[..., :, None] * centered[..., None, :], covariance, floor=0.018)
    assert_allclose(fit.intercept, 0)
    assert_allclose(fit.weight_draws, 1)
    assert fit.diagnostics["convergence"] != "established"


def test_point_mass_posterior_matches_integrated_dirichlet_mixture():
    # With one predictor row, integrating Dirichlet(2, 3) gives prior masses 2/5, 3/5.
    X = np.array([[-1.0], [-0.4], [0.2], [0.7], [1.0]])
    y = np.array([-0.5, -0.1, 0.2, 0.5, 0.4])
    sigma, variance = 0.8, 1.2**2
    null_log = np.log(0.4) + multivariate_normal.logpdf(y, cov=sigma**2 * np.eye(5))
    alt_log = np.log(0.6) + multivariate_normal.logpdf(
        y, cov=sigma**2 * np.eye(5) + variance * X @ X.T
    )
    null_probability = np.exp(null_log - np.logaddexp(null_log, alt_log))
    conditional_variance = 1 / (1 / variance + np.sum(X**2) / sigma**2)
    conditional_mean = conditional_variance * float((X.T @ y).item()) / sigma**2
    fit = fit_regression(
        X,
        y,
        sigma=sigma,
        scales=[1.2],
        dirichlet=[2, 3],
        fit_intercept=False,
        chains=4,
        draws=1000,
        warmup=500,
        seed=41,
    )
    _mc_check((fit.draws == 0).astype(float), [[null_probability]], floor=0.02)
    _mc_check(fit.draws, [[(1 - null_probability) * conditional_mean]], floor=0.02)
    _mc_check(
        fit.weight_draws, [(2 + null_probability) / 6, (4 - null_probability) / 6], floor=0.02
    )
    assert np.all(fit.diagnostics["component_switches"] > 0)


def test_gaussian_random_intercepts_match_joint_normal_block_posterior():
    X = np.array([[-1.0], [-0.3], [0.4], [-0.6], [0.3], [1.0]])
    y = np.array([-0.1, 0.3, 0.7, -0.8, -0.1, 0.2])
    groups = ["a", "a", "a", "b", "b", "b"]
    Z = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    design = np.column_stack((X, Z))
    sigma, tau = 0.6, 0.7
    covariance = np.linalg.inv(
        design.T @ design / sigma**2 + np.diag([1.0, 1 / tau**2, 1 / tau**2])
    )
    mean = covariance @ design.T @ y / sigma**2
    fit = fit_regression(
        X,
        y,
        groups=groups,
        sigma=sigma,
        random_scale=tau,
        estimate_random_scale=False,
        scales=[1],
        point_mass=False,
        fit_intercept=False,
        chains=4,
        draws=1000,
        warmup=500,
        seed=42,
    )
    joined = np.concatenate((fit.draws[..., 0], fit.random_draws[..., 0, 0]), axis=-1)
    _mc_check(joined, mean, floor=0.02)
    assert_allclose(fit.random_scale_draws, tau)
    assert fit.diagnostics["random_scale_prior"] == "fixed"


class _FixedRng:
    def standard_normal(self, size):
        return np.ones(size)

    def random(self):
        return 0.5

    def uniform(self, lower, upper):
        return (lower + upper) / 2


def test_elliptical_slice_constant_likelihood_accepts_first_full_angle_move():
    current = np.array([2.0, -1.0])
    value, loglike, evaluations = elliptical_slice(current, np.eye(2), lambda _: -3.0, _FixedRng())
    # The midpoint of the random initial [0, 2*pi] range is pi.
    assert_allclose(value, -current, atol=1e-14)
    assert loglike == -3 and evaluations == 1


def test_singular_gaussian_factor_and_slice_preserve_exact_support():
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    factor = gaussian_factor(covariance)
    assert_allclose(factor @ factor.T, covariance, atol=1e-14)
    value, _, _ = elliptical_slice(np.ones(2), factor, lambda _: 0.0, np.random.default_rng(1))
    assert_allclose(value[0], value[1], atol=1e-14)
    point, _, count = elliptical_slice(np.zeros(2), np.zeros((2, 2)), lambda _: 0.0, _FixedRng())
    assert_allclose(point, 0)
    assert count == 0


def test_slice_rejects_nonfinite_start_and_does_not_silently_exhaust():
    with pytest.raises(ValueError, match="finite starting"):
        elliptical_slice(np.zeros(1), np.eye(1), lambda _: -np.inf, _FixedRng())
    with pytest.raises(RuntimeError, match="exhausted"):
        elliptical_slice(np.zeros(1), np.eye(1), lambda _: 0.0, _FixedRng(), max_steps=0)


def test_constant_chain_diagnostics_do_not_claim_convergence():
    d = chain_diagnostics(np.zeros((4, 20, 1, 1)))
    assert bool(d["constant_coordinates"][0, 0])
    assert np.isnan(d["split_rhat"][0, 0])
    assert np.isnan(d["effective_sample_size"][0, 0])
    separate = np.broadcast_to(np.arange(4)[:, None, None], (4, 20, 1))
    assert np.isinf(chain_diagnostics(separate)["split_rhat"][0])
    assert not chain_diagnostics(np.zeros((1, 3)))["diagnostics_available"]


@pytest.fixture(scope="module")
def tiny_fits():
    X = np.array([[-1.0], [-0.4], [0.0], [0.3], [0.7], [1.0]])
    fits = {}
    for i, family in enumerate(FAMILIES):
        options = {}
        if family == "gaussian":
            y = np.array([-0.4, -0.1, 0.2, 0.3, 0.2, 0.7])
        elif family == "binomial":
            y = [0, 1, 0, 1, 1, 1]
        elif family in ("multinomial", "ordinal"):
            y = [0, 1, 0, 1, 2, 2]
            options["n_categories"] = 3
        elif family in ("cox", "weibull"):
            y = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0]
            options.update(event=[1, 0, 1, 1, 0, 1], entry=[0.0, 0.2, 0.5, 0.0, 1.0, 0.0])
        else:
            y = [0, 1, 0, 2, 1, 3]
        fits[family] = fit_regression(
            X, y, family=family, draws=12, warmup=4, chains=2, scales=[0.5], seed=100 + i, **options
        )
    return fits


@pytest.mark.parametrize("family", FAMILIES)
def test_each_direct_likelihood_executes_and_predicts(family, tiny_fits):
    fit = tiny_fits[family]
    expected_latent = 2 if family.startswith(("zero_", "hurdle_")) or family == "multinomial" else 1
    assert fit.draws.shape == (2, 12, 1, expected_latent)
    assert np.all(np.isfinite(fit.draws))
    prediction = fit.predict(np.array([[0.0], [0.5]]))
    assert prediction.shape == (2, 3 if family in ("ordinal", "multinomial") else 1)
    assert np.all(np.isfinite(prediction))
    if family in ("ordinal", "multinomial"):
        assert_allclose(prediction.sum(axis=1), 1)
    assert len(fit.to_records()) == expected_latent
    assert fit.interval().shape == (1, expected_latent, 2)
    assert fit.contrast(np.ones(expected_latent)).draws.shape == (2, 12, 1, 1)
    if family == "cox":
        assert_allclose(fit.intercept_draws, 0)
        with pytest.raises(ValueError, match="baseline"):
            fit.posterior_predictive([[0.0]])
    else:
        assert fit.posterior_predictive([[0.0]]).shape == (24, 1, 1)


def test_nonlinear_prediction_averages_transformed_draws(tiny_fits):
    fit = tiny_fits["binomial"]
    X = np.array([[0.7]])
    links = fit.draws[..., 0, 0] * 0.7 + fit.intercept_draws[..., 0]
    expected = expit(links).mean()
    assert_allclose(fit.predict(X, scale="probability"), [[expected]])
    assert abs(expected - expit(links.mean())) > 1e-6
    assert_allclose(fit.predict(X, trials=4), [[4 * expected]])


def test_prediction_uses_new_offsets_and_exposure_not_training_rows():
    fit = fit_regression(
        np.zeros((3, 1)),
        [0, 1, 2],
        family="poisson",
        exposure=[2, 3, 4],
        offset=[0.1, 0.2, 0.3],
        draws=8,
        warmup=2,
        chains=2,
        seed=4,
    )
    ordinary = fit.predict([[0.0]])
    assert_allclose(fit.predict([[0.0]], exposure=3, offset=np.log(2)), 6 * ordinary)


def test_multioutput_gaussian_shapes_and_cross_condition_contrasts():
    fit = fit_regression(
        [[-1.0], [0.0], [1.0]],
        [[0.0, 1.0], [1.0, 0.0], [2.0, -1.0]],
        draws=8,
        warmup=2,
        chains=2,
        seed=5,
    )
    assert fit.draws.shape == (2, 8, 1, 2)
    assert fit.predict([[0.0], [1.0]]).shape == (2, 2)
    contrast = fit.contrast([1, -1])
    assert_allclose(contrast.draws[..., 0], fit.draws[..., 0] - fit.draws[..., 1])


def test_random_slopes_new_group_policy_and_repeated_group_draws():
    X = np.array([[-1.0], [0.0], [0.5], [1.0]])
    z = np.column_stack((np.ones(4), X[:, 0]))
    fit = fit_regression(
        X,
        [0.2, 0.1, 0.5, 0.4],
        groups=["a", "a", "b", "b"],
        random_design=z,
        random_covariance=[[1.0, 0.2], [0.2, 0.5]],
        random_scale=0.8,
        estimate_random_scale=False,
        draws=12,
        warmup=4,
        chains=2,
        seed=6,
    )
    assert fit.random_draws.shape == (2, 12, 2, 2, 1)
    assert len(fit.random_effect_summary()) == 4
    with pytest.raises(ValueError, match="unknown group"):
        fit.predict([[0.0]], groups=["new"], random_design=[[1.0, 0.0]])
    eta = fit._predict_draws(
        [[0.0], [0.0]],
        groups=["new", "new"],
        random_design=[[1.0, 0.0], [1.0, 0.0]],
        allow_new_groups=True,
        seed=7,
    )
    assert_allclose(eta[:, 0], eta[:, 1])
    population = fit._predict_draws([[0.0]])
    assert_allclose(population[:, 0, 0], fit.intercept_draws.ravel())
    assert np.std(eta[:, 0, 0] - population[:, 0, 0]) > 0
    with pytest.raises(ValueError, match="positive definite"):
        fit_regression(
            X,
            [0.2, 0.1, 0.5, 0.4],
            groups=["a", "a", "b", "b"],
            random_design=z,
            random_covariance=np.ones((2, 2)),
            draws=4,
            warmup=0,
            chains=2,
        )


def test_estimated_group_scale_is_positive_and_reported():
    fit = fit_regression(
        [[-1.0], [0.0], [1.0], [0.4]],
        [0.2, 0.1, 0.4, 0.6],
        groups=["a", "a", "b", "b"],
        draws=8,
        warmup=2,
        chains=2,
        seed=8,
    )
    assert np.all(fit.random_scale_draws > 0)
    assert np.std(fit.random_scale_draws) > 0
    assert fit.diagnostics["random_scale_prior"] == "IG(shape=2,scale=1)"
    assert fit.diagnostics["random_scale_diagnostics"]["diagnostics_available"]
    assert fit.diagnostics["intercept_diagnostics"]["diagnostics_available"]
    assert fit.diagnostics["weight_diagnostics"]["diagnostics_available"]


def test_design_aliases_and_prior_only_columns_are_reported():
    fit = fit_regression(
        [[-1.0, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [0.1, 0.2, 0.5],
        groups=["a", "a", "b"],
        random_design=np.ones((3, 2)),
        draws=4,
        warmup=0,
        chains=2,
        seed=10,
    )
    d = fit.diagnostics
    assert d["fixed_design_rank"] == 2
    assert d["fixed_design_columns"] == 4
    assert d["random_design_rank"] == 1
    assert d["random_design_columns"] == 2
    assert len(d["identification_warnings"]) == 3


def test_constant_nonzero_predictor_requires_intercept_removed():
    with pytest.raises(ValueError, match="duplicates the fitted intercept"):
        fit_regression(np.ones((3, 1)), [0.1, 0.2, 0.5], draws=4, warmup=0, chains=2)
    fit = fit_regression(
        np.ones((3, 1)), [0.1, 0.2, 0.5], fit_intercept=False, draws=4, warmup=0, chains=2, seed=11
    )
    assert fit.diagnostics["fixed_design_rank"] == 1
    assert fit.diagnostics["intercept_diagnostics"] is None


def test_new_group_predictive_noise_does_not_reuse_random_effect_normal_stream():
    fit = fit_regression(
        [[-1.0], [0.0], [1.0], [0.4]],
        [0.2, 0.1, 0.4, 0.6],
        groups=["a", "a", "b", "b"],
        random_scale=0.8,
        estimate_random_scale=False,
        sigma=0.7,
        draws=20,
        warmup=2,
        chains=2,
        seed=9,
    )
    arguments = dict(groups=["new"], allow_new_groups=True, seed=72)
    eta = fit._predict_draws([[0.0]], **arguments)[:, 0, 0]
    base = fit._predict_draws([[0.0]])[:, 0, 0]
    outcomes = fit.posterior_predictive([[0.0]], **arguments)[:, 0, 0]
    group_normal = (eta - base) / 0.8
    error_normal = (outcomes - eta) / 0.7
    assert not np.allclose(group_normal, error_normal), (
        "new-group and observation noise reused identical normals"
    )


def test_cox_survival_averages_draw_specific_breslow_baselines(tiny_fits):
    fit = tiny_fits["cox"]
    beta = fit.draws.reshape(-1)
    times = np.array([0.0, 1.0, 2.0, 5.0])
    y = fit.likelihood.y[:, 0]
    event = fit.likelihood.event[:, 0].astype(bool)
    entry = fit.likelihood.entry[:, 0]
    expected = []
    for b in beta:
        h = np.zeros(len(times))
        for t in np.unique(y[event]):
            increment = (
                np.sum((y == t) & event) / np.exp(b * fit.X[(entry < t) & (y >= t), 0]).sum()
            )
            h[times >= t] += increment
        expected.append(np.exp(-h * np.exp(0.4 * b)))
    actual = fit.predict([[0.4]], scale="survival", time=times)
    assert_allclose(actual, np.mean(expected, axis=0)[None, :])
    assert_allclose(actual[:, 0], 1)
    assert np.all(np.diff(actual, axis=1) <= 0)
    intervals = fit.predict_interval([[0.4]], scale="survival", time=times, level=0.8)
    assert_allclose(intervals, np.quantile(expected, [0.1, 0.9], axis=0).T[None, :, :])


@pytest.mark.parametrize(
    "options",
    [
        {"time": [1.0], "exposure": 2},
        {"time": [1.0], "trials": 2},
        {"time": []},
        {"time": [[1.0, 2.0]]},
        {"time": [-1.0]},
        {"time": [np.nan]},
        {"time": [np.inf]},
    ],
)
def test_cox_survival_rejects_unsupported_options_and_invalid_times(tiny_fits, options):
    fit = tiny_fits["cox"]
    with pytest.raises((TypeError, ValueError)):
        fit.predict([[0.4]], scale="survival", **options)
    with pytest.raises((TypeError, ValueError)):
        fit.predict_interval([[0.4]], scale="survival", **options)


def test_cox_extreme_hazards_keep_survival_finite_and_one_at_zero(tiny_fits):
    fit = tiny_fits["cox"]
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = fit.predict([[10000.0]], scale="survival", time=[0.0, 1.0, 5.0])
    assert np.all(np.isfinite(prediction))
    assert_allclose(prediction[:, 0], 1)
    assert np.all((prediction >= 0) & (prediction <= 1))


def test_safe_json_roundtrip_preserves_draws_predictions_and_raw_training_data(tmp_path, tiny_fits):
    fit = tiny_fits["binomial"]
    path = tmp_path / "posterior.json"
    fit.save(path)
    loaded = load_regression_result(path)
    assert_allclose(loaded.draws, fit.draws)
    assert_allclose(loaded.weight_draws, fit.weight_draws)
    assert_allclose(loaded.predict([[0.2]]), fit.predict([[0.2]]))
    assert_allclose(loaded.likelihood.y, fit.likelihood.y)
    assert "NaN" not in path.read_text()
    with pytest.raises(FileExistsError):
        fit.save(path)


@pytest.mark.parametrize(
    "corruption", ["unknown", "draw_shape", "weights", "nonfinite", "duplicate"]
)
def test_safe_json_rejects_malformed_inputs(tmp_path, tiny_fits, corruption):
    path = tmp_path / "broken.json"
    tiny_fits["gaussian"].save(path)
    value = json.loads(path.read_text())
    if corruption == "unknown":
        value["payload"]["execute"] = "arbitrary"
    elif corruption == "draw_shape":
        value["payload"]["draws"] = [[[0.0]]]
    elif corruption == "weights":
        value["payload"]["weight_draws"][0][0] = [-1.0, 2.0]
    elif corruption == "nonfinite":
        path.write_text(path.read_text().replace('"version": 1', '"version": NaN'))
    else:
        path.write_text(path.read_text().replace('"version": 1', '"version": 1, "version": 1'))
    if corruption not in ("nonfinite", "duplicate"):
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_regression_result(path)


def test_draw_result_exact_summary_and_invalid_inputs():
    draws = np.array([0.0, 1.0, 2.0, 3.0, -1.0, 0.0, 1.0, 2.0]).reshape(2, 4, 1, 1)
    result = DrawResult(draws)
    assert_allclose(result.posterior_mean, [[1.0]])
    assert_allclose(result.lfdr, [[0.25]])
    assert_allclose(result.lfsr, [[0.375]])
    assert_allclose(result.cdf(0), [[0.375]])
    assert_allclose(result.quantile(0.5), [[1.0]])
    assert result.sample(3, seed=1).shape == (3, 1, 1)
    with pytest.raises(ValueError):
        result.quantile(1.1)
    with pytest.raises(ValueError):
        result.interval(0)
    with pytest.raises(ValueError):
        result.contrast([1, 2])
