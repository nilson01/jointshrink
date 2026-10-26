import numpy as np
import pytest
from numpy.testing import assert_allclose
from jointshrink import set_data
from jointshrink.patterns import (
    canonical_covariances,
    validate_covariances,
    select_strong,
    cov_pca,
    cov_ed,
    make_scale_grid,
    expand_covariances,
    estimate_null_correlation,
)


def test_canonical_matrices_are_known_and_psd():
    matrices = canonical_covariances(3)
    assert len(matrices) == 8
    assert_allclose(matrices["identity"], np.eye(3))
    assert_allclose(matrices["equal_effects"], np.ones((3, 3)))
    assert_allclose(matrices["singleton_2"], np.diag([0, 1, 0]))
    assert_allclose(matrices["simple_het_0.25"], 0.75 * np.eye(3) + 0.25)
    assert all(np.linalg.eigvalsh(m)[0] >= -1e-12 for m in matrices.values())


def test_normalization_preserves_correlation_structure_and_null():
    original = {"a": [[4, 2], [2, 1]], "null": np.zeros((2, 2))}
    normalized = validate_covariances(original, normalize=True)
    assert_allclose(normalized["a"], [[1, 0.5], [0.5, 0.25]])
    assert_allclose(normalized["null"], 0)
    with pytest.raises(ValueError):
        normalized["a"][0, 0] = 2


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"a": [[1, 2], [2, 1]]},
        {"a": [[1, 0], [1, 1]]},
        {"a": [[1, np.nan], [np.nan, 1]]},
        {"a": np.eye(2), "b": np.eye(3)},
        {"": np.eye(2)},
        np.ones(2),
    ],
)
def test_bad_patterns_fail(bad):
    with pytest.raises(ValueError):
        validate_covariances(bad)


@pytest.mark.parametrize("dimension", [1, 0, True, 2.5])
def test_invalid_dimensions_fail(dimension):
    with pytest.raises(ValueError):
        canonical_covariances(dimension)


def test_unknown_canonical_name_fails():
    with pytest.raises(ValueError):
        canonical_covariances(2, methods=["other"])


def test_sd_expansion_deduplicates_without_erasing_tiny_covariances():
    tiny = 1e-30 * np.eye(2)
    expanded = expand_covariances({"a": tiny, "b": tiny}, [1, 2], normalize=False)
    assert len(expanded) == 3
    assert_allclose(expanded["a@2"], 4 * tiny, atol=0)
    assert np.any(expanded["a@1"] > 0)
    no_null = expand_covariances({"a": np.eye(2)}, [2], point_mass=False)
    assert len(no_null) == 1
    assert_allclose(next(iter(no_null.values())), 4 * np.eye(2))


@pytest.mark.parametrize("scales", [[], [0], [-1], [np.inf], [[1]]])
def test_invalid_scale_vectors_fail(scales):
    with pytest.raises(ValueError):
        expand_covariances({"a": np.eye(2)}, scales)


def test_scale_grid_and_strong_selection_use_working_and_observed_values():
    data = set_data([[6, np.nan], [0.2, 0.4], [-4, 0]], [[2, 1], [1, 1], [1, 1]], alpha=1)
    assert select_strong(data).tolist() == [2]  # strict inequality: first row Z=3
    grid = make_scale_grid(data)
    assert np.all(grid > 0)
    assert_allclose(grid[1:] / grid[:-1], np.sqrt(2))
    assert grid[0] <= 0.1 and grid[-1] >= 4 - 1e-12
    with pytest.raises(ValueError):
        make_scale_grid(data, factor=1)


def test_pca_aggregate_recovers_second_moment_or_centered_covariance():
    values = np.array([[2, 1], [-1, 2], [3, -2], [0, 1]])
    data = set_data(values)
    uncentered = cov_pca(data, 2)
    assert_allclose(uncentered["PCA_total"], values.T @ values / 4)
    centered = values - values.mean(0)
    assert_allclose(cov_pca(data, 2, center=True)["PCA_total"], centered.T @ centered / 4)
    assert_allclose(np.trace(uncentered["PCA_1"]), 1)
    assert np.linalg.matrix_rank(uncentered["PCA_1"]) == 1


def test_pca_rejects_incomplete_or_degenerate_rows_and_invalid_components():
    data = set_data([[1, np.nan], [2, 3], [4, 5]])
    with pytest.raises(ValueError, match="complete"):
        cov_pca(data, 1)
    assert len(cov_pca(data, 1, subset=[1, 2])) == 2
    with pytest.raises(ValueError):
        cov_pca(set_data(np.zeros((3, 2))), 1)
    for subset in ([1, 1], [5], [], [1.5]):
        with pytest.raises(ValueError):
            cov_pca(data, 1, subset=subset)
    with pytest.raises(ValueError):
        cov_pca(data, 3, subset=[1, 2])


def test_ed_single_step_matches_gaussian_conditional_second_moment():
    values = np.array([[2, 0], [-2, 0], [0, 2], [0, -2]])
    data = set_data(values, 0.5)
    fitted = cov_ed(data, {"start": np.eye(2)}, maxiter=1, tol=1e-12)
    # U=I, V=.25 I gives posterior covariance .2 I and mean .8 y.
    expected = 0.2 * np.eye(2) + 0.8**2 * (values.T @ values / 4)
    assert_allclose(fitted.covariances["ED_start"], expected)
    assert fitted.diagnostics["iterations"] == 1
    assert fitted.diagnostics["reason"] == "maxiter"


def test_ed_corrects_measurement_noise_and_reaches_isotropic_mle():
    data = set_data([[2, 0], [-2, 0], [0, 2], [0, -2]], 0.5)
    fitted = cov_ed(data, {"start": np.eye(2)}, maxiter=300, tol=1e-12)
    # Observed covariance is 2 I; the Gaussian prior MLE subtracts .25 I.
    assert_allclose(fitted.covariances["ED_start"], 1.75 * np.eye(2), atol=1e-5)
    assert fitted.diagnostics["converged"]
    assert np.min(np.diff(fitted.diagnostics["log_likelihood"])) >= 0


def test_ed_rank_ridge_and_zero_component_contracts():
    data = set_data([[2, 1], [-2, -1], [1, 2], [-1, -2]], 0.5)
    fitted = cov_ed(data, {"a": np.eye(2)}, rank=1, ridge=0.1, maxiter=8)
    values = np.linalg.eigvalsh(fitted.covariances["ED_a"])
    assert abs(values[0]) < 1e-12 and values[1] >= 0.1 - 1e-12
    assert np.min(np.diff(fitted.diagnostics["log_likelihood"])) >= 0
    zero = cov_ed(data, {"null": np.zeros((2, 2))}, maxiter=5)
    assert_allclose(zero.covariances["ED_null"], 0)
    assert zero.diagnostics["converged"]
    full = cov_ed(data, {"a": np.zeros((2, 2))}, ridge=0.2, maxiter=3)
    assert np.linalg.eigvalsh(full.covariances["ED_a"])[0] >= 0.2 - 1e-12


@pytest.mark.parametrize(
    "options", [{"rank": 0}, {"rank": 3}, {"ridge": -1}, {"maxiter": 0}, {"tol": 0}]
)
def test_ed_bad_options_fail(options):
    with pytest.raises(ValueError):
        cov_ed(set_data([[1, 2], [3, 4]]), {"a": np.eye(2)}, **options)


def test_null_correlation_matches_centered_sample_correlation_then_shrinkage():
    values = np.array([[0.1, 0.4], [-0.2, 0.1], [0.3, -0.2], [-0.4, -0.3], [9, 9]])
    result = estimate_null_correlation(set_data(values), regularization=0.2)
    expected = 0.8 * np.corrcoef(values[:4], rowvar=False) + 0.2 * np.eye(2)
    assert_allclose(result.correlation, expected)
    assert result.row_indices.tolist() == [0, 1, 2, 3]
    assert np.linalg.eigvalsh(result.correlation)[0] > 0


def test_null_correlation_rejects_insufficient_and_constant_information():
    with pytest.raises(ValueError, match="at least"):
        estimate_null_correlation(set_data([[0.1, 0.2], [5, 5]]))
    with pytest.raises(ValueError, match="constant"):
        estimate_null_correlation(set_data([[0.1, 0.2], [0.1, 0.3], [0.1, 0.4]]))


def test_ed_heterogeneous_correlated_errors_matches_one_step_oracle():
    values = np.array([[2.0, 1.0], [-1.0, -0.5], [0.3, -0.2]])
    errors = np.array([[[0.5, 0.1], [0.1, 1]], [[1, -0.2], [-0.2, 0.7]], [[0.2, 0], [0, 0.8]]])
    initial = np.array([[1.2, 0.4], [0.4, 0.9]])
    moments = []
    for value, error in zip(values, errors):
        # Independently use precision addition, possible because this start is PD.
        covariance = np.linalg.inv(np.linalg.inv(initial) + np.linalg.inv(error))
        mean = covariance @ np.linalg.solve(error, value)
        moments.append(covariance + np.outer(mean, mean))
    fitted = cov_ed(set_data(values, error_covariance=errors), {"a": initial}, maxiter=1)
    assert_allclose(fitted.covariances["ED_a"], np.mean(moments, axis=0))
