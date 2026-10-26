"""Analytic posterior identities and boundary behavior."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from jointshrink import PosteriorResult


def scalar(weights, means, variances):
    return PosteriorResult(
        [weights], np.asarray(means)[None, :, None], np.asarray(variances)[None, :, None, None]
    )


def test_total_expectation_and_covariance():
    # Conditional means 0 and (1,-1), with conditional covariance I plus .5 off-diagonal.
    result = PosteriorResult(
        [[0.4, 0.6]], [[[0, 0], [1, -1]]], [[np.zeros((2, 2)), [[1, 0.5], [0.5, 1]]]]
    )
    assert_allclose(result.posterior_mean, [[0.6, -0.6]])
    assert_allclose(result.posterior_covariance, [[[0.84, 0.06], [0.06, 0.84]]])
    assert_allclose(result.posterior_sd, [[np.sqrt(0.84), np.sqrt(0.84)]])


def test_between_component_variance_is_not_lost_with_large_common_mean():
    result = scalar([0.5, 0.5], [1e12 - 1, 1e12 + 1], [1, 1])
    assert_allclose(result.posterior_mean, [[1e12]])
    assert_allclose(result.posterior_covariance, [[[2]]])


def test_coordinate_atoms_and_global_null_differ():
    result = PosteriorResult(
        [[0.2, 0.3, 0.5]],
        [[[0, 0], [0, 1], [2, 0]]],
        [[np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]],
    )
    assert_allclose(result.lfdr, [[0.5, 0.7]])
    assert_allclose(result.joint_null_probability, [0.2])
    assert_allclose(result.lfsr, [[0.5, 0.7]])
    assert_allclose(result.probability_positive, [[0.5, 0.3]])
    assert_array_equal(result.probability_negative, [[0, 0]])
    assert_allclose(result.probability_positive + result.probability_negative + result.lfdr, 1)


def test_zero_included_in_both_sign_errors():
    result = scalar([0.2, 0.3, 0.5], [-1, 0, 1], [0, 0, 0])
    assert_allclose(result.lfsr, [[0.5]])
    assert_allclose(result.lfdr, [[0.3]])
    assert_allclose(result.cdf(0), [[0.5]])
    assert_allclose(result.probability_negative, [[0.2]])
    assert_allclose(result.probability_positive, [[0.5]])


def test_all_null_lfsr_equals_one():
    result = scalar([1], [0], [0])
    assert_array_equal(result.lfsr, [[1]])
    assert_array_equal(result.lfdr, [[1]])
    assert_array_equal(result.interval(), [[[0, 0]]])


@pytest.mark.parametrize("variance", [1e-200, 1e-300])
def test_small_positive_variance_is_continuous(variance):
    result = scalar([1], [0], [variance])
    assert_array_equal(result.lfdr, [[0]])
    assert_allclose(result.lfsr, [[0.5]])
    assert_allclose(result.quantile(0.975) / np.sqrt(variance), [[norm.ppf(0.975)]], rtol=1e-12)
    assert_allclose(result.cdf(0), [[0.5]])
    assert np.any(result.sample(8, seed=4) != 0)


@pytest.mark.parametrize("mean", [-10.0, 10.0])
def test_sign_error_preserves_small_gaussian_tail(mean):
    result = scalar([1], [mean], [1])
    assert_allclose(result.lfsr, [[norm.cdf(-10)]], rtol=1e-12, atol=0)


def test_single_normal_quantiles_and_interval():
    result = scalar([1], [2], [9])
    probabilities = np.array([0.001, 0.025, 0.5, 0.975, 0.999])
    assert_allclose(
        result.quantile(probabilities)[:, 0, 0], norm.ppf(probabilities, loc=2, scale=3)
    )
    assert_allclose(result.interval()[0, 0], norm.ppf([0.025, 0.975], loc=2, scale=3))
    assert result.quantile(0.5).shape == (1, 1)
    assert result.quantile([]).shape == (0, 1, 1)


def test_mixed_cdf_jump_and_quantile():
    result = scalar([0.4, 0.6], [0, 1], [0, 1])
    jump_left = 0.6 * norm.cdf(-1)
    jump_right = 0.4 + jump_left
    assert_allclose(result.cdf(0), [[jump_right]])
    for probability in [jump_left, 0.3, jump_right]:
        assert result.quantile(probability)[0, 0] == 0
    p = 0.8
    expected = 1 + norm.ppf((p - 0.4) / 0.6)
    assert_allclose(result.quantile(p), [[expected]], atol=1e-12)


def test_discrete_generalized_inverse_chooses_first_point_on_plateau():
    result = scalar([0.5, 0.25, 0.25], [-3, 2, 9], [0, 0, 0])
    # F is .5 throughout [-3,2), so q(.5) is -3, not 2 or an interior point.
    assert_array_equal(
        result.quantile([0, 0.25, 0.5, 0.50001, 0.75, 0.9, 1])[:, 0, 0], [-3, -3, -3, 2, 2, 9, 9]
    )
    assert_array_equal(result.cdf([[-1]]), [[0.5]])


def test_discrete_duplicate_and_zero_weight_components():
    result = scalar([0.2, 0.3, 0, 0.5], [0, 0, -100, 1], [0, 0, 0, 0])
    assert_array_equal(result.quantile([0, 0.5, 0.75, 1])[:, 0, 0], [0, 0, 1, 1])


def test_symmetric_continuous_mixture_median():
    result = scalar([0.5, 0.5], [-2, 2], [1, 1])
    assert_allclose(result.quantile(0.5), 0, atol=1e-14)
    q = result.quantile([0.1, 0.9])[:, 0, 0]
    assert_allclose(q[0], -q[1], atol=1e-13)
    assert_allclose(result.cdf(q[0]), 0.1, atol=1e-13)


def test_singular_sampling_preserves_exact_linear_constraint():
    result = PosteriorResult([[1]], [[[2, 2]]], [[[[3, 3], [3, 3]]]])
    samples = result.sample(16, seed=41)
    assert samples.shape == (16, 1, 2)
    assert_allclose(samples[:, 0, 0], samples[:, 0, 1], atol=1e-14)
    assert_array_equal(samples, result.sample(16, seed=41))
    null = scalar([1], [5], [0])
    assert_array_equal(null.sample(3, seed=4), np.full((3, 1, 1), 5))


def test_contrast_uses_joint_covariance_and_propagates_missingness():
    result = PosteriorResult(
        [[1]],
        [[[2, 1]]],
        [[[[4, 1.5], [1.5, 1]]]],
        row_names=["gene"],
        condition_names=["a", "b"],
        observed=np.array([[True, False]]),
    )
    contrasts = result.contrast([[1, -1], [1, 1], [1, 0]], names=["difference", "sum", "a"])
    assert_allclose(contrasts.posterior_mean, [[1, 3, 2]])
    # Var(a-b)=4+1-2*1.5; Cov(a-b,a+b)=4-1.
    assert_allclose(contrasts.posterior_covariance, [[[2, 3, 2.5], [3, 8, 5.5], [2.5, 5.5, 4]]])
    assert_array_equal(contrasts.observed, [[False, False, True]])
    assert contrasts.row_names == ("gene",)
    assert contrasts.condition_names == ("difference", "sum", "a")


def test_shared_singular_contrast_has_exact_null_atom():
    result = PosteriorResult([[1]], [[[1, 1]]], [[[[2, 2], [2, 2]]]])
    difference = result.contrast([1, -1])
    assert_array_equal(difference.posterior_mean, [[0]])
    assert_array_equal(difference.posterior_covariance, [[[0]]])
    assert_array_equal(difference.lfdr, [[1]])


def test_sharing_uses_union_denominator_and_requires_same_sign():
    # Deterministic posterior: every nonzero effect is significant, zero is not.
    means = np.array([[2, 1], [1, -1], [1, 0], [0, 1], [0, 0]])
    result = PosteriorResult(np.ones((5, 1)), means[:, None, :], np.zeros((5, 1, 2, 2)))
    sign = result.sharing(mode="sign")
    magnitude = result.sharing(mode="magnitude")
    assert_array_equal(sign["denominator"], [[3, 4], [4, 3]])
    assert_array_equal(sign["numerator"], [[3, 1], [1, 3]])
    assert_allclose(sign["matrix"], [[1, 0.25], [0.25, 1]])
    assert_allclose(magnitude["matrix"], sign["matrix"])
    assert_array_equal(result.significant(), [0, 1, 2, 3])
    assert_array_equal(result.significant(any_condition=False), [0, 1])


def test_magnitude_factor_is_inclusive_and_not_sign_agnostic():
    means = np.array([[2, 1], [2.01, 1], [-2, 1]])
    result = PosteriorResult(np.ones((3, 1)), means[:, None, :], np.zeros((3, 1, 2, 2)))
    assert result.sharing()["matrix"][0, 1] == pytest.approx(1 / 3)
    assert result.sharing(mode="sign")["matrix"][0, 1] == pytest.approx(2 / 3)


def test_empty_sharing_denominator_is_explicit():
    result = scalar([1], [0], [0])
    summary = result.sharing()
    assert_array_equal(summary["denominator"], [[0]])
    assert np.isnan(summary["matrix"][0, 0])


def test_records_tables_and_summary():
    result = PosteriorResult(
        [[1]],
        [[[2, 0]]],
        [[[[0, 0], [0, 0]]]],
        row_names=["g"],
        condition_names=["a", "b"],
        observed=np.array([[True, False]]),
    )
    records = result.to_records()
    assert records[0] == dict(
        row="g",
        condition="a",
        posterior_mean=2.0,
        posterior_sd=0.0,
        lower=2.0,
        upper=2.0,
        lfsr=0.0,
        lfdr=0.0,
        observed=True,
    )
    assert records[1]["observed"] is False
    summary = result.summary()
    assert summary["significant_rows"] == 1
    assert summary["significant_by_condition"] == {"a": 1, "b": 0}
    assert summary["missing_values"] == 1
    assert "Conditional" in result.format_summary()
    pd = pytest.importorskip("pandas")
    assert isinstance(result.to_dataframe(), pd.DataFrame)
    assert result.to_dataframe().to_dict(orient="records") == records


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("quantile", [-0.1]),
        ("quantile", [1.1]),
        ("quantile", [np.nan]),
        ("quantile", [[[0.5]]]),
        ("interval", [0]),
        ("interval", [1]),
        ("sample", [0]),
        ("sample", [True]),
        ("sample", [1.5]),
        ("contrast", [[1, 2]]),
        ("cdf", [np.nan]),
    ],
)
def test_invalid_summary_arguments_rejected(method, arguments):
    result = scalar([1], [0], [1])
    with pytest.raises(ValueError):
        getattr(result, method)(*arguments)


def test_result_arrays_do_not_alias_inputs():
    weights = np.array([[1.0]])
    means = np.array([[[2.0]]])
    covariance = np.array([[[[1.0]]]])
    result = PosteriorResult(weights, means, covariance)
    means[:] = 99
    covariance[:] = 99
    assert_allclose(result.posterior_mean, [[2]])
    assert_allclose(result.posterior_covariance, [[[1]]])
    with pytest.raises(ValueError):
        result.component_means[:] = 1


def test_contrast_roundoff_from_nearly_singular_three_condition_covariance():
    # Fixed covariance from the illustrative learned-pattern fit. Direct expansion
    # of V11+V22-2 V12 loses 6.9e-18 to cancellation, although V is PSD to roundoff.
    covariance = [
        [0.057447090252467525, 0.0574470902524675, 0.05744709025246748],
        [0.0574470902524675, 0.05744709025246746, 0.05744709025246744],
        [0.05744709025246748, 0.05744709025246744, 0.05744709025246755],
    ]
    result = PosteriorResult([[1]], [[[1, 1, 1]]], [[covariance]])
    contrast = result.contrast([[1, -1, 0]])
    assert contrast.posterior_covariance[0, 0, 0] >= 0
    assert np.isfinite(contrast.posterior_sd).all()
    assert_allclose(contrast.posterior_mean, 0, atol=0)


def test_contrast_keeps_tiny_positive_variance_and_original_atoms():
    result = PosteriorResult([[1]], [[[0, 0]]], [[[[1e-300, 0], [0, 0]]]])
    contrast = result.contrast([[1, -1], [0, 1]])
    assert_allclose(contrast.posterior_covariance[0, 0, 0], 1e-300, rtol=1e-14, atol=0)
    assert_array_equal(contrast.lfdr, [[0, 1]])
    assert contrast.quantile(0.975)[0, 0] > 0
    assert contrast.quantile(0.975)[0, 1] == 0


def test_materially_indefinite_component_cannot_be_hidden_by_contrast():
    with pytest.raises(ValueError, match="positive semidefinite"):
        PosteriorResult([[1]], [[[0, 0]]], [[[[1, 1.01], [1.01, 1]]]])
