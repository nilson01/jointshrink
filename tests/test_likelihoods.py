"""Independent finite-case checks for regression likelihoods and prediction."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit
from scipy.stats import binom, logistic, multinomial, nbinom, norm, poisson, weibull_min

from jointshrink.likelihoods import FAMILIES, make_likelihood


@pytest.mark.parametrize("family", FAMILIES)
def test_family_constructor_roundtrip_and_sampler_support(family):
    options = {}
    y = [0, 1, 2]
    if family == "binomial":
        options["trials"] = [2, 3, 4]
    if family in {"cox", "weibull"}:
        y = [1, 2, 3]
        options.update(event=[1, 0, 1], entry=[0, 0.5, 1])
    ll = make_likelihood(y, family=family, **options)
    eta = np.linspace(-0.5, 0.5, 3 * ll.n_latent).reshape(3, ll.n_latent)
    restored = make_likelihood(**json.loads(json.dumps(ll.to_dict())))
    assert restored.loglike(eta) == ll.loglike(eta)
    assert np.isfinite(ll.loglike(eta))
    if family == "cox":
        with pytest.raises(ValueError, match="baseline"):
            ll.sample(eta, np.random.default_rng(1))
    else:
        sample = ll.sample(eta, np.random.default_rng(1))
        assert sample.shape == (3, 1)
        assert np.all(np.isfinite(sample))
        if family in {"multinomial", "ordinal"}:
            assert np.all((sample >= 0) & (sample < 3))
        elif family not in {"gaussian", "weibull"}:
            assert np.all(sample >= 0)
            assert_array_equal(sample, sample.astype(int))


def test_gaussian_normalized_logdensity_and_fixed_sigma():
    y = np.array([[1.0, -1.0], [2.0, 3.0]])
    eta = np.array([[0.2, 0.7], [-0.3, 1.0]])
    sigma = np.array([0.5, 2.0])
    ll = make_likelihood(y, sigma=sigma)
    assert_allclose(ll.loglike(eta), norm.logpdf(y, eta, sigma).sum(), rtol=1e-14)
    assert_array_equal(ll.predict(eta), eta)
    seed = 3
    assert_array_equal(
        ll.sample(eta, np.random.default_rng(seed)), np.random.default_rng(seed).normal(eta, sigma)
    )


def test_binomial_normalized_logmass_and_prediction_trials():
    y = np.array([0, 2, 5])[:, None]
    trials = np.array([2, 4, 8])[:, None]
    eta = np.array([-3.0, 0.2, 2.0])[:, None]
    ll = make_likelihood(y, family="binomial", trials=trials)
    assert_allclose(ll.loglike(eta), binom.logpmf(y, trials, expit(eta)).sum())
    assert_allclose(ll.predict(eta), expit(eta))  # training trials are not reused
    assert_allclose(ll.predict(eta, trials=10), 10 * expit(eta))
    assert_allclose(ll.predict(eta, scale="probability"), expit(eta))
    assert_allclose(ll.predict(np.zeros((5, 1)), trials=6), 3 * np.ones((5, 1)))


@pytest.mark.parametrize("family", ["poisson", "negative_binomial"])
def test_count_reference_logmass_and_exposure(family):
    y = np.array([0, 1, 8])[:, None]
    eta = np.array([-2.0, 0.2, 2.0])[:, None]
    exposure = np.array([0.1, 2.0, 4.0])[:, None]
    options = {"dispersion": 0.7} if family == "negative_binomial" else {}
    ll = make_likelihood(y, family=family, exposure=exposure, **options)
    mu = exposure * np.exp(eta)
    reference = poisson(mu) if family == "poisson" else nbinom(1 / 0.7, 1 / (1 + 0.7 * mu))
    assert_allclose(ll.loglike(eta), reference.logpmf(y).sum(), rtol=1e-13)
    assert_allclose(ll.predict(eta), np.exp(eta))
    assert_allclose(ll.predict(eta, exposure=3), 3 * np.exp(eta))
    assert_allclose(ll.predict(np.zeros((5, 1))), np.ones((5, 1)))


@pytest.mark.parametrize(
    "family",
    [
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
        "hurdle_poisson",
        "hurdle_negative_binomial",
    ],
)
def test_two_part_count_normalization_mean_and_reference(family):
    grid = np.arange(500)
    mu, zero_probability, alpha = 2.3, 0.37, 0.4
    logit_zero = np.log(zero_probability / (1 - zero_probability))
    is_nb = "negative_binomial" in family
    reference = nbinom(1 / alpha, 1 / (1 + alpha * mu)) if is_nb else poisson(mu)
    p = reference.pmf(grid)
    if family.startswith("hurdle_"):
        expected = (1 - zero_probability) * p / (1 - reference.pmf(0))
        expected[0] = zero_probability
    else:
        expected = (1 - zero_probability) * p
        expected[0] += zero_probability
    observed = []
    options = {"dispersion": alpha} if is_nb else {}
    for y in grid:
        ll = make_likelihood([y], family=family, **options)
        observed.append(np.exp(ll.loglike([[np.log(mu), logit_zero]])))
    assert_allclose(observed, expected, atol=1e-15, rtol=1e-11)
    assert_allclose(np.sum(observed), 1.0, atol=1e-13)
    ll = make_likelihood([0], family=family, **options)
    assert_allclose(ll.predict([[np.log(mu), logit_zero]])[0, 0], grid @ expected, rtol=1e-13)


@pytest.mark.parametrize("family", ["hurdle_poisson", "hurdle_negative_binomial"])
def test_hurdle_tiny_means_are_not_clipped(family):
    ll = make_likelihood([1], family=family)
    eta = np.array([[-1000.0, -1000.0]])
    # As the count mean tends to zero, a positive truncated count tends to one.
    assert_allclose(ll.loglike(eta), 0.0, atol=1e-12)
    assert_allclose(ll.predict(eta), [[1.0]], atol=1e-12)
    assert_array_equal(ll.sample(eta, np.random.default_rng(4)), [[1]])
    # Zero-inflation logits remain free to express probabilities near one.
    assert_allclose(ll.predict([[-1000.0, 1000.0]]), [[0.0]], atol=0.0)


def test_multinomial_loglikelihood_and_baseline_reference():
    eta = np.array([[1.0, -2.0], [-0.3, 0.2], [0.5, 0.5]])
    weights = np.exp(np.column_stack([eta, np.zeros(3)]))
    p = weights / weights.sum(axis=1, keepdims=True)
    y = np.array([0, 1, 2])
    ll = make_likelihood(y, family="multinomial", n_categories=3)
    assert_allclose(ll.predict(eta), p)
    ref = sum(multinomial.logpmf(np.eye(3, dtype=int)[yi], 1, pi) for yi, pi in zip(y, p))
    assert_allclose(ll.loglike(eta), ref)
    # Classes not seen in the supplied outcomes remain in explicit category space.
    absent = make_likelihood([0, 0], family="multinomial", n_categories=4)
    assert absent.n_latent == 3
    assert_allclose(absent.predict(np.zeros((2, 3))), 0.25)


def test_ordinal_loglikelihood_fixed_thresholds_and_extreme_predictors():
    thresholds = np.array([-1.0, 0.2, 2.0])
    eta = np.array([[-2.0], [0.0], [1.0], [3.0]])
    cumulative = logistic.cdf(thresholds[None, :] - eta)
    p = np.diff(np.column_stack([np.zeros(4), cumulative, np.ones(4)]), axis=1)
    ll = make_likelihood([0, 1, 2, 3], family="ordinal", n_categories=4, thresholds=thresholds)
    assert_allclose(ll.predict(eta), p)
    assert_allclose(ll.loglike(eta), np.log(p[np.arange(4), np.arange(4)]).sum())
    ext = ll.predict(np.array([[-1000.0], [1000.0]]))
    assert_allclose(ext.sum(axis=1), 1.0)
    assert_array_equal(np.argmax(ext, axis=1), [0, 3])
    # An improbable internal category retains a finite log probability.
    extreme = make_likelihood([1], family="ordinal", n_categories=4, thresholds=thresholds)
    assert np.isfinite(extreme.loglike([[1000.0]]))


def test_cox_breslow_ties_delayed_entry_and_baseline():
    # At t=1, entry=1 excludes row 2. At t=2, entry=2 excludes row 3.
    ll = make_likelihood([1, 2, 2, 3], family="cox", event=[1, 1, 1, 0], entry=[0, 0, 1, 2])
    eta = np.log([1.0, 2.0, 3.0, 4.0])[:, None]
    expected = -np.log(3) + np.log(2) + np.log(3) - 2 * np.log(5)
    assert_allclose(ll.loglike(eta), expected, rtol=1e-14)
    times, cumulative = ll.baseline_cumulative_hazard(eta)
    assert_array_equal(times, [1, 2])
    assert_allclose(cumulative, [1 / 3, 1 / 3 + 2 / 5])
    assert_allclose(ll.predict(eta), [[1], [2], [3], [4]])
    assert_allclose(ll.loglike(eta + 1000), expected, atol=1e-12)
    with pytest.raises(ValueError, match="unsupported"):
        ll.predict(eta, scale="survival", time=2)


def test_weibull_full_likelihood_matches_conditional_density_and_survival():
    t = np.array([1.1, 2.0, 3.0])[:, None]
    entry = np.array([0.0, 0.2, 1.0])[:, None]
    event = np.array([1, 0, 1])[:, None]
    eta = np.array([-0.2, 0.5, 1.0])[:, None]
    shape = 1.7
    ref = weibull_min(shape, scale=np.exp(-eta / shape))
    expected = np.where(event, ref.logpdf(t), ref.logsf(t)) - ref.logsf(entry)
    ll = make_likelihood(t, family="weibull", event=event, entry=entry, shape=shape)
    assert_allclose(ll.loglike(eta), expected.sum(), rtol=1e-14)
    assert_allclose(ll.predict(eta, scale="survival", time=t), ref.sf(t))
    assert_allclose(ll.predict(eta, scale="cumulative_hazard", time=t), -ref.logsf(t))
    # Conditional outcome draws use the untruncated, unconditional fitted model.
    seed = 11
    u = np.random.default_rng(seed).uniform(size=(3, 1))
    assert_allclose(ll.sample(eta, np.random.default_rng(seed)), ref.isf(u))


@pytest.mark.parametrize(
    "family",
    [
        "gaussian",
        "binomial",
        "poisson",
        "negative_binomial",
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
        "hurdle_poisson",
        "hurdle_negative_binomial",
    ],
)
def test_multiple_outcomes_sum_separate_likelihoods(family):
    y = np.array([[0, 1], [1, 2], [2, 0]])
    options = {"trials": 3} if family == "binomial" else {}
    joint = make_likelihood(y, family=family, **options)
    eta = np.array([[0.1, -0.2], [0.4, 0.6], [-0.5, 0.9]])
    two_part = joint.n_latent == 4
    if two_part:
        eta = np.column_stack([eta, np.array([[0.2, 0.4], [0.8, -0.9], [-0.4, -0.1]])])
    separate = 0.0
    for r in range(2):
        one = make_likelihood(y[:, r], family=family, **options)
        e = eta[:, [r, r + 2]] if two_part else eta[:, [r]]
        separate += one.loglike(e)
    assert_allclose(joint.loglike(eta), separate)
    assert joint.predict(eta).shape == (3, 2)


@pytest.mark.parametrize(
    "y,kwargs",
    [
        ([np.nan], {}),
        ([np.inf], {}),
        (["1"], {}),
        ([], {}),
        ([[[]]], {}),
        ([0], {"family": "other"}),
        ([0.5], {"family": "poisson"}),
        ([-1], {"family": "negative_binomial"}),
        ([2], {"family": "binomial"}),
        ([0], {"family": "binomial", "trials": 0.5}),
        ([0], {"sigma": 0}),
        ([0], {"family": "poisson", "exposure": 0}),
        ([0], {"family": "negative_binomial", "dispersion": -1}),
        ([0], {"family": "ordinal", "n_categories": True}),
        ([0], {"family": "ordinal", "n_categories": 1}),
        ([2], {"family": "ordinal", "n_categories": 2}),
        ([0], {"family": "ordinal", "n_categories": 3, "thresholds": [1, 0]}),
        ([[0, 1]], {"family": "multinomial"}),
        ([1], {"family": "cox"}),
        ([0], {"family": "cox", "event": [1]}),
        ([1], {"family": "cox", "event": [2]}),
        ([1], {"family": "cox", "event": [1], "entry": [1]}),
        ([1], {"family": "cox", "event": [1], "ties": "efron"}),
        ([1], {"family": "weibull", "event": [1], "shape": 0}),
        ([1], {"family": "weibull", "event": [1], "time": [2]}),
        ([1], {"trials": 2}),
        ([1], {"exposure": 2}),
        ([1], {"family": "poisson", "sigma": 2}),
    ],
)
def test_invalid_outcomes_and_inapplicable_nuisance_controls(y, kwargs):
    with pytest.raises(ValueError):
        make_likelihood(y, **kwargs)


@pytest.mark.parametrize("eta", [[1, 2], [[1, 2]], [[np.nan]], [[np.inf]]])
def test_eta_shape_and_finiteness(eta):
    with pytest.raises(ValueError):
        make_likelihood([0]).loglike(eta)


def test_fixed_row_sigma_new_rows_and_unsupported_predictions():
    ll = make_likelihood([0, 1], sigma=[1, 2])
    with pytest.raises(ValueError, match="row-specific sigma"):
        ll.sample(np.zeros((3, 1)), np.random.default_rng(1))
    with pytest.raises(ValueError, match="unsupported"):
        ll.predict(np.zeros((2, 1)), scale="probability")
    with pytest.raises(ValueError, match="trials"):
        ll.predict(np.zeros((2, 1)), trials=2)
    with pytest.raises(ValueError, match="exposure"):
        ll.predict(np.zeros((2, 1)), exposure=2)


def test_no_events_cox_baseline_and_loglikelihood():
    ll = make_likelihood([1, 2], family="cox", event=[0, 0])
    assert ll.loglike(np.zeros((2, 1))) == 0
    t, h = ll.baseline_cumulative_hazard(np.zeros((2, 1)))
    assert t.size == h.size == 0


@pytest.mark.parametrize("family", ["binomial", "poisson", "negative_binomial"])
def test_conditional_sampler_matches_numpy_reference(family):
    eta = np.array([[-0.7], [0.3], [1.2]])
    ll = make_likelihood([0, 1, 0], family=family)
    seed = 129
    rng = np.random.default_rng(seed)
    if family == "binomial":
        expected = rng.binomial(1, expit(eta))
    elif family == "poisson":
        expected = rng.poisson(np.exp(eta))
    else:
        expected = rng.negative_binomial(1, 1 / (1 + np.exp(eta)))
    assert_array_equal(ll.sample(eta, np.random.default_rng(seed)), expected)


@pytest.mark.parametrize("family", ["hurdle_poisson", "hurdle_negative_binomial"])
def test_hurdle_sampler_conditional_quantile_reference(family):
    eta = np.array([[-0.7, -0.3], [0.3, 0.4], [1.2, -1.0]])
    mu = np.exp(eta[:, :1])
    ll = make_likelihood([0, 1, 2], family=family)
    reference = poisson(mu) if family == "hurdle_poisson" else nbinom(1, 1 / (1 + mu))
    seed = 193
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(3, 1))
    p0 = reference.pmf(0)
    expected = reference.ppf(p0 + (1 - p0) * u)
    zero = rng.uniform(size=(3, 1)) < expit(eta[:, 1:])
    expected[zero] = 0
    assert_array_equal(ll.sample(eta, np.random.default_rng(seed)), expected)


@pytest.mark.parametrize("family", ["multinomial", "ordinal"])
def test_categorical_sampler_matches_probability_intervals(family):
    ll = make_likelihood([0, 1, 2], family=family)
    eta = (
        np.array([[0.3, -0.5], [1.0, -1.0], [0.2, 0.2]])
        if family == "multinomial"
        else np.array([[0.3], [1.0], [-0.7]])
    )
    p = ll.predict(eta)
    u = np.random.default_rng(29).uniform(size=(3, 1))
    expected = np.array(
        [np.searchsorted(np.cumsum(pi), ui, side="left") for pi, ui in zip(p, u[:, 0])]
    )[:, None]
    assert_array_equal(ll.sample(eta, np.random.default_rng(29)), expected)


@pytest.mark.parametrize("family", ["zero_inflated_poisson", "zero_inflated_negative_binomial"])
def test_zero_inflated_sampler_uses_count_component_and_independent_zero_gate(family):
    eta = np.array([[-0.7, -0.3], [0.3, 0.4], [1.2, -1.0]])
    mu = np.exp(eta[:, :1])
    rng = np.random.default_rng(35)
    count = (
        rng.poisson(mu)
        if family == "zero_inflated_poisson"
        else rng.negative_binomial(1, 1 / (1 + mu))
    )
    expected = np.where(rng.uniform(size=(3, 1)) < expit(eta[:, 1:]), 0, count)
    ll = make_likelihood([0, 1, 2], family=family)
    assert_array_equal(ll.sample(eta, np.random.default_rng(35)), expected)
