"""Multivariate adaptive shrinkage with fixed or learned covariance patterns."""

from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from scipy.special import logsumexp
from .data import MashData
from ._numeric import readonly, validate_psd, component_posterior, weight_fit, responsibilities
from .results import PosteriorResult


@dataclass(frozen=True, init=False)
class MashPrior:
    """A fitted Gaussian-mixture prior on the EE or EZ working scale.

    Covariances are already scale-expanded. Reuse never refits their weights
    or learns new patterns. Predictions condition on these fitted quantities.
    """

    covariances: np.ndarray
    names: tuple
    weights: np.ndarray
    alpha: int
    metadata: object

    def __init__(self, covariances, names, weights, *, alpha=0, metadata=None):
        cov = np.asarray(covariances, dtype=float)
        if cov.ndim != 3 or cov.shape[0] < 1 or cov.shape[1] != cov.shape[2]:
            raise ValueError("covariances must have shape (k,d,d)")
        cov = np.stack([validate_psd(x) for x in cov])
        labels = tuple(names)
        if len(labels) != len(cov) or len(set(labels)) != len(labels):
            raise ValueError("component names must be unique and match covariances")
        if any(not isinstance(x, str) or not x.strip() or not x.isprintable() for x in labels):
            raise ValueError("component names must be nonempty printable strings")
        w = np.asarray(weights, dtype=float)
        if w.shape != (len(cov),) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and nonnegative, one per covariance")
        if not np.isclose(w.sum(), 1, atol=1e-10, rtol=1e-10):
            raise ValueError("weights must sum to one")
        if alpha not in (0, 1):
            raise ValueError("alpha must be 0 or 1")
        object.__setattr__(self, "covariances", readonly(cov))
        object.__setattr__(self, "weights", readonly(w / w.sum()))
        object.__setattr__(self, "names", labels)
        object.__setattr__(self, "alpha", int(alpha))
        object.__setattr__(self, "metadata", MappingProxyType(dict(metadata or {})))

    def _check(self, data):
        if not isinstance(data, MashData):
            raise TypeError("data must be created with set_data")
        if data.n_conditions != self.covariances.shape[1] or data.alpha != self.alpha:
            raise ValueError("data dimension and EE/EZ scale must match the prior")

    def apply(self, data):
        """Condition this prior on new summary statistics without refitting."""
        self._check(data)
        return MashResult(data, self, diagnostics={"fixed_prior": True})

    def score_samples(self, data):
        """Log marginal density of each row, on its observed original scale."""
        self._check(data)
        ll, _, _ = component_posterior(data, self.covariances, compute_moments=False)
        _, score = responsibilities(ll, self.weights)
        return score - np.where(data.mask, np.log(data.scale), 0).sum(axis=1)

    def iter_apply(self, data, *, batch_size=128):
        """Yield posterior batches in input row order, without refitting.

        Reduce batch_size when the condition/component count is large.
        Each batch retains its original row labels.
        """
        self._check(data)
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        for start in range(0, data.n_effects, batch_size):
            yield self.apply(data.subset(slice(start, start + batch_size)))

    def save(self, path, *, overwrite=False):
        from .io import save_prior

        return save_prior(self, path, overwrite=overwrite)


class MashResult(PosteriorResult):
    """Conditional posterior results from a fitted multivariate prior."""

    def __init__(self, data, prior, *, diagnostics=None):
        prior._check(data)
        ll, means, covs = component_posterior(data, prior.covariances)
        weights, score = responsibilities(ll, prior.weights)
        means *= data.scale[:, None, :]
        covs *= data.scale[:, None, :, None] * data.scale[:, None, None, :]
        super().__init__(
            weights,
            means,
            covs,
            row_names=data.row_names,
            condition_names=data.condition_names,
            observed=data.mask,
            metadata={"alpha": data.alpha, "hyperparameters": "conditional"},
        )
        self.data = data
        self.prior = prior
        self.diagnostics = dict(diagnostics or {})
        self.log_likelihood = readonly(score - np.where(data.mask, np.log(data.scale), 0).sum(1))
        null = np.all(prior.covariances == 0, axis=(1, 2))
        alt = ~null
        if np.any(alt) and prior.weights[alt].sum() > 0:
            with np.errstate(divide="ignore"):
                alt_ll = logsumexp(
                    ll[:, alt] + np.log(prior.weights[alt] / prior.weights[alt].sum()), axis=1
                )
            # The global-null density is defined even when its fitted weight is zero.
            null_ll, _, _ = component_posterior(
                data, np.zeros((1, data.n_conditions, data.n_conditions)), compute_moments=False
            )
            self.log_bayes_factor = readonly(alt_ll - null_ll[:, 0])
        else:
            self.log_bayes_factor = readonly(np.full(data.n_effects, -np.inf))

    def shrink(self, data):
        """Apply the fitted prior to new rows, without learning from them."""
        return self.prior.apply(data)

    def score_samples(self, data):
        return self.prior.score_samples(data)

    def save(self, path, *, overwrite=False):
        from .io import save_result

        return save_result(self, path, overwrite=overwrite)


def fit_prior(
    data,
    covariances=None,
    *,
    scales=None,
    normalize=True,
    point_mass=True,
    null_weight=1,
    initial_weights=None,
    maxiter=10000,
    tol=1e-8,
):
    """Fit mixture weights and return a reusable prior, without posterior tensors.

    Use representative rows here. Strong-row selection is for learning
    covariance patterns, not for estimating their final mixture proportions.
    `scales` contains standard-deviation multipliers, squared in covariances.
    """
    from .patterns import canonical_covariances, expand_covariances, make_scale_grid

    if not isinstance(data, MashData):
        raise TypeError("data must be created with set_data")
    if covariances is None:
        covariances = canonical_covariances(data)
    if scales is None:
        scales = make_scale_grid(data)
    expanded = expand_covariances(covariances, scales, point_mass=point_mass, normalize=normalize)
    names = tuple(expanded)
    cov = np.stack(list(expanded.values()))
    if cov.shape[1:] != (data.n_conditions, data.n_conditions):
        raise ValueError("covariance dimension must match data conditions")
    ll, _, _ = component_posterior(data, cov, compute_moments=False)
    null = np.flatnonzero(np.all(cov == 0, axis=(1, 2)))
    weights, diagnostics = weight_fit(
        ll,
        null_index=None if len(null) == 0 else int(null[0]),
        null_weight=null_weight,
        initial_weights=initial_weights,
        maxiter=maxiter,
        tol=tol,
    )
    prior = MashPrior(
        cov,
        names,
        weights,
        alpha=data.alpha,
        metadata={
            "normalized": bool(normalize),
            "scale_multipliers": list(map(float, scales)),
            "weight_training_rows": data.n_effects,
            "fit_diagnostics": diagnostics,
        },
    )
    return prior


def fit(data, covariances=None, **options):
    """Fit mixture weights on all supplied rows and return posterior summaries.

    For larger analyses, use fit_prior followed by prior.iter_apply.
    """
    prior = fit_prior(data, covariances, **options)
    return MashResult(data, prior, diagnostics=prior.metadata["fit_diagnostics"])


def apply_prior(data, prior):
    """Apply a MashPrior to new data without estimating new parameters."""
    return prior.apply(data)


def score_samples(data, prior):
    """Evaluate observed-row marginal log densities under a fixed prior."""
    return prior.score_samples(data)


class Mash:
    """Configure a mixture fit; `fit` returns a separate MashResult.

    Pattern learning remains explicit through cov_pca and cov_ed.
    """

    def __init__(self, covariances=None, **options):
        self.covariances = covariances
        self.options = dict(options)

    def fit(self, data):
        return fit(data, self.covariances, **self.options)
