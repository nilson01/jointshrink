"""Posterior summaries for finite mixtures of multivariate normal distributions."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _names(names, count, prefix):
    if names is None:
        return tuple(f"{prefix}{i + 1}" for i in range(count))
    names = tuple(str(x) for x in names)
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"{prefix} names must have the required length and be unique")
    return names


class PosteriorResult:
    """A fitted Gaussian-mixture posterior, conditional on estimated hyperparameters.

    Component arrays have shapes (rows, components), (rows, components, conditions),
    and (rows, components, conditions, conditions). Singular covariance matrices
    represent exact coordinate or joint point masses. All quantities use original
    effect units. ``probability_negative`` and ``probability_positive`` are strict
    probabilities; ``lfsr`` includes zero in both possible sign errors.
    """

    def __init__(
        self,
        component_weights,
        component_means,
        component_covariances,
        *,
        row_names=None,
        condition_names=None,
        observed=None,
        metadata=None,
    ):
        weights = np.asarray(component_weights, dtype=float)
        means = np.asarray(component_means, dtype=float)
        cov = np.asarray(component_covariances, dtype=float)
        if means.ndim != 3 or min(means.shape) < 1:
            raise ValueError("component_means must have shape (rows, components, conditions)")
        n, k, d = means.shape
        if weights.shape != (n, k) or cov.shape != (n, k, d, d):
            raise ValueError("component weights, means and covariances have incompatible shapes")
        if not all(np.all(np.isfinite(x)) for x in (weights, means, cov)):
            raise ValueError("posterior components must be finite")
        if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1, rtol=1e-9, atol=1e-12):
            raise ValueError("component weights must be nonnegative and sum to one in each row")
        weights = weights / weights.sum(axis=1, keepdims=True)
        cleaned = np.empty_like(cov)
        for i in range(n):
            for j in range(k):
                matrix = cov[i, j]
                scale = np.max(np.abs(matrix))
                tolerance = 128 * np.finfo(float).eps * d * scale
                if np.max(np.abs(matrix - matrix.T)) > tolerance:
                    raise ValueError("component covariances must be symmetric")
                matrix = (matrix + matrix.T) / 2
                values, vectors = np.linalg.eigh(matrix)
                if values[0] < -tolerance:
                    raise ValueError("component covariances must be positive semidefinite")
                if values[0] < 0:
                    matrix = (vectors * np.maximum(values, 0)) @ vectors.T
                cleaned[i, j] = matrix
        if observed is None:
            observed = np.ones((n, d), dtype=bool)
        observed = np.asarray(observed)
        if observed.shape != (n, d) or observed.dtype.kind != "b":
            raise ValueError("observed must be a boolean array with shape (rows, conditions)")
        self.component_weights = _readonly(weights)
        self.component_means = _readonly(means)
        self.component_covariances = _readonly(cleaned)
        self.observed = _readonly(observed, bool)
        self.row_names = _names(row_names, n, "effect")
        self.condition_names = _names(condition_names, d, "condition")
        self.metadata = dict(metadata or {})

    @property
    def n_effects(self):
        return self.component_means.shape[0]

    @property
    def n_conditions(self):
        return self.component_means.shape[2]

    @property
    def posterior_mean(self):
        return np.einsum("nk,nkd->nd", self.component_weights, self.component_means)

    @property
    def posterior_covariance(self):
        delta = self.component_means - self.posterior_mean[:, None, :]
        return np.einsum(
            "nk,nkde->nde", self.component_weights, self.component_covariances
        ) + np.einsum("nk,nkd,nke->nde", self.component_weights, delta, delta)

    @property
    def posterior_sd(self):
        return np.sqrt(np.maximum(np.diagonal(self.posterior_covariance, axis1=-2, axis2=-1), 0))

    @property
    def _variances(self):
        return np.maximum(np.diagonal(self.component_covariances, axis1=-2, axis2=-1), 0)

    @property
    def lfdr(self):
        """Marginal posterior probability of an exactly zero effect."""
        atoms = (self._variances == 0) & (self.component_means == 0)
        return np.einsum("nk,nkd->nd", self.component_weights, atoms)

    @property
    def joint_null_probability(self):
        """Posterior probability that every effect in a row is exactly zero."""
        atoms = np.all(self._variances == 0, axis=2) & np.all(self.component_means == 0, axis=2)
        return np.sum(self.component_weights * atoms, axis=1)

    def _component_cdf(self, value, *, strict=False):
        value = np.broadcast_to(np.asarray(value, dtype=float), (self.n_effects, self.n_conditions))
        if np.any(np.isnan(value)):
            raise ValueError("CDF arguments cannot be NaN")
        means = self.component_means
        sd = np.sqrt(self._variances)
        positive = sd > 0
        standardized = np.zeros_like(means)
        np.divide(value[:, None, :] - means, sd, out=standardized, where=positive)
        atom_cdf = means < value[:, None, :] if strict else means <= value[:, None, :]
        return np.where(positive, ndtr(standardized), atom_cdf)

    def cdf(self, value):
        """Return marginal Pr(effect <= value), including any atom at value."""
        return np.einsum("nk,nkd->nd", self.component_weights, self._component_cdf(value))

    @property
    def probability_negative(self):
        return np.einsum("nk,nkd->nd", self.component_weights, self._component_cdf(0, strict=True))

    @property
    def probability_positive(self):
        # Evaluate the upper tail directly to retain very small probabilities.
        sd = np.sqrt(self._variances)
        z = np.zeros_like(sd)
        np.divide(self.component_means, sd, out=z, where=sd > 0)
        prob = np.where(sd > 0, ndtr(z), self.component_means > 0)
        return np.einsum("nk,nkd->nd", self.component_weights, prob)

    @property
    def lfsr(self):
        zero = self.lfdr
        return np.minimum(self.probability_negative + zero, self.probability_positive + zero)

    def quantile(self, probability):
        """Marginal generalized inverse CDF; a vector returns (probabilities, rows, conditions)."""
        probabilities = np.asarray(probability, dtype=float)
        if (
            probabilities.ndim > 1
            or np.any(~np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("probability must be a scalar or vector in [0, 1]")
        out = np.empty((probabilities.size, self.n_effects, self.n_conditions))
        for i in range(self.n_effects):
            for j in range(self.n_conditions):
                active = self.component_weights[i] > 0
                w = self.component_weights[i, active]
                mu = self.component_means[i, active, j]
                sd = np.sqrt(self._variances[i, active, j])
                continuous = sd > 0

                def cdf(x, strict=False):
                    c = np.empty(len(w))
                    c[continuous] = ndtr((x - mu[continuous]) / sd[continuous])
                    c[~continuous] = mu[~continuous] < x if strict else mu[~continuous] <= x
                    return float(w @ c)

                atoms = np.unique(mu[~continuous])
                for q, p in enumerate(probabilities.ravel()):
                    if p == 0:
                        value = -np.inf if np.any(continuous) else mu.min()
                    elif p == 1:
                        value = np.inf if np.any(continuous) else mu.max()
                    else:
                        jumps = [a for a in atoms if cdf(a, True) <= p <= cdf(a)]
                        if jumps:
                            value = jumps[0]
                        elif not np.any(continuous):
                            order = np.argsort(mu)
                            value = mu[order[np.searchsorted(np.cumsum(w[order]), p)]]
                        else:
                            # Component p-quantiles bracket the mixture p-quantile.
                            component_q = mu + sd * ndtri(p)
                            low, high = float(component_q.min()), float(component_q.max())
                            if low == high:
                                value = low
                            else:
                                # Scale the root coordinate so tiny effect units retain accuracy.
                                midpoint = low / 2 + high / 2
                                halfwidth = high / 2 - low / 2
                                value = midpoint + halfwidth * brentq(
                                    lambda t: cdf(midpoint + halfwidth * t) - p,
                                    -1,
                                    1,
                                    xtol=np.nextafter(0.0, 1.0),
                                    rtol=8 * np.finfo(float).eps,
                                )
                    out[q, i, j] = value
        return out[0] if probabilities.ndim == 0 else out

    def interval(self, level=0.95):
        """Equal-tailed marginal credible intervals, conditional on fitted hyperparameters."""
        if not np.isscalar(level) or not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be strictly between zero and one")
        return np.moveaxis(self.quantile([(1 - level) / 2, (1 + level) / 2]), 0, -1)

    def sample(self, size=1, seed=None):
        """Draw joint posterior samples with shape (size, rows, conditions)."""
        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < 1
        ):
            raise ValueError("size must be a positive integer")
        rng = np.random.default_rng(seed)
        output = np.empty((size, self.n_effects, self.n_conditions))
        for i in range(self.n_effects):
            components = rng.choice(
                len(self.component_weights[i]), size=size, p=self.component_weights[i]
            )
            for k in np.unique(components):
                chosen = np.flatnonzero(components == k)
                values, vectors = np.linalg.eigh(self.component_covariances[i, k])
                root = vectors * np.sqrt(np.maximum(values, 0))
                output[chosen, i] = (
                    self.component_means[i, k]
                    + rng.standard_normal((len(chosen), self.n_conditions)) @ root.T
                )
        return output

    def contrast(self, coefficients, *, names=None):
        """Return the joint posterior of linear combinations; rows of coefficients are contrasts."""
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients[None, :]
        if (
            coefficients.ndim != 2
            or coefficients.shape[1] != self.n_conditions
            or coefficients.shape[0] < 1
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("coefficients must have shape (contrasts, conditions) and be finite")
        means = np.einsum("ad,nkd->nka", coefficients, self.component_means)
        # Transform a PSD square root instead of subtracting nearly equal terms
        # in L V L.T. The Gram form preserves nonnegative contrast variances.
        # Positive eigenvalues, however small, remain part of the distribution.
        shape = (*self.component_weights.shape, len(coefficients), len(coefficients))
        cov = np.empty(shape)
        for i in range(self.n_effects):
            for k in range(self.component_weights.shape[1]):
                values, vectors = np.linalg.eigh(self.component_covariances[i, k])
                root = vectors * np.sqrt(np.maximum(values, 0))
                transformed_root = coefficients @ root
                cov[i, k] = transformed_root @ transformed_root.T
        # Observation labels distinguish contrasts involving an imputed coordinate.
        observed = np.all(self.observed[:, None, :] | (coefficients[None, :, :] == 0), axis=2)
        return PosteriorResult(
            self.component_weights,
            means,
            cov,
            row_names=self.row_names,
            condition_names=_names(names, len(coefficients), "contrast"),
            observed=observed,
            metadata={**self.metadata, "contrast_coefficients": coefficients.tolist()},
        )

    def _significant_mask(self, threshold, by):
        if by not in ("lfsr", "lfdr"):
            raise ValueError("by must be 'lfsr' or 'lfdr'")
        if not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return getattr(self, by) < threshold

    def significant(self, threshold=0.05, *, by="lfsr", any_condition=True):
        """Row indices with at least one (or all) marginal error rates below threshold."""
        mask = self._significant_mask(threshold, by)
        return np.flatnonzero(np.any(mask, axis=1) if any_condition else np.all(mask, axis=1))

    def sharing(self, *, factor=0.5, threshold=0.05, by="lfsr", mode="magnitude"):
        """Pairwise sharing of posterior means, following Urbut et al. (2019).

        Each denominator counts rows significant in either member of the pair.
        Sign sharing requires equal nonzero signs; magnitude sharing additionally
        requires the absolute means to be within a factor of 1/factor. These are
        descriptive plug-in proportions, not posterior sharing probabilities.
        """
        if not np.isscalar(factor) or not np.isfinite(factor) or not 0 < factor <= 1:
            raise ValueError("factor must be in (0, 1]")
        if mode not in ("sign", "magnitude"):
            raise ValueError("mode must be 'sign' or 'magnitude'")
        sig = self._significant_mask(threshold, by)
        means = self.posterior_mean
        d = self.n_conditions
        numerator = np.zeros((d, d), dtype=int)
        denominator = np.zeros((d, d), dtype=int)
        for i in range(d):
            for j in range(d):
                eligible = sig[:, i] | sig[:, j]
                same = (np.sign(means[:, i]) == np.sign(means[:, j])) & (means[:, i] != 0)
                if mode == "magnitude":
                    a, b = np.abs(means[:, i]), np.abs(means[:, j])
                    same &= (a >= factor * b) & (b >= factor * a)
                numerator[i, j] = np.sum(eligible & same)
                denominator[i, j] = np.sum(eligible)
        matrix = np.full((d, d), np.nan)
        np.divide(numerator, denominator, out=matrix, where=denominator > 0)
        return dict(
            matrix=matrix,
            numerator=numerator,
            denominator=denominator,
            condition_names=self.condition_names,
            factor=factor,
            threshold=threshold,
            by=by,
            mode=mode,
            definition="Posterior-mean sharing among rows significant "
            "in either condition; equal nonzero signs required. Empty denominators are NaN.",
        )

    def to_records(self, *, level=0.95):
        """Long-form records, one per row and condition."""
        interval, mean, sd = self.interval(level), self.posterior_mean, self.posterior_sd
        lfsr, lfdr = self.lfsr, self.lfdr
        return [
            dict(
                row=self.row_names[i],
                condition=self.condition_names[j],
                posterior_mean=float(mean[i, j]),
                posterior_sd=float(sd[i, j]),
                lower=float(interval[i, j, 0]),
                upper=float(interval[i, j, 1]),
                lfsr=float(lfsr[i, j]),
                lfdr=float(lfdr[i, j]),
                observed=bool(self.observed[i, j]),
            )
            for i in range(self.n_effects)
            for j in range(self.n_conditions)
        ]

    def to_dataframe(self, *, level=0.95):
        """Return a pandas table; requires the table extra."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install jointshrink[table] to create a pandas table") from exc
        return pd.DataFrame.from_records(self.to_records(level=level))

    def summary(self, *, threshold=0.05, by="lfsr"):
        """Compact dimensions, discovery counts and uncertainty convention."""
        sig = self._significant_mask(threshold, by)
        return dict(
            n_effects=self.n_effects,
            n_conditions=self.n_conditions,
            n_components=self.component_weights.shape[1],
            significant_rows=int(np.any(sig, axis=1).sum()),
            significant_by_condition=dict(zip(self.condition_names, sig.sum(axis=0).tolist())),
            threshold=threshold,
            by=by,
            uncertainty="Conditional on fitted covariance patterns and mixture weights.",
            missing_values=int((~self.observed).sum()),
        )

    def format_summary(self, *, threshold=0.05, by="lfsr"):
        summary = self.summary(threshold=threshold, by=by)
        row_word = "row" if summary["significant_rows"] == 1 else "rows"
        return (
            f"Multivariate adaptive shrinkage: {self.n_effects} effects, "
            f"{self.n_conditions} conditions\n"
            f"{summary['significant_rows']} {row_word} with {by} < {threshold:g}\n"
            f"{summary['uncertainty']}"
        )

    def __repr__(self):
        return f"PosteriorResult(n_effects={self.n_effects}, n_conditions={self.n_conditions})"
