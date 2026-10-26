"""Covariance patterns, scale grids, and error-aware pattern learning."""

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np
from scipy.special import logsumexp

from ._numeric import readonly, validate_psd, component_posterior


@dataclass(frozen=True)
class CovarianceFit:
    """Learned zero-mean covariance patterns and EM diagnostics."""

    covariances: dict
    weights: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class CorrelationEstimate:
    """Error-correlation estimate from rows with small absolute Z scores."""

    correlation: np.ndarray
    row_indices: np.ndarray
    diagnostics: dict


def _dimension(value):
    value = getattr(value, "n_conditions", value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
        raise ValueError("dimension must be an integer of at least two")
    return int(value)


def validate_covariances(patterns, *, normalize=False):
    """Validate named PSD matrices; optionally set each maximum variance to one.

    Singular and zero matrices are allowed. Normalization leaves zero matrices
    unchanged. Returned arrays are read-only copies.
    """
    if isinstance(patterns, Mapping):
        pairs = list(patterns.items())
    else:
        array = np.asarray(patterns, dtype=float)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3:
            raise ValueError("patterns must be a mapping or a matrix/stack of matrices")
        pairs = [(f"pattern_{i + 1}", value) for i, value in enumerate(array)]
    if not pairs:
        raise ValueError("at least one covariance pattern is required")
    output = {}
    dimension = None
    for name, value in pairs:
        if not isinstance(name, str) or not name or name in output:
            raise ValueError("pattern names must be nonempty unique strings")
        matrix = validate_psd(value, name=f"pattern {name}")
        if dimension is None:
            dimension = matrix.shape[0]
            _dimension(dimension)
        if matrix.shape != (dimension, dimension):
            raise ValueError("all patterns must have the same dimension")
        largest = np.max(np.diag(matrix))
        if normalize and largest > 0:
            matrix = matrix / largest
        output[name] = readonly(matrix)
    return output


def canonical_covariances(
    data_or_dimension, *, methods=("identity", "singletons", "equal_effects", "simple_het")
):
    """Return independent, condition-specific, equal, and correlated patterns."""
    d = _dimension(data_or_dimension)
    if isinstance(methods, str):
        methods = (methods,)
    output = {}
    for method in methods:
        if method == "identity":
            output["identity"] = np.eye(d)
        elif method == "singletons":
            for i in range(d):
                matrix = np.zeros((d, d))
                matrix[i, i] = 1
                output[f"singleton_{i + 1}"] = matrix
        elif method == "equal_effects":
            output["equal_effects"] = np.ones((d, d))
        elif method == "simple_het":
            for correlation in (0.25, 0.5, 0.75):
                output[f"simple_het_{correlation:g}"] = (1 - correlation) * np.eye(d) + correlation
        elif method == "null":
            output["null"] = np.zeros((d, d))
        else:
            raise ValueError(f"unknown canonical covariance method: {method!r}")
    return validate_covariances(output)


def _rows(data, subset, *, minimum=1):
    n = data.n_effects
    if subset is None:
        indices = np.arange(n)
    else:
        values = np.asarray(subset)
        if values.dtype.kind == "b":
            if values.shape != (n,):
                raise ValueError("subset mask must have one entry per row")
            indices = np.flatnonzero(values)
        else:
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError("subset must be a one-dimensional integer array or Boolean mask")
            indices = values.astype(int)
            if np.any(indices < 0) or np.any(indices >= n):
                raise ValueError("subset indices are out of bounds")
            if len(np.unique(indices)) != len(indices):
                raise ValueError("subset indices must not repeat rows")
    if len(indices) < minimum:
        raise ValueError(f"at least {minimum} selected rows are required")
    if not np.all(data.mask[indices]):
        raise ValueError("covariance learning requires complete selected rows")
    return indices


def select_strong(data, *, z_threshold=3):
    """Select rows with an observed absolute Z score above the threshold.

    This selects pattern-learning rows, not a representative sample for fitting
    final mixture weights. Missing coordinates do not contribute to selection.
    """
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be finite and positive")
    z = np.where(data.mask, np.abs(data.effects / data.standard_errors), -np.inf)
    return readonly(np.flatnonzero(np.max(z, axis=1) > z_threshold), dtype=int)


def cov_pca(data, n_components=2, *, subset=None, center=False):
    """Find rank-one PCs and their aggregate on the working-effect scale.

    The default uncentered SVD follows the R helper. Use ``center=True`` for
    column centering; use ``alpha=1`` data to work with Z scores. Selected rows
    must be complete. No observation-error correction is applied at this step.
    """
    indices = _rows(data, subset, minimum=2 if center else 1)
    d = data.n_conditions
    if isinstance(n_components, (bool, np.bool_)) or not isinstance(
        n_components, (int, np.integer)
    ):
        raise ValueError("n_components must be an integer")
    maximum = min(d, len(indices) - int(bool(center)))
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between one and {maximum}")
    values = np.array(data.working_effects[indices], copy=True)
    if center:
        values -= values.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(values, full_matrices=False)
    if singular_values[0] == 0:
        raise ValueError("selected rows contain no variation on the requested scale")
    output = {f"PCA_{i + 1}": np.outer(vectors[i], vectors[i]) for i in range(n_components)}
    selected = vectors[:n_components]
    output["PCA_total"] = (
        (selected.T * singular_values[:n_components] ** 2) @ selected / len(indices)
    )
    return validate_covariances(output)


def _project(matrix, rank, ridge):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.maximum(values, 0)
    if rank is not None:
        values[:-rank] = 0
    if ridge:
        if rank is None:
            values = np.maximum(values, ridge)
        else:
            values[-rank:] = np.maximum(values[-rank:], ridge)
    return (vectors * values) @ vectors.T


def cov_ed(data, initial, *, subset=None, maxiter=200, tol=1e-6, ridge=0, rank=None):
    """Learn a zero-mean noisy Gaussian mixture by extreme deconvolution.

    EM uses each row's known observation-error covariance. ``ridge`` is an
    explicit eigenvalue floor, and ``rank`` is an optional maximum component
    rank. Constrained proposals are accepted only if observed likelihood does
    not decrease. Diagnostics distinguish convergence, iteration limits and
    rejected proposals. Exact EM with singular starts cannot leave their initial
    support; use positive-definite starts or a positive ridge to avoid this.
    The output is not normalized or expanded over scales.
    """
    indices = _rows(data, subset)
    initial = validate_covariances(initial)
    d = data.n_conditions
    if next(iter(initial.values())).shape != (d, d):
        raise ValueError("initial patterns must match the number of conditions")
    if (
        isinstance(maxiter, (bool, np.bool_))
        or not isinstance(maxiter, (int, np.integer))
        or maxiter < 1
    ):
        raise ValueError("maxiter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("tol must be positive and ridge must be nonnegative, both finite")
    if rank is not None and (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or not 1 <= rank <= d
    ):
        raise ValueError("rank must be an integer between one and the number of conditions")
    selected = data.subset(indices)
    matrices = np.stack([_project(value, rank, ridge) for value in initial.values()])
    k = len(matrices)
    weights = np.full(k, 1 / k)

    def expectation(covariances, proportions):
        loglik, means, covs = component_posterior(selected, covariances)
        with np.errstate(divide="ignore"):
            logjoint = loglik + np.log(proportions)
        normalizer = logsumexp(logjoint, axis=1)
        return float(normalizer.sum()), np.exp(logjoint - normalizer[:, None]), means, covs

    objective, responsibilities, means, covs = expectation(matrices, weights)
    history = [objective]
    reason = "maxiter"
    converged = False
    for iteration in range(1, maxiter + 1):
        mass = responsibilities.sum(axis=0)
        new_weights = mass / len(indices)
        proposed = matrices.copy()
        for component in range(k):
            if mass[component] > 0:
                moments = covs[:, component] + np.einsum(
                    "ni,nj->nij", means[:, component], means[:, component]
                )
                updated = (
                    np.einsum("n,nij->ij", responsibilities[:, component], moments)
                    / mass[component]
                )
                proposed[component] = _project(updated, rank, ridge)
        candidate = expectation(proposed, new_weights)
        # Reject rounding or projection steps that reduce the fitted objective.
        if candidate[0] < objective:
            accepted = False
            for power in range(1, 26):
                fraction = 2.0**-power
                trial = np.stack(
                    [
                        _project((1 - fraction) * a + fraction * b, rank, ridge)
                        for a, b in zip(matrices, proposed)
                    ]
                )
                trial_weights = (1 - fraction) * weights + fraction * new_weights
                trial_result = expectation(trial, trial_weights)
                if trial_result[0] >= objective:
                    proposed, new_weights, candidate = trial, trial_weights, trial_result
                    accepted = True
                    break
            if not accepted:
                reason = "proposal_rejected"
                break
        gain = candidate[0] - objective
        matrices, weights = proposed, new_weights
        objective, responsibilities, means, covs = candidate
        history.append(objective)
        if gain <= tol * (1 + abs(history[-2])):
            reason, converged = "tolerance", True
            break
    output = {f"ED_{name}": readonly(value) for name, value in zip(initial, matrices)}
    diagnostics = {
        "converged": converged,
        "reason": reason,
        "iterations": len(history) - 1,
        "log_likelihood": history,
        "row_indices": indices.tolist(),
        "ridge": float(ridge),
        "rank": rank,
        "zero_mean": True,
        "scale": "working",
        "initial_ranks": [int(np.linalg.matrix_rank(v)) for v in initial.values()],
        "final_ranks": [int(np.linalg.matrix_rank(v)) for v in matrices],
    }
    return CovarianceFit(output, readonly(weights), diagnostics)


def make_scale_grid(data, *, factor=np.sqrt(2)):
    """Return positive standard-deviation multipliers spanning observed effects.

    The smallest scale is the smallest working standard error divided by ten.
    The largest covers the largest absolute working estimate (or twice the
    smallest scale when all estimates are zero). Adjacent scales differ by
    ``factor``; covariance multipliers are their squares.
    """
    if not np.isfinite(factor) or factor <= 1:
        raise ValueError("factor must be finite and greater than one")
    noise = np.sqrt(np.diagonal(data.working_covariance, axis1=-2, axis2=-1))
    noise = np.broadcast_to(noise, data.effects.shape)
    minimum = float(np.min(noise[data.mask])) / 10
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("working standard errors are too small for an automatic grid")
    maximum = max(float(np.max(np.abs(data.working_effects[data.mask]))), minimum * 2)
    steps = int(np.ceil((np.log(maximum) - np.log(minimum)) / np.log(factor)))
    if steps > 10000:
        raise ValueError("scale grid exceeds 10001 entries; use a larger factor")
    grid = np.exp(np.linspace(np.log(maximum) - steps * np.log(factor), np.log(maximum), steps + 1))
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError("automatic scales exceed floating-point limits; supply explicit scales")
    return readonly(grid)


def expand_covariances(patterns, scales, *, point_mass=True, normalize=True):
    """Expand normalized patterns by squared SD scales, retaining unique matrices."""
    patterns = validate_covariances(patterns, normalize=normalize)
    scales = np.asarray(scales, dtype=float)
    if scales.ndim != 1 or scales.size == 0 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must be a nonempty vector of finite positive SD multipliers")
    output = {}
    d = next(iter(patterns.values())).shape[0]
    if point_mass:
        output["null"] = readonly(np.zeros((d, d)))
    for index, scale in enumerate(scales):
        for name, pattern in patterns.items():
            with np.errstate(over="ignore", under="ignore"):
                matrix = pattern * scale * scale
            if np.any(~np.isfinite(matrix)) or (np.any(pattern != 0) and not np.any(matrix != 0)):
                raise ValueError("scaled covariance exceeds floating-point limits")
            # Exact equality avoids merging meaningfully distinct tiny patterns.
            if any(np.array_equal(matrix, old) for old in output.values()):
                continue
            candidate = f"{name}@{index + 1}"
            output[candidate] = readonly(matrix)
    return output


def estimate_null_correlation(data, *, z_threshold=2, minimum_rows=None, regularization=0.05):
    """Estimate centered Z-score correlation among complete, nullish rows.

    All absolute Z scores must be below ``z_threshold``. The empirical
    correlation is shrunk toward identity by ``regularization``. Selection can
    bias this estimate; it is not a test that the errors have this correlation.
    """
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError("z_threshold must be finite and positive")
    if not np.isfinite(regularization) or not 0 <= regularization <= 1:
        raise ValueError("regularization must be between zero and one")
    d = data.n_conditions
    if minimum_rows is None:
        minimum_rows = max(3, d + 1)
    if (
        isinstance(minimum_rows, (bool, np.bool_))
        or not isinstance(minimum_rows, (int, np.integer))
        or minimum_rows < 2
    ):
        raise ValueError("minimum_rows must be an integer of at least two")
    z = data.effects / data.standard_errors
    indices = np.flatnonzero(np.all(data.mask & (np.abs(z) < z_threshold), axis=1))
    if len(indices) < minimum_rows:
        raise ValueError(
            f"only {len(indices)} complete nullish rows; at least {minimum_rows} required"
        )
    if np.any(np.max(z[indices], axis=0) == np.min(z[indices], axis=0)):
        raise ValueError("nullish Z scores have a constant condition")
    centered = z[indices] - np.mean(z[indices], axis=0)
    covariance = centered.T @ centered / (len(indices) - 1)
    sd = np.sqrt(np.diag(covariance))
    if np.any(sd == 0):
        raise ValueError("nullish Z scores have a constant condition")
    empirical = covariance / np.outer(sd, sd)
    correlation = (1 - regularization) * empirical + regularization * np.eye(d)
    correlation = validate_psd(correlation, name="estimated correlation")
    np.fill_diagonal(correlation, 1)
    return CorrelationEstimate(
        readonly(correlation),
        readonly(indices, dtype=int),
        {
            "n_rows": len(indices),
            "z_threshold": float(z_threshold),
            "regularization": float(regularization),
            "centered": True,
            "minimum_eigenvalue": float(np.linalg.eigvalsh(correlation)[0]),
        },
    )
