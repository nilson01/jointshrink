"""Validated summary statistics on effect or standardized-effect scales."""

from dataclasses import dataclass
import numpy as np
from ._numeric import readonly


def _labels(values, n, prefix):
    labels = tuple(f"{prefix}{i + 1}" for i in range(n)) if values is None else tuple(values)
    if len(labels) != n or len(set(labels)) != n:
        raise ValueError(f"{prefix} labels must be unique and match the input dimension")
    if any(not isinstance(x, str) or not x.strip() or not x.isprintable() for x in labels):
        raise ValueError("labels must be nonempty printable strings")
    return labels


def _cov_array(value, n, d, name):
    a = np.asarray(value, dtype=float)
    if a.shape == (d, d):
        a = np.broadcast_to(a, (n, d, d)).copy()
    if a.shape != (n, d, d) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape (d,d) or (n,d,d) and be finite")
    if not np.allclose(a, a.swapaxes(-1, -2), rtol=1e-10, atol=0):
        raise ValueError(f"{name} must be symmetric")
    a = (a + a.swapaxes(-1, -2)) / 2
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return a


@dataclass(frozen=True, init=False)
class MashData:
    """Effect estimates and their known sampling covariance.

    Rows are independent effects; columns are conditions. Missing effect
    estimates are NaN. Supply positive standard errors for all conditions,
    including unobserved conditions, to define their output scale under EZ.
    """

    effects: np.ndarray
    standard_errors: np.ndarray
    error_covariance: np.ndarray
    mask: np.ndarray
    scale: np.ndarray
    working_effects: np.ndarray
    working_covariance: np.ndarray
    row_names: tuple
    condition_names: tuple
    alpha: int

    def __init__(
        self,
        effects,
        standard_errors=1,
        *,
        correlation=None,
        error_covariance=None,
        alpha=0,
        row_names=None,
        condition_names=None,
    ):
        b = np.asarray(effects, dtype=float)
        if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] < 2:
            raise ValueError("effects must have shape (n,d), with n>=1 and d>=2")
        if np.any(np.isinf(b)) or np.any(~np.any(np.isfinite(b), axis=1)):
            raise ValueError("each effect row needs an observed value; infinity is invalid")
        n, d = b.shape
        if alpha not in (0, 1):
            raise ValueError("alpha must be 0 (EE) or 1 (EZ)")
        se = np.asarray(standard_errors, dtype=float)
        if error_covariance is not None:
            if correlation is not None or se.ndim != 0 or float(se) != 1:
                raise ValueError("error_covariance replaces standard_errors and correlation")
            cov = _cov_array(error_covariance, n, d, "error_covariance")
            se = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
        else:
            try:
                se = np.broadcast_to(se, b.shape).copy()
            except ValueError as exc:
                raise ValueError("standard_errors must broadcast to effects") from exc
            if not np.all(np.isfinite(se)) or np.any(se <= 0):
                raise ValueError("standard errors must be finite and strictly positive")
            corr = _cov_array(
                np.eye(d) if correlation is None else correlation, n, d, "correlation"
            )
            if not np.allclose(np.diagonal(corr, axis1=1, axis2=2), 1, rtol=1e-10):
                raise ValueError("correlation must have unit diagonal")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                cov = corr * se[:, :, None] * se[:, None, :]
            cov = _cov_array(cov, n, d, "sampling covariance")
        scale = se if alpha == 1 else np.ones_like(se)
        working_cov = cov / scale[:, :, None] / scale[:, None, :]
        if not np.all(np.isfinite(working_cov)) or not np.all(np.isfinite(scale)):
            raise ValueError("input scales overflow the working covariance")
        with np.errstate(over="ignore", invalid="ignore"):
            working_effects = b / scale
        if not np.all(np.isfinite(working_effects[np.isfinite(b)])):
            raise ValueError("input scales overflow the working effects")
        for key, val in dict(
            effects=b,
            standard_errors=se,
            error_covariance=cov,
            scale=scale,
            working_effects=working_effects,
            working_covariance=working_cov,
        ).items():
            object.__setattr__(self, key, readonly(val))
        object.__setattr__(self, "mask", readonly(np.isfinite(b), dtype=bool))
        object.__setattr__(self, "row_names", _labels(row_names, n, "effect"))
        object.__setattr__(self, "condition_names", _labels(condition_names, d, "condition"))
        object.__setattr__(self, "alpha", int(alpha))

    @property
    def n_effects(self):
        return self.effects.shape[0]

    @property
    def n_conditions(self):
        return self.effects.shape[1]

    def subset(self, indices):
        """Select rows without changing their sampling covariance or scale."""
        idx = np.arange(self.n_effects)[indices]
        idx = np.atleast_1d(idx)
        return MashData(
            self.effects[idx],
            error_covariance=self.error_covariance[idx],
            alpha=self.alpha,
            row_names=[self.row_names[i] for i in idx],
            condition_names=self.condition_names,
        )


def set_data(effects, standard_errors=1, **options):
    """Construct :class:`MashData`; keyword options match its constructor."""
    return MashData(effects, standard_errors, **options)
