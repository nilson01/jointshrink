"""Native multivariate Student-t summary likelihood with Gaussian mixture priors.

A row has one shared Gamma precision, including when some conditions are
missing. The supplied error matrix is the t *scale* matrix, not its covariance.
Posterior calculations integrate over log precision using positive quadrature
weights. They do not convert t statistics to Gaussian scores.
"""

import warnings
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp, roots_legendre, ndtr

from .data import MashData
from .model import MashPrior
from .results import PosteriorResult
from ._numeric import readonly, responsibilities, weight_fit


def _degrees(df, n):
    raw = np.asarray(df)
    if raw.dtype.kind == "b" or any(
        isinstance(x, (bool, np.bool_)) for x in np.asarray(df, dtype=object).flat
    ):
        raise ValueError("df must be finite and positive, not Boolean")
    try:
        value = np.broadcast_to(np.asarray(df, dtype=float), (n,)).copy()
    except ValueError as exc:
        raise ValueError("df must be scalar or have one value per effect row") from exc
    if np.any(~np.isfinite(value)) or np.any(value <= 0) or np.any(value / 2 == 0):
        raise ValueError("df must be finite, positive and numerically representable")
    return value


def _settings(order, maximum, tolerance):
    for name, value in [("quadrature_order", order), ("max_quadrature_order", maximum)]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 4
        ):
            raise ValueError(f"{name} must be an integer of at least four")
    if maximum < order:
        raise ValueError("max_quadrature_order must be at least quadrature_order")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not np.isscalar(tolerance)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("quadrature_tol must be finite and positive")
    return int(order), int(maximum), float(tolerance)


class _Integral:
    """One observed row and Gaussian prior component, in whitened coordinates."""

    def __init__(self, y, v, u, obs, df):
        self.u, self.obs = u, obs
        self.d, self.m = len(u), len(obs)
        lower = np.linalg.cholesky(v)
        inverse = solve_triangular(lower, np.eye(len(obs)), lower=True)
        prior = inverse @ u[np.ix_(obs, obs)] @ inverse.T
        eigenvalues, vectors = np.linalg.eigh((prior + prior.T) / 2)
        eigenvalues = np.maximum(eigenvalues, 0)
        with np.errstate(divide="ignore"):
            self.logeig = np.log(eigenvalues)
        self.z = vectors.T @ inverse @ y
        self.cross = u[:, obs] @ inverse.T @ vectors
        self.back = vectors.T @ inverse
        self.logdet = 2 * np.log(np.diag(lower)).sum()
        self.a = df / 2
        self.width = 1 / np.sqrt(self.a) if self.a >= 1 else 1.0
        if self.a > 1e5:
            # Stable log(a**a / Gamma(a)) - a, via Stirling's expansion.
            self.constant = (
                0.5 * np.log(self.a / (2 * np.pi)) - (1 / self.a) / 12 + (1 / self.a) ** 3 / 360
            )
        else:
            self.constant = self.a * np.log(self.a) - self.a - gammaln(self.a)
        self.left, self.right = self._bounds()

    def evaluate(self, x, moments=False):
        t = np.atleast_1d(x) * self.width
        den = np.logaddexp(self.logeig[None, :], -t[:, None])
        with np.errstate(over="ignore", invalid="ignore"):
            quadratic = np.sum((self.z[None, :] * np.exp(-den / 2)) ** 2, axis=1)
            ll = -0.5 * (self.m * np.log(2 * np.pi) + self.logdet + den.sum(1) + quadratic)
            remainder = self.a * (t - np.expm1(t))
            small = np.abs(t) < 1e-3
            ts = t[small]
            remainder[small] = (
                -0.5
                * (np.sqrt(self.a) * ts) ** 2
                * (1 + ts / 3 + ts**2 / 12 + ts**3 / 60 + ts**4 / 360)
            )
            gamma = self.constant + remainder + np.log(self.width)
        values = ll + gamma
        values = np.where(np.isnan(values), -np.inf, values)
        if not moments:
            return values
        inverse = np.exp(-den)
        means = np.einsum("dm,qm,m->qd", self.cross, inverse, self.z)
        covariances = np.empty((len(t), self.d, self.d))
        for j in range(len(t)):
            gain = (self.cross * inverse[j]) @ self.back
            residual = np.eye(self.d)
            residual[:, self.obs] -= gain
            # Joseph form preserves zero/singular prior supports.
            noise = (self.cross * np.exp(-t[j] - 2 * den[j])) @ self.cross.T
            cov = residual @ self.u @ residual.T + noise
            covariances[j] = (cov + cov.T) / 2
        return values, means, covariances

    def _bounds(self):
        # Expand both tails and inspect a grid, rather than assuming a mode at 0.
        left, right = -1.0, 1.0
        for _ in range(80):
            grid = np.linspace(left, right, 65)
            values = self.evaluate(grid)
            peak = np.max(values)
            if np.isfinite(peak):
                imax = int(np.argmax(values))
                if 0 < imax < len(grid) - 1:
                    opt = minimize_scalar(
                        lambda x: -self.evaluate(x)[0],
                        bounds=(grid[imax - 1], grid[imax + 1]),
                        method="bounded",
                    )
                    peak = max(peak, -opt.fun)
                if values[0] < peak - 45 and values[-1] < peak - 45:
                    return left, right
            if not np.isfinite(peak) or values[0] >= peak - 45:
                left *= 2
            if not np.isfinite(peak) or values[-1] >= peak - 45:
                right *= 2
        raise FloatingPointError(
            "Student-t precision tails could not be bounded; inspect scales or df"
        )

    def integrate(self, order):
        nodes, weights = roots_legendre(order)
        half = (self.right - self.left) / 2
        x = (self.right + self.left) / 2 + half * nodes
        logvalues, mean, cov = self.evaluate(x, moments=True)
        joint = logvalues + np.log(weights) + np.log(half)
        score = logsumexp(joint)
        if not np.isfinite(score) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
            raise FloatingPointError("nonfinite Student-t quadrature; inspect scales and df")
        weights = np.exp(joint - score)
        averaged = weights @ mean
        second = np.einsum("q,qij->ij", weights, cov) + np.einsum(
            "q,qi,qj->ij", weights, mean, mean
        )
        # Sign CDFs join moment checks so sign summaries are monitored too.
        sd = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
        z = np.zeros_like(mean)
        np.divide(-mean, sd, out=z, where=sd > 0)
        cdf = weights @ np.where(sd > 0, ndtr(z), mean <= 0)
        statistic = np.concatenate(([score], averaged, second.ravel(), cdf))
        return score, weights, mean, cov, statistic


def _integrate(data, prior_covariances, df, order, maximum, tolerance):
    n, d, k = data.n_effects, data.n_conditions, len(prior_covariances)
    integrals = []
    for i in range(n):
        obs = np.flatnonzero(data.mask[i])
        integrals.append(
            [
                _Integral(
                    data.working_effects[i, obs],
                    data.working_covariance[i][np.ix_(obs, obs)],
                    u,
                    obs,
                    df[i],
                )
                for u in prior_covariances
            ]
        )
    previous, error, converged = None, None, False
    while True:
        if 8 * n * k * order * (d * d + d + 1) > 256 * 1024**2:
            raise MemoryError(
                "Student-t posterior components exceed 256 MiB; reduce rows, patterns or quadrature order"
            )
        ll = np.empty((n, k))
        weights = np.empty((n, k, order))
        means = np.empty((n, k, order, d))
        cov = np.empty((n, k, order, d, d))
        stats = []
        for i in range(n):
            row = []
            for j in range(k):
                ll[i, j], weights[i, j], means[i, j], cov[i, j], stat = integrals[i][j].integrate(
                    order
                )
                row.append(stat)
            stats.append(row)
        stats = np.asarray(stats)
        if previous is not None:
            error = float(np.max(np.abs(stats - previous) / (1 + np.abs(stats))))
            converged = error <= tolerance
        if converged or order >= maximum:
            break
        previous = stats
        order = min(2 * order, maximum)
    if not converged:
        warnings.warn(
            "Student-t quadrature did not meet its refinement tolerance; increase max_quadrature_order and compare results",
            RuntimeWarning,
            stacklevel=3,
        )
    diagnostics = dict(
        quadrature_order=order,
        quadrature_converged=bool(converged),
        quadrature_relative_change=error,
        quadrature_tolerance=tolerance,
        precision_tail_log_drop=45,
        quadrature_checked="Marginal log densities, first/second moments and marginal CDF at zero; not a rigorous error bound.",
    )
    return ll, weights, means, cov, diagnostics


class StudentTResult(PosteriorResult):
    """Quadrature posterior under an elliptical Student-t observation model.

    Gaussian component priors are fitted by marginal empirical Bayes. Uncertainty
    conditions on these fitted prior parameters and the supplied t scale and df.
    For df<=2 the error covariance does not exist; the scale matrix stays valid.
    """

    def __init__(self, data, prior, df, integrated, *, options, fit_diagnostics=None):
        ll, node_weights, means, cov, diagnostics = integrated
        responsibility, score = responsibilities(ll, prior.weights)
        n, k, q, d = means.shape
        means = means * data.scale[:, None, None, :]
        cov = cov * data.scale[:, None, None, :, None] * data.scale[:, None, None, None, :]
        super().__init__(
            (responsibility[:, :, None] * node_weights).reshape(n, k * q),
            means.reshape(n, k * q, d),
            cov.reshape(n, k * q, d, d),
            row_names=data.row_names,
            condition_names=data.condition_names,
            observed=data.mask,
            metadata={
                "alpha": data.alpha,
                "likelihood": "multivariate_student_t",
                "error_matrix": "scale, not covariance",
                "hyperparameters": "conditional",
            },
        )
        self.data, self.prior, self.df = data, prior, readonly(df)
        self.diagnostics = {**(fit_diagnostics or {}), **diagnostics}
        self.log_likelihood = readonly(score - np.where(data.mask, np.log(data.scale), 0).sum(1))
        self._quadrature_options = dict(options)

    def shrink(self, data, *, df, **quadrature_options):
        """Apply this Gaussian prior using new t scale matrices and explicit df."""
        return apply_student_prior(
            data, self.prior, df, **{**self._quadrature_options, **quadrature_options}
        )

    def score_samples(self, data, *, df, **quadrature_options):
        return self.shrink(data, df=df, **quadrature_options).log_likelihood

    def save(self, path, *, overwrite=False):
        """Save safe JSON inputs and fitted prior; loading repeats quadrature only."""
        from .io import _write, _data_payload, _prior_payload

        return _write(
            path,
            "student_result",
            dict(
                data=_data_payload(self.data),
                prior=_prior_payload(self.prior),
                df=self.df,
                quadrature_options=self._quadrature_options,
            ),
            overwrite,
        )

    def summary(self, **options):
        result = super().summary(**options)
        result.update(
            n_prior_components=len(self.prior.weights),
            likelihood="multivariate_student_t",
            uncertainty="Conditional on fitted Gaussian prior, supplied t scale matrices and df; numerical quadrature.",
            quadrature_converged=self.diagnostics["quadrature_converged"],
        )
        return result


def apply_student_prior(
    data, prior, df, *, quadrature_order=64, max_quadrature_order=256, quadrature_tol=1e-5
):
    """Condition a fixed Gaussian mixture prior on native multivariate t errors.

    ``data.error_covariance`` supplies the t scale matrix. Its covariance is
    df/(df-2) times this matrix only when df>2. One precision is shared across
    observed entries of each row. EE/EZ preprocessing retains its Jacobian.
    """
    if not isinstance(prior, MashPrior):
        raise TypeError("prior must be a MashPrior")
    prior._check(data)
    degrees = _degrees(df, data.n_effects)
    order, maximum, tolerance = _settings(quadrature_order, max_quadrature_order, quadrature_tol)
    options = dict(quadrature_order=order, max_quadrature_order=maximum, quadrature_tol=tolerance)
    integrated = _integrate(data, prior.covariances, degrees, order, maximum, tolerance)
    return StudentTResult(
        data, prior, degrees, integrated, options=options, fit_diagnostics={"fixed_prior": True}
    )


def fit_student_t(
    data,
    df,
    covariances=None,
    *,
    scales=None,
    normalize=True,
    point_mass=True,
    null_weight=1,
    initial_weights=None,
    quadrature_order=64,
    max_quadrature_order=256,
    quadrature_tol=1e-5,
    maxiter=10000,
    tol=1e-8,
):
    """Fit Gaussian prior weights under a native multivariate Student-t likelihood.

    ``df`` is finite and positive, scalar or one value per row. This is an
    elliptical t likelihood with a common row precision, not independent t
    errors for each condition. Prior scales are SD multipliers. Missing entries
    are marginalized using their exact observed block. Quadrature refinement
    diagnostics must be inspected; increasing order does not prove calibration.
    """
    from .patterns import canonical_covariances, expand_covariances, make_scale_grid

    if not isinstance(data, MashData):
        raise TypeError("data must be created with set_data")
    degrees = _degrees(df, data.n_effects)
    order, maximum, tolerance = _settings(quadrature_order, max_quadrature_order, quadrature_tol)
    if covariances is None:
        covariances = canonical_covariances(data)
    if scales is None:
        scales = make_scale_grid(data)
    expanded = expand_covariances(covariances, scales, point_mass=point_mass, normalize=normalize)
    cov = np.stack(list(expanded.values()))
    if cov.shape[1:] != (data.n_conditions, data.n_conditions):
        raise ValueError("covariance dimension must match data conditions")
    integrated = _integrate(data, cov, degrees, order, maximum, tolerance)
    ll = integrated[0]
    null = np.flatnonzero(np.all(cov == 0, axis=(1, 2)))
    weights, diagnostics = weight_fit(
        ll,
        null_index=int(null[0]) if len(null) else None,
        null_weight=null_weight,
        initial_weights=initial_weights,
        maxiter=maxiter,
        tol=tol,
    )
    prior = MashPrior(
        cov,
        tuple(expanded),
        weights,
        alpha=data.alpha,
        metadata={
            "likelihood": "multivariate_student_t",
            "weight_training_rows": data.n_effects,
            "fit_diagnostics": diagnostics,
        },
    )
    options = dict(quadrature_order=order, max_quadrature_order=maximum, quadrature_tol=tolerance)
    return StudentTResult(
        data, prior, degrees, integrated, options=options, fit_diagnostics=diagnostics
    )


def load_student_result(path):
    """Load a t result by applying its saved prior, without refitting weights."""
    from .io import _read, _keys, _restore_data, _restore_prior, _numeric_tree

    payload = _read(path, "student_result")
    _keys(payload, ("data", "prior", "df", "quadrature_options"))
    _numeric_tree(payload["df"])
    _keys(
        payload["quadrature_options"],
        ("quadrature_order", "max_quadrature_order", "quadrature_tol"),
    )
    return apply_student_prior(
        _restore_data(payload["data"]),
        _restore_prior(payload["prior"]),
        payload["df"],
        **payload["quadrature_options"],
    )
