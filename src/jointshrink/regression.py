"""Joint coefficient shrinkage with direct likelihoods and Gaussian random effects."""

from pathlib import Path
import json
import numpy as np
from ._numeric import readonly, validate_psd
from ._sampling import gaussian_factor, elliptical_slice, chain_diagnostics


def _positive_int(value, name, minimum=1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _matrix(value, n=None, name="X"):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a nonempty finite numeric matrix")
    if n is not None and a.shape[0] != n:
        raise ValueError(f"{name} row count must match outcomes")
    return a


def _offset(value, n, d):
    a = np.asarray(value, dtype=float)
    if a.shape == (n,):
        a = a[:, None]
    try:
        a = np.broadcast_to(a, (n, d)).copy()
    except ValueError as exc:
        raise ValueError("offset must broadcast to observations by latent predictors") from exc
    if not np.all(np.isfinite(a)):
        raise ValueError("offset must be finite")
    return a


def _labels(values, n, prefix):
    result = tuple(f"{prefix}{i + 1}" for i in range(n)) if values is None else tuple(values)
    if (
        len(result) != n
        or len(set(result)) != n
        or any(not isinstance(v, str) or not v.strip() or not v.isprintable() for v in result)
    ):
        raise ValueError(f"{prefix} names must be unique nonempty printable strings")
    return result


def _prior(d, covariances, scales, point_mass):
    if covariances is None:
        if d == 1:
            covariances = {"independent": np.eye(1)}
        else:
            from .patterns import canonical_covariances

            covariances = canonical_covariances(d)
    if hasattr(covariances, "items"):
        patterns = list(covariances.items())
    else:
        a = np.asarray(covariances, dtype=float)
        if a.ndim == 2:
            a = a[None]
        patterns = [(f"pattern{i + 1}", u) for i, u in enumerate(a)]
    scales = np.asarray(scales, dtype=float)
    if (
        scales.ndim != 1
        or len(scales) == 0
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0)
    ):
        raise ValueError("scales must be a positive finite SD-multiplier vector")
    names = []
    matrices = []
    if point_mass:
        names.append("null")
        matrices.append(np.zeros((d, d)))
    for name, u in patterns:
        u = validate_psd(u)
        if u.shape != (d, d):
            raise ValueError("prior pattern dimension must match latent predictors")
        for scale in scales:
            with np.errstate(over="ignore", under="ignore"):
                matrix = u * scale * scale
            if not np.all(np.isfinite(matrix)) or (np.any(u) and not np.any(matrix)):
                raise ValueError("prior scale exceeds floating-point limits")
            if not any(np.array_equal(matrix, old) for old in matrices):
                matrices.append(matrix)
                names.append(f"{name}@{scale:g}")
    if not matrices:
        raise ValueError("at least one prior component is needed")
    return tuple(names), readonly(matrices)


class DrawResult:
    """Marginal summaries of retained coefficient or contrast draws.

    Monte Carlo error and prior/likelihood assumptions remain relevant even
    when chain diagnostics appear satisfactory.
    """

    def __init__(self, draws, *, row_names=None, condition_names=None):
        x = np.asarray(draws, dtype=float)
        if x.ndim != 4 or min(x.shape) < 1 or not np.all(np.isfinite(x)):
            raise ValueError("draws must have shape (chains,draws,rows,conditions) and be finite")
        self.draws = readonly(x)
        self.row_names = _labels(row_names, x.shape[2], "coefficient")
        self.condition_names = _labels(condition_names, x.shape[3], "outcome")

    @property
    def posterior_mean(self):
        return self.draws.mean(axis=(0, 1))

    @property
    def coefficients(self):
        return self.posterior_mean

    @property
    def posterior_sd(self):
        return self.draws.reshape(-1, *self.draws.shape[2:]).std(axis=0, ddof=1)

    @property
    def posterior_covariance(self):
        flat = self.draws.reshape(-1, *self.draws.shape[2:])
        centered = flat - flat.mean(axis=0)
        return np.einsum("spr,spt->prt", centered, centered) / (len(flat) - 1)

    @property
    def lfdr(self):
        return (self.draws == 0).mean(axis=(0, 1))

    @property
    def lfsr(self):
        return np.minimum((self.draws <= 0).mean(axis=(0, 1)), (self.draws >= 0).mean(axis=(0, 1)))

    @property
    def joint_null_probability(self):
        return np.all(self.draws == 0, axis=3).mean(axis=(0, 1))

    def cdf(self, value):
        value = np.asarray(value, dtype=float)
        if np.any(np.isnan(value)):
            raise ValueError("CDF values cannot be NaN")
        return (self.draws <= value).mean(axis=(0, 1))

    def quantile(self, probability):
        p = np.asarray(probability, dtype=float)
        if p.ndim > 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be a scalar or vector in [0,1]")
        return np.quantile(self.draws.reshape(-1, *self.draws.shape[2:]), p, axis=0)

    def interval(self, level=0.95):
        if not np.isscalar(level) or not np.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        return np.moveaxis(self.quantile([(1 - level) / 2, (1 + level) / 2]), 0, -1)

    def sample(self, size=1, seed=None):
        size = _positive_int(size, "size")
        flat = self.draws.reshape(-1, *self.draws.shape[2:])
        return flat[np.random.default_rng(seed).integers(len(flat), size=size)]

    def contrast(self, coefficients, *, names=None):
        a = np.asarray(coefficients, dtype=float)
        if a.ndim == 1:
            a = a[None, :]
        if a.ndim != 2 or a.shape[1] != self.draws.shape[3] or not np.all(np.isfinite(a)):
            raise ValueError("contrast rows must match the latent outcome dimension")
        return DrawResult(
            np.einsum("cdpr,mr->cdpm", self.draws, a),
            row_names=self.row_names,
            condition_names=_labels(names, len(a), "contrast"),
        )

    def to_records(self, level=0.95):
        intervals = self.interval(level)
        mean, sd = self.posterior_mean, self.posterior_sd
        null, sign = self.lfdr, self.lfsr
        return [
            dict(
                term=name,
                outcome=outcome,
                mean=float(mean[j, k]),
                sd=float(sd[j, k]),
                lower=float(intervals[j, k, 0]),
                upper=float(intervals[j, k, 1]),
                lfdr=float(null[j, k]),
                lfsr=float(sign[j, k]),
            )
            for j, name in enumerate(self.row_names)
            for k, outcome in enumerate(self.condition_names)
        ]

    def to_dataframe(self, level=0.95):
        import pandas as pd

        return pd.DataFrame.from_records(self.to_records(level))

    def summary(self):
        return {
            "coefficients": self.posterior_mean,
            "posterior_sd": self.posterior_sd,
            "interval": self.interval(),
            "lfdr": self.lfdr,
            "lfsr": self.lfsr,
        }

    def format_summary(self):
        return "\n".join(
            f"{r['term']} / {r['outcome']}: {r['mean']:.4g} [{r['lower']:.4g}, {r['upper']:.4g}]"
            for r in self.to_records()
        )


class RegressionResult(DrawResult):
    """Direct-likelihood shrinkage regression and retained joint posterior draws."""

    def __init__(
        self,
        draws,
        *,
        intercept_draws,
        weight_draws,
        random_draws,
        random_scale_draws,
        likelihood,
        X,
        offset,
        groups,
        random_design,
        group_names,
        prior_covariances,
        prior_names,
        random_covariance,
        diagnostics,
        row_names=None,
        condition_names=None,
    ):
        super().__init__(draws, row_names=row_names, condition_names=condition_names)
        c, s, p, d = self.draws.shape
        shape = (c, s)
        X = _matrix(X, name="X")
        if X.shape[1] != p or len(X) != likelihood.n_observations or d != likelihood.n_latent:
            raise ValueError("stored design or outcome dimensions disagree with draws")
        z = np.asarray(random_design, dtype=float)
        if z.ndim != 2 or len(z) != len(X) or not np.all(np.isfinite(z)):
            raise ValueError("stored random design has invalid dimensions")
        g = len(group_names)
        q = z.shape[1]
        for name, value, target in [
            ("intercept_draws", intercept_draws, shape + (d,)),
            ("random_draws", random_draws, shape + (g, q, d)),
            ("random_scale_draws", random_scale_draws, shape),
        ]:
            array = np.asarray(value, dtype=float)
            if array.size == 0 and np.prod(target) == 0:
                array = array.reshape(target)
            if array.shape != target or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} has invalid shape or values")
            setattr(self, name, readonly(array))
        weights = np.asarray(weight_draws, dtype=float)
        covariance = np.asarray(prior_covariances, dtype=float)
        if (
            covariance.ndim != 3
            or covariance.shape[1:] != (d, d)
            or weights.shape != shape + (len(covariance),)
            or len(prior_names) != len(covariance)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.allclose(weights.sum(axis=-1), 1)
        ):
            raise ValueError("stored prior components or mixture weights disagree")
        for matrix in covariance:
            validate_psd(matrix)
        if np.any(self.random_scale_draws <= 0):
            raise ValueError("stored random scales must be positive")
        self.weight_draws = readonly(weights)
        self.likelihood = likelihood
        self.family = likelihood.family
        self.X = readonly(X)
        self.offset = readonly(_offset(offset, len(X), d))
        self.groups = None if groups is None else tuple(groups)
        self.random_design = readonly(random_design)
        self.group_names = _labels(group_names, g, "group")
        self.prior_covariances = readonly(prior_covariances)
        self.prior_names = tuple(prior_names)
        rcov = np.asarray(random_covariance, dtype=float)
        if rcov.size == 0 and g == 0:
            rcov = rcov.reshape(0, 0)
        if rcov.shape != (q * d, q * d) or not np.all(np.isfinite(rcov)):
            raise ValueError("stored random covariance has invalid shape")
        if g:
            np.linalg.cholesky(rcov)
            if groups is None or len(groups) != len(X) or set(groups) != set(group_names):
                raise ValueError("stored group labels disagree")
        self.random_covariance = readonly(rcov)
        self.diagnostics = diagnostics

    @property
    def intercept(self):
        return self.intercept_draws.mean(axis=(0, 1))

    def _predict_draws(
        self, X, *, groups=None, random_design=None, offset=0, allow_new_groups=False, seed=0
    ):
        X = _matrix(X, name="new X")
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("new predictor columns must match the fitted order")
        n = len(X)
        d = self.draws.shape[-1]
        flat = self.draws.reshape(-1, *self.draws.shape[2:])
        intercepts = self.intercept_draws.reshape(-1, d)
        eta = np.einsum("np,spr->snr", X, flat) + intercepts[:, None, :] + _offset(offset, n, d)
        if groups is not None:
            if not self.group_names:
                raise ValueError("this model has no group effects")
            groups = tuple(groups)
            if len(groups) != n or any(not isinstance(g, str) or not g for g in groups):
                raise ValueError("new group labels must match rows")
            q = self.random_design.shape[1]
            z = (
                np.ones((n, 1))
                if random_design is None and q == 1
                else _matrix(random_design, n, "new random_design")
            )
            if z.shape[1] != q:
                raise ValueError("new random-effect columns must match the fitted order")
            old = self.random_draws.reshape(-1, len(self.group_names), q, d)
            rng = np.random.default_rng(seed)
            extra = {}
            factor = gaussian_factor(self.random_covariance)
            tau = self.random_scale_draws.reshape(-1)
            for i, g in enumerate(groups):
                if g in self.group_names:
                    v = old[:, self.group_names.index(g)]
                elif allow_new_groups:
                    if g not in extra:
                        extra[g] = (
                            rng.normal(size=(len(flat), q * d)) @ factor.T * tau[:, None]
                        ).reshape(-1, q, d)
                    v = extra[g]
                else:
                    raise ValueError(
                        f"unknown group {g!r}; set allow_new_groups=True to draw new group effects"
                    )
                eta[:, i, :] += np.einsum("q,sqr->sr", z[i], v)
        return eta

    def predict(
        self,
        X,
        *,
        scale="response",
        groups=None,
        random_design=None,
        offset=0,
        allow_new_groups=False,
        seed=0,
        **options,
    ):
        """Average the requested prediction over retained posterior draws.

        With groups=None the random effect is fixed at zero. This is not the
        population-marginal response mean for a nonlinear link. Unknown groups
        are integrated by prior draws only when allow_new_groups=True.
        """
        eta = self._predict_draws(
            X,
            groups=groups,
            random_design=random_design,
            offset=offset,
            allow_new_groups=allow_new_groups,
            seed=seed,
        )
        return self._response_draws(eta, scale=scale, **options).mean(axis=0)

    def _response_draws(self, eta, *, scale="response", **options):
        """Transform each draw, including its fitted Cox baseline."""
        if self.family == "cox" and scale == "survival":
            time = np.asarray(options.pop("time", None), dtype=float)
            if options:
                raise ValueError(f"unsupported Cox survival options: {sorted(options)}")
            if time.ndim > 1 or time.size == 0 or np.any(~np.isfinite(time)) or np.any(time < 0):
                raise ValueError("survival prediction requires finite nonnegative time")
            training = self._predict_draws(
                self.X, groups=self.groups, random_design=self.random_design, offset=self.offset
            )
            out = []
            for e, t in zip(eta, training):
                baseline = self.likelihood.baseline_cumulative_hazard(t)
                times, cumulative = baseline
                if len(times) == 0:
                    raise ValueError(
                        "Cox absolute survival cannot be estimated without observed events"
                    )
                positions = np.searchsorted(times, time, side="right") - 1
                h = np.where(positions >= 0, cumulative[np.maximum(positions, 0)], 0)
                with np.errstate(divide="ignore", over="ignore", under="ignore"):
                    log_h = np.log(np.atleast_1d(h))
                    cumulative_risk = np.exp(e[:, 0, None] + log_h[None, :])
                    out.append(np.exp(-cumulative_risk))
            return np.asarray(out)
        return np.asarray([self.likelihood.predict(e, scale=scale, **options) for e in eta])

    def predict_interval(self, X, *, level=0.95, scale="response", **options):
        """Credible intervals for conditional response means, not future outcomes."""
        if not 0 < level < 1:
            raise ValueError("level must lie between zero and one")
        prediction_keys = ("groups", "random_design", "offset", "allow_new_groups", "seed")
        arguments = {k: options.pop(k) for k in prediction_keys if k in options}
        eta = self._predict_draws(X, **arguments)
        values = self._response_draws(eta, scale=scale, **options)
        return np.moveaxis(np.quantile(values, [(1 - level) / 2, (1 + level) / 2], axis=0), 0, -1)

    def posterior_predictive(self, X, *, seed=0, **options):
        """Draw future outcomes, retaining coefficient and group uncertainty."""
        keys = ("groups", "random_design", "offset", "allow_new_groups")
        arguments = {k: options.pop(k) for k in keys if k in options}
        rng = np.random.default_rng(seed)
        eta = self._predict_draws(X, seed=rng, **arguments)
        return np.asarray([self.likelihood.sample(e, rng, **options) for e in eta])

    def random_effect_summary(self, level=0.95):
        if not self.group_names:
            raise ValueError("this model has no group effects")
        value = self.random_draws.reshape(*self.random_draws.shape[:2], -1, self.draws.shape[-1])
        names = [
            f"{g}:random{q + 1}"
            for g in self.group_names
            for q in range(self.random_design.shape[1])
        ]
        return DrawResult(value, row_names=names, condition_names=self.condition_names).to_records(
            level
        )

    def summary(self):
        result = super().summary()
        result.update(
            family=self.family,
            intercept=self.intercept,
            diagnostics=self.diagnostics,
            conditional_parameters=self.likelihood.options,
        )
        return result

    def save(self, path, *, overwrite=False):
        save_regression_result(self, path, overwrite=overwrite)


def fit_regression(
    X,
    y,
    *,
    family="gaussian",
    covariances=None,
    scales=(0.1, 0.5, 1, 2),
    point_mass=True,
    draws=1000,
    warmup=500,
    chains=4,
    seed=0,
    fit_intercept=True,
    offset=0,
    groups=None,
    random_design=None,
    random_covariance=None,
    random_scale=1,
    estimate_random_scale=True,
    dirichlet=1,
    coefficient_names=None,
    outcome_names=None,
    **likelihood_options,
):
    """Fit raw outcomes using a shared Gaussian-mixture coefficient prior.

    Gaussian residual scale, NB2 dispersion, ordinal thresholds and Weibull
    shape are supplied likelihood parameters. They are held fixed. Mixing
    proportions have a Dirichlet prior and are sampled, rather than fitted by
    the Gaussian-summary empirical-Bayes optimizer. Inspect chain diagnostics.
    """
    from .likelihoods import make_likelihood

    likelihood = make_likelihood(y, family=family, **likelihood_options)
    n = likelihood.n_observations
    d = likelihood.n_latent
    X = _matrix(X, n)
    p = X.shape[1]
    draws = _positive_int(draws, "draws", 4)
    warmup = _positive_int(warmup, "warmup", 0)
    chains = _positive_int(chains, "chains", 2)
    off = _offset(offset, n, d)
    if covariances is None and family.startswith(("zero_inflated", "hurdle")):
        covariances = {"independent": np.eye(d)}
    if outcome_names is None:
        if family.startswith(("zero_inflated", "hurdle")):
            outcome_names = [f"count{i + 1}" for i in range(likelihood.n_outputs)] + [
                f"zero_logit{i + 1}" for i in range(likelihood.n_outputs)
            ]
        elif family == "multinomial":
            outcome_names = [f"category{i}_vs_{likelihood.n_categories - 1}" for i in range(d)]
    names, ucov = _prior(d, covariances, scales, point_mass)
    k = len(ucov)
    factors = [gaussian_factor(u) for u in ucov]
    concentration = np.broadcast_to(np.asarray(dirichlet, dtype=float), (k,)).copy()
    if not np.all(np.isfinite(concentration)) or np.any(concentration <= 0):
        raise ValueError("dirichlet must contain positive finite concentrations")
    if family == "cox":
        fit_intercept = False
    group_names = ()
    group_index = None
    if groups is not None:
        groups = tuple(groups)
        if len(groups) != n or any(
            not isinstance(g, str) or not g.strip() or not g.isprintable() for g in groups
        ):
            raise ValueError("groups must contain a nonempty printable string per observation")
        group_names = tuple(dict.fromkeys(groups))
        group_index = np.array([group_names.index(g) for g in groups])
        z = np.ones((n, 1)) if random_design is None else _matrix(random_design, n, "random_design")
    else:
        if random_design is not None or random_covariance is not None:
            raise ValueError("random-design options require group labels")
        z = np.empty((n, 0))
    q = z.shape[1]
    g = len(group_names)
    h = q * d
    if not np.isscalar(random_scale) or not np.isfinite(random_scale) or random_scale <= 0:
        raise ValueError("random_scale must be finite and positive")
    if g:
        rcov = np.eye(h) if random_covariance is None else validate_psd(random_covariance)
        if rcov.shape != (h, h):
            raise ValueError("random_covariance must match random columns times latent outcomes")
        try:
            rfactor = np.linalg.cholesky(rcov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("random_covariance must be positive definite") from exc
        rinverse = np.linalg.inv(rcov)
    else:
        rcov = np.empty((0, 0))
        rfactor = rcov
    constant_columns = np.all(X == X[0], axis=0)
    if fit_intercept and np.any(constant_columns & (X[0] != 0)):
        raise ValueError("constant nonzero predictor duplicates the fitted intercept")
    fixed_design = np.column_stack((np.ones(n), X)) if fit_intercept else X
    fixed_rank = int(np.linalg.matrix_rank(fixed_design))
    random_rank = int(np.linalg.matrix_rank(z)) if g else 0
    bytes_needed = 8 * chains * draws * (p * d + d + k + g * h + p + 2)
    if bytes_needed > 512 * 1024**2:
        raise MemoryError(
            "retained draws exceed 512 MiB; use fewer retained draws or a server-sized design"
        )
    out = np.empty((chains, draws, p, d))
    ints = np.empty((chains, draws, d))
    ws = np.empty((chains, draws, k))
    rand = np.empty((chains, draws, g, q, d))
    taus = np.empty((chains, draws))
    label_draws = np.empty((chains, draws, p), dtype=int)
    logtrace = np.empty((chains, draws))
    switches = np.zeros(chains, dtype=int)
    accepted = np.zeros(chains, dtype=int)
    attempts = np.zeros(chains, dtype=int)
    slice_evals = np.zeros(chains, dtype=int)
    streams = np.random.SeedSequence(seed).spawn(chains)
    for c, stream in enumerate(streams):
        rng = np.random.default_rng(stream)
        beta = np.zeros((p, d))
        intercept = np.zeros(d)
        weights = rng.dirichlet(concentration)
        labels = np.zeros(p, dtype=int)
        u = np.zeros((g, q, d))
        tau2 = float(random_scale) ** 2
        eta = off.copy()
        current = float(likelihood.loglike(eta))
        if not np.isfinite(current):
            raise ValueError("initial likelihood is nonfinite; inspect offsets and outcome scales")
        for step in range(warmup + draws):
            for j in rng.permutation(p):
                base = eta - X[:, j, None] * beta[j]
                candidate_label = int(rng.choice(k, p=weights))
                proposal = factors[candidate_label] @ rng.normal(size=d)
                candidate_eta = base + X[:, j, None] * proposal
                value = float(likelihood.loglike(candidate_eta))
                attempts[c] += 1
                if np.isfinite(value) and np.log(max(rng.random(), np.finfo(float).tiny)) < min(
                    0, value - current
                ):
                    switches[c] += int(labels[j] != candidate_label)
                    labels[j] = candidate_label
                    beta[j] = proposal
                    eta = candidate_eta
                    current = value
                    accepted[c] += 1
                base = eta - X[:, j, None] * beta[j]

                def block(value):
                    return likelihood.loglike(base + X[:, j, None] * value)

                beta[j], current, neval = elliptical_slice(beta[j], factors[labels[j]], block, rng)
                eta = base + X[:, j, None] * beta[j]
                slice_evals[c] += neval
            if fit_intercept:
                base = eta - intercept

                def intercept_likelihood(value):
                    return likelihood.loglike(base + value)

                intercept, current, neval = elliptical_slice(
                    intercept, 5 * np.eye(d), intercept_likelihood, rng
                )
                eta = base + intercept
                slice_evals[c] += neval
            if g:
                for index in rng.permutation(g):
                    rows = group_index == index
                    base = eta.copy()
                    base[rows] -= z[rows] @ u[index]

                    def group_likelihood(value):
                        candidate = base.copy()
                        candidate[rows] += z[rows] @ value.reshape(q, d)
                        return likelihood.loglike(candidate)

                    value, current, neval = elliptical_slice(
                        u[index].ravel(), np.sqrt(tau2) * rfactor, group_likelihood, rng
                    )
                    u[index] = value.reshape(q, d)
                    eta = base
                    eta[rows] += z[rows] @ u[index]
                    slice_evals[c] += neval
                if estimate_random_scale:
                    v = u.reshape(g, h)
                    quadratic = np.einsum("gi,ij,gj->", v, rinverse, v)
                    tau2 = 1 / rng.gamma(2 + g * h / 2, scale=1 / (1 + quadratic / 2))
            weights = rng.dirichlet(concentration + np.bincount(labels, minlength=k))
            if step >= warmup:
                s = step - warmup
                out[c, s] = beta
                ints[c, s] = intercept
                ws[c, s] = weights
                rand[c, s] = u
                taus[c, s] = np.sqrt(tau2)
                label_draws[c, s] = labels
                logtrace[c, s] = current
    diagnostic = chain_diagnostics(out)
    diagnostic.update(
        method="Gaussian-mixture prior MCMC: prior independence MH + elliptical slice + Dirichlet Gibbs",
        inference="fully Bayesian coefficient/mixing-weight model; not Gaussian-summary empirical Bayes",
        draws=draws,
        warmup=warmup,
        chains=chains,
        prior_proposal_acceptance=accepted / attempts,
        component_switches=switches,
        slice_likelihood_evaluations=slice_evals,
        component_labels=label_draws,
        log_likelihood=logtrace,
        fit_intercept=bool(fit_intercept),
        nuisance_parameters="sigma, dispersion, shape and ordinal thresholds held fixed",
        weight_diagnostics=chain_diagnostics(ws[..., None]),
        intercept_diagnostics=chain_diagnostics(ints[:, :, None, :]) if fit_intercept else None,
        random_scale_diagnostics=(
            chain_diagnostics(taus[:, :, None, None]) if g and estimate_random_scale else None
        ),
        fixed_design_rank=fixed_rank,
        fixed_design_columns=fixed_design.shape[1],
        random_design_rank=random_rank,
        random_design_columns=q,
        identification_warnings=(
            (
                ["fixed design is rank deficient; proper priors do not identify aliased effects"]
                if fixed_rank < fixed_design.shape[1]
                else []
            )
            + (["random design is rank deficient"] if g and random_rank < q else [])
            + (
                ["zero predictor columns receive prior-only coefficient information"]
                if np.any(constant_columns & (X[0] == 0))
                else []
            )
        ),
        random_scale_prior="IG(shape=2,scale=1)" if g and estimate_random_scale else "fixed",
        convergence="not established by successful execution",
    )
    result = RegressionResult(
        out,
        intercept_draws=ints,
        weight_draws=ws,
        random_draws=rand,
        random_scale_draws=taus,
        likelihood=likelihood,
        X=X,
        offset=off,
        groups=groups,
        random_design=z,
        group_names=group_names,
        prior_covariances=ucov,
        prior_names=names,
        random_covariance=rcov,
        diagnostics=diagnostic,
        row_names=coefficient_names,
        condition_names=outcome_names,
    )
    return result


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_regression_result(result, path, *, overwrite=False):
    """Save retained draws and training inputs as JSON; this retains raw data."""
    values = {
        name: getattr(result, name)
        for name in (
            "draws",
            "intercept_draws",
            "weight_draws",
            "random_draws",
            "random_scale_draws",
            "X",
            "offset",
            "groups",
            "random_design",
            "group_names",
            "prior_covariances",
            "prior_names",
            "random_covariance",
            "diagnostics",
            "row_names",
            "condition_names",
        )
    }
    values["likelihood"] = result.likelihood.to_dict()
    document = {"schema": "jointshrink-regression", "version": 1, "payload": _json(values)}
    content = json.dumps(document, allow_nan=False, indent=2) + "\n"
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(content)


def load_regression_result(path):
    """Restore stored posterior draws without sampling or executing objects."""
    from .likelihoods import make_likelihood

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out

    def reject(value):
        raise ValueError("nonfinite JSON constant")

    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream, object_pairs_hook=unique, parse_constant=reject)
    if (
        set(record) != {"schema", "version", "payload"}
        or record["schema"] != "jointshrink-regression"
        or record["version"] != 1
    ):
        raise ValueError("unsupported regression schema")
    payload = record["payload"]
    expected = {
        "draws",
        "intercept_draws",
        "weight_draws",
        "random_draws",
        "random_scale_draws",
        "X",
        "offset",
        "groups",
        "random_design",
        "group_names",
        "prior_covariances",
        "prior_names",
        "random_covariance",
        "diagnostics",
        "row_names",
        "condition_names",
        "likelihood",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("unexpected regression fields")
    info = payload.pop("likelihood")
    if not isinstance(info, dict) or "y" not in info or "family" not in info:
        raise ValueError("invalid stored likelihood")
    likelihood = make_likelihood(**info)
    return RegressionResult(likelihood=likelihood, **payload)


def fit_gaussian_regression(X, y, **options):
    """Direct Gaussian regression; sigma supplies fixed residual standard deviations."""
    return fit_regression(X, y, family="gaussian", **options)


def fit_logistic_regression(X, y, **options):
    """Binomial/logistic regression; use trials=1 for binary outcomes."""
    return fit_regression(X, y, family="binomial", **options)


def fit_poisson_regression(X, y, **options):
    """Poisson log-link regression with explicit exposure and offsets."""
    return fit_regression(X, y, family="poisson", **options)


def fit_negative_binomial_regression(X, y, **options):
    """NB2 regression conditional on supplied dispersion."""
    return fit_regression(X, y, family="negative_binomial", **options)


def fit_survival(X, time, event, *, family="cox", **options):
    """Cox partial-likelihood or fixed-shape Weibull PH regression."""
    if family not in ("cox", "weibull"):
        raise ValueError("survival family must be cox or weibull")
    return fit_regression(X, time, family=family, event=event, **options)


def fit_multinomial(X, y, **options):
    """Reference-category multinomial logistic regression."""
    return fit_regression(X, y, family="multinomial", **options)


def fit_ordinal(X, y, **options):
    """Cumulative-logit regression conditional on ordered cutpoints."""
    return fit_regression(X, y, family="ordinal", **options)


def fit_zero_inflated_poisson(X, y, **options):
    """Poisson count and structural-zero logistic predictors."""
    return fit_regression(X, y, family="zero_inflated_poisson", **options)


def fit_zero_inflated_negative_binomial(X, y, **options):
    """NB2 count and structural-zero logistic predictors."""
    return fit_regression(X, y, family="zero_inflated_negative_binomial", **options)


def fit_hurdle_poisson(X, y, **options):
    """Zero hurdle and positive-truncated Poisson regression."""
    return fit_regression(X, y, family="hurdle_poisson", **options)


def fit_hurdle_negative_binomial(X, y, **options):
    """Zero hurdle and positive-truncated NB2 regression."""
    return fit_regression(X, y, family="hurdle_negative_binomial", **options)
