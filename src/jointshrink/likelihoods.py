"""Validated likelihoods for regression with shared coefficient priors.

Nuisance parameters are supplied and held fixed. Outcome columns are
conditionally independent given their linear predictors; dependence can enter
through coefficient and random-effect priors.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, log_expit, logsumexp
from scipy.stats import nbinom, poisson

FAMILIES = (
    "gaussian",
    "binomial",
    "poisson",
    "negative_binomial",
    "multinomial",
    "ordinal",
    "zero_inflated_poisson",
    "zero_inflated_negative_binomial",
    "hurdle_poisson",
    "hurdle_negative_binomial",
    "cox",
    "weibull",
)
_COUNTS = {
    "poisson",
    "negative_binomial",
    "zero_inflated_poisson",
    "zero_inflated_negative_binomial",
    "hurdle_poisson",
    "hurdle_negative_binomial",
}
_TWO = {x for x in _COUNTS if x.startswith(("zero_", "hurdle_"))}
_SURVIVAL = {"cox", "weibull"}


def _array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    a = raw.astype(float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite; missing values are not supported")
    return a


def _matrix(value, shape, name, *, positive=False, integer=False):
    a = _array(value, name)
    if a.ndim == 1 and a.shape == (shape[0],) and shape[1] == 1:
        a = a[:, None]
    try:
        a = np.broadcast_to(a, shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or broadcast to {shape}") from exc
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if integer and (np.any(a < 0) or np.any(a != np.floor(a))):
        raise ValueError(f"{name} must contain nonnegative integers")
    return a


def _log1mexp(logp):
    """log(1-exp(logp)) for logp <= 0, including limiting zero."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(logp < -np.log(2), np.log1p(-np.exp(logp)), np.log(-np.expm1(logp)))


class Likelihood:
    """A fixed-nuisance likelihood; construct with :func:`make_likelihood`."""

    def __init__(
        self,
        y,
        *,
        family="gaussian",
        trials=1,
        exposure=1,
        dispersion=1,
        sigma=1,
        n_categories=None,
        thresholds=None,
        event=None,
        entry=0,
        time=None,
        shape=1,
        ties="breslow",
    ):
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        # Reject unrelated controls rather than silently implying support.
        controls = (
            (family == "binomial", trials, 1, "trials"),
            (family in _COUNTS, exposure, 1, "exposure"),
            ("negative_binomial" in family, dispersion, 1, "dispersion"),
            (family == "gaussian", sigma, 1, "sigma"),
            (family in _SURVIVAL, entry, 0, "entry"),
            (family == "weibull", shape, 1, "shape"),
        )
        for relevant, value, default, name in controls:
            if not relevant and not np.all(_array(value, name) == default):
                raise ValueError(f"{name} is not used for {family}")
        if family not in {"multinomial", "ordinal"} and n_categories is not None:
            raise ValueError("n_categories is used only for categorical outcomes")
        if family != "ordinal" and thresholds is not None:
            raise ValueError("thresholds are used only for ordinal outcomes")
        if family not in _SURVIVAL and (event is not None or time is not None):
            raise ValueError("event and time are used only for survival outcomes")
        if ties != "breslow":
            raise ValueError("only Breslow ties are supported")
        self.family = family
        a = _array(y, "y")
        if a.ndim not in (1, 2) or not a.size or a.shape[0] == 0:
            raise ValueError("y must be a nonempty vector or two-dimensional array")
        if a.ndim == 1:
            a = a[:, None]
        if a.shape[1] == 0:
            raise ValueError("y must have at least one outcome column")
        self.y = a.copy()
        self.n_observations, self.n_outputs = a.shape
        self.n_latent = self.n_outputs * (2 if family in _TWO else 1)
        self.options = {}
        if family in _COUNTS or family in {"binomial", "multinomial", "ordinal"}:
            if np.any(a < 0) or np.any(a != np.floor(a)):
                raise ValueError("y must contain nonnegative integer counts or category codes")
        if family == "gaussian":
            self.sigma = _matrix(sigma, a.shape, "sigma", positive=True)
            self.options["sigma"] = self.sigma.tolist()
        if family == "binomial":
            self.trials = _matrix(trials, a.shape, "trials", integer=True)
            if np.any(a > self.trials):
                raise ValueError("binomial y must not exceed trials")
            self.options["trials"] = self.trials.tolist()
        if family in _COUNTS:
            self.exposure = _matrix(exposure, a.shape, "exposure", positive=True)
            self.options["exposure"] = self.exposure.tolist()
            if "negative_binomial" in family:
                self.dispersion = _matrix(dispersion, a.shape, "dispersion", positive=True)
                self.options["dispersion"] = self.dispersion.tolist()
        if family in {"multinomial", "ordinal"}:
            if self.n_outputs != 1:
                raise ValueError("categorical outcomes must be a single vector")
            if isinstance(n_categories, (bool, np.bool_)):
                raise ValueError("n_categories must be an integer, not Boolean")
            k = (
                max(2, int(a.max()) + 1)
                if n_categories is None
                else _array(n_categories, "n_categories")
            )
            if np.ndim(k) != 0:
                raise ValueError("n_categories must be a scalar integer")
            k = float(k)
            if (
                isinstance(k, (bool, np.bool_))
                or not np.isscalar(k)
                or not np.isfinite(k)
                or int(k) != k
                or k < 2
            ):
                raise ValueError("n_categories must be an integer of at least two")
            self.n_categories = int(k)
            if np.any(a >= k):
                raise ValueError("category codes must be smaller than n_categories")
            self.options["n_categories"] = int(k)
            self.n_latent = int(k) - 1 if family == "multinomial" else 1
            if family == "ordinal":
                p = np.arange(1, k) / k
                t = np.log(p / (1 - p)) if thresholds is None else _array(thresholds, "thresholds")
                if t.shape != (k - 1,) or np.any(np.diff(t) <= 0):
                    raise ValueError(
                        "thresholds must be a strictly increasing vector of length n_categories-1"
                    )
                self.thresholds = t.copy()
                self.options["thresholds"] = t.tolist()
        if family in _SURVIVAL:
            if self.n_outputs != 1:
                raise ValueError("survival outcomes must be a single time vector")
            if time is not None:
                supplied = _matrix(time, a.shape, "time", positive=True)
                if not np.array_equal(supplied, a):
                    raise ValueError("time and y must describe the same observed times")
            if np.any(a <= 0):
                raise ValueError("observed times must be strictly positive")
            if event is None:
                raise ValueError("event is required: one for an event, zero for right censoring")
            self.event = _matrix(event, a.shape, "event", integer=True)
            if np.any(self.event > 1):
                raise ValueError("event must contain only zero and one")
            self.entry = _matrix(entry, a.shape, "entry")
            if np.any(self.entry < 0) or np.any(self.entry >= a):
                raise ValueError("entry must be nonnegative and smaller than observed time")
            self.options.update(event=self.event[:, 0].tolist(), entry=self.entry[:, 0].tolist())
            if family == "cox":
                if ties != "breslow":
                    raise ValueError("only Breslow ties are supported")
                self.options["ties"] = ties
            else:
                sh = _array(shape, "shape")
                if sh.ndim or sh <= 0:
                    raise ValueError("Weibull shape must be a strictly positive scalar")
                self.shape = float(sh)
                self.options["shape"] = self.shape
        self.y.setflags(write=False)

    def _eta(self, eta, *, training=False):
        a = _array(eta, "eta")
        if a.ndim != 2 or a.shape[1] != self.n_latent:
            raise ValueError(f"eta must have shape (n, {self.n_latent})")
        if training and a.shape[0] != self.n_observations:
            raise ValueError("eta must have one row per training observation")
        return a

    def _count_terms(self, logmu, y, dispersion=None):
        with np.errstate(over="ignore", invalid="ignore"):
            if "negative_binomial" not in self.family:
                mu = np.exp(logmu)
                lp = y * logmu - mu - gammaln(y + 1)
                lp0 = -mu
            else:
                r = 1 / dispersion
                lr = np.log(r)
                lp = (
                    gammaln(y + r)
                    - gammaln(r)
                    - gammaln(y + 1)
                    - r * np.logaddexp(0, logmu - lr)
                    - y * np.logaddexp(0, lr - logmu)
                )
                lp0 = -r * np.logaddexp(0, logmu - lr)
        return lp, lp0

    def _category_logprob(self, eta):
        if self.family == "multinomial":
            logits = np.column_stack((eta, np.zeros(len(eta))))
            return logits - logsumexp(logits, axis=1, keepdims=True)
        t = self.thresholds
        z = t[None, :] - eta
        out = np.empty((len(eta), self.n_categories))
        out[:, 0] = log_expit(z[:, 0])
        out[:, -1] = log_expit(-z[:, -1])
        if self.n_categories > 2:
            d = np.diff(t)[None, :]
            out[:, 1:-1] = (
                z[:, 1:]
                - np.logaddexp(0, z[:, 1:])
                - np.logaddexp(0, z[:, :-1])
                + np.log(-np.expm1(-d))
            )
        return out

    def loglike(self, eta):
        """Return the summed conditional log likelihood, including constants."""
        eta = self._eta(eta, training=True)
        y = self.y
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.family == "gaussian":
                val = (
                    -0.5 * ((y - eta) / self.sigma) ** 2
                    - np.log(self.sigma)
                    - 0.5 * np.log(2 * np.pi)
                )
            elif self.family == "binomial":
                val = (
                    gammaln(self.trials + 1)
                    - gammaln(y + 1)
                    - gammaln(self.trials - y + 1)
                    + y * log_expit(eta)
                    + (self.trials - y) * log_expit(-eta)
                )
            elif self.family in _COUNTS:
                count_eta = eta[:, : self.n_outputs]
                logmu = count_eta + np.log(self.exposure)
                val, lp0 = self._count_terms(logmu, y, getattr(self, "dispersion", None))
                if self.family in _TWO:
                    z = eta[:, self.n_outputs :]
                    if self.family.startswith("zero_"):
                        val = np.where(
                            y == 0,
                            np.logaddexp(log_expit(z), log_expit(-z) + lp0),
                            log_expit(-z) + val,
                        )
                    else:
                        norm = _log1mexp(lp0)
                        norm = np.where(lp0 == 0, logmu, norm)
                        val = np.where(y == 0, log_expit(z), log_expit(-z) + val - norm)
            elif self.family in {"multinomial", "ordinal"}:
                val = self._category_logprob(eta)[np.arange(len(y)), y[:, 0].astype(int)]
            elif self.family == "cox":
                total = 0.0
                for t in np.unique(y[self.event.astype(bool)]):
                    failed = (y[:, 0] == t) & self.event[:, 0].astype(bool)
                    risk = (self.entry[:, 0] < t) & (y[:, 0] >= t)
                    total += eta[failed, 0].sum() - failed.sum() * logsumexp(eta[risk, 0])
                return float(total)
            else:
                logt = np.log(y)
                # Integrate the baseline hazard only over each observed risk interval.
                logdelta = self.shape * logt + _log1mexp(self.shape * (np.log(self.entry) - logt))
                val = self.event * (np.log(self.shape) + (self.shape - 1) * logt + eta) - np.exp(
                    eta + logdelta
                )
            total = float(np.sum(val))
        return total if np.isfinite(total) else -np.inf

    def predict(self, eta, *, scale="response", trials=None, exposure=None, time=None):
        """Predict means, probabilities, links or Weibull survival.

        New-row trials and exposure default to one, never to training offsets.
        """
        eta = self._eta(eta)
        if trials is not None and self.family != "binomial":
            raise ValueError("trials is used only for binomial prediction")
        if exposure is not None and self.family not in _COUNTS:
            raise ValueError("exposure is used only for count prediction")
        if time is not None and self.family not in _SURVIVAL:
            raise ValueError("time is used only for survival prediction")
        if scale == "link":
            return eta.copy()
        n = len(eta)
        target = (n, self.n_outputs)
        if self.family in {"multinomial", "ordinal"}:
            if scale not in {"response", "probability"}:
                raise ValueError("categorical scale must be response, probability or link")
            return np.exp(self._category_logprob(eta))
        if self.family == "binomial":
            if scale == "probability":
                return expit(eta)
            if scale == "response":
                return _matrix(
                    1 if trials is None else trials, target, "trials", integer=True
                ) * expit(eta)
        elif self.family == "gaussian" and scale == "response":
            return eta.copy()
        elif self.family in _COUNTS and scale == "response":
            e = _matrix(1 if exposure is None else exposure, target, "exposure", positive=True)
            logmu = eta[:, : self.n_outputs] + np.log(e)
            mu = np.exp(logmu)
            if self.family in _TWO:
                factor = expit(-eta[:, self.n_outputs :])
                if self.family.startswith("hurdle_"):
                    d = (
                        self._prediction_parameter("dispersion", target)
                        if "negative_binomial" in self.family
                        else None
                    )
                    _, lp0 = self._count_terms(logmu, np.zeros(target), d)
                    norm = _log1mexp(lp0)
                    norm = np.where(lp0 == 0, logmu, norm)
                    return factor * np.exp(logmu - norm)
                return factor * mu
            return mu
        elif self.family in _SURVIVAL:
            if scale in {"response", "relative_hazard"}:
                return np.exp(eta)
            if self.family == "weibull" and scale in {"survival", "cumulative_hazard"}:
                if time is None:
                    raise ValueError("time is required for survival prediction")
                t = _matrix(time, target, "time")
                if np.any(t < 0):
                    raise ValueError("prediction time must be nonnegative")
                cumulative = t**self.shape * np.exp(eta)
                return np.exp(-cumulative) if scale == "survival" else cumulative
        raise ValueError(f"unsupported prediction scale {scale!r} for {self.family}")

    def _prediction_parameter(self, name, shape):
        a = getattr(self, name)
        if np.all(a == a[0:1]):
            return np.broadcast_to(a[0:1], shape)
        if a.shape == shape:
            return a
        raise ValueError(
            f"row-specific {name} cannot be reused for a different prediction row count"
        )

    def sample(self, eta, rng, *, trials=None, exposure=None, time=None):
        """Draw conditional outcomes; Cox draws require a separately fitted baseline."""
        eta = self._eta(eta)
        target = (len(eta), self.n_outputs)
        if trials is not None and self.family != "binomial":
            raise ValueError("trials is used only for binomial sampling")
        if exposure is not None and self.family not in _COUNTS:
            raise ValueError("exposure is used only for count sampling")
        if time is not None:
            raise ValueError(
                "outcome sampling generates times rather than conditioning on prediction time"
            )
        if self.family == "cox":
            raise ValueError(
                "Cox outcome draws require a baseline hazard and are not available here"
            )
        if self.family == "gaussian":
            return rng.normal(eta, self._prediction_parameter("sigma", target))
        if self.family == "binomial":
            tr = _matrix(1 if trials is None else trials, target, "trials", integer=True)
            return rng.binomial(tr.astype(np.int64), expit(eta))
        if self.family in {"multinomial", "ordinal"}:
            p = self.predict(eta, scale="probability")
            return (
                (rng.uniform(size=len(eta))[:, None] > np.cumsum(p, axis=1))
                .sum(axis=1)[:, None]
                .clip(max=self.n_categories - 1)
            )
        if self.family == "weibull":
            return (-np.log(rng.uniform(size=target)) * np.exp(-eta)) ** (1 / self.shape)
        e = _matrix(1 if exposure is None else exposure, target, "exposure", positive=True)
        logmu = eta[:, : self.n_outputs] + np.log(e)
        mu = np.exp(logmu)
        d = (
            self._prediction_parameter("dispersion", target)
            if "negative_binomial" in self.family
            else None
        )
        dist = poisson(mu) if d is None else nbinom(1 / d, 1 / (1 + d * mu))
        if self.family.startswith("hurdle_"):
            out = np.zeros(target, dtype=np.int64)
            lp, lp0 = self._count_terms(logmu, np.ones(target), d)
            norm = _log1mexp(lp0)
            norm = np.where(lp0 == 0, logmu, norm)
            # Conditional inverse CDF avoids rejection when positive counts are rare.
            u = rng.uniform(size=target)
            ordinary = lp0 < np.log(0.99)
            if np.any(ordinary):
                p0 = np.exp(lp0[ordinary])
                q = np.minimum(np.nextafter(1.0, 0.0), p0 + (1 - p0) * u[ordinary])
                out[ordinary] = (
                    poisson.ppf(q, mu[ordinary])
                    if d is None
                    else nbinom.ppf(q, 1 / d[ordinary], 1 / (1 + d[ordinary] * mu[ordinary]))
                ).astype(np.int64)
            for ix in zip(*np.where(~ordinary)):
                k, mass = 1, float(np.exp(lp[ix] - norm[ix]))
                cumulative = mass
                while u[ix] > cumulative:
                    k += 1
                    if d is None:
                        mass *= mu[ix] / k
                    else:
                        r = 1 / d[ix]
                        mass *= (k - 1 + r) / k * mu[ix] / (r + mu[ix])
                    cumulative += mass
                    if mass == 0:
                        break
                out[ix] = k
        else:
            out = dist.rvs(random_state=rng)
        if self.family in _TWO:
            out = np.where(rng.uniform(size=target) < expit(eta[:, self.n_outputs :]), 0, out)
        return np.asarray(out).reshape(target)

    def baseline_cumulative_hazard(self, eta):
        """Breslow event times and cumulative baseline jumps, including each event."""
        if self.family != "cox":
            raise ValueError("Breslow baseline is defined only for Cox likelihoods")
        eta = self._eta(eta, training=True)
        times = np.unique(self.y[self.event.astype(bool)])
        increments = []
        for t in times:
            failed = (self.y[:, 0] == t) & self.event[:, 0].astype(bool)
            risk = (self.entry[:, 0] < t) & (self.y[:, 0] >= t)
            increments.append(np.exp(np.log(failed.sum()) - logsumexp(eta[risk, 0])))
        return times, np.cumsum(increments)

    def to_dict(self):
        """Return constructor data containing JSON-compatible values only."""
        return {"y": self.y.tolist(), "family": self.family, **self.options}


def make_likelihood(
    y,
    *,
    family="gaussian",
    trials=1,
    exposure=1,
    dispersion=1,
    sigma=1,
    n_categories=None,
    thresholds=None,
    event=None,
    entry=0,
    time=None,
    shape=1,
    ties="breslow",
):
    """Build a likelihood with explicit fixed nuisance parameters.

    Negative-binomial dispersion uses variance ``mu + dispersion * mu**2``.
    Zero/hurdle logits give the structural-zero probability. Ordinal predictors
    shift a logistic latent variable with fixed, increasing cutpoints.
    """
    return Likelihood(
        y,
        family=family,
        trials=trials,
        exposure=exposure,
        dispersion=dispersion,
        sigma=sigma,
        n_categories=n_categories,
        thresholds=thresholds,
        event=event,
        entry=entry,
        time=time,
        shape=shape,
        ties=ties,
    )
