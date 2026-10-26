"""Gaussian conditioning and constrained finite-mixture calculations."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp


def readonly(x, dtype=float):
    value = np.array(x, dtype=dtype, copy=True)
    value.setflags(write=False)
    return value


def validate_psd(matrix, name="covariance"):
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    scale = np.max(np.abs(a))
    if np.max(np.abs(a - a.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    a = (a + a.T) / 2
    ev, q = np.linalg.eigh(a)
    if ev[0] < -1e-10 * max(np.max(np.abs(ev)), np.finfo(float).tiny):
        raise ValueError(f"{name} must be positive semidefinite")
    if ev[0] < 0:
        a = (q * np.maximum(ev, 0)) @ q.T
        a = (a + a.T) / 2
    return a


def component_posterior(data, covariances, *, compute_moments=True):
    """Return observed-block log densities and conditional Gaussian moments.

    All returned moments use the data's working scale. Singular prior
    covariances are allowed; observed sampling covariance is positive definite.
    """
    u = np.asarray(covariances, dtype=float)
    n, d = data.working_effects.shape
    if u.ndim != 3 or u.shape[1:] != (d, d) or len(u) == 0:
        raise ValueError("component covariances must have shape (k,d,d)")
    ll = np.empty((n, len(u)))
    if compute_moments and 8 * n * len(u) * (d * d + d + 1) > 256 * 1024**2:
        raise MemoryError("posterior components exceed 256 MiB; use fit_prior and prior.iter_apply")
    means = np.empty((n, len(u), d)) if compute_moments else None
    covs = np.empty((n, len(u), d, d)) if compute_moments else None
    for i in range(n):
        obs = np.flatnonzero(data.mask[i])
        y = data.working_effects[i, obs]
        v = data.working_covariance[i][np.ix_(obs, obs)]
        for k, uk in enumerate(u):
            c = uk[np.ix_(obs, obs)] + v
            f = cho_factor(c, lower=True, check_finite=True)
            solve_y = cho_solve(f, y)
            ll[i, k] = -0.5 * (
                len(obs) * np.log(2 * np.pi) + 2 * np.log(np.diag(f[0])).sum() + y @ solve_y
            )
            if not compute_moments:
                continue
            cross = uk[:, obs]
            gain = cho_solve(f, cross.T).T
            means[i, k] = gain @ y
            # Joseph form avoids subtracting almost equal prior covariances.
            residual_map = np.eye(d)
            residual_map[:, obs] -= gain
            conditional = residual_map @ uk @ residual_map.T + gain @ v @ gain.T
            # Relative roundoff cleanup; strictly positive eigenvalues stay positive.
            conditional = (conditional + conditional.T) / 2
            vals, vec = np.linalg.eigh(conditional)
            prior_scale = max(np.max(np.abs(uk)), np.finfo(float).tiny)
            if vals[0] < -1e-8 * prior_scale:
                raise FloatingPointError("conditional covariance lost positive semidefiniteness")
            covs[i, k] = (vec * np.maximum(vals, 0)) @ vec.T
    if not np.all(np.isfinite(ll)) or (compute_moments and not np.all(np.isfinite(means))):
        raise FloatingPointError("nonfinite Gaussian calculation; inspect input scales")
    return ll, means, covs


def weight_fit(
    loglik, *, null_index=None, null_weight=1, initial_weights=None, maxiter=10000, tol=1e-8
):
    """Maximize log mixture likelihood plus (null_weight-1)*log(pi_null)."""
    a = np.asarray(loglik, dtype=float)
    if a.ndim != 2 or not np.all(np.isfinite(a)) or min(a.shape) == 0:
        raise ValueError("loglik must be a nonempty finite matrix")
    if not np.isfinite(null_weight) or null_weight < 1:
        raise ValueError("null_weight must be finite and at least one")
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    n, k = a.shape
    penalty = float(null_weight) - 1
    if penalty and (null_index is None or not 0 <= null_index < k):
        raise ValueError("a null penalty requires a zero covariance component")
    if initial_weights is None:
        w = np.full(k, 1 / k)
    else:
        w = np.asarray(initial_weights, dtype=float).copy()
        if w.shape != (k,) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("initial weights must be a finite nonnegative vector")
        w /= w.sum()
    if penalty and w[null_index] <= 0:
        raise ValueError("the penalized null component needs positive initial weight")
    adjusted_start = bool(np.any(w == 0))
    if adjusted_start:
        # A component initialized at zero cannot enter through an EM update.
        w = (1 - 1e-12) * w + 1e-12 / k
    shifts = a.max(axis=1)
    density = np.exp(a - shifts[:, None])

    def objective(x):
        mix = density @ x
        if np.any(mix <= 0) or (penalty and x[null_index] <= 0):
            return -np.inf
        val = np.log(mix).sum() + shifts.sum()
        if penalty:
            val += penalty * np.log(x[null_index])
        return float(val)

    trace = [objective(w)]
    # EM gives a feasible warm start; constrained optimization checks boundary weights.
    for _ in range(min(int(maxiter), 200)):
        resp, _ = responsibilities(a, w)
        counts = resp.sum(axis=0)
        if penalty:
            counts[null_index] += penalty
        proposal = counts / (n + penalty)
        value = objective(proposal)
        if value < trace[-1] - 1e-9 * max(1, abs(trace[-1])):
            raise FloatingPointError("mixture EM objective decreased")
        delta = np.max(np.abs(proposal - w))
        w = proposal
        trace.append(value)
        if delta <= tol:
            break

    def fun(x):
        return -objective(x) / n

    def jac(x):
        mix = np.maximum(density @ x, np.finfo(float).tiny)
        g = (density / mix[:, None]).sum(axis=0)
        if penalty:
            g[null_index] += penalty / max(x[null_index], np.finfo(float).tiny)
        return -g / n

    opt = None
    remaining = int(maxiter) - (len(trace) - 1)
    if remaining > 0 and k > 1:
        bounds = [(0, 1)] * k
        if penalty:
            bounds[null_index] = (np.finfo(float).eps, 1)
        opt = minimize(
            fun,
            w,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(k)},
            options={"maxiter": remaining, "ftol": min(tol * tol, 1e-12)},
        )
        candidate = np.maximum(opt.x, 0)
        candidate /= candidate.sum()
        if np.isfinite(objective(candidate)) and objective(candidate) >= trace[-1] - 1e-10:
            w = candidate
            trace.append(objective(w))
    gradient = -jac(w)
    lam = (n + penalty) / n
    active = w > max(1e-8, tol)
    residual = max(
        float(np.maximum(gradient - lam, 0).max()), float(np.max(np.abs(gradient[active] - lam)))
    )
    diagnostics = dict(
        converged=bool(residual <= max(1e-6, 10 * tol)),
        kkt_residual=residual,
        objective_history=trace,
        objective=trace[-1],
        iterations=len(trace) - 1 + (0 if opt is None else int(opt.nit)),
        null_weight=float(null_weight),
        interior_start_adjusted=adjusted_start,
        optimizer_message="EM" if opt is None else str(opt.message),
    )
    return readonly(w), diagnostics


def responsibilities(loglik, weights):
    with np.errstate(divide="ignore"):
        joint = loglik + np.log(weights)[None, :]
    normalizer = logsumexp(joint, axis=1)
    return np.exp(joint - normalizer[:, None]), normalizer
