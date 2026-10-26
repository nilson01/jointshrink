"""Gaussian-prior Markov chain updates and chain diagnostics."""

import numpy as np
from ._numeric import validate_psd


def gaussian_factor(covariance):
    cov = validate_psd(covariance)
    values, vectors = np.linalg.eigh(cov)
    return vectors * np.sqrt(np.maximum(values, 0))


def elliptical_slice(current, factor, loglike, rng, *, max_steps=1000):
    """Elliptical slice update for a zero-mean (possibly singular) Gaussian prior.

    Implements the full-angle bracket algorithm of Murray et al. (2010).
    The current state must lie in the support of the supplied prior.
    """
    current = np.asarray(current, dtype=float)
    direction = factor @ rng.standard_normal(factor.shape[1])
    old = float(loglike(current))
    if not np.isfinite(old):
        raise ValueError("elliptical slice sampling requires a finite starting log likelihood")
    if not np.any(factor):
        return current.copy(), old, 0
    height = old + np.log(max(rng.random(), np.finfo(float).tiny))
    angle = rng.uniform(0, 2 * np.pi)
    lower, upper = angle - 2 * np.pi, angle
    for steps in range(1, max_steps + 1):
        proposal = current * np.cos(angle) + direction * np.sin(angle)
        value = float(loglike(proposal))
        if np.isfinite(value) and value >= height:
            return proposal, value, steps
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError("elliptical slice bracket exhausted; no approximate update was accepted")


def chain_diagnostics(draws):
    """Split R-hat and a conservative initial-positive-pair autocorrelation ESS.

    These finite-chain diagnostics are screening tools, not convergence proofs.
    """
    x = np.asarray(draws, dtype=float)
    if x.ndim < 2 or x.shape[0] < 2 or x.shape[1] < 4:
        return {"split_rhat": None, "effective_sample_size": None, "diagnostics_available": False}
    chains, samples = x.shape[:2]
    half = samples // 2
    flat = np.concatenate((x[:, :half], x[:, -half:]), axis=0).reshape(2 * chains, half, -1)
    within = flat.var(axis=1, ddof=1).mean(axis=0)
    between = half * flat.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (half - 1) / half * within + between / half
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / within)
    constant = np.all(flat == flat[0, 0], axis=(0, 1))
    rhat[constant] = np.nan
    centered = flat - flat.mean(axis=1, keepdims=True)
    size = 1 << (2 * half - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(spectrum * spectrum.conjugate(), n=size, axis=1)[:, :half].real / half
    mean_acov = acov.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 1 - (within[None, :] - mean_acov) / var_hat[None, :]
    rho[0] = 1
    ess = np.empty(flat.shape[2])
    total = flat.shape[0] * half
    for j in range(flat.shape[2]):
        if constant[j] or not np.isfinite(var_hat[j]) or var_hat[j] <= 0:
            ess[j] = np.nan
            continue
        pairs = []
        for lag in range(0, half - 1, 2):
            pair = rho[lag, j] + rho[lag + 1, j]
            if not np.isfinite(pair) or pair < 0:
                break
            pairs.append(pair if not pairs else min(pair, pairs[-1]))
        tau = max(1, -1 + 2 * sum(pairs))
        ess[j] = min(total, total / tau)
    shape = x.shape[2:]
    return {
        "split_rhat": rhat.reshape(shape),
        "effective_sample_size": ess.reshape(shape),
        "constant_coordinates": constant.reshape(shape),
        "diagnostics_available": True,
    }
