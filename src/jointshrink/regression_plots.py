"""Plots of posterior coefficients, mixing proportions and sampling diagnostics."""

import numpy as np


def _axis(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    return ax


def _column(result, condition):
    if isinstance(condition, str):
        if condition not in result.condition_names:
            raise ValueError("unknown coefficient outcome name")
        return result.condition_names.index(condition)
    if (
        isinstance(condition, (bool, np.bool_))
        or not isinstance(condition, (int, np.integer))
        or not 0 <= condition < len(result.condition_names)
    ):
        raise ValueError("condition must be a valid name or integer index")
    return int(condition)


def plot_coefficients(result, *, condition=0, level=0.95, ax=None):
    """Show posterior coefficient means and equal-tail credible intervals."""
    c = _column(result, condition)
    ax = _axis(ax)
    interval = result.interval(level)[:, c]
    pos = np.arange(len(result.row_names))
    ax.hlines(pos, interval[:, 0], interval[:, 1], color="#0072B2")
    ax.scatter(result.posterior_mean[:, c], pos, color="#0072B2", s=18)
    ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_yticks(pos, result.row_names)
    ax.invert_yaxis()
    ax.set(xlabel="Coefficient", title=result.condition_names[c])
    return ax


def plot_chains(result, *, term=0, condition=0, ax=None):
    """Display all retained coefficient chains without smoothing."""
    c = _column(result, condition)
    if isinstance(term, str):
        term = result.row_names.index(term)
    if (
        isinstance(term, (bool, np.bool_))
        or not isinstance(term, (int, np.integer))
        or not 0 <= term < len(result.row_names)
    ):
        raise ValueError("term must be a valid coefficient name or index")
    ax = _axis(ax)
    for chain, values in enumerate(result.draws[:, :, term, c]):
        ax.plot(np.arange(len(values)), values, linewidth=0.7, label=f"Chain{chain + 1}")
    ax.set(
        xlabel="Retained draw",
        ylabel="Coefficient",
        title=f"{result.row_names[term]} / {result.condition_names[c]}",
    )
    ax.legend(frameon=False)
    return ax


def plot_regression_weights(result, *, level=0.95, ax=None):
    """Show posterior mixture weights and their equal-tail credible intervals."""
    if not 0 < level < 1:
        raise ValueError("level must lie between zero and one")
    flat = result.weight_draws.reshape(-1, len(result.prior_names))
    intervals = np.quantile(flat, [(1 - level) / 2, (1 + level) / 2], axis=0)
    mean = flat.mean(axis=0)
    pos = np.arange(len(mean))
    ax = _axis(ax)
    ax.hlines(pos, intervals[0], intervals[1], color="#0072B2")
    ax.scatter(mean, pos, color="#0072B2", s=16)
    ax.set_yticks(pos, result.prior_names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Mixture proportion")
    return ax


def plot_random_effects(result, *, condition=0, level=0.95, ax=None):
    """Show posterior random-effect means and equal-tail credible intervals."""
    if not result.group_names:
        raise ValueError("the fitted model has no random effects")
    c = _column(result, condition)
    from .regression import DrawResult

    d = len(result.condition_names)
    q = result.random_design.shape[1]
    values = result.random_draws.reshape(*result.random_draws.shape[:2], -1, d)
    labels = [f"{g}:random{j + 1}" for g in result.group_names for j in range(q)]
    draws = DrawResult(values, row_names=labels, condition_names=result.condition_names)
    return plot_coefficients(draws, condition=c, level=level, ax=ax)
