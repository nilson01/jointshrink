"""Plots of fitted quantities with no display, saving or global-style changes.

Matplotlib is imported only when a plot is requested. Each function returns its
axes so labels, layout and export remain under the caller's control.
"""

from collections.abc import Mapping
import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install jointshrink[plot] to use plotting functions") from error
    return plt


def _axis(ax=None):
    plt = _pyplot()
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    return ax


def _condition(result, condition):
    names = tuple(result.condition_names)
    if isinstance(condition, str):
        try:
            return names.index(condition)
        except ValueError as error:
            raise ValueError(f"Unknown condition: {condition}") from error
    if isinstance(condition, (bool, np.bool_)) or not isinstance(condition, (int, np.integer)):
        raise ValueError("condition must be a condition name or integer index")
    if not 0 <= condition < len(names):
        raise ValueError("condition index is out of range")
    return int(condition)


def _indices(result, indices):
    n = result.posterior_mean.shape[0]
    if indices is None:
        return np.arange(n)
    raw = np.asarray(indices)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("indices must be a nonempty vector of row indices")
    if raw.size == 0 or np.any(raw < 0) or np.any(raw >= n):
        raise ValueError("indices must contain valid row indices")
    return raw


def plot_covariances(prior_or_mapping, *, axes=None, ncols=3, condition_names=None):
    """Show covariance patterns on a common scale centered at zero.

    Returns a one-dimensional axes array. Pattern values retain their supplied
    covariance units; no per-panel normalization is performed.
    """
    if isinstance(prior_or_mapping, Mapping):
        names = list(prior_or_mapping)
        covariances = np.asarray(list(prior_or_mapping.values()), dtype=float)
    else:
        names = list(prior_or_mapping.names)
        covariances = np.asarray(prior_or_mapping.covariances, dtype=float)
    if (
        covariances.ndim != 3
        or covariances.shape[0] == 0
        or covariances.shape[1] != covariances.shape[2]
        or len(names) != covariances.shape[0]
        or not np.isfinite(covariances).all()
    ):
        raise ValueError("Supply nonempty, finite square covariance patterns")
    if isinstance(ncols, bool) or not isinstance(ncols, (int, np.integer)) or ncols < 1:
        raise ValueError("ncols must be a positive integer")
    if condition_names is not None and len(condition_names) != covariances.shape[1]:
        raise ValueError("condition_names must match the covariance dimension")
    if axes is None:
        plt = _pyplot()
        columns = min(ncols, len(names))
        _, grid = plt.subplots(
            int(np.ceil(len(names) / columns)),
            columns,
            squeeze=False,
            layout="constrained",
            figsize=(3 * columns, 2.8 * int(np.ceil(len(names) / columns))),
        )
        flat = grid.ravel()
    else:
        flat = np.asarray(axes, dtype=object).reshape(-1)
        if len(flat) < len(names):
            raise ValueError("Provide at least one axis per covariance pattern")
    limit = float(np.max(np.abs(covariances)))
    limit = limit if limit > 0 else 1.0
    for index, (name, covariance) in enumerate(zip(names, covariances)):
        ax = flat[index]
        image = ax.imshow(
            covariance, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest"
        )
        ax.set_title(str(name))
        labels = (
            list(condition_names)
            if condition_names is not None
            else np.arange(covariance.shape[0]) + 1
        )
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set(xlabel="Condition", ylabel="Condition")
        ax.figure.colorbar(image, ax=ax, label="Covariance")
    for ax in flat[len(names) :]:
        ax.set_visible(False)
    return flat[: len(names)]


def plot_weights(prior_or_result, *, ax=None):
    """Show every fitted mixture weight, including zero-weight components."""
    prior = getattr(prior_or_result, "prior", prior_or_result)
    weights = np.asarray(prior.weights, dtype=float)
    names = list(prior.names)
    if weights.ndim != 1 or len(names) != len(weights) or not np.isfinite(weights).all():
        raise ValueError("Prior weights and component names must agree")
    ax = _axis(ax)
    position = np.arange(len(weights))
    ax.barh(position, weights, color="#0072B2", edgecolor="#222222", linewidth=0.5)
    ax.set_yticks(position, names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Mixture weight", ylabel="Prior component")
    return ax


def plot_sharing(result, *, factor=0.5, threshold=0.05, by="lfsr", mode="magnitude", ax=None):
    """Show pairwise sharing fractions from result.sharing.

    Empty comparisons are gray, not zero. The result.sharing return value gives
    each comparison's numerator, denominator and precise selection definition.
    """
    summary = result.sharing(factor=factor, threshold=threshold, by=by, mode=mode)
    matrix = np.asarray(summary["matrix"], dtype=float)
    names = list(summary["condition_names"])
    ax = _axis(ax)
    from matplotlib import colormaps

    cmap = colormaps["cividis"].with_extremes(bad="#dddddd")
    image = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set(xlabel="Condition", ylabel="Condition", title=f"Effect sharing ({mode})")
    if len(names) <= 12:
        for row in range(len(names)):
            for column in range(len(names)):
                value = matrix[row, column]
                label = f"{value:.2f}" if np.isfinite(value) else "—"
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and value < 0.45 else "black",
                )
    ax.figure.colorbar(image, ax=ax, label="Sharing fraction")
    return ax


def plot_intervals(result, *, condition=0, indices=None, level=0.95, ax=None):
    """Show marginal equal-tail posterior intervals and posterior means.

    Intervals condition on the fitted prior, including estimated mixture
    weights. Rows are never silently selected by statistical significance.
    """
    column = _condition(result, condition)
    rows = _indices(result, indices)
    interval = result.interval(level=level)[rows, column, :]
    mean = result.posterior_mean[rows, column]
    ax = _axis(ax)
    position = np.arange(len(rows))
    # A mixture mean need not lie inside its equal-tail interval.
    ax.hlines(position, interval[:, 0], interval[:, 1], color="#0072B2", linewidth=1.5)
    ax.scatter(mean, position, marker="o", color="#0072B2", s=18, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_yticks(position, [result.row_names[index] for index in rows])
    ax.invert_yaxis()
    ax.set(
        xlabel="Effect",
        ylabel="Row",
        title=(
            f"{result.condition_names[column]}: {100 * level:g}% posterior intervals (fitted prior)"
        ),
    )
    return ax


def plot_diagnostics(result_or_covfit, *, ax=None):
    """Show the recorded optimization objective, without smoothing."""
    diagnostics = result_or_covfit.diagnostics
    key = next(
        (
            name
            for name in (
                "objective_history",
                "objective_trace",
                "log_likelihood_trace",
                "log_likelihood",
                "trace",
            )
            if name in diagnostics
        ),
        None,
    )
    if key is None:
        raise ValueError("No objective trace is available in these diagnostics")
    values = np.asarray(diagnostics[key], dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("The objective trace must be a nonempty finite vector")
    ax = _axis(ax)
    ax.plot(np.arange(len(values)), values, color="#0072B2", marker="o", markersize=3, linewidth=1)
    ax.set(
        xlabel="Recorded iteration",
        ylabel="Log likelihood"
        if key in ("log_likelihood_trace", "log_likelihood")
        else "Optimization objective",
    )
    return ax


def plot_effects(result, condition=0, *, ax=None):
    """Compare observed estimates with posterior means in original units.

    Missing observations are omitted; posterior predictions for those entries
    are available from result.posterior_mean. No uncertainty band is implied.
    """
    column = _condition(result, condition)
    if not hasattr(result, "data"):
        raise ValueError("Observed-versus-posterior plots require fitted data")
    observed = result.data.mask[:, column]
    if not np.any(observed):
        raise ValueError("This condition has no observed estimates")
    x = result.data.effects[observed, column]
    y = result.posterior_mean[observed, column]
    ax = _axis(ax)
    ax.scatter(x, y, color="#0072B2", marker="o", s=18, alpha=0.8)
    lower, upper = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower, upper = lower - padding, upper + padding
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#555555", linewidth=0.8)
    ax.set(
        xlabel="Observed effect estimate",
        ylabel="Posterior mean",
        title=str(result.condition_names[column]),
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    ax.set_aspect("equal", adjustable="box")
    return ax
