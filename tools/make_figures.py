"""Rebuild two documentation figures and their fixed illustrative source tables."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import os
import scipy
import runpy

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
}
COLORS = ["#0072B2", "#D55E00", "#009E73"]
CAPTIONS = {
    "covariance-learning": (
        "Covariance learning and mixture fitting for 24 constructed rows in three conditions. "
        "(a–c) An equal-effects pattern, the second principal-component pattern and its "
        "extreme-deconvolution refinement; each displayed matrix is divided by its largest "
        "diagonal entry. (d) Fitted mixture weights summed over covariance sources, including "
        "the global null. (e) Observed-data log likelihood during covariance learning on "
        "16 strong rows, stopping at the 300-iteration limit. (f) The mixture objective fitted on all 24 rows. The input table "
        "is illustrative; these traces do not measure statistical calibration."
    ),
    "posterior-summary": (
        "Posterior summaries for the same fixed input. (a) Supplied estimates "
        r"$\hat b_{jr}$ and (b) posterior means $\hat{\hat b}_{jr}$ use a common color scale. "
        "(c) Pairwise magnitude sharing among rows with lfsr < 0.05 in either condition; "
        "shared posterior means have equal signs and magnitudes within a factor of two. "
        "(d) Posterior means and 95% equal-tail intervals in condition 1 for six specified "
        r"rows. (e) The corresponding summaries of $b_{j1}-b_{j2}$. "
        "(f) Supplied estimates versus posterior means in all three conditions. "
        "Intervals condition on the fitted covariance patterns and mixture weights."
    ),
}


def table(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel(ax, label, title):
    ax.set_title(title, loc="left", pad=9)
    ax.text(
        -0.16,
        1.09,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        ha="left",
        va="bottom",
    )


def heatmap(ax, values, *, limit, label, row_ticks=None):
    image = ax.imshow(
        values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest", aspect="auto"
    )
    ax.set_xticks(np.arange(3), ["1", "2", "3"])
    if row_ticks is None:
        ax.set_yticks(np.arange(3), ["1", "2", "3"])
        ax.set_ylabel("Condition")
    else:
        ax.set_yticks(row_ticks, [str(j + 1) for j in row_ticks])
        ax.set_ylabel(r"Row $j$")
    ax.set_xlabel(r"Condition $r$")
    ax.figure.colorbar(image, ax=ax, label=label, fraction=0.07, pad=0.04)
    return image


def export(fig, output, name):
    for extension in ["pdf", "png"]:
        kwargs = {"dpi": 320, "facecolor": "white"}
        if extension == "pdf":
            kwargs["metadata"] = {
                "Title": name,
                "Creator": "jointshrink tools/make_figures.py",
                "CreationDate": None,
                "ModDate": None,
            }
        fig.savefig(output / f"{name}.{extension}", **kwargs)
    plt.close(fig)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    analysis = runpy.run_path(str(ROOT / "examples" / "learn_patterns.py"))
    data, strong, pca, learned, result = analysis["run_analysis"]()
    contrast = result.contrast([[1.0, -1.0, 0.0]], names=["r=1 minus r=2"])
    sharing = result.sharing()
    selected = np.array([0, 6, 8, 10, 12, 16])
    interval = result.interval(0.95)
    contrast_interval = contrast.interval(0.95)
    input_rows, summary_rows = [], []
    for j in range(data.n_effects):
        for r in range(data.n_conditions):
            input_rows.append(
                dict(
                    j=j + 1,
                    r=r + 1,
                    bhat=data.effects[j, r],
                    shat=data.standard_errors[j, r],
                    strong=int(j in strong),
                )
            )
            summary_rows.append(
                dict(
                    j=j + 1,
                    r=r + 1,
                    posterior_mean=result.posterior_mean[j, r],
                    posterior_sd=result.posterior_sd[j, r],
                    lower=interval[j, r, 0],
                    upper=interval[j, r, 1],
                    lfsr=result.lfsr[j, r],
                )
            )
    table(output / "illustrative-input.csv", ["j", "r", "bhat", "shat", "strong"], input_rows)
    table(output / "posterior-summaries.csv", list(summary_rows[0]), summary_rows)
    table(
        output / "contrast-summaries.csv",
        ["j", "posterior_mean", "lower", "upper"],
        [
            dict(
                j=j + 1,
                posterior_mean=contrast.posterior_mean[j, 0],
                lower=contrast_interval[j, 0, 0],
                upper=contrast_interval[j, 0, 1],
            )
            for j in range(data.n_effects)
        ],
    )
    patterns = [np.ones((3, 3)), pca["PCA_2"], learned.covariances["ED_PCA_2"]]
    patterns = [p / np.max(np.diag(p)) for p in patterns]
    weights = dict(null=0.0, canonical=0.0, PCA=0.0, ED=0.0)
    for name, weight in zip(result.prior.names, result.prior.weights):
        source = (
            "null"
            if name == "null"
            else "ED"
            if name.startswith("ED_")
            else "PCA"
            if name.startswith("PCA_")
            else "canonical"
        )
        weights[source] += float(weight)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6), layout="constrained")
        for ax, matrix, letter, title in zip(
            axes[0],
            patterns,
            "abc",
            ["Equal-effects pattern", "Principal component 2", "Error-aware refinement"],
        ):
            heatmap(ax, matrix, limit=1.0, label=r"$U_k\,/\,\max_r U_{k,rr}$")
            panel(ax, letter, title)
        ax = axes[1, 0]
        ax.bar(
            np.arange(4), list(weights.values()), color="#0072B2", edgecolor="black", linewidth=0.4
        )
        ax.set_xticks(np.arange(4), ["Null", "Canonical", "PCA", "ED"], rotation=25, ha="right")
        ax.set(ylim=(0, 1), ylabel=r"Summed $\hat\pi_{k,l}$", xlabel="Covariance source")
        panel(ax, "d", "Mixture weights")
        ax = axes[1, 1]
        ax.plot(learned.diagnostics["log_likelihood"], color="#0072B2")
        ax.set(xlabel="EM iteration", ylabel="Log likelihood")
        panel(ax, "e", "Covariance learning")
        ax = axes[1, 2]
        ax.plot(result.diagnostics["objective_history"], color="#0072B2")
        ax.set(xlabel="Recorded iteration", ylabel="Log likelihood")
        panel(ax, "f", "Mixture fitting")
        export(fig, output, "covariance-learning")

        fig, axes = plt.subplots(2, 3, figsize=(7.1, 5.1), layout="constrained")
        limit = float(max(np.max(np.abs(data.effects)), np.max(np.abs(result.posterior_mean))))
        for ax, values, letter, title, color_label in [
            (axes[0, 0], data.effects, "a", "Supplied estimates", r"$\hat b_{jr}$"),
            (axes[0, 1], result.posterior_mean, "b", "Posterior means", r"$\hat{\hat b}_{jr}$"),
        ]:
            heatmap(ax, values, limit=limit, label=color_label, row_ticks=np.arange(0, 24, 4))
            panel(ax, letter, title)
        ax = axes[0, 2]
        cmap = plt.get_cmap("cividis").with_extremes(bad="#dddddd")
        image = ax.imshow(
            np.ma.masked_invalid(sharing["matrix"]), vmin=0, vmax=1, cmap=cmap, aspect="auto"
        )
        for r in range(3):
            for s in range(3):
                value = sharing["matrix"][r, s]
                ax.text(
                    s,
                    r,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.45 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(3), ["1", "2", "3"])
        ax.set_yticks(range(3), ["1", "2", "3"])
        ax.set(xlabel="Condition", ylabel="Condition")
        fig.colorbar(image, ax=ax, fraction=0.07, pad=0.04, label="Sharing fraction")
        panel(ax, "c", "Magnitude sharing")
        bounds = np.concatenate(
            [
                interval[selected, 0].ravel(),
                contrast_interval[selected, 0].ravel(),
                result.posterior_mean[selected, 0],
                contrast.posterior_mean[selected, 0],
            ]
        )
        low, high = float(bounds.min()), float(bounds.max())
        padding = 0.07 * (high - low)
        for ax, means, intervals, letter, title, xlabel in [
            (
                axes[1, 0],
                result.posterior_mean[:, 0],
                interval[:, 0],
                "d",
                "Condition 1",
                r"$b_{j1}$",
            ),
            (
                axes[1, 1],
                contrast.posterior_mean[:, 0],
                contrast_interval[:, 0],
                "e",
                "Condition contrast",
                r"$b_{j1}-b_{j2}$",
            ),
        ]:
            ax.hlines(
                np.arange(6),
                intervals[selected, 0],
                intervals[selected, 1],
                color="#0072B2",
                linewidth=1.5,
            )
            ax.scatter(means[selected], np.arange(6), color="#0072B2", marker="o", s=13, zorder=3)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
            ax.set_yticks(range(6), [str(j + 1) for j in selected])
            ax.invert_yaxis()
            ax.set(xlabel=xlabel, ylabel=r"Row $j$", xlim=(low - padding, high + padding))
            panel(ax, letter, title)
        ax = axes[1, 2]
        for r, (color, marker) in enumerate(zip(COLORS, ["o", "s", "^"])):
            ax.scatter(
                data.effects[:, r],
                result.posterior_mean[:, r],
                s=13,
                color=color,
                marker=marker,
                label=f"$r={r + 1}$",
                alpha=0.85,
            )
        ax.plot([-2.1, 2.1], [-2.1, 2.1], color="#777777", linestyle="--", linewidth=0.7)
        ax.set(
            xlabel=r"$\hat b_{jr}$",
            ylabel=r"$\hat{\hat b}_{jr}$",
            xlim=(-2.1, 2.1),
            ylim=(-2.1, 2.1),
        )
        ax.legend(loc="upper left", frameon=False, fontsize=6, handletextpad=0.2, borderaxespad=0.2)
        panel(ax, "f", "Adaptive shrinkage")
        export(fig, output, "posterior-summary")
    information = {
        "kind": "fixed illustrative input; not real data or a repeated-dataset study",
        "source": "examples/learn_patterns.py",
        "J": 24,
        "R": 3,
        "alpha": 0,
        "error_correlation": "identity",
        "strong_rows": strong.tolist(),
        "strong_z_threshold": 2.5,
        "pca_center": False,
        "pca_components": 2,
        "ed_start_diagonal_addition": 0.1,
        "ed_maxiter": 300,
        "ed_tol": 1e-7,
        "scales_sd": [0.1, 0.25, 0.5, 1.0, 2.0],
        "omega_variance": [0.01, 0.0625, 0.25, 1.0, 4.0],
        "covariance_diagnostics": learned.diagnostics,
        "mixture_diagnostics": result.diagnostics,
        "fit_options": {
            "maxiter": 3000,
            "tol": 1e-7,
            "normalize": True,
            "point_mass": True,
            "null_weight": 1,
        },
        "expanded_prior": {
            "names": list(result.prior.names),
            "weights": result.prior.weights.tolist(),
            "covariances": result.prior.covariances.tolist(),
        },
        "mixture_weights_by_source": weights,
        "displayed_pattern_names": ["equal_effects", "PCA_2", "ED_PCA_2"],
        "displayed_covariances": [p.tolist() for p in patterns],
        "sharing": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sharing.items()},
        "selected_interval_rows_zero_based": selected.tolist(),
        "interval_level": 0.95,
        "uncertainty": "equal-tail posterior intervals conditional on fitted patterns and weights",
        "reference": "Urbut et al. (2019), doi:10.1038/s41588-018-0268-8, equations 2–4,8; Figures1–2",
        "design_transfer": "Matched heatmap scales, panel lettering and sparse axes; no reference data or artwork reused",
        "captions": CAPTIONS,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
            "randomness": "none",
            "thread_settings": {
                name: os.environ.get(name)
                for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "examples" / "learn_patterns.py",
                *sorted((ROOT / "src" / "jointshrink").glob("*.py")),
            ]
        },
        "alt_text": {
            "covariance-learning": "Six panels show three normalized covariance patterns, mixture weights by source, and two optimization traces for fixed illustrative estimates.",
            "posterior-summary": "Paired heatmaps compare supplied estimates and posterior means; further panels show pairwise sharing, posterior intervals, contrasts and a shrinkage scatterplot.",
        },
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.suffix in (".csv", ".pdf", ".png"):
            information["files"][path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (output / "figure-data.json").write_text(
        json.dumps(information, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "files": list(information["files"]),
                "covariance_stop": learned.diagnostics["reason"],
                "mixture_converged": result.diagnostics["converged"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "figures")
    main(parser.parse_args().output)
