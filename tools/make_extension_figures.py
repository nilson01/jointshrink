"""Rebuild fixed-input likelihood and conditional-response illustrations.

Run from a source checkout with jointshrink and its plotting extra installed.
These constructed examples are not a repeated-dataset validation study.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import platform

import numpy as np
import scipy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jointshrink import MashPrior, set_data, apply_student_prior, fit_regression
from make_figures import STYLE

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN = "#0072B2", "#B34424", "#007A55"


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _panel(ax, label, title):
    ax.set_title(title, loc="left", pad=8)
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _export(fig, output, name):
    for suffix in ("pdf", "png"):
        options = {"dpi": 320, "facecolor": "white"}
        if suffix == "pdf":
            options["metadata"] = {
                "Title": name,
                "Creator": "jointshrink tools/make_extension_figures.py",
                "CreationDate": None,
                "ModDate": None,
            }
        fig.savefig(output / f"{name}.{suffix}", **options)
    plt.close(fig)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = np.column_stack((np.linspace(-3, 3, 31), np.full(31, 0.5)))
    prior_covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    prior = MashPrior([prior_covariance], ["shared"], [1.0])
    gaussian_data = set_data(inputs, error_covariance=0.5 * np.eye(2))
    student_data = set_data(inputs, error_covariance=0.25 * np.eye(2))
    gaussian = prior.apply(gaussian_data)
    student = apply_student_prior(
        student_data, prior, 4, quadrature_order=64, max_quadrature_order=256, quadrature_tol=1e-5
    )
    if not student.diagnostics["quadrature_converged"]:
        raise RuntimeError("Student-t figure quadrature did not meet the specified tolerance")
    summary_rows = []
    for j, values in enumerate(inputs):
        for model, result in (("gaussian", gaussian), ("student_t4", student)):
            for r in range(2):
                summary_rows.append(
                    dict(
                        row=j + 1,
                        condition=r + 1,
                        model=model,
                        supplied_condition1=values[0],
                        supplied_condition2=values[1],
                        posterior_mean=result.posterior_mean[j, r],
                        posterior_sd=result.posterior_sd[j, r],
                    )
                )
    _table(output / "extension-summary-values.csv", summary_rows)
    _table(
        output / "extension-summary-inputs.csv",
        [dict(row=j + 1, condition1=v[0], condition2=v[1]) for j, v in enumerate(inputs)],
    )

    x = np.array([-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0])
    X = x[:, None]
    specifications = [
        ("gaussian", [-0.8, -0.4, -0.1, 0.2, 0.3, 0.6, 0.7, 1.1], {"sigma": 0.4}),
        ("binomial", [1, 2, 2, 4, 4, 6, 6, 7], {"trials": 8}),
        ("poisson", [0, 1, 1, 2, 1, 3, 3, 4], {"exposure": 2}),
        (
            "weibull",
            [0.2, 0.6, 0.4, 0.8, 1.1, 0.5, 1.3, 1.0],
            {"event": [1, 1, 0, 1, 0, 1, 1, 0], "shape": 1.3},
        ),
    ]
    settings = dict(chains=2, draws=500, warmup=300, point_mass=False, scales=[1.0])
    fits = {}
    diagnostics = {}
    observations = []
    prediction_rows = []
    grid = np.linspace(-1, 1, 61)[:, None]
    curves = {}
    for i, (family, y, controls) in enumerate(specifications):
        fit = fit_regression(X, y, family=family, seed=123 + i, **settings, **controls)
        fits[family] = fit
        diagnostics[family] = fit.diagnostics
        for j, value in enumerate(y):
            observations.append(
                dict(
                    family=family,
                    row=j + 1,
                    x=x[j],
                    y=value,
                    event=controls.get("event", [""] * len(x))[j],
                    sigma=controls.get("sigma", ""),
                    trials=controls.get("trials", ""),
                    exposure=controls.get("exposure", ""),
                    shape=controls.get("shape", ""),
                )
            )
        if family != "weibull":
            options = {"scale": "probability"} if family == "binomial" else {}
            if family == "poisson":
                options["exposure"] = 2
            mean = fit.predict(grid, **options)[:, 0]
            interval = fit.predict_interval(grid, level=0.95, **options)[:, 0]
            curves[family] = (grid[:, 0], mean, interval)
            for value, m, bounds in zip(grid[:, 0], mean, interval):
                prediction_rows.append(
                    dict(family=family, x=value, time="", mean=m, lower=bounds[0], upper=bounds[1])
                )
        else:
            time = np.linspace(0, 1.5, 61)
            for value in [-1.0, 0.0, 1.0]:
                new = np.full((len(time), 1), value)
                options = dict(scale="survival", time=time)
                mean = fit.predict(new, **options)[:, 0]
                interval = fit.predict_interval(new, level=0.95, **options)[:, 0]
                curves[(family, value)] = (time, mean, interval)
                for t, m, bounds in zip(time, mean, interval):
                    prediction_rows.append(
                        dict(
                            family=family, x=value, time=t, mean=m, lower=bounds[0], upper=bounds[1]
                        )
                    )
    _table(output / "extension-regression-inputs.csv", observations)
    _table(output / "extension-regression-predictions.csv", prediction_rows)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
        for label, result, color, style in [
            ("Gaussian", gaussian, BLUE, "-"),
            (r"Student-$t_4$", student, ORANGE, "--"),
        ]:
            axes[0].plot(
                inputs[:, 0], result.posterior_mean[:, 0], label=label, color=color, linestyle=style
            )
            axes[1].plot(
                inputs[:, 0], result.posterior_sd[:, 0], label=label, color=color, linestyle=style
            )
        axes[0].axhline(0, color="#888888", linewidth=0.6, linestyle=":")
        axes[0].legend(frameon=False, fontsize=7)
        axes[0].set(xlabel=r"$\hat b_{j1}$", ylabel=r"$\hat{\hat b}_{j1}$")
        axes[1].set(xlabel=r"$\hat b_{j1}$", ylabel=r"$\mathrm{SD}(b_{j1}\mid\hat{\mathbf{b}}_j)$")
        _panel(axes[0], "a", "Posterior mean")
        _panel(axes[1], "b", "Posterior standard deviation")
        _export(fig, output, "extension-summary-likelihoods")

        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), layout="constrained")
        for ax, family, letter, title, ylabel in [
            (axes[0, 0], "gaussian", "a", "Gaussian", r"$\mathrm{E}(Y\mid x)$"),
            (axes[0, 1], "binomial", "b", "Binomial", r"$p(x)$"),
            (axes[1, 0], "poisson", "c", "Poisson", r"$\mathrm{E}(Y\mid x)$"),
        ]:
            predictor, mean, interval = curves[family]
            ax.fill_between(
                predictor, interval[:, 0], interval[:, 1], color=BLUE, alpha=0.16, linewidth=0
            )
            ax.plot(predictor, mean, color=BLUE)
            values = np.asarray(fits[family].likelihood.y)[:, 0]
            if family == "binomial":
                values = values / 8
                ax.set_ylim(-0.03, 1.03)
            elif family == "poisson":
                ax.set_ylim(bottom=-0.15)
            ax.scatter(
                x, values, s=18, facecolor="white", edgecolor="#252525", linewidth=0.75, zorder=4
            )
            ax.set(xlabel=r"$x$", ylabel=ylabel, xlim=(-1.07, 1.07))
            _panel(ax, letter, title)
        ax = axes[1, 1]
        for value, color, style in zip([-1.0, 0.0, 1.0], [BLUE, ORANGE, GREEN], ["-", "--", ":"]):
            time, mean, interval = curves[("weibull", value)]
            ax.fill_between(
                time, interval[:, 0], interval[:, 1], color=color, alpha=0.10, linewidth=0
            )
            ax.plot(time, mean, color=color, linestyle=style, label=f"$x={value:g}$")
        time = np.asarray(specifications[3][1])
        event = np.asarray(specifications[3][2]["event"], dtype=bool)
        ax.scatter(time[event], np.full(event.sum(), 1.035), marker="|", s=18, color="#252525")
        ax.scatter(
            time[~event],
            np.full((~event).sum(), 1.035),
            marker="+",
            s=18,
            color="#252525",
            linewidth=0.8,
        )
        ax.set(xlabel=r"$t$", ylabel=r"$S(t\mid x)$", xlim=(0, 1.5), ylim=(-0.03, 1.08))
        ax.legend(frameon=False, fontsize=6.5, loc="lower left")
        _panel(ax, "d", "Weibull proportional hazards")
        _export(fig, output, "extension-regression-responses")

    manifest = {
        "kind": "fixed constructed illustrations; not real data or a calibration study",
        "summary": {
            "rows": len(inputs),
            "conditions": 2,
            "prior_covariance": prior_covariance,
            "prior_weights": [1.0],
            "fixed_condition2_estimate": 0.5,
            "gaussian_error_covariance": 0.5 * np.eye(2),
            "student_df": 4,
            "student_error_scale": 0.25 * np.eye(2),
            "comparison": "same prior and error covariance; Student-t scale differs from covariance",
            "quadrature_options": {
                "quadrature_order": 64,
                "max_quadrature_order": 256,
                "quadrature_tol": 1e-5,
            },
            "quadrature_diagnostics": student.diagnostics,
        },
        "regression": {
            "settings": settings,
            "seeds": dict(zip(fits, range(123, 127))),
            "observations_per_family": 8,
            "predictors": 1,
            "fixed_controls": {name: controls for name, _, controls in specifications},
            "diagnostics": diagnostics,
            "uncertainty": "95% equal-tail posterior intervals for conditional means or survival probabilities; fixed nuisance parameters",
            "survival": "five events and three right-censored observations; event times shown as ticks and censoring times as plus signs along upper rug",
        },
        "captions": {
            "extension-summary-likelihoods": "Posterior summaries under Gaussian and native multivariate Student-t4 errors for 31 constructed pairs of estimates. The second supplied estimate is fixed at 0.5. Both likelihoods have error covariance 0.5 I2; the Student-t scale matrix is 0.25 I2. A single fixed Gaussian prior has variances one and correlation 0.6. Panels show the posterior mean and standard deviation in condition 1. No prior weights are estimated in this illustration.",
            "extension-regression-responses": "Conditional response summaries from four separate constructed datasets, each with eight observations and one predictor. Lines average transformed posterior draws; shaded regions are 95% equal-tail credible intervals for conditional means, binomial probability or survival, not future observations. Open circles show Gaussian outcomes, observed binomial fractions with eight trials, and Poisson counts with exposure two. Gaussian residual SD is 0.4. Weibull shape is fixed at 1.3; curves share the same time axis and use x=-1, 0 and 1. The upper rug marks five event times with ticks and three right-censoring times with plus signs. Fits use one Gaussian coefficient-prior component, intercept SD five, two chains, 300 warmup iterations and 500 retained draws per chain. These examples illustrate interfaces and do not measure calibration.",
        },
        "alt_text": {
            "extension-summary-likelihoods": "Two line plots compare posterior means and posterior standard deviations for condition 1 as its supplied estimate varies; condition 2 is held fixed. Solid blue denotes Gaussian errors and dashed orange denotes Student-t4 errors.",
            "extension-regression-responses": "Four panels show Gaussian, binomial and Poisson conditional response curves with raw observations, followed by Weibull survival curves at three predictor values. Shaded credible bands describe conditional means or survival; a time rug distinguishes events and right censoring.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "thread_settings": {
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                ROOT / "tools" / "make_figures.py",
                *sorted((ROOT / "src" / "jointshrink").glob("*.py")),
            ]
        },
        "files": {},
    }
    for path in sorted(output.glob("extension-*")):
        if path.suffix in {".csv", ".pdf", ".png"}:
            manifest["files"][path.name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (output / "extension-figure-data.json").write_text(
        json.dumps(_json(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "figures": 2,
                "student_quadrature_converged": student.diagnostics["quadrature_converged"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "figures")
    main(parser.parse_args().output)
