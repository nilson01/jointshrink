"""Learn covariance patterns from fixed illustrative summary statistics.

The supplied values are constructed examples, not observations from a study.
"""

import numpy as np
from jointshrink import set_data, canonical_covariances, select_strong, cov_pca, cov_ed, fit


def illustrative_data():
    """Return 24 fixed rows spanning shared and condition-specific estimates."""
    effects = np.array(
        [
            [1.8, 1.6, 1.9],
            [1.3, 1.6, 1.4],
            [-1.8, -1.5, -1.7],
            [-1.2, -1.4, -1.6],
            [1.5, 1.2, 1.1],
            [-1.1, -1.3, -1.0],
            [2.0, -0.1, 0.2],
            [-1.8, 0.1, -0.2],
            [0.1, 1.8, -0.1],
            [-0.2, -1.7, 0.1],
            [0.1, -0.2, 1.9],
            [-0.1, 0.1, -1.8],
            [1.5, -1.4, 0.1],
            [-1.6, 1.5, -0.2],
            [1.2, -1.3, 0.2],
            [-1.4, 1.1, -0.1],
            [0.15, -0.2, 0.1],
            [-0.25, 0.1, -0.1],
            [0.1, 0.2, -0.2],
            [-0.1, -0.15, 0.25],
            [0.2, 0.1, -0.15],
            [-0.2, 0.15, 0.1],
            [0.3, -0.2, 0.15],
            [-0.1, -0.2, -0.15],
        ]
    )
    se = np.tile([0.35, 0.45, 0.55], (len(effects), 1))
    se[1::4] *= 1.15
    return set_data(
        effects,
        se,
        row_names=[f"j={j + 1}" for j in range(len(effects))],
        condition_names=["r=1", "r=2", "r=3"],
    )


def run_analysis():
    """Fit one explicit canonical/PCA/extreme-deconvolution workflow."""
    data = illustrative_data()
    strong = select_strong(data, z_threshold=2.5)
    pca = cov_pca(data, n_components=2, subset=strong, center=False)
    starts = {
        name: value + 0.1 * np.eye(data.n_conditions)
        for name, value in pca.items()
        if name != "PCA_total"
    }
    learned = cov_ed(data, starts, subset=strong, maxiter=300, tol=1e-7)
    patterns = {**canonical_covariances(data), **pca, **learned.covariances}
    scales = np.array([0.1, 0.25, 0.5, 1.0, 2.0])  # Standard-deviation multipliers.
    result = fit(data, patterns, scales=scales, maxiter=3000, tol=1e-7)
    return data, strong, pca, learned, result


if __name__ == "__main__":
    _, strong, _, learned, result = run_analysis()
    print(f"Pattern-learning rows: {len(strong)}; mixture-weight rows: {result.data.n_effects}")
    print(f"Covariance learning: {learned.diagnostics['reason']}")
    print(result.format_summary())
    contrast = result.contrast([[1.0, -1.0, 0.0]], names=["r=1 minus r=2"])
    print("First-row contrast posterior mean:", float(contrast.posterior_mean[0, 0]))
