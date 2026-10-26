"""Fit two raw count outcomes with a shared coefficient prior."""

import numpy as np
from jointshrink import fit_poisson_regression


def run_analysis():
    X = np.array([[-1.0], [-0.7], [-0.4], [-0.1], [0.2], [0.5], [0.8], [1.0]])
    y = np.array([[0, 1], [1, 0], [0, 2], [1, 1], [1, 3], [2, 2], [2, 4], [3, 3]])
    return fit_poisson_regression(
        X,
        y,
        coefficient_names=["x"],
        outcome_names=["A", "B"],
        scales=(0.5, 1.0),
        draws=600,
        warmup=300,
        chains=2,
        seed=42,
    )


if __name__ == "__main__":
    result = run_analysis()
    print(result.format_summary())
    print("Expected counts at x=0:", result.predict([[0.0]]))
    print("Split R-hat:", result.diagnostics["split_rhat"])
