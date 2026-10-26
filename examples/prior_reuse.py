"""Reuse a learned prior for new rows, including a partially observed row."""

from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from jointshrink import load_prior, save_prior, set_data
from learn_patterns import run_analysis


if __name__ == "__main__":
    _, _, _, _, fitted = run_analysis()
    new_data = set_data(
        [[1.1, 1.0, np.nan], [0.1, -0.2, 0.15]],
        [0.35, 0.45, 0.55],
        row_names=["partially observed", "small estimates"],
        condition_names=fitted.condition_names,
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "prior.json"
        save_prior(fitted.prior, path)
        restored = load_prior(path)
        posterior = restored.apply(new_data)
        print(posterior.format_summary())
        print("First-row posterior means:", posterior.posterior_mean[0])
        print("Observed coordinates:", new_data.mask[0])
