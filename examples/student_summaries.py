"""Apply a fixed Gaussian effect prior under native Student-t sampling errors."""

import numpy as np
from jointshrink import MashData, MashPrior, apply_student_prior


def run_analysis():
    effects = np.array([[0.2, 0.1], [1.4, 1.1], [3.5, -0.2]])
    # For Student-t errors, this is the scale matrix, not their covariance.
    data = MashData(effects, error_covariance=0.25 * np.eye(2))
    prior = MashPrior([np.zeros((2, 2)), np.eye(2)], ["null", "independent"], [0.2, 0.8])
    return apply_student_prior(data, prior, df=4)


if __name__ == "__main__":
    result = run_analysis()
    print(result.posterior_mean)
    print(result.diagnostics)
