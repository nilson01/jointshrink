"""Compare a fresh optional R export with the public Python workflow."""

import json
import argparse
import numpy as np
from jointshrink import set_data, fit, MashPrior, expand_covariances, cov_pca


def compare(record):
    outputs = []
    for case in record["cases"]:
        data = set_data(
            case["effects"],
            case["standard_errors"],
            correlation=case["correlation"],
            alpha=case["alpha"],
        )
        if case["fixed"]:
            expanded = expand_covariances(case["patterns"], [1], normalize=False)
            result = MashPrior(
                list(expanded.values()), list(expanded), case["weights"], alpha=data.alpha
            ).apply(data)
        else:
            result = fit(
                data,
                case["patterns"],
                scales=case["grid"],
                normalize=False,
                null_weight=case["null_weight"],
            )
        tolerance = 2e-10 if case["fixed"] else 2e-5
        errors = {}
        for name in ("posterior_mean", "posterior_sd", "posterior_covariance", "lfsr", "lfdr"):
            expected = np.asarray(case[name])
            actual = getattr(result, name)
            np.testing.assert_allclose(
                actual, expected, atol=tolerance, rtol=tolerance, err_msg=case["name"] + ":" + name
            )
            errors[name] = float(np.max(np.abs(actual - expected)))
        if not case["fixed"]:
            np.testing.assert_allclose(result.prior.weights, case["weights"], atol=2e-5, rtol=2e-5)
        outputs.append({"case": case["name"], "absolute_errors": errors})
    # PCA uses EE working effects for this reference.
    pca = cov_pca(
        set_data(case["effects"], case["standard_errors"], correlation=case["correlation"]), 2
    )
    for actual, expected in zip(pca.values(), record["pca"]["patterns"].values()):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", help="JSON from tools/export_mashr_reference.R")
    args = parser.parse_args()
    with open(args.export, encoding="utf-8") as stream:
        record = json.load(stream)
    print(json.dumps(compare(record), indent=2))
