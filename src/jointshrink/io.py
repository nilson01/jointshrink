"""Versioned JSON persistence for data, priors and fitted results.

Files contain numeric data and descriptive metadata, never executable objects.
Loading a result reapplies its saved prior; it does not refit mixture weights.
"""

from pathlib import Path
from collections.abc import Mapping
import json
import math
import numpy as np

_SCHEMA = "jointshrink"
_VERSION = 1


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("Metadata must contain only JSON values with string keys")


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Invalid or unexpected JSON fields")


def _numeric_tree(value):
    if isinstance(value, list):
        for item in value:
            _numeric_tree(item)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Numeric arrays must contain numbers, without nulls or booleans")


def _alpha(value):
    if type(value) is not int or value not in (0, 1):
        raise ValueError("alpha must be the integer 0 or 1")


def _prior_payload(prior):
    return _json_value(
        dict(
            covariances=prior.covariances,
            names=prior.names,
            weights=prior.weights,
            alpha=prior.alpha,
            metadata=prior.metadata,
        )
    )


def _data_payload(data):
    covariance = data.error_covariance
    effects = [
        [
            None if not data.mask[i, j] else float(data.effects[i, j])
            for j in range(data.n_conditions)
        ]
        for i in range(data.n_effects)
    ]
    return _json_value(
        dict(
            effects=effects,
            standard_errors=data.standard_errors,
            error_covariance=covariance,
            alpha=data.alpha,
            row_names=data.row_names,
            condition_names=data.condition_names,
        )
    )


def _read(path, kind):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        record = json.load(stream, parse_constant=reject_constant, object_pairs_hook=unique_fields)
    _keys(record, ("schema", "version", "kind", "payload"))
    if (
        record["schema"] != _SCHEMA
        or type(record["version"]) is not int
        or record["version"] != _VERSION
        or (kind is not None and record["kind"] != kind)
    ):
        raise ValueError("Unsupported file schema, version or object kind")
    _json_value(record)
    return (record["kind"], record["payload"]) if kind is None else record["payload"]


def _write(path, kind, payload, overwrite):
    record = dict(schema=_SCHEMA, version=_VERSION, kind=kind, payload=payload)
    content = json.dumps(_json_value(record), allow_nan=False, indent=2) + "\n"
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(content)


def _restore_prior(payload):
    from .model import MashPrior

    _keys(payload, ("covariances", "names", "weights", "alpha", "metadata"))
    _numeric_tree(payload["covariances"])
    _numeric_tree(payload["weights"])
    _alpha(payload["alpha"])
    if not isinstance(payload["metadata"], dict) or not isinstance(payload["names"], list):
        raise ValueError("Prior metadata must be an object and names must be a list")
    return MashPrior(
        payload["covariances"],
        payload["names"],
        payload["weights"],
        alpha=payload["alpha"],
        metadata=payload["metadata"],
    )


def _restore_data(payload):
    from .data import set_data

    _keys(
        payload,
        ("effects", "standard_errors", "error_covariance", "alpha", "row_names", "condition_names"),
    )
    _numeric_tree(payload["standard_errors"])
    _numeric_tree(payload["error_covariance"])
    _alpha(payload["alpha"])
    if not isinstance(payload["row_names"], list) or not isinstance(
        payload["condition_names"], list
    ):
        raise ValueError("Row and condition names must be lists")
    effects = payload["effects"]
    if not isinstance(effects, list) or not all(isinstance(row, list) for row in effects):
        raise ValueError("Effects must be a rectangular array")
    for row in effects:
        if any(
            value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)))
            for value in row
        ):
            raise ValueError("Effects must contain numbers or missing-value nulls")
    data = set_data(
        [[np.nan if value is None else value for value in row] for row in effects],
        error_covariance=payload["error_covariance"],
        alpha=payload["alpha"],
        row_names=payload["row_names"],
        condition_names=payload["condition_names"],
    )
    saved_se = np.asarray(payload["standard_errors"], dtype=float)
    if saved_se.shape != data.standard_errors.shape or not np.allclose(
        saved_se, data.standard_errors, rtol=1e-12, atol=0
    ):
        raise ValueError("Saved standard errors disagree with error covariance")
    return data


def save_prior(prior, path, *, overwrite=False):
    """Save a fitted prior; refuse an existing path unless overwrite=True."""
    _write(path, "prior", _prior_payload(prior), overwrite)


def load_prior(path):
    """Load a prior through its validated public constructor."""
    return _restore_prior(_read(path, "prior"))


def save_data(data, path, *, overwrite=False):
    """Save observations, labels and error covariance; missing effects use null."""
    _write(path, "data", _data_payload(data), overwrite)


def load_data(path):
    """Load data through set_data, retaining its observed-value mask."""
    return _restore_data(_read(path, "data"))


def save_result(result, path, *, overwrite=False):
    """Save Gaussian or native-t summary results with their correct likelihood."""
    from .student import StudentTResult
    from .model import MashResult

    if isinstance(result, StudentTResult):
        return result.save(path, overwrite=overwrite)
    if not isinstance(result, MashResult):
        raise TypeError("use save_regression_result for raw-outcome regression results")
    _write(
        path,
        "result",
        dict(
            data=_data_payload(result.data),
            prior=_prior_payload(result.prior),
            diagnostics=_json_value(result.diagnostics),
        ),
        overwrite,
    )


def load_result(path):
    """Recompute posterior summaries from a saved prior without optimization."""
    kind, payload = _read(path, None)
    if kind == "student_result":
        from .student import load_student_result

        return load_student_result(path)
    if kind != "result":
        raise ValueError("file is not a supported summary result")
    _keys(payload, ("data", "prior", "diagnostics"))
    if not isinstance(payload["diagnostics"], dict):
        raise ValueError("Result diagnostics must be a mapping")
    from .model import MashResult

    return MashResult(
        _restore_data(payload["data"]),
        _restore_prior(payload["prior"]),
        diagnostics=payload["diagnostics"],
    )
