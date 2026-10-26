"""Persistence and plot behavior on fixed, independently inspectable inputs."""

import copy
import json

import numpy as np
import pytest

from jointshrink import (
    MashPrior,
    MashResult,
    set_data,
    save_data,
    load_data,
    save_prior,
    load_prior,
    save_result,
    load_result,
    plot_covariances,
    plot_weights,
    plot_sharing,
    plot_intervals,
    plot_diagnostics,
    plot_effects,
)


@pytest.fixture
def data():
    return set_data(
        [[1.0, 2.0], [np.nan, -1.0], [0.0, 0.2]],
        [[0.5, 0.8], [0.4, 0.9], [1.0, 1.2]],
        correlation=[[1.0, 0.3], [0.3, 1.0]],
        row_names=["x", "y", "z"],
        condition_names=["A", "B"],
    )


def prior(alpha=0):
    return MashPrior(
        [np.zeros((2, 2)), np.eye(2), np.ones((2, 2))],
        ["null", "independent", "shared"],
        [0.2, 0.3, 0.5],
        alpha=alpha,
        metadata={"setting": "fixed", "nested": {"values": [None, True, 1]}},
    )


@pytest.fixture
def result(data):
    return MashResult(data, prior(), diagnostics={"objective_history": [-12.0, -10.0, -9.9]})


@pytest.mark.parametrize("alpha", [0, 1])
def test_data_correlated_missing_roundtrip(tmp_path, data, alpha):
    original = set_data(
        data.effects,
        error_covariance=data.error_covariance,
        alpha=alpha,
        row_names=data.row_names,
        condition_names=data.condition_names,
    )
    path = tmp_path / "d.json"
    save_data(original, path)
    restored = load_data(path)
    assert json.loads(path.read_text())["payload"]["effects"][1][0] is None
    assert restored.alpha == original.alpha
    assert restored.row_names == original.row_names
    assert restored.condition_names == original.condition_names
    for name in [
        "effects",
        "standard_errors",
        "error_covariance",
        "mask",
        "scale",
        "working_effects",
        "working_covariance",
    ]:
        np.testing.assert_allclose(
            getattr(original, name), getattr(restored, name), rtol=1e-14, atol=0, equal_nan=True
        )
    assert not restored.effects.flags.writeable


@pytest.mark.parametrize("alpha", [0, 1])
def test_prior_result_roundtrip_without_optimization(tmp_path, data, alpha, monkeypatch):
    import jointshrink.model as model
    import jointshrink._numeric as numeric

    p = prior(alpha)
    d = set_data(
        data.effects,
        error_covariance=data.error_covariance,
        alpha=alpha,
        row_names=data.row_names,
        condition_names=data.condition_names,
    )
    r = MashResult(d, p, diagnostics={"fixed_prior": True, "objective_history": [-2.0, -1.0]})
    ppath, rpath = tmp_path / "prior.json", tmp_path / "result.json"
    save_prior(p, ppath)
    save_result(r, rpath)

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading must not optimize")

    monkeypatch.setattr(model, "weight_fit", forbidden)
    monkeypatch.setattr(numeric, "weight_fit", forbidden)
    monkeypatch.setattr(model, "fit", forbidden)
    restored_p = load_prior(ppath)
    restored_r = load_result(rpath)
    assert restored_p.names == p.names
    assert dict(restored_p.metadata) == dict(p.metadata)
    assert restored_r.diagnostics == r.diagnostics
    for name in [
        "component_weights",
        "posterior_mean",
        "posterior_sd",
        "posterior_covariance",
        "lfdr",
        "lfsr",
        "log_likelihood",
        "log_bayes_factor",
    ]:
        np.testing.assert_allclose(getattr(restored_r, name), getattr(r, name), rtol=1e-13)
    np.testing.assert_allclose(restored_p.score_samples(d), p.score_samples(d))


@pytest.mark.parametrize(
    "save,load,kind",
    [
        (save_data, load_data, "data"),
        (save_prior, load_prior, "prior"),
        (save_result, load_result, "result"),
    ],
)
def test_overwrite_guard_and_kind(tmp_path, data, result, save, load, kind):
    objects = {"data": data, "prior": result.prior, "result": result}
    path = tmp_path / "state.json"
    save(objects[kind], path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save(objects[kind], path)
    assert path.read_bytes() == original
    save(objects[kind], path, overwrite=True)
    load(path)
    wrong = load_prior if kind != "prior" else load_data
    with pytest.raises(ValueError):
        wrong(path)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema":"jointshrink","schema":"jointshrink","version":1,"kind":"data","payload":{}}',
        '{"schema":"jointshrink","version":1,"kind":"data","payload":NaN}',
        '{"schema":"jointshrink","version":1,"kind":"data","payload":Infinity}',
        '{"schema":"jointshrink","version":1,"kind":"data","payload":1e400}',
        "[]",
        "null",
        '{"schema":"other","version":1,"kind":"data","payload":{}}',
        '{"schema":"jointshrink","version":true,"kind":"data","payload":{}}',
        '{"schema":"jointshrink","version":2,"kind":"data","payload":{}}',
        '{"schema":"jointshrink","version":1,"kind":"data","payload":{},"exec":"x"}',
    ],
)
def test_hostile_envelope_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_data(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("standard_errors", [[None, 1], [1, 1], [1, 1]]),
        ("error_covariance", [[[True, 0], [0, 1]]] * 3),
        ("effects", [[True, 2], [1, -1], [0, 0.2]]),
        ("effects", [[None, None], [1, -1], [0, 0.2]]),
        ("effects", [[1, 2, 3], [1, -1], [0, 0.2]]),
        ("alpha", True),
        ("alpha", "0"),
        ("row_names", "xyz"),
        ("standard_errors", [[5, 0.8], [0.4, 0.9], [1, 1.2]]),
        ("error_covariance", [[[1, 2], [2, 1]]] * 3),
    ],
)
def test_invalid_data_payload_rejected(tmp_path, data, field, value):
    path = tmp_path / "data.json"
    save_data(data, path)
    record = json.loads(path.read_text())
    record["payload"][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_data(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("covariances", [[[None, 0], [0, 1]]]),
        ("weights", [True, 0, 0]),
        ("weights", [1, 1, 1]),
        ("alpha", False),
        ("metadata", []),
        ("names", ["a", "a", "c"]),
        ("covariances", [[[1, 2], [2, 1]]] * 3),
    ],
)
def test_invalid_prior_payload_rejected(tmp_path, field, value):
    path = tmp_path / "prior.json"
    save_prior(prior(), path)
    record = json.loads(path.read_text())
    record["payload"][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_prior(path)


def test_unknown_nested_fields_and_diagnostics_rejected(tmp_path, data, result):
    path = tmp_path / "r.json"
    save_result(result, path)
    record = json.loads(path.read_text())
    extra = copy.deepcopy(record)
    extra["payload"]["prior"]["code"] = "print(1)"
    path.write_text(json.dumps(extra))
    with pytest.raises(ValueError):
        load_result(path)
    record["payload"]["diagnostics"] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_result(path)


def test_non_json_metadata_rejected(tmp_path):
    for value in [float("inf"), object(), {1: "nonstring-key"}]:
        p = MashPrior([np.eye(2)], ["I"], [1], metadata={"x": value})
        with pytest.raises(ValueError):
            save_prior(p, tmp_path / "state.json")


@pytest.fixture
def plt():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


def test_all_plot_paths_and_styles(plt, result):
    import matplotlib as mpl

    before = dict(mpl.rcParams)
    for function in [plot_weights, plot_sharing, plot_intervals, plot_diagnostics, plot_effects]:
        _, supplied = plt.subplots()
        returned = function(result, ax=supplied)
        assert returned is supplied
        supplied.figure.canvas.draw()
        assert supplied.get_xlabel()
        assert supplied.get_ylabel()
    axes = plot_covariances(result.prior, condition_names=result.condition_names)
    assert axes.shape == (3,)
    assert [ax.get_title() for ax in axes] == list(result.prior.names)
    limits = [ax.images[0].get_clim() for ax in axes]
    assert all(limits[0] == limit for limit in limits)
    assert limits[0][0] == -limits[0][1]
    assert [x.get_text() for x in axes[0].get_xticklabels()] == ["A", "B"]
    for ax in axes:
        ax.figure.canvas.draw()
    assert dict(mpl.rcParams) == before


def test_plot_rows_missing_observations_and_names(plt, result):
    ax = plot_effects(result, condition="A")
    assert len(ax.collections[0].get_offsets()) == 2
    assert ax.get_title() == "A"
    ax = plot_intervals(result, condition="B", indices=[2, 0], level=0.8)
    assert [label.get_text() for label in ax.get_yticklabels()] == ["z", "x"]
    assert "80%" in ax.get_title()
    assert "posterior" in ax.get_title()
    np.testing.assert_allclose(
        ax.collections[1].get_offsets()[:, 0], result.posterior_mean[[2, 0], 1]
    )


def test_empty_sharing_is_masked_not_zero(plt, data):
    null = MashPrior([np.zeros((2, 2))], ["null"], [1])
    r = null.apply(data)
    ax = plot_sharing(r)
    assert np.ma.getmaskarray(ax.images[0].get_array()).all()
    assert all(text.get_text() == "—" for text in ax.texts)
    ax.figure.canvas.draw()


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (plot_intervals, {"indices": []}),
        (plot_intervals, {"indices": [-1]}),
        (plot_intervals, {"indices": [0.5]}),
        (plot_intervals, {"condition": True}),
        (plot_intervals, {"condition": "unknown"}),
        (plot_intervals, {"condition": 2}),
        (plot_intervals, {"level": 1.1}),
        (plot_effects, {"condition": -1}),
    ],
)
def test_plot_invalid_selection(plt, result, function, kwargs):
    with pytest.raises(ValueError):
        function(result, **kwargs)


def test_unavailable_plot_inputs_rejected(plt):
    data = set_data([[1, np.nan], [2, np.nan]], condition_names=["A", "B"])
    result = prior().apply(data)
    with pytest.raises(ValueError, match="no observed"):
        plot_effects(result, condition="B")
    with pytest.raises(ValueError, match="No objective trace"):
        plot_diagnostics(result)
    with pytest.raises(ValueError):
        plot_covariances({})
    with pytest.raises(ValueError):
        plot_covariances(prior(), condition_names=["one"])
    with pytest.raises(ValueError):
        plot_covariances(prior(), axes=plt.subplots()[1])


def test_invalid_save_preserves_existing_file(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("original")
    invalid = MashPrior([np.eye(2)], ["I"], [1], metadata={"x": float("nan")})
    with pytest.raises(ValueError):
        save_prior(invalid, path, overwrite=True)
    assert path.read_text() == "original"


def test_import_does_not_require_plotting_dependency():
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys,jointshrink; assert 'matplotlib' not in sys.modules; "
            "assert 'pandas' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_supplied_covariance_axes_and_plain_mapping(plt):
    _, axes = plt.subplots(1, 3)
    returned = plot_covariances({"I": np.eye(2), "shared": np.ones((2, 2))}, axes=axes)
    assert len(returned) == 2
    assert returned[0] is axes[0]
    assert not axes[2].get_visible()
    assert [label.get_text() for label in axes[0].get_xticklabels()] == ["1", "2"]


@pytest.mark.parametrize(
    "key,label",
    [("objective_history", "Optimization objective"), ("log_likelihood", "Log likelihood")],
)
def test_diagnostic_trace_values_and_label(plt, result, key, label):
    from types import SimpleNamespace

    trace = [-3.0, -2.0, -1.5]
    ax = plot_diagnostics(SimpleNamespace(diagnostics={key: trace}))
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), trace)
    assert ax.get_ylabel() == label
