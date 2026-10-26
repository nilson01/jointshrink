"""Pinned installed-R fixtures; fixture generation is optional, checks are not."""

import json
from pathlib import Path
import importlib.util


def test_six_installed_mashr_cases_and_pca():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mashr_check", root / "tools/check_mashr_reference.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    outputs = mod.compare(json.loads((root / "tests/fixtures/mashr-0.2.79.json").read_text()))
    assert len(outputs) == 6
