"""Gated external-reference test: jaxerobulk CORE/NCAR vs the eORCA1 run.

Runs validation/eorca1_bulk.py against the eORCA1 record and the CORE.v2
normal-year forcing when both are made available through the environment
(JAXEROBULK_EORCA1_DIR, JAXEROBULK_CNYF_DIR); otherwise the whole module
skips (the data is never committed to this repository). Asserts the
documented verdict bar (validation/README.md) and consistency with the
reference run (validation/reference_results.json).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "validation" / "eorca1_bulk.py"
REFERENCE = REPO / "validation" / "reference_results.json"

EORCA1_DIR = os.environ.get("JAXEROBULK_EORCA1_DIR")
CNYF_DIR = os.environ.get("JAXEROBULK_CNYF_DIR")

pytest.importorskip("xarray")
pytestmark = pytest.mark.skipif(
    not EORCA1_DIR or not CNYF_DIR,
    reason="eORCA1 record / CNYF forcing not provided "
           "(set JAXEROBULK_EORCA1_DIR and JAXEROBULK_CNYF_DIR)")


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    out_dir = tmp_path_factory.mktemp("eorca1-validation")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--eorca1-dir", EORCA1_DIR, "--cnyf-dir", CNYF_DIR,
         "--out-dir", str(out_dir)],
        capture_output=True, text=True, check=True)
    return json.loads((out_dir / "eorca1-validation.json").read_text())


def _row(payload: dict, variant: str, var: str) -> dict:
    return payload["variants"][variant][var]


MATCHED = "V3 = V2 + NEMO heat-content terms (matched)"


def test_matched_qns_verdict_bar(result: dict) -> None:
    qns = _row(result, MATCHED, 0)
    assert abs(qns["bias"]) <= 5.0, qns
    assert qns["corr"] >= 0.92, qns


def test_matched_wind_stress_correlation(result: dict) -> None:
    utau = _row(result, MATCHED, 1)
    vtau = _row(result, MATCHED, 2)
    assert utau["corr"] >= 0.9, utau
    assert vtau["corr"] >= 0.9, vtau


def test_emp_anomaly_pattern(result: dict) -> None:
    emp = _row(result, "E1 = emp anomaly-form (fwb correction proxy)", 0)
    assert emp["corr"] >= 0.5, emp


def test_consistent_with_reference_run(result: dict) -> None:
    ref = json.loads(REFERENCE.read_text())
    for variant in result["variants"]:
        for i in range(len(result["variants"][variant])):
            got = _row(result, variant, i)
            want = _row(ref, variant, i)
            assert got["n"] == want["n"], (variant, i)
            assert abs(got["bias"] - want["bias"]) <= 0.5, (variant, i, got, want)
            assert abs(got["corr"] - want["corr"]) <= 0.01, (variant, i, got, want)
