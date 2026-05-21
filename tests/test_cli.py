"""Smoke tests for the CLI."""

import subprocess
import sys
import tempfile
import os

import pytest


def _run_cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "jaxerobulk.cli"] + list(args),
        capture_output=True, text=True, timeout=30,
    )
    return result


class TestToyCommand:
    def test_toy_default(self):
        r = _run_cli("toy")
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "coare3p6" in r.stdout
        assert "Cd=" in r.stdout

    def test_toy_ncar(self):
        r = _run_cli("toy", "--algorithm", "ncar")
        assert r.returncode == 0
        assert "ncar" in r.stdout

    def test_toy_with_skin(self):
        r = _run_cli("toy", "--skin", "--rad-sw", "400", "--rad-lw", "350")
        assert r.returncode == 0
        assert "dT_cs" in r.stdout

    def test_toy_custom_sst(self):
        r = _run_cli("toy", "--sst", "290", "--t-air", "288")
        assert r.returncode == 0
        assert "290.0K" in r.stdout


class TestCompareCommand:
    def test_compare_default(self):
        r = _run_cli("compare")
        assert r.returncode == 0
        assert "coare3p6" in r.stdout
        assert "ncar" in r.stdout
        assert "ecmwf" in r.stdout

    def test_compare_specific(self):
        r = _run_cli("compare", "-a", "ncar", "ecmwf")
        assert r.returncode == 0
        assert "ncar" in r.stdout
        assert "ecmwf" in r.stdout


class TestComputeCommand:
    def test_compute_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("sst,t_air,q,u_wind,v_wind,slp\n")
            f.write("300.0,298.0,0.018,5.0,0.0,101000.0\n")
            f.write("295.0,293.0,0.015,8.0,1.0,101000.0\n")
            csv_in = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_out = f.name

        try:
            r = _run_cli("compute", "--input-csv", csv_in, "--output-csv", csv_out)
            assert r.returncode == 0, f"stderr: {r.stderr}"
            assert "Wrote 2 rows" in r.stdout

            with open(csv_out) as f:
                content = f.read()
            assert "Cd" in content
            assert "Qlat" in content
        finally:
            os.unlink(csv_in)
            os.unlink(csv_out)

    def test_compute_missing_csv(self):
        r = _run_cli("compute")
        assert r.returncode != 0
