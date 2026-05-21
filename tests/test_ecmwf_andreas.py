"""Tests for ECMWF and Andreas bulk algorithms."""

import jax
import jax.numpy as jnp
import pytest

from jaxerobulk.ecmwf import turb_ecmwf
from jaxerobulk.andreas import turb_andreas, u_star_andreas
from jaxerobulk.thermodynamics import q_sat
from jaxerobulk.ncar import turb_ncar
from jaxerobulk.coare import turb_coare3p6


def _make_inputs():
    zt = 2.0
    zu = 10.0
    sst = jnp.float64(295.15)
    t_zt = jnp.float64(291.15)
    q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
    q_zt = jnp.float64(0.01)
    U_zu = jnp.float64(5.0)
    return zt, zu, sst, t_zt, q_s, q_zt, U_zu


class TestTurbEcmwf:
    def test_unstable(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        result = turb_ecmwf(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0
        assert float(result["Ch"]) > 0
        assert float(result["Ce"]) > 0
        assert float(result["u_star"]) > 0

    def test_stable(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(288.15)
        t_zt = jnp.float64(293.15)
        q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.005)
        U_zu = jnp.float64(5.0)
        result = turb_ecmwf(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0

    def test_physical_range(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        result = turb_ecmwf(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) < 0.01
        assert float(result["Ch"]) < 0.01
        assert float(result["Ce"]) < 0.01


class TestUStarAndreas:
    def test_at_5ms(self):
        result = float(u_star_andreas(jnp.float64(5.0)))
        assert result > 0

    def test_at_10ms(self):
        result = float(u_star_andreas(jnp.float64(10.0)))
        assert result > 0.3

    def test_differentiable(self):
        grad_f = jax.grad(u_star_andreas)
        result = grad_f(jnp.float64(10.0))
        assert jnp.isfinite(result)


class TestTurbAndreas:
    def test_unstable(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(268.15)
        t_zt = jnp.float64(263.15)
        q_s = q_sat(sst, jnp.float64(101000.0), l_ice=True) * 0.98
        q_zt = jnp.float64(0.002)
        U_zu = jnp.float64(5.0)
        result = turb_andreas(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0
        assert float(result["Ch"]) > 0
        assert float(result["Ce"]) > 0

    def test_stable(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(263.15)
        t_zt = jnp.float64(268.15)
        q_s = q_sat(sst, jnp.float64(101000.0), l_ice=True) * 0.98
        q_zt = jnp.float64(0.001)
        U_zu = jnp.float64(5.0)
        result = turb_andreas(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0


class TestCrossAlgorithmComparison:
    def test_ecmwf_vs_coare3p6_similar(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        ecmwf = turb_ecmwf(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        coare = turb_coare3p6(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        cd_ecmwf = float(ecmwf["Cd"])
        cd_coare = float(coare["Cd"])
        assert abs(cd_ecmwf - cd_coare) / cd_coare < 0.5


class TestEcmwfDifferentiability:
    def test_grad_sst(self):
        def f(sst):
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_ecmwf(2.0, 10.0, sst, jnp.float64(291.15), q_s, jnp.float64(0.01), jnp.float64(5.0))
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(295.15))
        assert jnp.isfinite(result)

    def test_grad_wind(self):
        def f(U):
            sst = jnp.float64(295.15)
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_ecmwf(2.0, 10.0, sst, jnp.float64(291.15), q_s, jnp.float64(0.01), U)
            return result["u_star"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(5.0))
        assert jnp.isfinite(result)


class TestAndreasDifferentiability:
    def test_grad_sst(self):
        def f(sst):
            q_s = q_sat(sst, jnp.float64(101000.0), l_ice=True) * 0.98
            result = turb_andreas(2.0, 10.0, sst, jnp.float64(263.15), q_s, jnp.float64(0.002), jnp.float64(5.0))
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(268.15))
        assert jnp.isfinite(result)
