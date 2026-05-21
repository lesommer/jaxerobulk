"""Tests for COARE 3.0 and COARE 3.6 bulk algorithms."""

import jax
import jax.numpy as jnp
import pytest

from jaxerobulk.coare import (
    charn_coare3p0,
    charn_coare3p6,
    turb_coare3p0,
    turb_coare3p6,
)
from jaxerobulk.thermodynamics import q_sat
from jaxerobulk.ncar import turb_ncar


class TestCharnCoare3p0:
    def test_low_wind(self):
        result = float(charn_coare3p0(jnp.float64(5.0)))
        assert abs(result - 0.011) < 1e-10

    def test_mid_wind(self):
        result = float(charn_coare3p0(jnp.float64(14.0)))
        expected = 0.011 + (0.018 - 0.011) * (14.0 - 10.0) / 8.0
        assert abs(result - expected) < 1e-10

    def test_high_wind(self):
        result = float(charn_coare3p0(jnp.float64(25.0)))
        assert abs(result - 0.018) < 1e-10

    def test_differentiable(self):
        grad_f = jax.grad(charn_coare3p0)
        result = grad_f(jnp.float64(14.0))
        assert jnp.isfinite(result)


class TestCharnCoare3p6:
    def test_zero_wind(self):
        result = float(charn_coare3p6(jnp.float64(0.0)))
        assert result == 0.0

    def test_negative(self):
        result = float(charn_coare3p6(jnp.float64(2.0)))
        assert result == 0.0

    def test_10ms(self):
        result = float(charn_coare3p6(jnp.float64(10.0)))
        expected = max(min(0.0017 * 10.0 - 0.005, 0.028), 0.0)
        assert abs(result - expected) < 1e-10

    def test_capped(self):
        result = float(charn_coare3p6(jnp.float64(50.0)))
        assert abs(result - 0.028) < 1e-10

    def test_differentiable(self):
        grad_f = jax.grad(charn_coare3p6)
        result = grad_f(jnp.float64(10.0))
        assert jnp.isfinite(result)


def _make_inputs():
    zt = 2.0
    zu = 10.0
    sst = jnp.float64(295.15)
    t_zt = jnp.float64(291.15)
    q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
    q_zt = jnp.float64(0.01)
    U_zu = jnp.float64(5.0)
    return zt, zu, sst, t_zt, q_s, q_zt, U_zu


class TestTurbCoare3p0:
    def test_unstable(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        result = turb_coare3p0(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0
        assert float(result["Ch"]) > 0
        assert float(result["Ce"]) > 0
        assert float(result["u_star"]) > 0
        assert float(result["Ubzu"]) > 0

    def test_stable(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(288.15)
        t_zt = jnp.float64(293.15)
        q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.005)
        U_zu = jnp.float64(5.0)
        result = turb_coare3p0(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0

    def test_zt_eq_zu(self):
        zu = 10.0
        zt = 10.0
        sst = jnp.float64(295.15)
        t_zt = jnp.float64(291.15)
        q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.01)
        U_zu = jnp.float64(5.0)
        result = turb_coare3p0(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0


class TestTurbCoare3p6:
    def test_unstable(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        result = turb_coare3p6(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
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
        result = turb_coare3p6(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert float(result["Cd"]) > 0


class TestCoareCrossComparison:
    def test_coare_vs_ncar_similar_cd(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        coare = turb_coare3p0(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        ncar = turb_ncar(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        cd_coare = float(coare["Cd"])
        cd_ncar = float(ncar["Cd"])
        assert abs(cd_coare - cd_ncar) / cd_ncar < 1.0

    def test_coare3p0_vs_3p6_similar(self):
        zt, zu, sst, t_zt, q_s, q_zt, U_zu = _make_inputs()
        r30 = turb_coare3p0(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        r36 = turb_coare3p6(zt, zu, sst, t_zt, q_s, q_zt, U_zu)
        assert abs(float(r30["Cd"]) - float(r36["Cd"])) / float(r30["Cd"]) < 0.5


class TestCoareDifferentiability:
    def test_coare3p0_grad_sst(self):
        def f(sst):
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_coare3p0(2.0, 10.0, sst, jnp.float64(291.15), q_s, jnp.float64(0.01), jnp.float64(5.0))
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(295.15))
        assert jnp.isfinite(result)

    def test_coare3p6_grad_sst(self):
        def f(sst):
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_coare3p6(2.0, 10.0, sst, jnp.float64(291.15), q_s, jnp.float64(0.01), jnp.float64(5.0))
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(295.15))
        assert jnp.isfinite(result)

    def test_coare3p0_grad_wind(self):
        def f(U):
            sst = jnp.float64(295.15)
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_coare3p0(2.0, 10.0, sst, jnp.float64(291.15), q_s, jnp.float64(0.01), U)
            return result["u_star"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(5.0))
        assert jnp.isfinite(result)
