"""Tests for stability functions, first guess, and NCAR algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxerobulk.stability import (
    psi_m_coare,
    psi_h_coare,
    psi_m_ncar,
    psi_h_ncar,
    psi_m_ecmwf,
    psi_h_ecmwf,
    psi_m_andreas,
    psi_h_andreas,
)
from jaxerobulk.first_guess import first_guess_coare
from jaxerobulk.ncar import cd_n10_ncar, ch_n10_ncar, ce_n10_ncar, turb_ncar
from jaxerobulk.constants import vkarmn
from jaxerobulk.thermodynamics import q_sat


class TestPsiMCoare:
    def test_near_neutral(self):
        result = float(psi_m_coare(jnp.float64(0.0)))
        assert abs(result) < 0.01

    def test_unstable_negative(self):
        result = float(psi_m_coare(jnp.float64(-1.0)))
        assert result > 0

    def test_stable_positive(self):
        result = float(psi_m_coare(jnp.float64(1.0)))
        assert result < 0

    def test_array(self):
        zeta = jnp.linspace(-5, 5, 21)
        result = psi_m_coare(zeta)
        assert result.shape == (21,)
        assert float(result[0]) > 0
        assert float(result[-1]) < 0


class TestPsiHCoare:
    def test_near_neutral(self):
        result = float(psi_h_coare(jnp.float64(0.0)))
        assert abs(result) < 0.01

    def test_unstable_negative(self):
        result = float(psi_h_coare(jnp.float64(-1.0)))
        assert result > 0

    def test_stable_positive(self):
        result = float(psi_h_coare(jnp.float64(1.0)))
        assert result < 0


class TestPsiMNcar:
    def test_neutral(self):
        result = float(psi_m_ncar(jnp.float64(0.0)))
        assert abs(result) < 1e-10

    def test_unstable(self):
        result = float(psi_m_ncar(jnp.float64(-1.0)))
        assert result > 0

    def test_stable(self):
        result = float(psi_m_ncar(jnp.float64(1.0)))
        assert result == pytest.approx(-5.0, abs=1e-10)


class TestPsiHNcar:
    def test_neutral(self):
        result = float(psi_h_ncar(jnp.float64(0.0)))
        assert abs(result) < 1e-10

    def test_stable(self):
        result = float(psi_h_ncar(jnp.float64(1.0)))
        assert result == pytest.approx(-5.0, abs=1e-10)


class TestPsiMEcmwf:
    def test_near_neutral(self):
        result = float(psi_m_ecmwf(jnp.float64(0.0)))
        assert abs(result) < 0.01

    def test_unstable(self):
        result = float(psi_m_ecmwf(jnp.float64(-1.0)))
        assert result > 0

    def test_stable(self):
        result = float(psi_m_ecmwf(jnp.float64(1.0)))
        assert result < 0


class TestPsiHEcmwf:
    def test_near_neutral(self):
        result = float(psi_h_ecmwf(jnp.float64(0.0)))
        assert abs(result) < 0.01


class TestPsiMAndreas:
    def test_neutral(self):
        result = float(psi_m_andreas(jnp.float64(0.0)))
        assert abs(result) < 1e-10

    def test_stable_bounded(self):
        result = float(psi_m_andreas(jnp.float64(5.0)))
        assert jnp.isfinite(result)
        assert result < 0

    def test_unstable(self):
        result = float(psi_m_andreas(jnp.float64(-1.0)))
        assert result > 0


class TestPsiHAndreas:
    def test_neutral(self):
        result = float(psi_h_andreas(jnp.float64(0.0)))
        assert abs(result) < 1e-10


class TestStabilityDifferentiability:
    @pytest.mark.parametrize(
        "func",
        [psi_m_coare, psi_h_coare, psi_m_ncar, psi_h_ncar, psi_m_ecmwf, psi_h_ecmwf, psi_m_andreas, psi_h_andreas],
    )
    def test_grad(self, func):
        grad_f = jax.grad(func)
        result = grad_f(jnp.float64(-1.0))
        assert jnp.isfinite(result)

    @pytest.mark.parametrize(
        "func",
        [psi_m_coare, psi_h_coare, psi_m_ncar, psi_h_ncar],
    )
    def test_forward_reverse_consistency(self, func):
        x = jnp.float64(-0.5)
        fwd = jax.jacfwd(func)(x)
        rev = jax.jacrev(func)(x)
        assert jnp.allclose(fwd, rev, rtol=1e-10)


class TestPsiCrossAlgorithmComparison:
    def test_all_small_at_neutral(self):
        zeta = jnp.float64(0.0)
        for func in [psi_m_coare, psi_m_ncar, psi_m_ecmwf, psi_m_andreas]:
            assert abs(float(func(zeta))) < 0.01

    def test_ncar_coare_similar_unstable(self):
        zeta = jnp.float64(-0.5)
        pm_ncar = float(psi_m_ncar(zeta))
        pm_coare = float(psi_m_coare(zeta))
        assert abs(pm_ncar - pm_coare) / abs(pm_coare) < 0.5


class TestCdN10Ncar:
    def test_at_5ms(self):
        result = float(cd_n10_ncar(jnp.float64(5.0)))
        assert 1.0e-3 < result < 2.0e-3

    def test_at_10ms(self):
        result = float(cd_n10_ncar(jnp.float64(10.0)))
        assert 1.0e-3 < result < 2.0e-3

    def test_at_20ms(self):
        result = float(cd_n10_ncar(jnp.float64(20.0)))
        assert 1.5e-3 < result < 3.0e-3

    def test_above_33ms(self):
        result = float(cd_n10_ncar(jnp.float64(40.0)))
        assert abs(result - 2.34e-3) < 1e-6

    def test_physical_range(self):
        U = jnp.linspace(1, 30, 50)
        cd = cd_n10_ncar(U)
        assert jnp.all(cd > 0)
        assert jnp.all(cd < 0.01)

    def test_differentiable(self):
        grad_f = jax.grad(cd_n10_ncar)
        result = grad_f(jnp.float64(10.0))
        assert jnp.isfinite(result)


class TestChCeN10Ncar:
    def test_ch_stable(self):
        result = float(ch_n10_ncar(jnp.float64(0.04), jnp.float64(1.0)))
        assert result > 0

    def test_ch_unstable(self):
        result = float(ch_n10_ncar(jnp.float64(0.04), jnp.float64(0.0)))
        assert result > 0

    def test_ce(self):
        result = float(ce_n10_ncar(jnp.float64(0.04)))
        assert result > 0

    def test_ch_stable_lt_unstable(self):
        ch_stab = float(ch_n10_ncar(jnp.float64(0.04), jnp.float64(1.0)))
        ch_unstab = float(ch_n10_ncar(jnp.float64(0.04), jnp.float64(0.0)))
        assert ch_unstab > ch_stab


class TestTurbNcar:
    def test_unstable_conditions(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(295.15)
        t_zt = jnp.float64(291.15)
        ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.01)
        U_zu = jnp.float64(5.0)

        result = turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, U_zu)
        assert float(result["Cd"]) > 0
        assert float(result["Ch"]) > 0
        assert float(result["Ce"]) > 0
        assert float(result["u_star"]) > 0

    def test_stable_conditions(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(290.15)
        t_zt = jnp.float64(295.15)
        ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.005)
        U_zu = jnp.float64(5.0)

        result = turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, U_zu)
        assert float(result["Cd"]) > 0
        assert float(result["Ch"]) > 0
        assert float(result["Ce"]) > 0

    def test_cd_positive_and_reasonable(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(295.15)
        t_zt = jnp.float64(291.15)
        ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.01)

        for W in [3.0, 5.0, 10.0, 15.0]:
            result = turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, jnp.float64(W))
            assert float(result["Cd"]) > 0
            assert float(result["Cd"]) < 0.01

    def test_zt_eq_zu(self):
        zu = 10.0
        zt = 10.0
        sst = jnp.float64(295.15)
        t_zt = jnp.float64(291.15)
        ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.01)
        U_zu = jnp.float64(5.0)

        result = turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, U_zu)
        assert float(result["Cd"]) > 0


class TestTurbNcarDifferentiability:
    def test_grad_sst(self):
        def f(sst):
            ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_ncar(2.0, 10.0, sst, jnp.float64(291.15), ssq, jnp.float64(0.01), jnp.float64(5.0))
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(295.15))
        assert jnp.isfinite(result)

    def test_grad_wind(self):
        def f(U):
            sst = jnp.float64(295.15)
            ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
            result = turb_ncar(2.0, 10.0, sst, jnp.float64(291.15), ssq, jnp.float64(0.01), U)
            return result["Cd"]

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(5.0))
        assert jnp.isfinite(result)


class TestFirstGuessCoare:
    def test_basic(self):
        zt = 2.0
        zu = 10.0
        sst = jnp.float64(295.15)
        t_zt = jnp.float64(291.15)
        ssq = q_sat(sst, jnp.float64(101000.0)) * 0.98
        q_zt = jnp.float64(0.01)
        U_zu = jnp.float64(5.0)
        pcharn = jnp.float64(0.014)

        pus, pts, pqs, t_zu, q_zu, Ubzu, pz0 = first_guess_coare(
            zt, zu, sst, t_zt, pssq=ssq, q_zt=q_zt, U_zu=U_zu, pcharn=pcharn
        )

        assert float(pus) > 0
        assert float(Ubzu) > 0
        assert float(pz0) > 0
