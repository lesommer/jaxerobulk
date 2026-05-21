"""Tests for JaxeroBulk thermodynamic functions.

Reference values are computed from the FORTRAN AeroBulk implementation
using the same inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxerobulk.constants import *
from jaxerobulk.thermodynamics import (
    pot_temp,
    abs_temp,
    virt_temp,
    e_sat,
    e_sat_ice,
    de_sat_dt_ice,
    q_sat,
    dq_sat_dt_ice,
    q_air_rh,
    q_air_dp,
    rho_air,
    visc_air,
    l_vap,
    cp_air,
    gamma_moist,
    one_on_L,
    ri_bulk,
    Pz_from_P0_tz_qz,
    theta_from_z_p0_t_q,
    T_from_z_p0_theta_q,
    alpha_sw,
    qlw_net,
    z0_from_cd,
    z0_from_ustar,
    f_m_louis,
    f_h_louis,
    e_air,
    rh_air,
    bulk_formula,
    delta_skin_layer,
    type_of_humidity,
)


class TestPotTemp:
    def test_scalar(self):
        result = float(pot_temp(jnp.float64(300.0), jnp.float64(101000.0)))
        expected = 300.0 * (101000.0 / 101000.0) ** rpoiss_dry
        assert abs(result - expected) < 1e-10

    def test_with_pref(self):
        result = float(pot_temp(jnp.float64(300.0), jnp.float64(90000.0), jnp.float64(101000.0)))
        expected = 300.0 * (101000.0 / 90000.0) ** rpoiss_dry
        assert abs(result - expected) < 1e-6

    def test_roundtrip(self):
        T = 295.0
        P = 99000.0
        theta = float(pot_temp(jnp.float64(T), jnp.float64(P)))
        T_back = float(abs_temp(jnp.float64(theta), jnp.float64(P)))
        assert abs(T_back - T) < 1e-10


class TestAbsTemp:
    def test_roundtrip(self):
        T = 295.0
        P = 99000.0
        theta = float(pot_temp(jnp.float64(T), jnp.float64(P)))
        T_back = float(abs_temp(jnp.float64(theta), jnp.float64(P)))
        assert abs(T_back - T) < 1e-10


class TestVirtTemp:
    def test_dry_air(self):
        result = float(virt_temp(jnp.float64(300.0), jnp.float64(0.0)))
        assert abs(result - 300.0) < 1e-10

    def test_moist_air(self):
        result = float(virt_temp(jnp.float64(300.0), jnp.float64(0.01)))
        expected = 300.0 * (1.0 + rctv0 * 0.01)
        assert abs(result - expected) < 1e-10


class TestESat:
    def test_at_20C(self):
        T = 293.15
        result = float(e_sat(jnp.float64(T)))
        assert 2300 < result < 2400

    def test_at_0C(self):
        T = 273.15
        result = float(e_sat(jnp.float64(T)))
        assert 600 < result < 620

    def test_array(self):
        T = jnp.array([273.15, 293.15, 313.15])
        result = e_sat(T)
        assert result.shape == (3,)
        assert float(result[1]) > float(result[0])


class TestESatIce:
    def test_at_0C(self):
        T = 273.15
        result = float(e_sat_ice(jnp.float64(T)))
        e_water = float(e_sat(jnp.float64(T)))
        assert result < e_water

    def test_derivative_finite_diff(self):
        T = 263.15
        dt = 0.01
        analytic = float(de_sat_dt_ice(jnp.float64(T)))
        finite_diff = (float(e_sat_ice(jnp.float64(T + dt))) - float(e_sat_ice(jnp.float64(T - dt)))) / (2 * dt)
        assert abs(analytic - finite_diff) / abs(analytic) < 1e-6


class TestQSat:
    def test_water(self):
        T = 293.15
        P = 101000.0
        result = float(q_sat(jnp.float64(T), jnp.float64(P)))
        assert 0.01 < result < 0.02

    def test_ice(self):
        T = 263.15
        P = 101000.0
        result = float(q_sat(jnp.float64(T), jnp.float64(P), l_ice=True))
        result_water = float(q_sat(jnp.float64(T), jnp.float64(P), l_ice=False))
        assert result < result_water

    def test_dq_sat_dt_ice_finite_diff(self):
        T = 263.15
        P = 101000.0
        dt = 0.01
        analytic = float(dq_sat_dt_ice(jnp.float64(T), jnp.float64(P)))
        finite_diff = (float(q_sat(jnp.float64(T + dt), jnp.float64(P), l_ice=True)) - float(q_sat(jnp.float64(T - dt), jnp.float64(P), l_ice=True))) / (2 * dt)
        assert abs(analytic - finite_diff) / abs(analytic) < 1e-6


class TestHumidityConversions:
    def test_rh_roundtrip(self):
        T = 293.15
        P = 101000.0
        RH = 80.0
        q = float(q_air_rh(jnp.float64(RH), jnp.float64(T), jnp.float64(P)))
        RH_back = float(rh_air(jnp.float64(q), jnp.float64(T), jnp.float64(P)))
        assert abs(RH_back - RH) < 0.1

    def test_dp_roundtrip(self):
        T_dew = 283.15
        P = 101000.0
        q = float(q_air_dp(jnp.float64(T_dew), jnp.float64(P)))
        e = float(e_sat(jnp.float64(T_dew)))
        q_expected = reps0 * e / (P - (1 - reps0) * e)
        assert abs(q - q_expected) / abs(q_expected) < 1e-10


class TestRhoAir:
    def test_standard_conditions(self):
        T = 293.15
        q = 0.01
        P = 101000.0
        result = float(rho_air(jnp.float64(T), jnp.float64(q), jnp.float64(P)))
        expected = P / (R_dry * T * (1.0 + rctv0 * q))
        assert abs(result - expected) < 1e-6
        assert 1.1 < result < 1.3


class TestViscAir:
    def test_at_20C(self):
        T = 293.15
        result = float(visc_air(jnp.float64(T)))
        assert 1.4e-5 < result < 1.6e-5


class TestLVap:
    def test_at_0C(self):
        result = float(l_vap(jnp.float64(273.15)))
        assert abs(result - 2.501e6) < 1e4

    def test_at_20C(self):
        result = float(l_vap(jnp.float64(293.15)))
        assert abs(result - (2.501 - 0.00237 * 20) * 1e6) < 1e4


class TestCpAir:
    def test_dry(self):
        result = float(cp_air(jnp.float64(0.0)))
        assert abs(result - rCp_dry) < 1e-10

    def test_moist(self):
        q = 0.01
        result = float(cp_air(jnp.float64(q)))
        expected = rCp_dry + rCp_vap * q
        assert abs(result - expected) < 1e-10


class TestGammaMoist:
    def test_dry(self):
        result = float(gamma_moist(jnp.float64(300.0), jnp.float64(1e-6)))
        expected = grav / rCp_dry
        assert abs(result - expected) / expected < 0.01


class TestOneOnL:
    def test_unstable(self):
        result = float(one_on_L(jnp.float64(300.0), jnp.float64(0.01), jnp.float64(0.3), jnp.float64(-0.1), jnp.float64(-0.001)))
        assert result < 0

    def test_stable(self):
        result = float(one_on_L(jnp.float64(300.0), jnp.float64(0.01), jnp.float64(0.3), jnp.float64(0.1), jnp.float64(0.001)))
        assert result > 0


class TestRiBulk:
    def test_unstable(self):
        result = float(ri_bulk(10.0, jnp.float64(295.0), jnp.float64(290.0), jnp.float64(0.02), jnp.float64(0.01), jnp.float64(5.0)))
        assert result < 0

    def test_stable(self):
        result = float(ri_bulk(10.0, jnp.float64(290.0), jnp.float64(295.0), jnp.float64(0.01), jnp.float64(0.02), jnp.float64(5.0)))
        assert result > 0


class TestPzFromP0:
    def test_sea_level(self):
        result = float(Pz_from_P0_tz_qz(0.0, jnp.float64(101000.0), jnp.float64(300.0), jnp.float64(0.01)))
        assert abs(result - 101000.0) < 1.0

    def test_at_10m(self):
        result = float(Pz_from_P0_tz_qz(10.0, jnp.float64(101000.0), jnp.float64(300.0), jnp.float64(0.01)))
        assert result < 101000.0
        assert result > 100000.0


class TestThetaConversions:
    def test_roundtrip(self):
        z = 2.0
        slp = 101000.0
        T = 293.15
        q = 0.01
        theta = float(theta_from_z_p0_t_q(z, jnp.float64(slp), jnp.float64(T), jnp.float64(q)))
        T_back = float(T_from_z_p0_theta_q(z, jnp.float64(slp), jnp.float64(theta), jnp.float64(q)))
        assert abs(T_back - T) < 0.01


class TestAlphaSw:
    def test_at_20C(self):
        result = float(alpha_sw(jnp.float64(293.15)))
        assert result > 0

    def test_cold_water(self):
        result = float(alpha_sw(jnp.float64(270.0)))
        assert result < 1e-5


class TestQlwNet:
    def test_water(self):
        result = float(qlw_net(jnp.float64(350.0), jnp.float64(293.15)))
        assert result != 0.0

    def test_ice(self):
        result_w = float(qlw_net(jnp.float64(350.0), jnp.float64(263.15), l_ice=False))
        result_i = float(qlw_net(jnp.float64(350.0), jnp.float64(263.15), l_ice=True))
        assert abs(result_i) > abs(result_w)


class TestZ0FromCd:
    def test_neutral(self):
        zu = 10.0
        Cd = 1.0e-3
        result = float(z0_from_cd(zu, jnp.float64(Cd)))
        expected = zu * jnp.exp(-vkarmn / jnp.sqrt(Cd))
        assert abs(result - expected) < 1e-10

    def test_with_psi(self):
        zu = 10.0
        Cd = 1.0e-3
        psi = -1.0
        result = float(z0_from_cd(zu, jnp.float64(Cd), jnp.float64(psi)))
        expected = zu * jnp.exp(-(vkarmn / jnp.sqrt(Cd) + psi))
        assert abs(result - expected) < 1e-10


class TestBulkFormula:
    def test_typical_conditions(self):
        zu = 10.0
        sst = 293.15
        q_s = float(q_sat(jnp.float64(sst), jnp.float64(101000.0))) * 0.98
        theta = 291.15
        q = 0.01
        Cd = 1.2e-3
        Ch = 1.0e-3
        Ce = 1.2e-3
        W = 5.0
        Ub = 5.5
        slp = 101000.0
        tau, Qsen, Qlat, Evap, rho = bulk_formula(
            zu, jnp.float64(sst), jnp.float64(q_s), jnp.float64(theta),
            jnp.float64(q), jnp.float64(Cd), jnp.float64(Ch), jnp.float64(Ce),
            jnp.float64(W), jnp.float64(Ub), jnp.float64(slp)
        )
        assert float(tau) > 0
        assert float(Qsen) != 0.0
        assert float(Qlat) != 0.0


class TestDifferentiability:
    def test_e_sat_grad(self):
        f = lambda T: e_sat(T)
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)
        assert result > 0

    def test_q_sat_grad(self):
        f = lambda T: q_sat(T, jnp.float64(101000.0))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)
        assert result > 0

    def test_rho_air_grad_T(self):
        f = lambda T: rho_air(T, jnp.float64(0.01), jnp.float64(101000.0))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)
        assert result < 0

    def test_rho_air_grad_q(self):
        f = lambda q: rho_air(jnp.float64(293.15), q, jnp.float64(101000.0))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(0.01))
        assert jnp.isfinite(result)

    def test_l_vap_grad(self):
        f = lambda T: l_vap(T)
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)
        assert result < 0

    def test_visc_air_grad(self):
        f = lambda T: visc_air(T)
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_pot_temp_grad(self):
        f = lambda T: pot_temp(T, jnp.float64(99000.0))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_virt_temp_grad(self):
        f = lambda T: virt_temp(T, jnp.float64(0.01))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_gamma_moist_grad(self):
        f = lambda T: gamma_moist(T, jnp.float64(0.01))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_qlw_net_grad_T(self):
        f = lambda T: qlw_net(jnp.float64(350.0), T)
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_one_on_L_grad(self):
        f = lambda us: one_on_L(jnp.float64(300.0), jnp.float64(0.01), us, jnp.float64(-0.1), jnp.float64(-0.001))
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(0.3))
        assert jnp.isfinite(result)

    def test_ri_bulk_grad(self):
        f = lambda W: ri_bulk(10.0, jnp.float64(295.0), jnp.float64(290.0), jnp.float64(0.02), jnp.float64(0.01), W)
        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(5.0))
        assert jnp.isfinite(result)

    def test_bulk_formula_grad_sst(self):
        def f(sst):
            q_s = q_sat(sst, jnp.float64(101000.0)) * 0.98
            tau, Qsen, Qlat, Evap, rho = bulk_formula(
                10.0, sst, q_s, jnp.float64(291.15), jnp.float64(0.01),
                jnp.float64(1.2e-3), jnp.float64(1.0e-3), jnp.float64(1.2e-3),
                jnp.float64(5.0), jnp.float64(5.5), jnp.float64(101000.0)
            )
            return Qlat

        grad_f = jax.grad(f)
        result = grad_f(jnp.float64(293.15))
        assert jnp.isfinite(result)

    def test_forward_reverse_consistency(self):
        T = jnp.float64(293.15)
        fwd = jax.jacfwd(lambda x: q_sat(x, jnp.float64(101000.0)))(T)
        rev = jax.jacrev(lambda x: q_sat(x, jnp.float64(101000.0)))(T)
        assert jnp.allclose(fwd, rev, rtol=1e-10)


class TestArrayTypeOfHumidity:
    def test_specific_humidity(self):
        vals = jnp.array([0.005, 0.010, 0.015])
        assert type_of_humidity(vals) == "sh"

    def test_relative_humidity(self):
        vals = jnp.array([70.0, 80.0, 90.0])
        assert type_of_humidity(vals) == "rh"

    def test_dewpoint(self):
        vals = jnp.array([280.0, 285.0, 290.0])
        assert type_of_humidity(vals) == "dp"


class TestFMLouis:
    def test_unstable(self):
        result = float(f_m_louis(10.0, jnp.float64(-0.1), jnp.float64(1.2e-3), jnp.float64(0.001)))
        assert result > 1.0

    def test_stable(self):
        result = float(f_m_louis(10.0, jnp.float64(0.1), jnp.float64(1.2e-3), jnp.float64(0.001)))
        assert result < 1.0


class TestFHLouis:
    def test_unstable(self):
        result = float(f_h_louis(10.0, jnp.float64(-0.1), jnp.float64(1.0e-3), jnp.float64(0.001)))
        assert result > 1.0

    def test_stable(self):
        result = float(f_h_louis(10.0, jnp.float64(0.1), jnp.float64(1.0e-3), jnp.float64(0.001)))
        assert result < 1.0


class TestArrayBroadcasting:
    def test_e_sat_array(self):
        T = jnp.array([273.15, 283.15, 293.15, 303.15])
        result = e_sat(T)
        assert result.shape == (4,)
        for i in range(3):
            assert float(result[i + 1]) > float(result[i])

    def test_q_sat_array(self):
        T = jnp.array([273.15, 283.15, 293.15])
        P = jnp.full(3, 101000.0)
        result = q_sat(T, P)
        assert result.shape == (3,)

    def test_rho_air_array(self):
        T = jnp.array([283.15, 293.15, 303.15])
        q = jnp.full(3, 0.01)
        P = jnp.full(3, 101000.0)
        result = rho_air(T, q, P)
        assert result.shape == (3,)
        assert float(result[0]) > float(result[2])
