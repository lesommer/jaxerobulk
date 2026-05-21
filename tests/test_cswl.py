"""Tests for cool-skin and warm-layer parameterizations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


class TestCSCoare:
    """Tests for CS_COARE (cool-skin, COARE version)."""

    def test_cs_coare_negative_qnsol(self):
        """Cool-skin dT should be negative when ocean loses heat."""
        from jaxerobulk.skin_coare import cs_coare
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(200.0)
        pQnsol = jnp.float64(-100.0)
        pustar = jnp.float64(0.3)
        pQlat = jnp.float64(-100.0)
        dT = cs_coare(pQsw, pQnsol, pustar, sst, pQlat)
        assert dT < 0, "Cool-skin dT should be negative when Qnsol < 0"

    def test_cs_coare_positive_qnsol(self):
        """Cool-skin dT can be positive when ocean gains heat (rare)."""
        from jaxerobulk.skin_coare import cs_coare
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(800.0)
        pQnsol = jnp.float64(50.0)
        pustar = jnp.float64(0.05)
        pQlat = jnp.float64(-5.0)
        dT = cs_coare(pQsw, pQnsol, pustar, sst, pQlat)
        assert dT > 0, "Cool-skin dT should be positive when Qnsol > 0 with strong solar"

    def test_cs_coare_magnitude(self):
        """Cool-skin dT magnitude should be < 1 K for typical conditions."""
        from jaxerobulk.skin_coare import cs_coare
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(200.0)
        pQnsol = jnp.float64(-100.0)
        pustar = jnp.float64(0.3)
        pQlat = jnp.float64(-100.0)
        dT = cs_coare(pQsw, pQnsol, pustar, sst, pQlat)
        assert jnp.abs(dT) < 1.0, "Cool-skin dT should be < 1 K"

    def test_cs_coare_stronger_wind_smaller_dt(self):
        """Stronger wind should reduce cool-skin thickness (more mixing)."""
        from jaxerobulk.skin_coare import cs_coare
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(200.0)
        pQnsol = jnp.float64(-100.0)
        pQlat = jnp.float64(-100.0)
        dT_weak = cs_coare(pQsw, pQnsol, jnp.float64(0.1), sst, pQlat)
        dT_strong = cs_coare(pQsw, pQnsol, jnp.float64(0.5), sst, pQlat)
        assert jnp.abs(dT_strong) < jnp.abs(dT_weak), "Stronger wind => smaller |dT_cs|"

    def test_cs_coare_array(self):
        """Test with array inputs."""
        from jaxerobulk.skin_coare import cs_coare
        sst = jnp.array([295.0, 300.0, 305.0])
        pQsw = jnp.array([100.0, 200.0, 300.0])
        pQnsol = jnp.array([-80.0, -100.0, -120.0])
        pustar = jnp.array([0.2, 0.3, 0.4])
        pQlat = jnp.array([-80.0, -100.0, -120.0])
        dT = cs_coare(pQsw, pQnsol, pustar, sst, pQlat)
        assert dT.shape == (3,)
        assert jnp.all(dT < 0)

    def test_cs_coare_differentiable(self):
        """Test that cs_coare is differentiable."""
        from jaxerobulk.skin_coare import cs_coare
        def f(sst):
            return cs_coare(jnp.float64(200.0), jnp.float64(-100.0), jnp.float64(0.3), sst, jnp.float64(-100.0))
        grad_fn = jax.grad(f)
        g = grad_fn(jnp.float64(300.0))
        assert jnp.isfinite(g)


class TestCSEcmwf:
    """Tests for CS_ECMWF (cool-skin, ECMWF version)."""

    def test_cs_ecmwf_negative_qnsol(self):
        """Cool-skin dT should be negative when ocean loses heat."""
        from jaxerobulk.skin_ecmwf import cs_ecmwf
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(200.0)
        pQnsol = jnp.float64(-100.0)
        pustar = jnp.float64(0.3)
        dT = cs_ecmwf(pQsw, pQnsol, pustar, sst)
        assert dT < 0

    def test_cs_ecmwf_vs_coare(self):
        """ECMWF cool-skin should be similar but not identical to COARE.

        Key difference: ECMWF uses 0.065 solar absorption coefficient
        vs COARE's 0.137.
        """
        from jaxerobulk.skin_coare import cs_coare
        from jaxerobulk.skin_ecmwf import cs_ecmwf
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(200.0)
        pQnsol = jnp.float64(-100.0)
        pustar = jnp.float64(0.3)
        dT_coare = cs_coare(pQsw, pQnsol, pustar, sst, jnp.float64(-100.0))
        dT_ecmwf = cs_ecmwf(pQsw, pQnsol, pustar, sst)
        assert jnp.abs(dT_coare - dT_ecmwf) < 0.5, "CS difference between COARE/ECMWF should be moderate"

    def test_cs_ecmwf_differentiable(self):
        """Test that cs_ecmwf is differentiable."""
        from jaxerobulk.skin_ecmwf import cs_ecmwf
        def f(sst):
            return cs_ecmwf(jnp.float64(200.0), jnp.float64(-100.0), jnp.float64(0.3), sst)
        grad_fn = jax.grad(f)
        g = grad_fn(jnp.float64(300.0))
        assert jnp.isfinite(g)


class TestWLEcmwf:
    """Tests for WL_ECMWF (warm-layer, Zeng & Beljaars 2005)."""

    def test_wl_ecmwf_warming(self):
        """Warm-layer dT should be positive with net positive heat input."""
        from jaxerobulk.skin_ecmwf import wl_ecmwf
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(800.0)
        pQnsol = jnp.float64(50.0)
        pustar = jnp.float64(0.1)
        dT_wl = jnp.float64(0.0)
        Hz_wl = jnp.float64(3.0)
        rdt = 3600.0
        gdept_1d = 1.0
        dT_new, Hz_new = wl_ecmwf(pQsw, pQnsol, pustar, sst, dT_wl, Hz_wl, rdt, gdept_1d)
        assert dT_new >= 0, "Warm-layer dT should be non-negative"

    def test_wl_ecmwf_zero_solar(self):
        """With zero solar and negative Qns, warm-layer should not develop."""
        from jaxerobulk.skin_ecmwf import wl_ecmwf
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(0.0)
        pQnsol = jnp.float64(-100.0)
        pustar = jnp.float64(0.3)
        dT_wl = jnp.float64(0.0)
        Hz_wl = jnp.float64(3.0)
        rdt = 3600.0
        gdept_1d = 1.0
        dT_new, Hz_new = wl_ecmwf(pQsw, pQnsol, pustar, sst, dT_wl, Hz_wl, rdt, gdept_1d)
        assert dT_new == 0.0, "Warm-layer should stay zero with no heating"

    def test_wl_ecmwf_depth_constant(self):
        """ECMWF warm-layer depth should remain constant (rd0=3m)."""
        from jaxerobulk.skin_ecmwf import wl_ecmwf
        sst = jnp.float64(300.0)
        pQsw = jnp.float64(500.0)
        pQnsol = jnp.float64(50.0)
        pustar = jnp.float64(0.1)
        dT_wl = jnp.float64(0.5)
        Hz_wl = jnp.float64(3.0)
        rdt = 3600.0
        gdept_1d = 1.0
        dT_new, Hz_new = wl_ecmwf(pQsw, pQnsol, pustar, sst, dT_wl, Hz_wl, rdt, gdept_1d)
        assert Hz_new == 3.0, "ECMWF warm-layer depth is constant"

    def test_wl_ecmwf_differentiable(self):
        """Test that wl_ecmwf is differentiable."""
        from jaxerobulk.skin_ecmwf import wl_ecmwf
        def f(sst):
            dT, _ = wl_ecmwf(jnp.float64(500.0), jnp.float64(50.0), jnp.float64(0.1), sst, jnp.float64(0.0), jnp.float64(3.0), 3600.0, 1.0)
            return dT
        grad_fn = jax.grad(f)
        g = grad_fn(jnp.float64(300.0))
        assert jnp.isfinite(g)


class TestPhiEcmwf:
    """Tests for the PHI stability function used in ECMWF warm-layer."""

    def test_phi_stable(self):
        """PHI should be > 1 for stable conditions (zeta > 0)."""
        from jaxerobulk.skin_ecmwf import _phi_ecmwf
        zeta = jnp.float64(1.0)
        phi = _phi_ecmwf(zeta)
        assert phi > 1.0

    def test_phi_unstable(self):
        """PHI should be < 1 for unstable conditions (zeta < 0)."""
        from jaxerobulk.skin_ecmwf import _phi_ecmwf
        zeta = jnp.float64(-1.0)
        phi = _phi_ecmwf(zeta)
        assert phi < 1.0

    def test_phi_neutral(self):
        """PHI should be ~1 for neutral conditions (zeta ~ 0)."""
        from jaxerobulk.skin_ecmwf import _phi_ecmwf
        zeta = jnp.float64(0.0)
        phi = _phi_ecmwf(zeta)
        assert jnp.abs(phi - 1.0) < 0.01


class TestCSWLIntegration:
    """Tests for CSWL integration with bulk algorithms."""

    def test_coare3p6_with_skin(self):
        """Test COARE 3.6 with cool-skin/warm-layer enabled."""
        from jaxerobulk.coare import turb_coare3p6
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_s = jnp.float64(0.022)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        pQsw = jnp.float64(200.0)
        prad_lw = jnp.float64(350.0)
        pslp = jnp.float64(101000.0)
        result = turb_coare3p6(
            2.0, 10.0, sst, t_zt, q_s, q_zt, U_zu,
            l_use_cs=True, l_use_wl=True,
            pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
        )
        assert "dT_cs" in result
        assert "dT_wl" in result
        assert "T_s" in result
        assert jnp.isfinite(result["Cd"])
        assert jnp.isfinite(result["dT_cs"])
        assert jnp.isfinite(result["dT_wl"])

    def test_ecmwf_with_skin(self):
        """Test ECMWF with cool-skin/warm-layer enabled."""
        from jaxerobulk.ecmwf import turb_ecmwf
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_s = jnp.float64(0.022)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        pQsw = jnp.float64(200.0)
        prad_lw = jnp.float64(350.0)
        pslp = jnp.float64(101000.0)
        result = turb_ecmwf(
            2.0, 10.0, sst, t_zt, q_s, q_zt, U_zu,
            l_use_cs=True, l_use_wl=True,
            pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
        )
        assert "dT_cs" in result
        assert "dT_wl" in result
        assert "T_s" in result
        assert jnp.isfinite(result["Cd"])
        assert jnp.isfinite(result["dT_cs"])
        assert jnp.isfinite(result["dT_wl"])

    def test_api_with_skin(self):
        """Test the high-level API with l_use_skin."""
        from jaxerobulk.api import aerobulk_model
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        V_zu = jnp.float64(0.0)
        slp = jnp.float64(101000.0)
        rad_sw = jnp.float64(200.0)
        rad_lw = jnp.float64(350.0)
        result = aerobulk_model(
            2.0, 10.0, sst, t_zt, q_zt, U_zu, V_zu, slp,
            "coare3p6", l_use_skin=True, rad_sw=rad_sw, rad_lw=rad_lw,
        )
        assert "dT_cs" in result
        assert "dT_wl" in result
        assert "T_s" in result
        assert jnp.isfinite(result["Qlat"])
        assert jnp.isfinite(result["Qsen"])

    def test_api_without_skin(self):
        """Test the high-level API without skin (backward compatibility)."""
        from jaxerobulk.api import aerobulk_model
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        V_zu = jnp.float64(0.0)
        slp = jnp.float64(101000.0)
        result = aerobulk_model(
            2.0, 10.0, sst, t_zt, q_zt, U_zu, V_zu, slp, "coare3p6"
        )
        assert "Cd" in result
        assert "dT_cs" not in result
        assert jnp.isfinite(result["Qlat"])

    def test_coare3p0_with_skin(self):
        """Test COARE 3.0 with cool-skin/warm-layer enabled."""
        from jaxerobulk.coare import turb_coare3p0
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_s = jnp.float64(0.022)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        pQsw = jnp.float64(200.0)
        prad_lw = jnp.float64(350.0)
        pslp = jnp.float64(101000.0)
        result = turb_coare3p0(
            2.0, 10.0, sst, t_zt, q_s, q_zt, U_zu,
            l_use_cs=True, l_use_wl=True,
            pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
        )
        assert "dT_cs" in result
        assert "dT_wl" in result
        assert jnp.isfinite(result["Cd"])

    def test_wl_state_persistence(self):
        """Test warm-layer state can be passed and updated."""
        from jaxerobulk.coare import turb_coare3p6
        sst = jnp.float64(300.0)
        t_zt = jnp.float64(298.0)
        q_s = jnp.float64(0.022)
        q_zt = jnp.float64(0.018)
        U_zu = jnp.float64(5.0)
        pQsw = jnp.float64(500.0)
        prad_lw = jnp.float64(350.0)
        pslp = jnp.float64(101000.0)
        plon = jnp.float64(0.0)

        wl_state_0 = {
            "dT_wl": jnp.float64(0.0),
            "Hz_wl": jnp.float64(20.0),
            "Qnt_ac": jnp.float64(0.0),
            "Tau_ac": jnp.float64(0.0),
        }
        result1 = turb_coare3p6(
            2.0, 10.0, sst, t_zt, q_s, q_zt, U_zu,
            l_use_wl=True, pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
            plon=plon, isecday_utc=43200, rdt=3600.0, gdept_1d=1.0,
            wl_state=wl_state_0,
        )
        wl_state_1 = {
            "dT_wl": result1["dT_wl"],
            "Hz_wl": result1["Hz_wl"],
            "Qnt_ac": result1["Qnt_ac"],
            "Tau_ac": result1["Tau_ac"],
        }
        result2 = turb_coare3p6(
            2.0, 10.0, sst, t_zt, q_s, q_zt, U_zu,
            l_use_wl=True, pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
            plon=plon, isecday_utc=46800, rdt=3600.0, gdept_1d=1.0,
            wl_state=wl_state_1,
        )
        assert jnp.isfinite(result2["dT_wl"])
        assert jnp.isfinite(result2["Hz_wl"])


class TestCSWLDifferentiability:
    """Differentiability tests for CSWL."""

    def test_cs_coare_grad_sst(self):
        """Test gradient of cs_coare w.r.t. SST."""
        from jaxerobulk.skin_coare import cs_coare

        def f(sst):
            return cs_coare(jnp.float64(200.0), jnp.float64(-100.0), jnp.float64(0.3), sst, jnp.float64(-100.0))

        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)

    def test_cs_ecmwf_grad_sst(self):
        """Test gradient of cs_ecmwf w.r.t. SST."""
        from jaxerobulk.skin_ecmwf import cs_ecmwf

        def f(sst):
            return cs_ecmwf(jnp.float64(200.0), jnp.float64(-100.0), jnp.float64(0.3), sst)

        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)

    def test_wl_ecmwf_grad_sst(self):
        """Test gradient of wl_ecmwf w.r.t. SST."""
        from jaxerobulk.skin_ecmwf import wl_ecmwf

        def f(sst):
            dT, _ = wl_ecmwf(jnp.float64(500.0), jnp.float64(50.0), jnp.float64(0.1), sst, jnp.float64(0.0), jnp.float64(3.0), 3600.0, 1.0)
            return dT

        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)

    def test_coare3p6_cswl_grad_sst(self):
        """Test gradient through COARE 3.6 with CSWL w.r.t. SST."""
        from jaxerobulk.coare import turb_coare3p6

        def f(sst):
            result = turb_coare3p6(
                2.0, 10.0, sst, jnp.float64(298.0), jnp.float64(0.022), jnp.float64(0.018), jnp.float64(5.0),
                l_use_cs=True, l_use_wl=True,
                pQsw=jnp.float64(200.0), prad_lw=jnp.float64(350.0), pslp=jnp.float64(101000.0),
            )
            return result["Cd"]

        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)

    def test_ecmwf_cswl_grad_sst(self):
        """Test gradient through ECMWF with CSWL w.r.t. SST."""
        from jaxerobulk.ecmwf import turb_ecmwf

        def f(sst):
            result = turb_ecmwf(
                2.0, 10.0, sst, jnp.float64(298.0), jnp.float64(0.022), jnp.float64(0.018), jnp.float64(5.0),
                l_use_cs=True, l_use_wl=True,
                pQsw=jnp.float64(200.0), prad_lw=jnp.float64(350.0), pslp=jnp.float64(101000.0),
            )
            return result["Cd"]

        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)
