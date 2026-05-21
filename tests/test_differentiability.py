"""Comprehensive differentiability test suite.

Tests forward-mode (jacfwd), reverse-mode (grad/jacrev), and consistency
between the two for all algorithms and key input variables.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxerobulk.api import aerobulk_model, Algorithm
from jaxerobulk.ncar import turb_ncar
from jaxerobulk.coare import turb_coare3p0, turb_coare3p6
from jaxerobulk.ecmwf import turb_ecmwf
from jaxerobulk.andreas import turb_andreas
from jaxerobulk.skin_coare import cs_coare, wl_coare
from jaxerobulk.skin_ecmwf import cs_ecmwf, wl_ecmwf

ALGORITHMS = ["ncar", "coare3p0", "coare3p6", "ecmwf", "andreas"]
ALGORITHMS_WITH_CSWL = ["coare3p0", "coare3p6", "ecmwf"]

DEFAULT_INPUTS = dict(
    sst=300.0,
    t_zt=298.0,
    hum_zt=0.018,
    U_zu=5.0,
    V_zu=0.0,
    slp=101000.0,
)


def _make_input(**overrides):
    d = dict(DEFAULT_INPUTS, **overrides)
    return {k: jnp.float64(v) for k, v in d.items()}


class TestTurbAlgorithmGradSST:
    """Test jax.grad through each algorithm w.r.t. SST."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_sst_cd(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Cd"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g), f"{algo} grad Cd w.r.t. SST is NaN"

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_sst_qlat(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Qlat"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g), f"{algo} grad Qlat w.r.t. SST is NaN"

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_sst_qsen(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Qsen"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g), f"{algo} grad Qsen w.r.t. SST is NaN"


class TestTurbAlgorithmGradWind:
    """Test jax.grad through each algorithm w.r.t. wind speed."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_wind_cd(self, algo):
        def f(U):
            inp = _make_input(U_zu=U)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Cd"]
        g = jax.grad(f)(jnp.float64(5.0))
        assert jnp.isfinite(g), f"{algo} grad Cd w.r.t. wind is NaN"


class TestTurbAlgorithmGradTAir:
    """Test jax.grad through each algorithm w.r.t. air temperature."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_tair_cd(self, algo):
        def f(t_zt):
            inp = _make_input(t_zt=t_zt)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Ch"]
        g = jax.grad(f)(jnp.float64(298.0))
        assert jnp.isfinite(g), f"{algo} grad Ch w.r.t. T_air is NaN"


class TestForwardReverseConsistency:
    """Check forward-mode and reverse-mode gradients agree."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_fwd_rev_sst_cd(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Cd"]
        g_fwd = jax.jacfwd(f)(jnp.float64(300.0))
        g_rev = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g_fwd) and jnp.isfinite(g_rev)
        np.testing.assert_allclose(float(g_fwd), float(g_rev), rtol=1e-5,
                                   err_msg=f"{algo}: fwd/rev mismatch for Cd/SST")

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_fwd_rev_sst_qlat(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Qlat"]
        g_fwd = jax.jacfwd(f)(jnp.float64(300.0))
        g_rev = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g_fwd) and jnp.isfinite(g_rev)
        np.testing.assert_allclose(float(g_fwd), float(g_rev), rtol=1e-5,
                                   err_msg=f"{algo}: fwd/rev mismatch for Qlat/SST")


class TestCSWLDifferentiability:
    """Differentiability through algorithms with CSWL enabled."""

    @pytest.mark.parametrize("algo", ALGORITHMS_WITH_CSWL)
    def test_cswl_grad_sst_cd(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(
                2.0, 10.0, **inp, algo=algo,
                l_use_skin=True,
                rad_sw=jnp.float64(200.0), rad_lw=jnp.float64(350.0),
            )
            return result["Cd"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g), f"{algo}+CSWL grad Cd w.r.t. SST is NaN"

    @pytest.mark.parametrize("algo", ALGORITHMS_WITH_CSWL)
    def test_cswl_grad_sst_qlat(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(
                2.0, 10.0, **inp, algo=algo,
                l_use_skin=True,
                rad_sw=jnp.float64(200.0), rad_lw=jnp.float64(350.0),
            )
            return result["Qlat"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g), f"{algo}+CSWL grad Qlat w.r.t. SST is NaN"

    @pytest.mark.parametrize("algo", ALGORITHMS_WITH_CSWL)
    def test_cswl_fwd_rev_consistency(self, algo):
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(
                2.0, 10.0, **inp, algo=algo,
                l_use_skin=True,
                rad_sw=jnp.float64(200.0), rad_lw=jnp.float64(350.0),
            )
            return result["Cd"]
        g_fwd = jax.jacfwd(f)(jnp.float64(300.0))
        g_rev = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g_fwd) and jnp.isfinite(g_rev)
        np.testing.assert_allclose(float(g_fwd), float(g_rev), rtol=1e-5,
                                   err_msg=f"{algo}+CSWL: fwd/rev mismatch for Cd/SST")


class TestGradMagnitudeSanity:
    """Check that gradient magnitudes are physically reasonable."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_grad_sst_cd_magnitude(self, algo):
        """dCd/dSST should be O(1e-5) or smaller."""
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return result["Cd"]
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.abs(g) < 1e-3, f"{algo}: |dCd/dSST| = {float(jnp.abs(g)):.2e} is too large"


class TestSkinGrad:
    """Differentiability through skin functions."""

    def test_cs_coare_grad_all(self):
        for var_name, val in [("sst", 300.0), ("Qsw", 200.0), ("Qnsol", -100.0), ("ustar", 0.3)]:
            def f(x):
                kwargs = dict(
                    pQsw=jnp.float64(200.0), pQnsol=jnp.float64(-100.0),
                    pustar=jnp.float64(0.3), pSST=jnp.float64(300.0), pQlat=jnp.float64(-100.0),
                )
                kwargs[{0: "pSST", 1: "pQsw", 2: "pQnsol", 3: "pustar"}[
                    ["sst", "Qsw", "Qnsol", "ustar"].index(var_name)]] = x
                return cs_coare(**kwargs)
            g = jax.grad(f)(jnp.float64(val))
            assert jnp.isfinite(g), f"cs_coare grad w.r.t. {var_name} is NaN"

    def test_cs_ecmwf_grad_all(self):
        for var_name, val in [("sst", 300.0), ("Qsw", 200.0), ("Qnsol", -100.0), ("ustar", 0.3)]:
            def f(x):
                kwargs = dict(
                    pQsw=jnp.float64(200.0), pQnsol=jnp.float64(-100.0),
                    pustar=jnp.float64(0.3), pSST=jnp.float64(300.0),
                )
                kwargs[{0: "pSST", 1: "pQsw", 2: "pQnsol", 3: "pustar"}[
                    ["sst", "Qsw", "Qnsol", "ustar"].index(var_name)]] = x
                return cs_ecmwf(**kwargs)
            g = jax.grad(f)(jnp.float64(val))
            assert jnp.isfinite(g), f"cs_ecmwf grad w.r.t. {var_name} is NaN"

    def test_wl_ecmwf_grad_sst(self):
        def f(sst):
            dT, _ = wl_ecmwf(
                jnp.float64(500.0), jnp.float64(50.0), jnp.float64(0.1),
                sst, jnp.float64(0.0), jnp.float64(3.0), 3600.0, 1.0,
            )
            return dT
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)

    def test_wl_coare_grad_sst(self):
        def f(sst):
            dT, _, _, _ = wl_coare(
                jnp.float64(500.0), jnp.float64(50.0), jnp.float64(0.03),
                sst, jnp.float64(0.0), 43200,
                jnp.float64(0.0), jnp.float64(20.0), jnp.float64(0.0), jnp.float64(0.0),
                3600.0, 1.0,
            )
            return dT
        g = jax.grad(f)(jnp.float64(300.0))
        assert jnp.isfinite(g)


class TestJacobian:
    """Test full Jacobian computation via jax.jacfwd."""

    @pytest.mark.parametrize("algo", ALGORITHMS)
    def test_jacobian_sst_exists(self, algo):
        """Test that we can compute a Jacobian of multiple outputs w.r.t. SST."""
        def f(sst):
            inp = _make_input(sst=sst)
            result = aerobulk_model(2.0, 10.0, **inp, algo=algo)
            return jnp.array([result["Cd"], result["Ch"], result["Ce"]])

        J = jax.jacfwd(f)(jnp.float64(300.0))
        assert J.shape == (3,)
        assert jnp.all(jnp.isfinite(J)), f"{algo} Jacobian has NaN"
