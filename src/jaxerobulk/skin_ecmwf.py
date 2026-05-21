"""Cool-skin and warm-layer parameterizations (ECMWF, Zeng & Beljaars 2005).

Port of mod_skin_ecmwf.f90 to JAX. All functions are pure and differentiable.
Warm-layer state is passed explicitly (no global mutable arrays).
"""

import jax
import jax.numpy as jnp

from jaxerobulk.constants import grav, vkarmn, rk0_w, rho0_w, rCp0_w, rnu0_w, sq_radrw
from jaxerobulk.thermodynamics import alpha_sw, delta_skin_layer

rd0 = 3.0
zRhoCp_w = rho0_w * rCp0_w
rNuwl0 = 0.5


@jax.custom_jvp
def _safe_sqrt_where(x):
    """sqrt(x) where x > 0, else 0. Gradient-safe at x=0."""
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


@_safe_sqrt_where.defjvp
def _safe_sqrt_where_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    result = jnp.where(x > 0, jnp.sqrt(x), 0.0)
    dresult = jnp.where(x > 0, 0.5 / jnp.sqrt(jnp.maximum(x, 1e-30)) * dx, 0.0)
    return result, dresult


def _phi_ecmwf(pzeta):
    """Stability function PHI (Takaya et al. 2010, Eq. 5)."""
    zzt2 = pzeta * pzeta
    ztf = jnp.where(pzeta > 0, 1.0, 0.0)
    return (
        ztf * (1.0 + (5.0 * pzeta + 4.0 * zzt2) / (1.0 + 3.0 * pzeta + 0.25 * zzt2))
        + (1.0 - ztf) / jnp.sqrt(1.0 - 16.0 * (-jnp.abs(pzeta)))
    )


def cs_ecmwf(pQsw, pQnsol, pustar, pSST):
    """Cool-skin parameterization (Fairall et al. 1996, ECMWF version).

    Parameters
    ----------
    pQsw : array
        Net solar radiation into the ocean [W/m^2] (>= 0)
    pQnsol : array
        Net non-solar heat flux into the ocean [W/m^2] (normally < 0)
    pustar : array
        Friction velocity [m/s]
    pSST : array
        Bulk SST [K]

    Returns
    -------
    dT_cs : array
        Temperature difference due to cool-skin effect [K]
    """
    zalpha = alpha_sw(pSST)
    zQabs = pQnsol

    zdelta = delta_skin_layer(zalpha, zQabs, pustar)

    for _ in range(4):
        zfr = jnp.maximum(
            0.065 + 11.0 * zdelta - 6.6e-5 / zdelta * (1.0 - jnp.exp(-zdelta / 8.0e-4)),
            0.01,
        )
        zQabs = pQnsol + zfr * pQsw
        zdelta = delta_skin_layer(zalpha, zQabs, pustar)

    return zQabs * zdelta / rk0_w


def wl_ecmwf(pQsw, pQnsol, pustar, pSST, dT_wl, Hz_wl, rdt, gdept_1d, pustk=None):
    """Warm-layer scheme (Zeng & Beljaars 2005, ECMWF).

    Parameters
    ----------
    pQsw : array
        Net solar radiation into the ocean [W/m^2] (>= 0)
    pQnsol : array
        Net non-solar heat flux into the ocean [W/m^2] (normally < 0)
    pustar : array
        Friction velocity [m/s]
    pSST : array
        Bulk SST [K]
    dT_wl : array
        Previous warm-layer dT [K]
    Hz_wl : array
        Previous warm-layer depth [m]
    rdt : float
        Time step [s]
    gdept_1d : float
        Depth of first T-point [m]
    pustk : array or None
        Surface Stokes velocity [m/s] (optional)

    Returns
    -------
    dT_wl_new, Hz_wl_new : arrays
        Updated warm-layer state
    """
    zHwl = Hz_wl

    flg = jnp.where(gdept_1d > zHwl, 1.0, 0.0)
    ztcorr = flg + (1.0 - flg) * gdept_1d / zHwl
    zdTwl_b = jnp.maximum(dT_wl / ztcorr, 0.0)

    zalpha = alpha_sw(pSST)

    zfr = (
        1.0
        - 0.28 * jnp.exp(-71.5 * zHwl)
        - 0.27 * jnp.exp(-2.8 * zHwl)
        - 0.45 * jnp.exp(-0.07 * zHwl)
    )

    zQabs = zfr * pQsw + pQnsol

    zusw = jnp.maximum(pustar, 1.0e-4) * sq_radrw
    zusw2 = zusw * zusw

    if pustk is not None:
        zLa = jnp.sqrt(zusw / jnp.maximum(pustk, 1.0e-6))
    else:
        zLa = 0.3
    zfLa = jnp.maximum(zLa ** (-2.0 / 3.0), 1.0)

    zwf = jnp.where(zQabs > 0, 1.0, 0.0)

    zcst1 = vkarmn * grav * zalpha

    zL2 = zcst1 * zQabs / (zRhoCp_w * zusw2 * zusw)

    zcst2 = zcst1 / (5.0 * zHwl * zusw2)

    zcst0 = rdt * (rNuwl0 + 1.0) / zHwl

    zA = zcst0 * zQabs / (rNuwl0 * zRhoCp_w)

    zcst3 = -zcst0 * vkarmn * zusw * zfLa

    zdTwl_n = zdTwl_b
    for _ in range(10):
        zdTwl_n = 0.5 * (zdTwl_n + zdTwl_b)

        zL1 = _safe_sqrt_where(zdTwl_n * zcst2)

        zeta = (1.0 - zwf) * zHwl * zL1 + zwf * zHwl * zL2

        zB = zcst3 / _phi_ecmwf(zeta)

        zdTwl_n = jnp.maximum(zdTwl_b + zA + zB * zdTwl_n, 0.0)

    dT_wl_new = zdTwl_n * ztcorr

    return dT_wl_new, Hz_wl
