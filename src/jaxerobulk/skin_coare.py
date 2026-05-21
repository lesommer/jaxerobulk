"""Cool-skin and warm-layer parameterizations (COARE, Fairall et al. 1996/2019).

Port of mod_skin_coare.f90 to JAX. All functions are pure and differentiable.
Warm-layer state is passed explicitly (no global mutable arrays).
"""

import jax.numpy as jnp
from functools import partial

from jaxerobulk.constants import grav, vkarmn, rk0_w, rho0_w, rCp0_w, rnu0_w
from jaxerobulk.thermodynamics import alpha_sw, delta_skin_layer

Hwl_max = 20.0
Rich0 = 0.65
zfr0 = 0.5


def cs_coare(pQsw, pQnsol, pustar, pSST, pQlat):
    """Cool-skin parameterization (Fairall et al. 1996, COARE 3.6).

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
    pQlat : array
        Latent heat flux [W/m^2]

    Returns
    -------
    dT_cs : array
        Temperature difference due to cool-skin effect [K]
    """
    zalpha = alpha_sw(pSST)
    zQabs = pQnsol

    zdelta = delta_skin_layer(zalpha, zQabs, pustar, Qlat=pQlat)

    for _ in range(4):
        zfr = jnp.maximum(
            0.137 + 11.0 * zdelta - 6.6e-5 / zdelta * (1.0 - jnp.exp(-zdelta / 8.0e-4)),
            0.01,
        )
        zQabs = pQnsol + zfr * pQsw
        zdelta = delta_skin_layer(zalpha, zQabs, pustar, Qlat=pQlat)

    return zQabs * zdelta / rk0_w


def wl_coare(pQsw, pQnsol, pTau, pSST, plon, isd, dT_wl, Hz_wl, Qnt_ac, Tau_ac, rdt, gdept_1d, iwait=0):
    """Warm-layer scheme (COARE 3.6, Fairall et al. 2019).

    Parameters
    ----------
    pQsw : array
        Net solar radiation into the ocean [W/m^2] (>= 0)
    pQnsol : array
        Net non-solar heat flux into the ocean [W/m^2] (normally < 0)
    pTau : array
        Wind stress [N/m^2]
    pSST : array
        Bulk SST [K]
    plon : array
        Longitude [deg.E]
    isd : int
        Current UTC time in seconds since 00h of current day
    dT_wl : array
        Previous warm-layer dT [K]
    Hz_wl : array
        Previous warm-layer depth [m]
    Qnt_ac : array
        Previous accumulated heat [J/m^2]
    Tau_ac : array
        Previous accumulated momentum [N.s/m^2]
    rdt : float
        Time step [s]
    gdept_1d : float
        Depth of first T-point [m]
    iwait : int
        If != 0, don't update accumulated fluxes (in iteration loop)

    Returns
    -------
    dT_wl, Hz_wl, Qnt_ac, Tau_ac : arrays
        Updated warm-layer state
    """
    zalpha = alpha_sw(pSST)

    zcd1 = jnp.sqrt(2.0 * Rich0 * rCp0_w / (zalpha * grav * rho0_w))
    zcd2 = jnp.sqrt(2.0 * zalpha * grav / (Rich0 * rho0_w)) / rCp0_w**1.5

    rlag_gw_h = -1.0 * jnp.mod((360.0 - jnp.mod(plon, 360.0)) / 15.0, 24.0)
    rlag_gw_h_val = jnp.minimum(jnp.abs(rlag_gw_h), jnp.abs(jnp.mod(rlag_gw_h, 24.0)))
    rlag_gw_h = -1.0 * jnp.where(rlag_gw_h + 12.0 >= 0, rlag_gw_h_val, -rlag_gw_h_val)
    ilag_gw_s = (rlag_gw_h * 3600.0).astype(jnp.int32)
    isd_sol = jnp.mod(isd + ilag_gw_s, 24 * 3600)
    rhr_sol = isd_sol / 3600.0

    l_destroy_wl = (rhr_sol > 4.0) & (rhr_sol <= 6.5)
    l_exit = l_destroy_wl

    zdTwl = dT_wl
    zHwl = jnp.maximum(jnp.minimum(Hz_wl, Hwl_max), 0.1)

    zqac = Qnt_ac
    ztac = Tau_ac

    zfr = zfr0

    zfr_nl = 1.0 - (
        0.28 * 0.014 * (1.0 - jnp.exp(-zHwl / 0.014))
        + 0.27 * 0.357 * (1.0 - jnp.exp(-zHwl / 0.357))
        + 0.45 * 12.82 * (1.0 - jnp.exp(-zHwl / 12.82))
    ) / zHwl

    zQabs = zfr_nl * pQsw + pQnsol

    l_exit = jnp.where(l_exit, True, (jnp.abs(zdTwl) < 1.0e-6) & (zQabs <= 0.0))
    l_exit = jnp.where(l_exit, True, (Qnt_ac + zQabs * rdt) <= 0.0)
    l_destroy_wl = jnp.where(l_exit, l_destroy_wl | ((Qnt_ac + zQabs * rdt) <= 0.0), l_destroy_wl)

    ztac_iter = Tau_ac + jnp.maximum(0.002, pTau) * rdt

    zHwl_iter = zHwl
    zfr_iter = zfr_nl
    zQabs_iter = zQabs
    zqac_iter = zqac

    for _ in range(5):
        zfr_iter = 1.0 - (
            0.28 * 0.014 * (1.0 - jnp.exp(-zHwl_iter / 0.014))
            + 0.27 * 0.357 * (1.0 - jnp.exp(-zHwl_iter / 0.357))
            + 0.45 * 12.82 * (1.0 - jnp.exp(-zHwl_iter / 12.82))
        ) / zHwl_iter
        zQabs_iter = zfr_iter * pQsw + pQnsol
        zqac_iter = Qnt_ac + zQabs_iter * rdt
        zHwl_iter = jnp.maximum(
            jnp.minimum(Hwl_max, zcd1 * ztac_iter / jnp.sqrt(jnp.maximum(zqac_iter, 1.0e-20))),
            0.1,
        )

    zqac_iter = jnp.where(l_exit, zqac, zqac_iter)
    ztac_iter = jnp.where(l_exit, ztac, ztac_iter)
    zHwl_iter = jnp.where(l_exit, zHwl, zHwl_iter)

    l_destroy_wl_final = l_destroy_wl | (jnp.where(l_exit, False, zqac_iter <= 0.0))

    zdTwl_iter = jnp.where(
        l_exit,
        zdTwl,
        zcd2 * jnp.maximum(zqac_iter, 0.0)**1.5 / jnp.maximum(ztac_iter, 1.0e-20),
    )

    flg = jnp.where(gdept_1d > zHwl_iter, 1.0, 0.0)
    zdTwl_iter = zdTwl_iter * (flg + (1.0 - flg) * gdept_1d / zHwl_iter)

    zdTwl_out = jnp.where(l_destroy_wl_final, 0.0, zdTwl_iter)
    zHwl_out = jnp.where(l_destroy_wl_final, Hwl_max, zHwl_iter)
    zqac_out = jnp.where(l_destroy_wl_final, 0.0, zqac_iter)
    ztac_out = jnp.where(l_destroy_wl_final, 0.0, ztac_iter)
    zfr_out = jnp.where(l_destroy_wl_final, 0.75, zfr_iter)

    do_update = iwait == 0
    dT_wl_new = jnp.where(do_update, zdTwl_out, dT_wl)
    Hz_wl_new = jnp.where(do_update, zHwl_out, Hz_wl)
    Qnt_ac_new = jnp.where(do_update, zqac_out, Qnt_ac)
    Tau_ac_new = jnp.where(do_update, ztac_out, Tau_ac)

    return dT_wl_new, Hz_wl_new, Qnt_ac_new, Tau_ac_new
