"""COARE 3.0 and COARE 3.6 bulk algorithms.

With optional cool-skin/warm-layer parameterizations.
"""

import jax.numpy as jnp
from jaxerobulk.constants import grav, vkarmn, vkarmn2, Cx_min, z0_sea_max, rdct_qsat_salt
from jaxerobulk.thermodynamics import visc_air, one_on_L, q_sat, update_qnsol_tau
from jaxerobulk.stability import psi_m_coare, psi_h_coare
from jaxerobulk.first_guess import first_guess_coare


def charn_coare3p0(pwnd):
    """Charnock parameter for COARE 3.0.

    Piecewise: 0.011 for U<10, linear ramp 10-18, 0.018 for U>18.
    """
    zgt10 = jnp.where(pwnd >= 10.0, 1.0, 0.0)
    zgt18 = jnp.where(pwnd >= 18.0, 1.0, 0.0)
    return (
        (1.0 - zgt10) * 0.011
        + zgt10 * (
            (1.0 - zgt18) * (0.011 + (0.018 - 0.011) * (pwnd - 10.0) / 8.0)
            + zgt18 * 0.018
        )
    )


def charn_coare3p6(pwnd):
    """Charnock parameter for COARE 3.6.

    Linear: max(min(0.0017*U - 0.005, 0.028), 0) (Edson et al. 2013 Eq. 13).
    """
    return jnp.maximum(jnp.minimum(0.0017 * pwnd - 0.005, 0.028), 0.0)


def _turb_coare_core(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, charn_func, Beta0, z0t_exponent, z0t_prefactor, z0t_max, nb_iter=5,
                     l_use_cs=False, l_use_wl=False, pQsw=None, prad_lw=None, pslp=None,
                     isecday_utc=12, plon=None, rdt=3600.0, gdept_1d=1.0,
                     wl_state=None):
    """Core COARE iteration loop (shared by 3.0 and 3.6).

    Parameters
    ----------
    z0t_exponent : float
        0.6 for COARE 3.0, 0.72 for COARE 3.6
    z0t_prefactor : float
        5.5e-5 for COARE 3.0, 5.8e-5 for COARE 3.6
    z0t_max : float
        1.1e-4 for COARE 3.0, 1.6e-4 for COARE 3.6
    Beta0 : float
        1.25 for COARE 3.0, 1.2 for COARE 3.6
    l_use_cs : bool
        Enable cool-skin parameterization
    l_use_wl : bool
        Enable warm-layer parameterization
    pQsw : array or None
        Net solar radiation into ocean [W/m^2] (needed for CSWL)
    prad_lw : array or None
        Downwelling longwave radiation [W/m^2] (needed for CSWL)
    pslp : array or None
        Sea-level pressure [Pa] (needed for CSWL)
    isecday_utc : int
        Current UTC time in seconds since midnight (for warm-layer)
    plon : array or None
        Longitude [deg.E] (for warm-layer)
    rdt : float
        Time step [s] (for warm-layer)
    gdept_1d : float
        Depth of first T-point [m] (for warm-layer)
    wl_state : dict or None
        Warm-layer state dict with keys dT_wl, Hz_wl, Qnt_ac, Tau_ac
    """
    zi0 = 600.0
    zeta_abs_max = 50.0

    l_zt_equal_zu = abs(zu - zt) < 0.01
    zm_ztzu = 0.0 if l_zt_equal_zu else 1.0

    zlog_10 = jnp.log(jnp.float64(10.0))
    zlog_zt = jnp.log(jnp.float64(zt))
    zlog_zu = jnp.log(jnp.float64(zu))

    pcharn = charn_func(U_zu)

    zSST = T_s
    zus, zts, zqs, t_zu, q_zu, zUbzu, zz0 = first_guess_coare(
        zt, zu, T_s, t_zt, pssq=q_s, q_zt=q_zt, U_zu=U_zu, pcharn=pcharn
    )

    zlog_z0 = jnp.log(zz0)
    zNu_a = visc_air(t_zu)

    zdt = jnp.sign(t_zu - T_s) * jnp.maximum(jnp.abs(t_zu - T_s), 1.0e-9)
    zdq = jnp.sign(q_zu - q_s) * jnp.maximum(jnp.abs(q_zu - q_s), 1.0e-12)

    zdT_cs = jnp.zeros_like(T_s)
    zT_s = T_s
    zq_s = q_s

    if l_use_wl and wl_state is not None:
        z_dT_wl = wl_state["dT_wl"]
        z_Hz_wl = wl_state["Hz_wl"]
        z_Qnt_ac = wl_state["Qnt_ac"]
        z_Tau_ac = wl_state["Tau_ac"]
    elif l_use_wl:
        z_dT_wl = jnp.zeros_like(T_s)
        z_Hz_wl = jnp.full_like(T_s, 20.0)
        z_Qnt_ac = jnp.zeros_like(T_s)
        z_Tau_ac = jnp.zeros_like(T_s)

    for jit in range(nb_iter):
        zus2 = zus * zus

        z1oL = one_on_L(t_zu, q_zu, zus, zts, zqs)
        z1oL = jnp.sign(z1oL) * jnp.minimum(jnp.abs(z1oL), 200.0)

        zgust2 = Beta0**2 * zus2 * jnp.maximum(-zi0 * z1oL / vkarmn, 0.0) ** (2.0 / 3.0)
        zUbzu = jnp.maximum(jnp.sqrt(U_zu**2 + zgust2), 0.2)

        zzta_u = zu * z1oL
        zzta_u = jnp.sign(zzta_u) * jnp.minimum(jnp.abs(zzta_u), zeta_abs_max)
        zzta_t = zt * z1oL
        zzta_t = jnp.sign(zzta_t) * jnp.minimum(jnp.abs(zzta_t), zeta_abs_max)

        zUn10 = zus / vkarmn * (zlog_10 - zlog_z0)

        zz0 = charn_func(zUn10) * zus2 / grav + 0.11 * zNu_a / zus
        zz0 = jnp.minimum(jnp.maximum(jnp.abs(zz0), 1.0e-9), 1.0)
        zlog_z0 = jnp.log(zz0)

        zRe_r_inv = (zNu_a / (zz0 * zus)) ** z0t_exponent
        zz0t = jnp.minimum(z0t_max, z0t_prefactor * zRe_r_inv)
        zz0t = jnp.minimum(jnp.maximum(jnp.abs(zz0t), 1.0e-9), 1.0)
        zlog_z0t = jnp.log(zz0t)

        ztmp0 = psi_h_coare(zzta_u)
        ztmp1 = vkarmn / (zlog_zu - zlog_z0t - ztmp0)

        zts = zdt * ztmp1
        zqs = zdq * ztmp1
        zus = jnp.maximum(
            zUbzu * vkarmn / (zlog_zu - zlog_z0 - psi_m_coare(zzta_u)),
            1.0e-9,
        )

        if not l_zt_equal_zu:
            ztmp1 = zlog_zt - zlog_zu + ztmp0 - psi_h_coare(zzta_t)
            t_zu = t_zt - zm_ztzu * zts / vkarmn * ztmp1
            q_zu = q_zt - zm_ztzu * zqs / vkarmn * ztmp1

        if l_use_cs:
            from jaxerobulk.skin_coare import cs_coare
            zQns, zTau, zQlat = update_qnsol_tau(
                zu, zT_s, zq_s, t_zu, q_zu, zus, zts, zqs, U_zu, zUbzu, pslp, prad_lw
            )
            zdT_cs = cs_coare(pQsw, zQns, zus, zSST, zQlat)
            zT_s = zSST + zdT_cs
            if l_use_wl:
                zT_s = zT_s + z_dT_wl
            zq_s = rdct_qsat_salt * q_sat(jnp.maximum(zT_s, 200.0), pslp)

        if l_use_wl:
            from jaxerobulk.skin_coare import wl_coare
            zQns, zTau, _ = update_qnsol_tau(
                zu, zT_s, zq_s, t_zu, q_zu, zus, zts, zqs, U_zu, zUbzu, pslp, prad_lw
            )
            iwait = jit % (nb_iter)
            z_dT_wl, z_Hz_wl, z_Qnt_ac, z_Tau_ac = wl_coare(
                pQsw, zQns, zTau, zSST, plon if plon is not None else jnp.zeros_like(T_s),
                isecday_utc, z_dT_wl, z_Hz_wl, z_Qnt_ac, z_Tau_ac, rdt, gdept_1d, iwait=iwait
            )
            zT_s = zSST + z_dT_wl
            if l_use_cs:
                zT_s = zT_s + zdT_cs
            zq_s = rdct_qsat_salt * q_sat(jnp.maximum(zT_s, 200.0), pslp)

        if l_use_cs or l_use_wl or (not l_zt_equal_zu):
            zdt = jnp.sign(t_zu - zT_s) * jnp.maximum(jnp.abs(t_zu - zT_s), 1.0e-9)
            zdq = jnp.sign(q_zu - zq_s) * jnp.maximum(jnp.abs(q_zu - zq_s), 1.0e-12)
        else:
            zdt = jnp.sign(t_zu - zT_s) * jnp.maximum(jnp.abs(t_zu - zT_s), 1.0e-9)
            zdq = jnp.sign(q_zu - zq_s) * jnp.maximum(jnp.abs(q_zu - zq_s), 1.0e-12)

    ztmp0 = zus / zUbzu
    Cd = jnp.maximum(ztmp0**2, Cx_min)
    Ch = jnp.maximum(ztmp0 * zts / zdt, Cx_min)
    Ce = jnp.maximum(ztmp0 * zqs / zdq, Cx_min)

    ztmp0_neut = 1.0 / (zlog_zu - zlog_z0)
    CdN = jnp.maximum(vkarmn2 * ztmp0_neut**2, Cx_min)
    ztmp1_neut = vkarmn2 * ztmp0_neut / (zlog_zu - zlog_z0t)
    ChN = jnp.maximum(ztmp1_neut, Cx_min)
    CeN = jnp.maximum(ztmp1_neut, Cx_min)

    result = {
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "t_zu": t_zu,
        "q_zu": q_zu,
        "Ubzu": zUbzu,
        "CdN": CdN,
        "ChN": ChN,
        "CeN": CeN,
        "z0": zz0,
        "u_star": zus,
        "L": jnp.where(jnp.abs(z1oL) > 0, 1.0 / z1oL, 1.0e10),
        "UN10": zus / vkarmn * (zlog_10 - zlog_z0),
    }

    if l_use_cs:
        result["dT_cs"] = zdT_cs
        result["T_s"] = zT_s
        result["q_s"] = zq_s

    if l_use_wl:
        result["dT_wl"] = z_dT_wl
        result["Hz_wl"] = z_Hz_wl
        result["Qnt_ac"] = z_Qnt_ac
        result["Tau_ac"] = z_Tau_ac
        if "T_s" not in result:
            result["T_s"] = zT_s
            result["q_s"] = zq_s

    return result


def turb_coare3p0(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, nb_iter=5,
                  l_use_cs=False, l_use_wl=False, pQsw=None, prad_lw=None, pslp=None,
                  isecday_utc=12, plon=None, rdt=3600.0, gdept_1d=1.0, wl_state=None):
    """COARE 3.0 bulk algorithm (Fairall et al. 2003).

    With optional cool-skin/warm-layer parameterizations.
    """
    return _turb_coare_core(
        zt, zu, T_s, t_zt, q_s, q_zt, U_zu,
        charn_func=charn_coare3p0,
        Beta0=1.25,
        z0t_exponent=0.6,
        z0t_prefactor=5.5e-5,
        z0t_max=1.1e-4,
        nb_iter=nb_iter,
        l_use_cs=l_use_cs, l_use_wl=l_use_wl,
        pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
        isecday_utc=isecday_utc, plon=plon, rdt=rdt, gdept_1d=gdept_1d,
        wl_state=wl_state,
    )


def turb_coare3p6(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, nb_iter=5,
                  l_use_cs=False, l_use_wl=False, pQsw=None, prad_lw=None, pslp=None,
                  isecday_utc=12, plon=None, rdt=3600.0, gdept_1d=1.0, wl_state=None):
    """COARE 3.6 bulk algorithm (Edson et al. 2013).

    With optional cool-skin/warm-layer parameterizations.
    """
    return _turb_coare_core(
        zt, zu, T_s, t_zt, q_s, q_zt, U_zu,
        charn_func=charn_coare3p6,
        Beta0=1.2,
        z0t_exponent=0.72,
        z0t_prefactor=5.8e-5,
        z0t_max=1.6e-4,
        nb_iter=nb_iter,
        l_use_cs=l_use_cs, l_use_wl=l_use_wl,
        pQsw=pQsw, prad_lw=prad_lw, pslp=pslp,
        isecday_utc=isecday_utc, plon=plon, rdt=rdt, gdept_1d=gdept_1d,
        wl_state=wl_state,
    )
