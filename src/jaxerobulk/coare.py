"""COARE 3.0 and COARE 3.6 bulk algorithms.

Without cool-skin/warm-layer (added in Iteration 5).
"""

import jax.numpy as jnp
from jaxerobulk.constants import grav, vkarmn, vkarmn2, Cx_min, z0_sea_max, rdct_qsat_salt
from jaxerobulk.thermodynamics import visc_air, one_on_L, q_sat
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


def _turb_coare_core(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, charn_func, Beta0, z0t_exponent, z0t_prefactor, z0t_max, nb_iter=5):
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
    """
    zi0 = 600.0
    zeta_abs_max = 50.0

    l_zt_equal_zu = abs(zu - zt) < 0.01
    zm_ztzu = 0.0 if l_zt_equal_zu else 1.0

    zlog_10 = jnp.log(jnp.float64(10.0))
    zlog_zt = jnp.log(jnp.float64(zt))
    zlog_zu = jnp.log(jnp.float64(zu))

    pcharn = charn_func(U_zu)

    zus, zts, zqs, t_zu, q_zu, zUbzu, zz0 = first_guess_coare(
        zt, zu, T_s, t_zt, pssq=q_s, q_zt=q_zt, U_zu=U_zu, pcharn=pcharn
    )

    zlog_z0 = jnp.log(zz0)
    zNu_a = visc_air(t_zu)

    zdt = jnp.sign(t_zu - T_s) * jnp.maximum(jnp.abs(t_zu - T_s), 1.0e-9)
    zdq = jnp.sign(q_zu - q_s) * jnp.maximum(jnp.abs(q_zu - q_s), 1.0e-12)

    for _ in range(nb_iter):
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

        zdt = jnp.sign(t_zu - T_s) * jnp.maximum(jnp.abs(t_zu - T_s), 1.0e-9)
        zdq = jnp.sign(q_zu - q_s) * jnp.maximum(jnp.abs(q_zu - q_s), 1.0e-12)

    ztmp0 = zus / zUbzu
    Cd = jnp.maximum(ztmp0**2, Cx_min)
    Ch = jnp.maximum(ztmp0 * zts / zdt, Cx_min)
    Ce = jnp.maximum(ztmp0 * zqs / zdq, Cx_min)

    ztmp0_neut = 1.0 / (zlog_zu - zlog_z0)
    CdN = jnp.maximum(vkarmn2 * ztmp0_neut**2, Cx_min)
    ztmp1_neut = vkarmn2 * ztmp0_neut / (zlog_zu - zlog_z0t)
    ChN = jnp.maximum(ztmp1_neut, Cx_min)
    CeN = jnp.maximum(ztmp1_neut, Cx_min)

    return {
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


def turb_coare3p0(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, nb_iter=5):
    """COARE 3.0 bulk algorithm (Fairall et al. 2003).

    Without cool-skin/warm-layer.
    """
    return _turb_coare_core(
        zt, zu, T_s, t_zt, q_s, q_zt, U_zu,
        charn_func=charn_coare3p0,
        Beta0=1.25,
        z0t_exponent=0.6,
        z0t_prefactor=5.5e-5,
        z0t_max=1.1e-4,
        nb_iter=nb_iter,
    )


def turb_coare3p6(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, nb_iter=5):
    """COARE 3.6 bulk algorithm (Edson et al. 2013).

    Without cool-skin/warm-layer.
    """
    return _turb_coare_core(
        zt, zu, T_s, t_zt, q_s, q_zt, U_zu,
        charn_func=charn_coare3p6,
        Beta0=1.2,
        z0t_exponent=0.72,
        z0t_prefactor=5.8e-5,
        z0t_max=1.6e-4,
        nb_iter=nb_iter,
    )
