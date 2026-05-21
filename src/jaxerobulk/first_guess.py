"""COARE first-guess routine shared by COARE 3.0, COARE 3.6, and ECMWF."""

import jax.numpy as jnp
from jaxerobulk.constants import grav, vkarmn, vkarmn2
from jaxerobulk.thermodynamics import visc_air, ri_bulk
from jaxerobulk.stability import psi_m_coare, psi_h_coare


def first_guess_coare(zt, zu, psst, t_zt, pssq, q_zt, U_zu, pcharn):
    """First guess of u*, theta*, q* for COARE algorithms.

    Parameters
    ----------
    zt : float
        Height for temperature and humidity [m]
    zu : float
        Height for wind speed [m]
    psst : array
        SST [K]
    t_zt : array
        Potential air temperature at zt [K]
    pssq : array
        Saturation specific humidity at SST [kg/kg]
    q_zt : array
        Specific humidity at zt [kg/kg]
    U_zu : array
        Scalar wind speed at zu [m/s]
    pcharn : array
        Charnock parameter

    Returns
    -------
    (pus, pts, pqs, t_zu, q_zu, Ubzu, pz0)
    """
    l_zt_equal_zu = abs(zu - zt) < 0.01

    t_zu = jnp.maximum(t_zt, 180.0)
    q_zu = jnp.maximum(q_zt, 1.0e-6)

    zz0 = 1.0e-4

    zlog_10 = jnp.log(10.0)
    zlog_zt = jnp.log(jnp.float64(zt))
    zlog_zu = jnp.log(jnp.float64(zu))
    zc_a = 0.035 * jnp.log(10.0 / zz0) / jnp.log(zu / zz0)
    zc_b = 0.004 * 600.0 * 1.2**3

    zdt = jnp.sign(t_zu - psst) * jnp.maximum(jnp.abs(t_zu - psst), 1.0e-9)
    zdq = jnp.sign(q_zu - pssq) * jnp.maximum(jnp.abs(q_zu - pssq), 1.0e-12)

    zNu_a = visc_air(t_zu)
    zUb = jnp.sqrt(U_zu**2 + 0.5**2)

    zus = zc_a * zUb

    zz0 = pcharn * zus**2 / grav + 0.11 * zNu_a / zus
    zz0 = jnp.minimum(jnp.maximum(jnp.abs(zz0), 1.0e-8), 1.0)
    zlog_z0 = jnp.log(zz0)

    zCd = (vkarmn / (zlog_zu - zlog_z0)) ** 2
    z1_o_sqrt_Cd10 = (zlog_10 - zlog_z0) / vkarmn

    zz0t = 10.0 / jnp.exp(vkarmn / (0.00115 * z1_o_sqrt_Cd10))
    zz0t = jnp.minimum(jnp.maximum(jnp.abs(zz0t), 1.0e-8), 1.0)
    zlog_z0t = jnp.log(zz0t)

    zRib = ri_bulk(zu, psst, t_zu, pssq, q_zu, zUb)

    zcc = vkarmn2 / (zCd * (zlog_zt - zlog_z0t))
    zcc_ri = zcc * zRib
    z1_o_Ribcu = -zc_b / zu
    zstab = jnp.where(zRib >= 0, 1.0, 0.0)

    zzeta_u = (
        (1.0 - zstab) * zcc_ri / (1.0 + zRib * z1_o_Ribcu)
        + zstab * (zcc_ri + 27.0 / 9.0 * zRib**2)
    )

    zus = jnp.maximum(
        zUb * vkarmn / (zlog_zu - zlog_z0 - psi_m_coare(zzeta_u)),
        1.0e-9,
    )
    ztmp = vkarmn / (zlog_zu - zlog_z0t - psi_h_coare(zzeta_u))
    zts = zdt * ztmp
    zqs = zdq * ztmp

    if not l_zt_equal_zu:
        zzeta_t = zt * zzeta_u / zu
        zprf = jnp.log(jnp.float64(zt / zu)) + psi_h_coare(zzeta_u) - psi_h_coare(zzeta_t)
        t_zu = t_zt - zts / vkarmn * zprf
        q_zu = q_zt - zqs / vkarmn * zprf
        q_zu = jnp.maximum(q_zu, 0.0)

        zdt = jnp.sign(t_zu - psst) * jnp.maximum(jnp.abs(t_zu - psst), 1.0e-9)
        zdq = jnp.sign(q_zu - pssq) * jnp.maximum(jnp.abs(q_zu - pssq), 1.0e-12)
        zts = zdt * ztmp
        zqs = zdq * ztmp

    pz0 = pcharn * zus**2 / grav + 0.11 * zNu_a / zus
    pz0 = jnp.minimum(jnp.maximum(jnp.abs(pz0), 1.0e-8), 1.0)

    return zus, zts, zqs, t_zu, q_zu, zUb, pz0
