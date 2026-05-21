"""ECMWF (IFS Cy40r1) bulk algorithm.

Without cool-skin/warm-layer (added in Iteration 5).
"""

import jax.numpy as jnp
from jaxerobulk.constants import grav, vkarmn, vkarmn2, Cx_min, rdct_qsat_salt
from jaxerobulk.thermodynamics import visc_air, one_on_L, ri_bulk, q_sat, z0_from_cd
from jaxerobulk.stability import psi_m_ecmwf, psi_h_ecmwf
from jaxerobulk.first_guess import first_guess_coare


charn0_ecmwf = 0.018
_alpha_M = 0.11
_alpha_H = 0.40
_alpha_Q = 0.62


def turb_ecmwf(zt, zu, T_s, t_zt, q_s, q_zt, U_zu, nb_iter=5):
    """ECMWF bulk algorithm (IFS Cy40r1).

    Without cool-skin/warm-layer.
    """
    zi0 = 1000.0
    Beta0 = 1.0

    l_zt_equal_zu = abs(zu - zt) < 0.01
    zm_ztzu = 0.0 if l_zt_equal_zu else 1.0

    zlog_10 = jnp.log(jnp.float64(10.0))
    zlog_zu = jnp.log(jnp.float64(zu))
    zlog_ztu = jnp.log(jnp.float64(zt / zu))

    zus, zts, zqs, t_zu, q_zu, zUbzu, zz0 = first_guess_coare(
        zt, zu, T_s, t_zt, pssq=q_s, q_zt=q_zt, U_zu=U_zu, pcharn=jnp.full_like(U_zu, charn0_ecmwf) if isinstance(U_zu, jnp.ndarray) else jnp.float64(charn0_ecmwf)
    )

    zlog_z0 = jnp.log(zz0)
    zNu_a = visc_air(t_zt)

    zdt = jnp.sign(t_zu - T_s) * jnp.maximum(jnp.abs(t_zu - T_s), 1.0e-9)
    zdq = jnp.sign(q_zu - q_s) * jnp.maximum(jnp.abs(q_zu - q_s), 1.0e-12)

    z1oL = one_on_L(t_zu, q_zu, zus, zts, zqs)
    zzeta_u = zu * z1oL
    zzeta_t = zt * z1oL

    zz0t = jnp.minimum(
        jnp.maximum(
            jnp.abs(1.0 / (0.1 * jnp.exp(vkarmn / (0.00115 / (vkarmn / (zlog_10 - zlog_z0)))))),
            1.0e-9,
        ),
        1.0,
    )
    zlog_z0t = jnp.log(zz0t)

    zFm = zlog_zu - zlog_z0 - psi_m_ecmwf(zzeta_u) + psi_m_ecmwf(zz0 * z1oL)
    zpsi_h_u = psi_h_ecmwf(zzeta_u)
    zFh = zlog_zu - zlog_z0t - zpsi_h_u + psi_h_ecmwf(zz0t * z1oL)

    for _ in range(nb_iter):
        zRib = ri_bulk(zu, T_s, t_zu, q_s, q_zu, zUbzu)

        z1oL = zRib * zFm**2 / zFh / zu
        z1oL = jnp.sign(z1oL) * jnp.minimum(jnp.abs(z1oL), 200.0)

        zzeta_u = zu * z1oL
        zzeta_u = jnp.clip(zzeta_u, -50.0, 5.0)
        zpsi_m_u = psi_m_ecmwf(zzeta_u)
        zpsi_h_u = psi_h_ecmwf(zzeta_u)

        zzeta_t = zt * z1oL
        zzeta_t = jnp.clip(zzeta_t, -50.0, 5.0)
        zpsi_h_t = psi_h_ecmwf(zzeta_t)

        zFm = zlog_zu - zlog_z0 - zpsi_m_u + psi_m_ecmwf(zz0 * z1oL)

        zus = zUbzu * vkarmn / zFm
        zus2 = zus**2
        ztmp0 = zNu_a / zus
        zz0 = jnp.minimum(jnp.abs(_alpha_M * ztmp0 + charn0_ecmwf * zus2 / grav), 0.001)
        zz0t = jnp.minimum(jnp.abs(_alpha_H * ztmp0), 0.001)
        zz0q = jnp.minimum(jnp.abs(_alpha_Q * ztmp0), 0.001)

        zlog_z0 = jnp.log(zz0)
        zlog_z0t = jnp.log(zz0t)
        zlog_z0q = jnp.log(zz0q)

        zpsi_m_z0 = psi_m_ecmwf(zz0 * z1oL)
        zpsi_h_z0t = psi_h_ecmwf(zz0t * z1oL)
        zpsi_h_z0q = psi_h_ecmwf(zz0q * z1oL)

        zgust2 = Beta0**2 * zus2 * jnp.maximum(-zi0 * z1oL / vkarmn, 0.0) ** (2.0 / 3.0)
        zUbzu = jnp.maximum(jnp.sqrt(U_zu**2 + zgust2), 0.2)

        ztmp0 = zpsi_h_u - zpsi_h_z0t
        ztmp1 = vkarmn / (zlog_zu - zlog_z0t - ztmp0)
        zts = zdt * ztmp1
        ztmp1 = zlog_ztu + ztmp0 - zpsi_h_t + zpsi_h_z0t
        t_zu = t_zt - zm_ztzu * zts / vkarmn * ztmp1

        ztmp0 = zpsi_h_u - zpsi_h_z0q
        ztmp1 = vkarmn / (zlog_zu - zlog_z0q - ztmp0)
        zqs = zdq * ztmp1
        ztmp1 = zlog_ztu + ztmp0 - zpsi_h_t + zpsi_h_z0q
        q_zu = jnp.maximum(q_zt - zm_ztzu * zqs / vkarmn * ztmp1, 0.0)

        zFm = zlog_zu - zlog_z0 - zpsi_m_u + zpsi_m_z0
        zFh = zlog_zu - zlog_z0t - zpsi_h_u + zpsi_h_z0t

        zdt = jnp.sign(t_zu - T_s) * jnp.maximum(jnp.abs(t_zu - T_s), 1.0e-9)
        zdq = jnp.sign(q_zu - q_s) * jnp.maximum(jnp.abs(q_zu - q_s), 1.0e-12)

    zFq = zlog_zu - zlog_z0q - zpsi_h_u + zpsi_h_z0q
    Cd = jnp.maximum(vkarmn2 / (zFm**2), Cx_min)
    Ch = jnp.maximum(vkarmn2 / (zFm * zFh), Cx_min)
    Ce = jnp.maximum(vkarmn2 / (zFm * zFq), Cx_min)

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
