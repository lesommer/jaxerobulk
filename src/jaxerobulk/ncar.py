"""NCAR (Large & Yeager 2004/2008) bulk algorithm."""

import jax.numpy as jnp
from jaxerobulk.constants import vkarmn, vkarmn2, Cx_min, z0_sea_max
from jaxerobulk.thermodynamics import one_on_L, virt_temp, z0_from_cd, un10_from_cd
from jaxerobulk.stability import psi_m_ncar, psi_h_ncar


def cd_n10_ncar(pw10):
    """Neutral drag coefficient at 10m from wind speed (L&Y 2008 Eq. 11)."""
    zw = pw10
    zw6 = zw**6
    zgt33 = jnp.where(zw >= 33.0, 1.0, 0.0)
    result = 1.0e-3 * (
        (1.0 - zgt33) * (2.7 / zw + 0.142 + zw / 13.09 - 3.14807e-10 * zw6)
        + zgt33 * 2.34
    )
    return jnp.maximum(result, Cx_min)


def ch_n10_ncar(psqrt_cdn10, pstab):
    """Neutral sensible heat coefficient at 10m (L&Y 2008 Eq. 9/12)."""
    result = 1.0e-3 * psqrt_cdn10 * (18.0 * pstab + 32.7 * (1.0 - pstab))
    return jnp.maximum(result, Cx_min)


def ce_n10_ncar(psqrt_cdn10):
    """Neutral evaporation coefficient at 10m (L&Y 2008 Eq. 9/13)."""
    result = 1.0e-3 * 34.6 * psqrt_cdn10
    return jnp.maximum(result, Cx_min)


def turb_ncar(zt, zu, sst, t_zt, ssq, q_zt, U_zu, nb_iter=5):
    """NCAR bulk algorithm: compute transfer coefficients.

    Parameters
    ----------
    zt : float
        Height for temperature and humidity [m]
    zu : float
        Height for wind speed [m]
    sst : array
        SST [K]
    t_zt : array
        Potential air temperature at zt [K]
    ssq : array
        Saturation specific humidity at SST [kg/kg]
    q_zt : array
        Specific humidity at zt [kg/kg]
    U_zu : array
        Scalar wind speed at zu [m/s]
    nb_iter : int
        Number of iterations (default 5)

    Returns
    -------
    dict with keys: Cd, Ch, Ce, t_zu, q_zu, Ubzu, CdN, ChN, CeN, z0, u_star, L, UN10
    """
    l_zt_equal_zu = abs(zu - zt) < 0.01

    Ubzu = jnp.maximum(0.5, U_zu)

    zlog1 = jnp.log(jnp.float64(zt / zu))
    zlog2 = jnp.log(jnp.float64(zu / 10.0))

    zstab0 = jnp.where(
        virt_temp(t_zt, q_zt) - virt_temp(sst, ssq) >= 0, 1.0, 0.0
    )

    zCdN = cd_n10_ncar(Ubzu)
    zsqrt_CdN = jnp.sqrt(zCdN)

    Cd = zCdN
    Ce = ce_n10_ncar(zsqrt_CdN)
    Ch = ch_n10_ncar(zsqrt_CdN, zstab0)
    zsqrt_Cd = zsqrt_CdN

    t_zu = jnp.maximum(t_zt, 180.0)
    q_zu = jnp.maximum(q_zt, 1.0e-6)

    for _ in range(nb_iter):
        zdt = t_zu - sst
        zdq = q_zu - ssq

        zus = zsqrt_Cd * Ubzu
        zts = Ch / zsqrt_Cd * zdt
        zqs = Ce / zsqrt_Cd * zdq

        z1oL = one_on_L(t_zu, q_zu, zus, zts, zqs)

        zeta_u = zu * z1oL
        zeta_u = jnp.sign(zeta_u) * jnp.minimum(jnp.abs(zeta_u), 10.0)

        if not l_zt_equal_zu:
            zeta_t = zt * z1oL
            zeta_t = jnp.sign(zeta_t) * jnp.minimum(jnp.abs(zeta_t), 10.0)
            ztmp = zlog1 + psi_h_ncar(zeta_u) - psi_h_ncar(zeta_t)
            t_zu = t_zt - zts / vkarmn * ztmp
            q_zu = q_zt - zqs / vkarmn * ztmp
            q_zu = jnp.maximum(q_zu, 0.0)

        zpsi_m = psi_m_ncar(zeta_u)
        zUn10 = jnp.maximum(
            0.25,
            un10_from_cd(zu, Ubzu, Cd, zpsi_m),
        )
        zCdN = cd_n10_ncar(zUn10)
        zsqrt_CdN = jnp.sqrt(zCdN)

        ztmp = 1.0 + zsqrt_CdN / vkarmn * (zlog2 - zpsi_m)
        Cd = jnp.maximum(zCdN / (ztmp * ztmp), Cx_min)

        zsqrt_Cd = jnp.sqrt(Cd)
        ztmp = (zlog2 - psi_h_ncar(zeta_u)) / vkarmn / zsqrt_CdN
        ztmp2 = zsqrt_Cd / zsqrt_CdN

        zstab = jnp.where(zeta_u >= 0, 1.0, 0.0)
        zChN = 1.0e-3 * zsqrt_CdN * (18.0 * zstab + 32.7 * (1.0 - zstab))
        zCeN = 1.0e-3 * 34.6 * zsqrt_CdN

        Ch = jnp.maximum(zChN * ztmp2 / (1.0 + zChN * ztmp), Cx_min)
        Ce = jnp.maximum(zCeN * ztmp2 / (1.0 + zCeN * ztmp), Cx_min)

    return {
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "t_zu": t_zu,
        "q_zu": q_zu,
        "Ubzu": Ubzu,
        "CdN": zCdN,
        "ChN": zChN,
        "CeN": zCeN,
        "z0": jnp.minimum(z0_from_cd(zu, zCdN), z0_sea_max),
        "u_star": zus,
        "L": jnp.where(jnp.abs(z1oL) > 0, 1.0 / z1oL, 1.0e10),
        "UN10": zUn10,
    }
