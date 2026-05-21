"""Andreas et al. (2015) bulk algorithm for sea ice."""

import jax.numpy as jnp
from jaxerobulk.constants import vkarmn, vkarmn2, Cx_min, z0_sea_max
from jaxerobulk.thermodynamics import visc_air, one_on_L, ri_bulk, z0_from_cd, un10_from_ustar, q_sat
from jaxerobulk.stability import psi_m_andreas, psi_h_andreas
from jaxerobulk.thermodynamics import z0tq_lkb


_rRi_max = 0.15
_rCs_min = 0.35e-3


def u_star_andreas(pun10):
    """Friction velocity from neutral 10m wind speed (Andreas et al. 2015 Eq. 2.2)."""
    za = pun10 - 8.271
    zt = za + jnp.sqrt(0.12 * za**2 + 0.181)
    return 0.239 + 0.0433 * zt


def turb_andreas(zt, zu, sst, t_zt, ssq, q_zt, U_zu, nb_iter=5):
    """Andreas bulk algorithm for sea ice (Andreas et al. 2015).

    Parameters
    ----------
    zt : float
        Height for temperature and humidity [m]
    zu : float
        Height for wind speed [m]
    sst : array
        Ice surface temperature [K]
    t_zt : array
        Potential air temperature at zt [K]
    ssq : array
        Saturation specific humidity at surface temperature [kg/kg]
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

    Ubzu = jnp.maximum(0.25, U_zu)

    UN10 = Ubzu
    Cd = jnp.full_like(Ubzu, 1.1e-3) if isinstance(Ubzu, jnp.ndarray) else jnp.float64(1.1e-3)
    Ch = jnp.full_like(Ubzu, 1.1e-3) if isinstance(Ubzu, jnp.ndarray) else jnp.float64(1.1e-3)
    Ce = jnp.full_like(Ubzu, 1.1e-3) if isinstance(Ubzu, jnp.ndarray) else jnp.float64(1.1e-3)
    t_zu = t_zt
    q_zu = q_zt

    zsqrt_Cd = jnp.sqrt(Cd)
    t_star = Ch / zsqrt_Cd * (t_zu - sst)
    q_star = Ce / zsqrt_Cd * (q_zu - ssq)

    RiB = ri_bulk(zu, sst, t_zu, ssq, q_zu, Ubzu)

    for _ in range(nb_iter):
        u_star = jnp.where(
            RiB < _rRi_max,
            u_star_andreas(UN10),
            jnp.sqrt(Cx_min) * Ubzu,
        )

        zeta_u = zu * one_on_L(t_zu, q_zu, u_star, t_star, q_star)

        ztmp0 = u_star / Ubzu
        Cd = jnp.maximum(ztmp0**2, Cx_min)

        z0 = jnp.minimum(
            z0_from_cd(zu, Cd, psi_m_andreas(zeta_u)),
            z0_sea_max,
        )

        zRe_r = z0 * u_star / visc_air(t_zu)
        z0t = z0tq_lkb(1, zRe_r, z0)
        z0q = z0tq_lkb(2, zRe_r, z0)

        zpsi_h = psi_h_andreas(zeta_u)
        t_star = (t_zu - sst) * vkarmn / (jnp.log(jnp.float64(zu)) - jnp.log(z0t) - zpsi_h)
        q_star = (q_zu - ssq) * vkarmn / (jnp.log(jnp.float64(zu)) - jnp.log(z0q) - zpsi_h)

        if not l_zt_equal_zu:
            zeta_t = zeta_u / zu * zt
            zprf = jnp.log(jnp.float64(zt / zu)) + psi_h_andreas(zeta_u) - psi_h_andreas(zeta_t)
            t_zu = t_zt - t_star / vkarmn * zprf
            q_zu = q_zt - q_star / vkarmn * zprf
            RiB = ri_bulk(zu, sst, t_zu, ssq, q_zu, Ubzu)

        UN10 = jnp.maximum(
            0.1,
            un10_from_ustar(zu, Ubzu, u_star, psi_m_andreas(zeta_u)),
        )

    ztmp0 = u_star / Ubzu
    Cd = jnp.maximum(ztmp0**2, Cx_min)

    zdt = jnp.sign(t_zu - sst) * jnp.maximum(jnp.abs(t_zu - sst), 1.0e-6)
    zdq = jnp.sign(q_zu - ssq) * jnp.maximum(jnp.abs(q_zu - ssq), 1.0e-9)
    Ch = jnp.maximum(ztmp0 * t_star / zdt, _rCs_min)
    Ce = jnp.maximum(ztmp0 * q_star / zdq, _rCs_min)

    zlog_zu_z0 = 1.0 / jnp.log(jnp.float64(zu) / z0)
    CdN = jnp.maximum(vkarmn2 * zlog_zu_z0**2, Cx_min)

    zRe_r = z0 * u_star / visc_air(t_zu)
    ChN = vkarmn2 * zlog_zu_z0 / jnp.log(jnp.float64(zu) / z0tq_lkb(1, zRe_r, z0))
    CeN = vkarmn2 * zlog_zu_z0 / jnp.log(jnp.float64(zu) / z0tq_lkb(2, zRe_r, z0))

    return {
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "t_zu": t_zu,
        "q_zu": q_zu,
        "Ubzu": Ubzu,
        "CdN": CdN,
        "ChN": ChN,
        "CeN": CeN,
        "z0": z0,
        "u_star": u_star,
        "L": zu / zeta_u,
        "UN10": un10_from_ustar(zu, Ubzu, u_star, psi_m_andreas(zeta_u)),
    }
