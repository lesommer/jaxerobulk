"""Physical thermodynamic functions for the marine boundary layer.

Port of mod_phymbl.f90 to JAX. All functions are pure and differentiable.
They operate elementwise on JAX arrays via broadcasting.
"""

import jax
import jax.numpy as jnp
from functools import partial

from jaxerobulk.constants import (
    grav,
    R_dry,
    R_vap,
    R_gas,
    rCp_dry,
    rCp_vap,
    rCp0_w,
    reps0,
    rctv0,
    rt0,
    rtt0,
    rLevap,
    rLsub,
    rnu0_air,
    rnu0_w,
    rk0_w,
    rho0_w,
    rho0_a,
    Patm,
    stefan,
    emiss_w,
    emiss_i,
    vkarmn,
    vkarmn2,
    rgamma_dry,
    rpoiss_dry,
    radrw,
    sq_radrw,
    rcst_cs,
    z0_sea_max,
    rdct_qsat_salt,
    _rAg_i,
    _rBg_i,
    _rCg_i,
    _rDg_i,
    _rc2_louis,
    _ram_louis,
    _rah_louis,
    _repsilon,
    ref_sha_min,
    ref_sha_max,
    ref_dpt_min,
    ref_dpt_max,
    ref_rlh_min,
    ref_rlh_max,
)


def pot_temp(pTa, pPz, pPref=None):
    """Potential temperature from absolute temperature and pressure.

    Poisson's equation: theta = T * (P_ref / P_z)^(R_dry / Cp_dry)
    """
    zPref = Patm if pPref is None else pPref
    return pTa * (zPref / pPz) ** rpoiss_dry


def abs_temp(pThta, pPz, pPref=None):
    """Absolute temperature from potential temperature and pressure."""
    zPref = Patm if pPref is None else pPref
    return pThta / jnp.maximum((zPref / pPz) ** rpoiss_dry, 1.0e-9)


def virt_temp(pTa, pqa):
    """Virtual temperature: T_v = T * (1 + 0.608 * q)."""
    return pTa * (1.0 + rctv0 * pqa)


def e_sat(pTa):
    """Saturation vapor pressure over water [Pa] using Goff (1957)."""
    zta = jnp.maximum(pTa, 180.0)
    ztmp = rt0 / zta
    exponent = (
        10.79574 * (1.0 - ztmp)
        - 5.028 * jnp.log10(zta / rt0)
        + 1.50475e-4 * (1.0 - 10.0 ** (-8.2969 * (zta / rt0 - 1.0)))
        + 0.42873e-3 * (10.0 ** (4.76955 * (1.0 - ztmp)) - 1.0)
        + 0.78614
    )
    return 100.0 * 10.0**exponent


def e_sat_ice(pTa):
    """Saturation vapor pressure over ice [Pa] using Goff formula."""
    zta = jnp.maximum(pTa, 180.0)
    ztmp = rtt0 / zta
    zle = (
        _rAg_i * (ztmp - 1.0)
        + _rBg_i * jnp.log10(ztmp)
        + _rCg_i * (1.0 - zta / rtt0)
        + _rDg_i
    )
    return 100.0 * 10.0**zle


def de_sat_dt_ice(pTa):
    """Derivative of saturation vapor pressure over ice w.r.t. temperature [Pa/K]."""
    zta = jnp.maximum(pTa, 180.0)
    zde = -(_rAg_i * rtt0) / (zta * zta) - _rBg_i / (zta * jnp.log(10.0)) - _rCg_i / rtt0
    return jnp.log(10.0) * zde * e_sat_ice(zta)


def q_sat(pTa, pslp, l_ice=False):
    """Saturation specific humidity [kg/kg] from temperature and pressure."""
    ze_s = jnp.where(l_ice, e_sat_ice(pTa), e_sat(pTa))
    return reps0 * ze_s / (pslp - (1.0 - reps0) * ze_s)


def dq_sat_dt_ice(pTa, pslp):
    """Derivative of q_sat_ice w.r.t. temperature [1/K]."""
    ze_s = e_sat_ice(pTa)
    zde_s_dt = de_sat_dt_ice(pTa)
    ztmp = (reps0 - 1.0) * ze_s + pslp
    return reps0 * pslp * zde_s_dt / (ztmp * ztmp)


def q_air_rh(prha, pTa, pslp):
    """Specific humidity [kg/kg] from relative humidity [%], temperature, and pressure."""
    ze = 0.01 * prha * e_sat(pTa)
    return ze * reps0 / jnp.maximum(pslp - (1.0 - reps0) * ze, 1.0)


def q_air_dp(da, slp):
    """Specific humidity [kg/kg] from dew-point temperature [K] and pressure [Pa]."""
    ze = jnp.maximum(e_sat(da), 0.0)
    return ze * reps0 / jnp.maximum(slp - (1.0 - reps0) * ze, 1.0)


def rho_air(pTa, pqa, pslp):
    """Density of moist air [kg/m^3], floored at 0.8."""
    return jnp.maximum(pslp / (R_dry * pTa * (1.0 + rctv0 * pqa)), 0.8)


def visc_air(pTa):
    """Kinematic viscosity of air [m^2/s] from temperature [K]."""
    ztc = pTa - rt0
    ztc2 = ztc * ztc
    return 1.326e-5 * (1.0 + 6.542e-3 * ztc + 8.301e-6 * ztc2 - 4.84e-9 * ztc2 * ztc)


def l_vap(psst):
    """Latent heat of vaporization [J/kg] from water temperature [K]."""
    return (2.501 - 0.00237 * (psst - rt0)) * 1.0e6


def cp_air(pqa):
    """Specific heat of moist air [J/K/kg] from specific humidity."""
    return rCp_dry + rCp_vap * pqa


def gamma_moist(pTa, pqa):
    """Moist adiabatic lapse rate [K/m]."""
    zta = jnp.maximum(pTa, 180.0)
    zqa = jnp.maximum(pqa, 1.0e-6)
    zwa = zqa / (1.0 - zqa)
    ziRT = 1.0 / (R_dry * zta)
    zLvap = l_vap(pTa)
    return grav * (1.0 + zLvap * zwa * ziRT) / (rCp_dry + zLvap**2 * zwa * reps0 * ziRT / zta)


def one_on_L(pThta, pqa, pus, pts, pqs):
    """Inverse Obukhov length [1/m] from turbulent scales."""
    zqa = 1.0 + rctv0 * pqa
    val = grav * vkarmn * (pts * zqa + rctv0 * pThta * pqs) / jnp.maximum(pus**2 * pThta * zqa, 1.0e-9)
    return jnp.sign(val) * jnp.minimum(jnp.abs(val), 200.0)


def ri_bulk(pz, psst, pThta, pssq, pqa, pub, pTa_layer=None, pqa_layer=None):
    """Bulk Richardson number."""
    zsstv = virt_temp(psst, pssq)
    zdthv = virt_temp(pThta, pqa) - zsstv
    if pTa_layer is not None and pqa_layer is not None:
        ztv = virt_temp(pTa_layer, pqa_layer)
    else:
        ztv = 0.5 * (zsstv + virt_temp(pThta - rgamma_dry * pz, pqa))
    return grav * zdthv * pz / (ztv * pub**2)


def Pz_from_P0_tz_qz(pz, pslp, pTa, pqa, l_ice=False):
    """Pressure at height z [Pa] from sea-level pressure, temperature, and humidity.

    Uses barometric equation iterated 3 times.
    """
    zpa = pslp
    for _ in range(3):
        zqsat = q_sat(pTa, zpa, l_ice=l_ice)
        zf = pqa / zqsat
        zxm = (1.0 - zf) * 28.9647e-3 + zf * 18.0153e-3
        zpa = pslp * jnp.exp(-grav * zxm * pz / (R_gas * pTa))
    return zpa


def theta_from_z_p0_t_q(pz, pslp, pTa, pqa):
    """Potential temperature at height z [K] from sea-level pressure, absolute temperature, and humidity."""
    zPz = Pz_from_P0_tz_qz(pz, pslp, pTa, pqa)
    return pot_temp(pTa, zPz, pPref=pslp)


def T_from_z_p0_theta_q(pz, pslp, pThta, pqa):
    """Absolute temperature at height z [K] from sea-level pressure, potential temperature, and humidity.

    Iterates 4 times.
    """
    zTa = pThta - rgamma_dry * pz
    for _ in range(4):
        zPz = Pz_from_P0_tz_qz(pz, pslp, zTa, pqa)
        zTa = abs_temp(pThta, zPz, pPref=pslp)
    return zTa


def alpha_sw(psst):
    """Thermal expansion coefficient of seawater [1/K]."""
    return 2.1e-5 * jnp.maximum(psst - rt0 + 3.2, 0.0) ** 0.79


def qlw_net(pdwlw, pts, l_ice=False):
    """Net longwave radiative flux [W/m^2]."""
    zemiss = jnp.where(l_ice, emiss_i, emiss_w)
    zt2 = pts * pts
    return zemiss * (pdwlw - stefan * zt2 * zt2)


def z0_from_cd(pzu, pCd, ppsi=None):
    """Roughness length [m] from drag coefficient."""
    if ppsi is None:
        return pzu * jnp.exp(-vkarmn / jnp.sqrt(pCd))
    else:
        return pzu * jnp.exp(-(vkarmn / jnp.sqrt(pCd) + ppsi))


def z0_from_ustar(pzu, pus, puzu):
    """Roughness length [m] from friction velocity and wind speed."""
    return pzu * jnp.exp(-vkarmn * puzu / pus)


def un10_from_cd(pzu, pUb, pCd, ppsi):
    """Neutral wind speed at 10m [m/s] from drag coefficient."""
    z0 = z0_from_cd(pzu, pCd, ppsi=ppsi)
    return jnp.sqrt(pCd) * pUb / vkarmn * jnp.log(10.0 / z0)


def un10_from_cdn(pzu, pUb, pCdn, ppsi):
    """Neutral wind speed at 10m [m/s] from neutral drag coefficient."""
    return pUb / (1.0 + jnp.sqrt(pCdn) / vkarmn * (jnp.log(pzu / 10.0) - ppsi))


def un10_from_ustar(pzu, pUzu, pus, ppsi):
    """Neutral wind speed at 10m [m/s] from friction velocity."""
    return pUzu - pus / vkarmn * (jnp.log(pzu / 10.0) - ppsi)


def f_m_louis(pzu, pRib, pCdn, pz0):
    """Stability correction for momentum (Louis 1979)."""
    zstab = jnp.where(pRib >= 0, 1.0, 0.0)
    ztu = pRib / (1.0 + 3.0 * _rc2_louis * pCdn * jnp.sqrt(jnp.abs(-pRib * (pzu / pz0 + 1.0))))
    zts = pRib / jnp.sqrt(jnp.abs(1.0 + pRib))
    unstable = 1.0 - _ram_louis * ztu
    stable = 1.0 / (1.0 + _ram_louis * zts)
    return (1.0 - zstab) * unstable + zstab * stable


def f_h_louis(pzu, pRib, pChn, pz0):
    """Stability correction for heat (Louis 1979)."""
    zstab = jnp.where(pRib >= 0, 1.0, 0.0)
    ztu = pRib / (1.0 + 3.0 * _rc2_louis * pChn * jnp.sqrt(jnp.abs(-pRib * (pzu / pz0 + 1.0))))
    zts = pRib / jnp.sqrt(jnp.abs(1.0 + pRib))
    unstable = 1.0 - _rah_louis * ztu
    stable = 1.0 / (1.0 + _rah_louis * zts)
    return (1.0 - zstab) * unstable + zstab * stable


def e_air(pqa, pslp):
    """Vapor pressure of air [Pa] from specific humidity and pressure.

    Iterative solver.
    """
    e_old = pqa * pslp / reps0
    for _ in range(20):
        ee = pqa / reps0 * (pslp - (1.0 - reps0) * e_old)
        e_old = ee
    return ee


def rh_air(pqa, pTa, pslp):
    """Relative humidity [%] from specific humidity, temperature, and pressure."""
    return 100.0 * e_air(pqa, pslp) / e_sat(pTa)


def q_sat_crude(pts, prhoa):
    """Crude estimate of saturation specific humidity."""
    return 640380.0 / prhoa * jnp.exp(-5107.4 / pts)


def bulk_formula(pzu, pts, pqs, pThta, pqa, pCd, pCh, pCe, pwnd, pUb, pslp, l_ice=False):
    """Core bulk formula computation.

    Returns (tau, Qsen, Qlat, Evap, rho_a).
    """
    zta = pThta - rgamma_dry * pzu
    zrho = rho_air(zta, pqa, pslp)
    zrho = rho_air(zta, pqa, pslp - zrho * grav * pzu)

    zUrho = pUb * jnp.maximum(zrho, 1.0)

    pTau = zUrho * pCd * pwnd

    zevap = zUrho * pCe * (pqa - pqs)
    pQsen = zUrho * pCh * (pThta - pts) * cp_air(pqa)

    pQlat = jnp.where(l_ice, rLsub * zevap, l_vap(pts) * zevap)
    pEvap = jnp.where(l_ice, jnp.minimum(zevap, 0.0), zevap)

    return pTau, pQsen, pQlat, pEvap, zrho


def update_qnsol_tau(pzu, pts, pqs, pThta, pqa, pust, ptst, pqst, pwnd, pUb, pslp, prlw, l_ice=False):
    """Compute non-solar heat flux and wind stress from turbulent scales."""
    zdt = pThta - pts
    zdt = jnp.sign(zdt) * jnp.maximum(jnp.abs(zdt), 1.0e-9)
    zdq = pqa - pqs
    zdq = jnp.sign(zdq) * jnp.maximum(jnp.abs(zdq), 1.0e-12)

    zz0 = pust / pUb
    zCd = zz0 * zz0
    zCh = zz0 * ptst / zdt
    zCe = zz0 * pqst / zdq

    tau, Qsen, Qlat, _, _ = bulk_formula(
        pzu, pts, pqs, pThta, pqa, zCd, zCh, zCe, pwnd, pUb, pslp, l_ice=l_ice
    )
    Qlw = qlw_net(prlw, pts, l_ice=l_ice)
    Qns = Qlat + Qsen + Qlw
    return Qns, tau, Qlat


def delta_skin_layer(palpha, pQd, pustar_a, Qlat=None):
    """Thickness of the viscous cool-skin layer [m] (Fairall et al. 1996)."""
    zQd = pQd
    if Qlat is not None:
        zQd = pQd + 0.026 * jnp.minimum(Qlat, 0.0) * rCp0_w / rLevap / palpha

    ztf = jnp.where(zQd > 0, 1.0, 0.0)

    zusw = jnp.maximum(pustar_a, 1.0e-4) * sq_radrw
    zusw2 = zusw * zusw

    zlamb = 6.0 * (1.0 + jnp.maximum(palpha * rcst_cs / (zusw2**2) * zQd, 0.0) ** 0.75) ** (-1.0 / 3.0)

    ztmp = rnu0_w / zusw
    return (1.0 - ztf) * zlamb * ztmp + ztf * jnp.minimum(6.0 * ztmp, 0.007)


def type_of_humidity(Xval):
    """Auto-detect humidity type from value ranges.

    Returns 'sh' (specific), 'rh' (relative), or 'dp' (dew-point).
    """
    zmean = jnp.mean(Xval)
    zmin = jnp.min(Xval)
    zmax = jnp.max(Xval)

    is_sh = (zmean >= ref_sha_min) & (zmean < ref_sha_max) & (zmin >= ref_sha_min) & (zmax < ref_sha_max)
    is_dp = (zmean >= ref_dpt_min) & (zmean < ref_dpt_max) & (zmin >= ref_dpt_min) & (zmax < ref_dpt_max)
    is_rh = (zmean >= ref_rlh_min) & (zmean <= ref_rlh_max) & (zmin >= ref_rlh_min) & (zmax <= ref_rlh_max)

    if is_sh:
        return "sh"
    elif is_dp:
        return "dp"
    elif is_rh:
        return "rh"
    else:
        raise ValueError(
            f"Cannot identify humidity type: mean={float(zmean):.4f}, "
            f"min={float(zmin):.4f}, max={float(zmax):.4f}"
        )


def z0tq_lkb(iflag, pRer, pz0):
    """Scalar roughness length for temperature or humidity using Liu-Katsaros-Businger (1979).

    Parameters
    ----------
    iflag : int
        1 for temperature (z0t), 2 for humidity (z0q)
    pRer : array
        Roughness Reynolds number [z0*u*/nu]
    pz0 : array
        Momentum roughness length [m]
    """
    XA = jnp.array([
        [0.177, 0.292],
        [1.376, 1.808],
        [1.026, 1.393],
        [1.625, 1.956],
        [4.661, 4.994],
        [34.904, 30.709],
        [1667.19, 1448.68],
        [5.88e5, 2.98e5],
    ])
    XB = jnp.array([
        [0.0, 0.0],
        [0.929, 0.826],
        [-0.599, -0.528],
        [-1.018, -0.870],
        [-1.475, -1.297],
        [-2.067, -1.845],
        [-2.907, -2.682],
        [-3.935, -3.616],
    ])
    XRAN = jnp.array([0.0, 0.11, 0.825, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0])

    idx = iflag - 1
    a_col = XA[:, idx]
    b_col = XB[:, idx]

    result = jnp.full_like(pRer, -999.0)
    in_range = (pRer > 0) & (pRer < 1000.0)

    jm = jnp.ones_like(pRer, dtype=jnp.int32)
    for i in range(8):
        found = (pRer > XRAN[i]) & (pRer <= XRAN[i + 1])
        jm = jnp.where(found & in_range, i, jm)

    a_val = a_col[jm]
    b_val = b_col[jm]
    z0tq = a_val * pRer**b_val * pz0 / jnp.maximum(pRer, 1.0e-30)

    result = jnp.where(in_range, z0tq, result)
    return jnp.minimum(jnp.maximum(jnp.abs(result), 1.0e-9), 0.05)
