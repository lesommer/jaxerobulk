"""High-level API for JaxeroBulk.

Provides a unified interface matching the FORTRAN AEROBULK_MODEL subroutine.
"""

from enum import Enum

import jax.numpy as jnp

from jaxerobulk.constants import rdct_qsat_salt, Cx_min
from jaxerobulk.thermodynamics import q_sat, bulk_formula, type_of_humidity
from jaxerobulk.ncar import turb_ncar
from jaxerobulk.coare import turb_coare3p0, turb_coare3p6
from jaxerobulk.ecmwf import turb_ecmwf
from jaxerobulk.andreas import turb_andreas


class Algorithm(Enum):
    COARE3P0 = "coare3p0"
    COARE3P6 = "coare3p6"
    NCAR = "ncar"
    ECMWF = "ecmwf"
    ANDREAS = "andreas"


_ALGORITHM_MAP = {
    Algorithm.COARE3P0: turb_coare3p0,
    Algorithm.COARE3P6: turb_coare3p6,
    Algorithm.NCAR: turb_ncar,
    Algorithm.ECMWF: turb_ecmwf,
    Algorithm.ANDREAS: turb_andreas,
}


def aerobulk_compute(
    zt: float,
    zu: float,
    sst,
    t_zt,
    hum_zt,
    U_zu,
    V_zu,
    slp,
    algo: Algorithm,
    hum_type: str = None,
    nb_iter: int = 5,
    l_ice: bool = False,
):
    """Compute bulk transfer coefficients and adjust T/q to wind height.

    Parameters
    ----------
    zt : float
        Height for temperature and humidity [m]
    zu : float
        Height for wind speed [m]
    sst : array
        Sea surface temperature [K]
    t_zt : array
        Potential air temperature at zt [K]
    hum_zt : array
        Humidity at zt (specific, relative, or dew-point)
    U_zu : array
        Zonal wind at zu [m/s]
    V_zu : array
        Meridional wind at zu [m/s]
    slp : array
        Sea-level pressure [Pa]
    algo : Algorithm
        Bulk algorithm to use
    hum_type : str, optional
        Humidity type: 'sh' (specific), 'rh' (relative), 'dp' (dew-point).
        If None, auto-detected.
    nb_iter : int
        Number of iterations (default 5)
    l_ice : bool
        Whether over ice (affects q_sat calculation)

    Returns
    -------
    dict with keys: Cd, Ch, Ce, t_zu, q_zu, Ubzu, u_star, tau, Qlat, Qsen, Evap
    """
    if hum_type is None:
        hum_type = type_of_humidity(hum_zt)

    if hum_type == "sh":
        q_zt = hum_zt
    elif hum_type == "rh":
        q_zt = q_sat(t_zt, slp) * hum_zt / 100.0
    elif hum_type == "dp":
        from jaxerobulk.thermodynamics import q_air_dp

        q_zt = q_air_dp(hum_zt, slp)
    else:
        raise ValueError(f"Unknown humidity type: {hum_type}")

    q_s = rdct_qsat_salt * q_sat(sst, slp, l_ice=l_ice)

    W_zu = jnp.sqrt(U_zu**2 + V_zu**2)

    turb_func = _ALGORITHM_MAP[algo]
    result = turb_func(zt, zu, sst, t_zt, q_s, q_zt, W_zu, nb_iter=nb_iter)

    Cd = result["Cd"]
    Ch = result["Ch"]
    Ce = result["Ce"]
    Ubzu = result["Ubzu"]

    tau_mag, Qsen, Qlat, Evap, rho_a = bulk_formula(
        zu, sst, q_s, result["t_zu"], result["q_zu"], Cd, Ch, Ce, W_zu, Ubzu, slp, l_ice=l_ice
    )

    tau_x = tau_mag / jnp.maximum(W_zu, 1e-10) * U_zu
    tau_y = tau_mag / jnp.maximum(W_zu, 1e-10) * V_zu

    return {
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "t_zu": result["t_zu"],
        "q_zu": result["q_zu"],
        "Ubzu": Ubzu,
        "u_star": result["u_star"],
        "tau_x": tau_x,
        "tau_y": tau_y,
        "Qlat": Qlat,
        "Qsen": Qsen,
        "Evap": Evap,
        "rho_air": rho_a,
        "L": result.get("L"),
        "z0": result.get("z0"),
        "UN10": result.get("UN10"),
    }


def aerobulk_model(
    zt: float,
    zu: float,
    sst,
    t_zt,
    hum_zt,
    U_zu,
    V_zu,
    slp,
    algo: str,
    hum_type: str = None,
    nb_iter: int = 5,
    l_ice: bool = False,
):
    """High-level API matching FORTRAN AEROBULK_MODEL.

    Parameters
    ----------
    algo : str
        Algorithm name: 'coare3p0', 'coare3p6', 'ncar', 'ecmwf', 'andreas'
    """
    algo_enum = Algorithm(algo.lower())
    return aerobulk_compute(
        zt, zu, sst, t_zt, hum_zt, U_zu, V_zu, slp, algo_enum, hum_type, nb_iter, l_ice
    )
