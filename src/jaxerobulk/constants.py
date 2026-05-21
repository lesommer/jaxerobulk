"""Physical constants for AeroBulk / JaxeroBulk.

All values match the FORTRAN reference implementation (mod_const.f90).
"""

import jax.numpy as jnp

grav = 9.8
rpi = 3.141592653589793
two_pi = 2.0 * rpi
to_rad = rpi / 180.0

R_earth = 6.37e6
rtilt_earth = 23.5
Sol0 = 1366.0
roce_alb0 = 0.066
rice_alb0 = 0.8

emiss_w = 0.98
emiss_i = 0.996
stefan = 5.67e-8

rt0 = 273.15
rtt0 = 273.16

rCp0_w = 4190.0
rho0_w = 1025.0
rnu0_w = 1.0e-6
rk0_w = 0.6

rCp0_a = 1015.0
rCp_dry = 1005.0
rCp_vap = 1860.0

R_dry = 287.05
R_vap = 461.495
R_gas = 8.314510

rmm_dryair = 28.9647e-3
rmm_water = 18.0153e-3
rmm_ratio = rmm_water / rmm_dryair

rpoiss_dry = R_dry / rCp_dry
rgamma_dry = grav / rCp_dry

reps0 = R_dry / R_vap
rctv0 = R_vap / R_dry - 1.0

rnu0_air = 1.5e-5

rLevap = 2.46e6
rLsub = 2.834e6

Patm = 101000.0
rho0_a = 1.2

vkarmn = 0.4
vkarmn2 = 0.4 * 0.4
rdct_qsat_salt = 0.98
z0_sea_max = 0.0025

rcst_cs = -16.0 * 9.80665 * rho0_w * rCp0_w * rnu0_w**3 / (rk0_w**2)
radrw = rho0_a / rho0_w
sq_radrw = jnp.sqrt(radrw)

Cx_min = 0.1e-3

rCd_ice = 1.4e-3

ref_sst_min = 270.0
ref_sst_max = 320.0
ref_taa_min = 180.0
ref_taa_max = 330.0
ref_sha_min = 0.0
ref_sha_max = 0.08
ref_dpt_min = 150.0
ref_dpt_max = 330.0
ref_rlh_min = 0.0
ref_rlh_max = 100.0
ref_slp_min = 80000.0
ref_slp_max = 110000.0
ref_wnd_min = 0.0
ref_wnd_max = 50.0
ref_rsw_min = 0.0
ref_rsw_max = 1500.0
ref_rlw_min = 0.0
ref_rlw_max = 750.0
ref_tau_max = 10.0

_rAg_i = -9.09718
_rBg_i = -3.56654
_rCg_i = 0.876793
_rDg_i = jnp.log10(6.1071)

_rc_louis = 5.0
_rc2_louis = _rc_louis**2
_ram_louis = 2.0 * _rc_louis
_rah_louis = 3.0 * _rc_louis

_repsilon = 1.0e-6
