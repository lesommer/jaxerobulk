# JaxeroBulk Project Status

## Last Updated: 2026-05-21

## Current State

Iterations 1-2 completed. Core thermodynamics, stability functions, and NCAR algorithm are implemented and tested.

## Implementation Progress

| Iteration | Description | Status |
|-----------|-------------|--------|
| 1 | Project scaffolding, constants, core thermodynamics | **Complete** |
| 2 | Stability functions, NCAR algorithm | **Complete** |
| 3 | COARE 3.0 and COARE 3.6 | Pending |
| 4 | ECMWF algorithm | Pending |
| 5 | Cool-skin and warm-layer | Pending |
| 6 | Andreas algorithm, high-level API | Pending |
| 7 | CLI, comprehensive differentiability | Pending |

## Module Status

| Module | File | Status | Tests | Differentiable |
|--------|------|--------|-------|---------------|
| Constants | `constants.py` | Complete | Indirect | N/A |
| Thermodynamics | `thermodynamics.py` | Complete | 61 passing | Forward + Reverse |
| Stability | `stability.py` | Complete | 51 passing (incl. NCAR) | Forward + Reverse |
| First guess | `first_guess.py` | Complete | Included in NCAR tests | Partial |
| NCAR | `ncar.py` | Complete | 51 passing | Forward + Reverse |

## Functions Implemented

### constants.py
All physical constants from `mod_const.f90`.

### thermodynamics.py
- `pot_temp`, `abs_temp`, `virt_temp` — temperature conversions
- `e_sat`, `e_sat_ice`, `de_sat_dt_ice` — saturation vapor pressure (Goff 1957)
- `q_sat`, `dq_sat_dt_ice` — saturation specific humidity
- `q_air_rh`, `q_air_dp` — humidity conversions
- `rho_air` — air density
- `visc_air` — kinematic viscosity of air
- `l_vap`, `cp_air` — thermodynamic properties
- `gamma_moist` — moist adiabatic lapse rate
- `one_on_L` — inverse Obukhov length
- `ri_bulk` — bulk Richardson number
- `Pz_from_P0_tz_qz` — pressure at height z
- `theta_from_z_p0_t_q`, `T_from_z_p0_theta_q` — potential/absolute temperature conversion at height
- `alpha_sw` — thermal expansion of seawater
- `qlw_net` — net longwave flux
- `z0_from_cd`, `z0_from_ustar` — roughness length utilities
- `f_m_louis`, `f_h_louis` — Louis (1979) stability corrections
- `e_air`, `rh_air` — vapor pressure / relative humidity from specific humidity
- `bulk_formula` — core bulk flux computation
- `update_qnsol_tau` — non-solar heat flux and wind stress from turbulent scales
- `delta_skin_layer` — cool-skin thickness (Fairall 1996)
- `type_of_humidity` — auto-detect humidity type
- `z0tq_lkb` — Liu-Katsaros-Businger scalar roughness

### stability.py
- `psi_m_coare`, `psi_h_coare` — COARE 3.0/3.6 stability functions
- `psi_m_ncar`, `psi_h_ncar` — NCAR stability functions (Paulson + -5z)
- `psi_m_ecmwf`, `psi_h_ecmwf` — ECMWF stability functions
- `psi_m_andreas`, `psi_h_andreas` — Andreas stability functions (Grachev et al. 2007 stable)

### first_guess.py
- `first_guess_coare` — COARE first guess of u*, theta*, q* (shared by COARE and ECMWF)

### ncar.py
- `cd_n10_ncar` — neutral drag coefficient (L&Y 2008 Eq. 11)
- `ch_n10_ncar` — neutral sensible heat coefficient
- `ce_n10_ncar` — neutral evaporation coefficient
- `turb_ncar` — full NCAR bulk algorithm iteration

## Differentiability

All thermodynamic and stability functions are differentiable in both forward and reverse mode.
NCAR `turb_ncar` is differentiable w.r.t. SST and wind speed (verified).
Forward/reverse Jacobian consistency checked for all stability functions.

## Known Issues

- `e_air` uses a fixed 20-iteration loop instead of the FORTRAN convergence check
  (needed for JAX compatibility). Accuracy is sufficient for typical inputs.
- COARE/ECMWF stable psi functions have a small offset at zeta=0 (by design, from
  the Beljaars-Holtslag formulation with constant 8.525)
