# JaxeroBulk Project Status

## Last Updated: 2026-05-21

## Current State

Iterations 1-7 completed. All algorithms, cool-skin/warm-layer, high-level API, CLI, and comprehensive differentiability are implemented and tested.

## Implementation Progress

| Iteration | Description | Status |
|-----------|-------------|--------|
| 1 | Project scaffolding, constants, core thermodynamics | **Complete** |
| 2 | Stability functions, NCAR algorithm | **Complete** |
| 3 | COARE 3.0 and COARE 3.6 | **Complete** |
| 4 | ECMWF algorithm, Andreas, high-level API | **Complete** |
| 5 | Cool-skin and warm-layer | **Complete** |
| 6 | Andreas algorithm, high-level API | **Complete** (merged into iter4) |
| 7 | CLI, comprehensive differentiability | **Complete** |

## Module Status

| Module | File | Status | Tests | Differentiable |
|--------|------|--------|-------|---------------|
| Constants | `constants.py` | Complete | Indirect | N/A |
| Thermodynamics | `thermodynamics.py` | Complete | 61 passing | Forward + Reverse |
| Stability | `stability.py` | Complete | 51 passing (incl. NCAR) | Forward + Reverse |
| First guess | `first_guess.py` | Complete | Included in NCAR tests | Partial |
| NCAR | `ncar.py` | Complete | 51 passing | Forward + Reverse |
| COARE | `coare.py` | Complete | 19 passing | Forward + Reverse (incl. CSWL) |
| ECMWF | `ecmwf.py` | Complete | 12 passing | Forward + Reverse (incl. CSWL) |
| Andreas | `andreas.py` | Complete | 12 passing | Forward + Reverse |
| API | `api.py` | Complete | Included | Forward + Reverse |
| Cool-skin COARE | `skin_coare.py` | Complete | 27 passing (CSWL) | Forward + Reverse |
| Cool-skin ECMWF | `skin_ecmwf.py` | Complete | 27 passing (CSWL) | Forward + Reverse |
| CLI | `cli.py` | Complete | 8 passing | N/A |

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

### coare.py
- `charn_coare3p0`, `charn_coare3p6` — Charnock parameters
- `turb_coare3p0`, `turb_coare3p6` — full COARE bulk algorithms with optional CSWL

### ecmwf.py
- `turb_ecmwf` — ECMWF bulk algorithm with optional CSWL

### andreas.py
- `turb_andreas` — Andreas bulk algorithm for sea ice

### skin_coare.py
- `cs_coare` — cool-skin parameterization (Fairall et al. 1996, COARE version, uses Qlat)
- `wl_coare` — warm-layer scheme (Fairall et al. 2019, COARE 3.6)

### skin_ecmwf.py
- `cs_ecmwf` — cool-skin parameterization (Fairall et al. 1996, ECMWF version, 0.065 solar absorption)
- `wl_ecmwf` — warm-layer scheme (Zeng & Beljaars 2005, fixed depth)
- `_phi_ecmwf` — stability function PHI (Takaya et al. 2010)

### api.py
- `aerobulk_model`, `aerobulk_compute` — high-level API matching FORTRAN AEROBULK_MODEL
- `Algorithm` — enum for algorithm selection
- Full CSWL support via `l_use_skin`, `rad_sw`, `rad_lw` parameters
- Warm-layer state management via `wl_state` dict

### cli.py
- `aerobulk-toy` — interactive single-point exploration with all algorithm options
- `aerobulk-compare` — cross-algorithm comparison table at a single point
- `aerobulk-compute` — batch computation from CSV forcing files

## Differentiability

All thermodynamic and stability functions are differentiable in both forward and reverse mode.
All bulk algorithms (NCAR, COARE 3.0/3.6, ECMWF, Andreas) are differentiable w.r.t. SST, wind speed, and air temperature.
CSWL parameterizations are differentiable using `jax.custom_jvp` for gradient-safe `sqrt(x) where x > 0` pattern.
COARE/ECMWF with CSWL enabled are differentiable w.r.t. SST (verified).
Forward/reverse gradient consistency verified for all algorithms (rtol < 1e-5).
Gradient magnitudes are physically reasonable (dCd/dSST < 1e-3).
Full Jacobian computation verified via `jax.jacfwd` for all algorithms.

## Known Issues

- `e_air` uses a fixed 20-iteration loop instead of the FORTRAN convergence check
  (needed for JAX compatibility). Accuracy is sufficient for typical inputs.
- COARE/ECMWF stable psi functions have a small offset at zeta=0 (by design, from
  the Beljaars-Holtslag formulation with constant 8.525)
- `wl_coare` uses `jnp.maximum(zqac, 0.0)**1.5` instead of `zqac**1.5 * sign(zqac)` to
  avoid NaN gradients when zqac=0. This means the warm-layer dT is always non-negative,
  consistent with the physical model.
