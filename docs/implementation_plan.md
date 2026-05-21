# Implementation Plan for JaxeroBulk

## Strategy

Port AeroBulk from FORTRAN90 to Python/JAX/Equinox bottom-up, following the
dependency graph of the original modules. Each iteration produces a pip-installable,
test-backed, functional package. All code is written to be JAX-differentiable
from the start (no `if` on traced values, pure functions, `jax.lax` control flow).

Reference implementation: `../aerobulk/` (tag 1.0.0).

---

## Iteration 1 — Project Scaffolding, Constants, Core Thermodynamics

**Branch**: `iter1-thermo`

**Modules to implement**:
- `src/jaxerobulk/__init__.py`
- `src/jaxerobulk/constants.py` — all physical constants from `mod_const.f90`
- `src/jaxerobulk/thermodynamics.py` — all public functions from `mod_phymbl.f90`:
  - `pot_temp`, `abs_temp`, `virt_temp`
  - `pressure_at_z`, `theta_from_z_p0_t_q`, `t_from_z_p0_theta_q`
  - `rho_air`, `visc_air`, `l_vap`, `cp_air`, `gamma_moist`
  - `one_on_L` (inverse Obukhov length)
  - `ri_bulk` (bulk Richardson number)
  - `e_sat`, `e_sat_ice`, `q_sat`, `dq_sat_dt_ice`
  - `q_air_rh`, `q_air_dp`
  - `rho_air`, `e_air`, `rh_air`
  - `bulk_formula` (the core flux computation)
  - `alpha_sw`, `qlw_net`
  - `z0_from_cd`, `z0_from_ustar`
  - `type_of_humidity`, `check_unit_consistency`
  - `delta_skin_layer`

**Tests**:
- Unit tests for each thermodynamic function against FORTRAN reference values
- Test `bulk_formula` with known inputs/outputs
- Test humidity conversion roundtrips
- Cross-validation: `e_sat` → `q_sat` → `q_air_rh` → `rh_air` consistency
- Finite-difference verification of `dq_sat_dt_ice` against analytical

**Deliverables**:
- `pyproject.toml` with JAX, Equinox, scipy dependencies
- pip-installable package
- `pytest -q` passes
- Updated README.md

---

## Iteration 2 — Stability Functions and NCAR Algorithm

**Branch**: `iter2-ncar`

**Modules to implement**:
- `src/jaxerobulk/stability.py` — stability correction functions:
  - `psi_m_coare`, `psi_h_coare` (from `mod_common_coare.f90`)
  - `psi_m_ncar`, `psi_h_ncar`
  - `psi_m_ecmwf`, `psi_h_ecmwf`
  - `psi_m_andreas`, `psi_h_andreas`
- `src/jaxerobulk/roughness.py` — roughness length utilities:
  - `z0tq_lkb` (Liu-Katsaros-Businger)
  - `f_m_louis`, `f_h_louis` (Louis 1979 stability)
- `src/jaxerobulk/ncar.py` — NCAR bulk algorithm:
  - `cd_n10_ncar`, `ch_n10_ncar`, `ce_n10_ncar`
  - `turb_ncar` (full iteration loop)
- `src/jaxerobulk/first_guess.py` — COARE first guess routine:
  - `first_guess_coare`

**Tests**:
- Test all ψ_m, ψ_h against FORTRAN-computed values on a ζ grid (reference from `test_psi_stab`)
- Test NCAR neutral coefficients against FORTRAN sweep
- Test `turb_ncar` with hardcoded SST/T_air/wind inputs from `example_call_aerobulk`
- Differentiability: `jax.grad` through `turb_ncar` for Cd, Ch, Ce w.r.t. SST, T_air, wind

**Deliverables**:
- pip-installable package with NCAR algorithm functional
- `pytest -q` passes
- Updated README.md and project_status.md

---

## Iteration 3 — COARE 3.0 and COARE 3.6 Algorithms

**Branch**: `iter3-coare`

**Modules to implement**:
- `src/jaxerobulk/coare.py` — COARE algorithms:
  - `charn_coare3p0`, `charn_coare3p6`
  - `turb_coare3p0` (iteration loop, without CSWL)
  - `turb_coare3p6` (iteration loop, without CSWL)
  - `charn_coare3p6_wave` (wave-dependent Charnock, optional)
- `src/jaxerobulk/neutral.py` — neutral 10m coefficient computation:
  - `turb_neutral_10m`

**Tests**:
- Test COARE 3.0/3.6 transfer coefficients against FORTRAN reference
- Test Charnock parameter values at key wind speeds
- Compare COARE and NCAR under identical conditions (cross-algorithm sanity)
- Test `turb_neutral_10m` sweeps against FORTRAN `test_coef_n10` output
- Differentiability tests for COARE routines

**Deliverables**:
- pip-installable with NCAR + COARE 3.0 + COARE 3.6
- `pytest -q` passes
- Updated README.md and project_status.md

---

## Iteration 4 — ECMWF Algorithm

**Branch**: `iter4-ecmwf`

**Modules to implement**:
- `src/jaxerobulk/ecmwf.py`:
  - `turb_ecmwf` (iteration loop, without CSWL)
  - Profile-based transfer coefficient computation
  - ζ capping logic

**Tests**:
- Test ECMWF transfer coefficients against FORTRAN reference
- Cross-algorithm comparison (ECMWF vs COARE vs NCAR)
- Test bulk Richardson number stability initialization
- Differentiability tests

**Deliverables**:
- pip-installable with NCAR + COARE 3.0/3.6 + ECMWF (without CSWL)
- `pytest -q` passes
- Updated README.md and project_status.md

---

## Iteration 5 — Cool-Skin and Warm-Layer

**Branch**: `iter5-cswl`

**Modules to implement**:
- `src/jaxerobulk/skin_coare.py`:
  - `cs_coare` (cool-skin)
  - `wl_coare` (warm-layer, with prognostic state)
- `src/jaxerobulk/skin_ecmwf.py`:
  - `cs_ecmwf` (cool-skin)
  - `wl_ecmwf` (warm-layer, fixed depth, Zeng & Beljaars)
- Update COARE 3.0/3.6/ECMWF to integrate CSWL when enabled
- Equinox state management for warm-layer prognostic variables

**Tests**:
- Test cool-skin ΔT against FORTRAN reference values
- Test warm-layer evolution over a diurnal cycle
- Test full COARE+CSWL and ECMWF+CSWL flux computation
- Test `aerobulk_model` with `l_use_skin=True`
- Differentiability through cool-skin and warm-layer

**Deliverables**:
- pip-installable with all ocean algorithms + CSWL
- `pytest -q` passes
- Updated README.md and project_status.md

---

## Iteration 6 — Andreas Algorithm and High-Level API

**Branch**: `iter6-api`

**Modules to implement**:
- `src/jaxerobulk/andreas.py`:
  - `u_star_andreas`
  - `turb_andreas` (iteration loop)
- `src/jaxerobulk/api.py` — high-level API mirroring `mod_aerobulk.f90`:
  - `aerobulk_init` — input validation, humidity detection
  - `aerobulk_model` — main computation entry point
  - `aerobulk_bye` — cleanup (may be no-op in JAX)
  - `aerobulk_compute` — dispatch to algorithm

**Tests**:
- Test Andreas against FORTRAN reference
- Test `aerobulk_model` for all 5 algorithms with example_call_aerobulk inputs
- Test humidity auto-detection
- Test input validation
- Full differentiability verification for `aerobulk_model`

**Deliverables**:
- Complete API matching FORTRAN `AEROBULK_MODEL` interface
- pip-installable with all 5 algorithms
- `pytest -q` passes
- Updated README.md and project_status.md

---

## Iteration 7 — CLI and Comprehensive Differentiability

**Branch**: `iter7-cli-diff`

**Modules to implement**:
- `src/jaxerobulk/cli.py` — CLI replicating FORTRAN binaries:
  - `aerobulk-toy` — interactive single-point exploration
  - `aerobulk-compute` — batch computation from NetCDF forcing
  - `aerobulk-compare` — cross-algorithm comparison
- Comprehensive differentiability test suite:
  - Forward-mode (`jax.jacfwd`) for all algorithms
  - Reverse-mode (`jax.grad`, `jax.jacrev`) for all algorithms
  - Gradient consistency checks (forward vs reverse)
  - Gradient magnitude sanity checks

**Tests**:
- CLI smoke tests
- Full differentiability matrix: algorithm × input_variable × mode
- Regression test: compare JaxeroBulk output to FORTRAN reference for
  the PAPA station forcing (if available) or the idealized test cases
- Plot generation using JaxeroBulk diagnostic scripts

**Deliverables**:
- Full CLI surface
- Proven differentiability for all routines
- pip-installable, fully functional package
- Updated README.md with usage, CLI reference, algorithm description
- Updated project_status.md with differentiability status
- Plots generated by JaxeroBulk diagnostic scripts in README.md

---

## Cross-Iteration Requirements

At the end of every iteration:
1. `pytest -q` passes
2. Package is `pip install .`-able
3. `README.md` is current (description, usage, algorithm, status)
4. `docs/project_status.md` is updated (including differentiability status)
5. Branch is merged to main after verification
6. All changes committed and pushed to remote

## Differentiability Strategy

All functions are written to be JAX-differentiable from day one:
- Pure functions (no side effects, no mutable state)
- Use `jax.lax.cond`, `jax.lax.select` instead of Python `if` on traced values
- Use `jax.lax.while_loop` or bounded loops instead of Python `while`
- Warm-layer state managed via Equinox modules, not global arrays
- All iteration loops have a fixed maximum number of iterations
- No `print`, file I/O, or other side effects inside traced code
