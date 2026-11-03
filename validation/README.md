# External-reference validation: the eORCA1 CORE-forced run

This directory holds **Gate A** of the neutral-bench validation chain: an
external-reference validation of jaxerobulk's CORE/NCAR bulk formulae
against a NEMO run that was *forced with them* — the millennial eORCA1
record of de Lavergne et al. (2026, *JAMES* 18(2),
doi:10.1029/2024MS004824; Zenodo record 14041098, CC-BY-4.0). Its TRA
restart (year-1001 end state, 1000-12-31) carries the model's own
surface fluxes (`qns_b`, `emp_b`, `utau_b`, `vtau_b`) computed by NEMO
4.2's `sbcblk` with `ln_NCAR` from the CORE.v2 normal-year (CNYF)
forcing; `validation/eorca1_bulk.py` recomputes them from the same
forcing and the model's own SST through jaxerobulk and compares.

No data is committed to this repository. Both datasets are external:

| Dataset | Source | Licence / citation |
| ------- | ------ | ------------------ |
| eORCA1 record (`data_repository`) | zenodo.org/records/14041098 (~1.34 GB zip; unpack to `EORCA1_DIR`) | CC-BY-4.0 — cite de Lavergne et al. (2026), doi:10.1029/2024MS004824 |
| CORE.v2 normal-year forcing (CNYF) | `ftp://data1.gfdl.noaa.gov/1/users/Niki.Zadeh/COREv2/data_1948_2006/NormalYearForcing/CORRECTED` (files `t_10/q_10/u_10/v_10/slp/ncar_rad/ncar_precip .15JUNE2009.nc`) | cite Large & Yeager (2004, 2009) and the CORE.v2 dataset documentation |

## Run configuration (read off the record's namelists)

- bulk: NEMO 4.2 generic `sbcblk` with `ln_NCAR` (the CORE/NCAR
  formulation of Large & Yeager — the legacy `sbc_core` was merged into
  `sbcblk` in 4.2); reference `rn_zqt = rn_zu = 10 m`.
- calendar: noleap (`nn_leapy=0`; the TRA segment is 11680 steps x
  2700 s = exactly 365 days), so the restart snapshot falls on forcing
  day 365 of the CNYF year — no calendar phase margin.
- all `sn_*` forcings have time interpolation OFF: the model consumed
  literal records — the 6-hourly state fields at Dec 31 03/09/15/21 UTC
  (the 21 UTC record covers the final timesteps), the daily radiation of
  day 365, the December record of the monthly precipitation fields.
- winds remapped with masked **bicubic** weights and rotated to the grid
  directions; the other fields bilinearly (the probe samples bilinearly
  by default, bicubically in variant V4; rotation is negligible south of
  the north fold).
- `qns_b` (NEMO 4.2 `blk_oce_2`) = net longwave (Kirchhoff form,
  emissivity 0.98) + sensible + latent − snow·rLfus − evap·rcp·SST +
  rain·rcp·Tair + snow·rcpi·min(Tair, 0 °C).
- `emp_b` = (evap − precip) − runoff + SSS-restoring erp + ice
  freshwater − the `nn_fwb=1` global-mean correction. The run's runoff
  (`runoff-icb_DaiTrenberth_Depoorter.nc`) and restoring climatology are
  not part of the Zenodo record — emp is compared in anomaly form
  (variant E1) and stays a pattern-level diagnostic.
- the run's forcing files are `*_CORE2_fill` variants, not shipped with
  the record; the probe uses the GFDL CNYF originals (documented
  assumption: same fields over open ocean).

## The variant matrix (reference run, 2026-11-03)

The comparison climbs from a naive protocol to the matched protocol step
by step, so every residual term is attributed. n = 50,596 ice-free
open-water cells (`sst_m > −1 °C`, |lat| ≤ 64°):

| Variant | qns bias (W/m²) | qns rms | qns corr | utau corr | vtau corr |
| ------- | -------------- | ------- | -------- | --------- | --------- |
| V0 day-mean inputs, 3-term qns (naive) | +12.13 | 50.0 | 0.9233 | 0.7883 | 0.7724 |
| V1 mean of per-record fluxes | +5.17 | 41.1 | 0.9465 | 0.8347 | 0.8252 |
| V2 last 6-h record (Dec 31 21 UTC) | +2.73 | 12.8 | 0.9954 | 0.9922 | 0.9898 |
| **V3 = V2 + NEMO heat-content terms (matched)** | **+1.92** | **12.0** | **0.9956** | 0.9922 | 0.9898 |
| V4 = V3 + bicubic wind | −1.87 | 12.0 | 0.9956 | 0.9926 | 0.9896 |
| V5 = V3, strict mask (`sst_m > 0.5 °C`) | +1.78 | 10.4 | 0.9968 | 0.9924 | 0.9899 |
| E1 emp anomaly-form (fwb proxy) | — | 3.4e-5 | 0.7433 | — | — |

Notes: the old committed probe's +12.13 W/m² qns bias (V0) is dominated
by the averaging protocol (V0→V1: +12.13 → +5.17), then by comparing a
day-mean against the instantaneous restart field (V1→V2: +5.17 →
+2.73, rms 41 → 12.8); the NEMO heat-content terms close another 0.8
W/m² (V2→V3). The bilinear/bicubic wind remaps bracket the residual mean
(V3 +1.92, V4 −1.87). emp was also repaired on the probe side:
jaxerobulk's `Evap` is downward-positive (NEMO's `pevp` is upward).

Regression of the matched Δqns (V3) on {1, LWDN, σT⁴, |Qlat|, |Qsen|}:
R² = 0.036, residual sigma 11.7 W/m² — no uniform offset signature
(intercept −2.8 W/m²; only ~2–3% turbulent-amplitude loading consistent
with the remap convention).

## Verdict bar

The matched comparison **V3 passes if** |qns bias| ≤ 5 W/m² and
corr ≥ 0.92, with no uniform regression signature.
Reference result: bias **+1.92 W/m²**, corr **0.9956** — **PASS**.
(The bar is intentionally generous: the residual includes the
bilinear-vs-bicubic remap bracket and the `CORE2_fill` vs CNYF lineage
assumption, not just bulk-formula physics.)

## Running

Script (full matrix, maps + JSON to `--out-dir`):

```bash
uv run --extra validation python validation/eorca1_bulk.py \
    --eorca1-dir <EORCA1_DIR> --cnyf-dir <CNYF_DIR> --out-dir eorca1-out
```

Gated test (`tests/test_validation_eorca1.py`): skips unless
`JAXEROBULK_EORCA1_DIR` and `JAXEROBULK_CNYF_DIR` are set; asserts the
verdict bar on a fresh run and compares against
`validation/reference_results.json`. `pytest -q` stays green without
the data.

## Provenance

The protocol and the variant decomposition were developed in the
neutral-bench reproduction package
(`2026_variationnal-neutral-density-repro`, commit `36cf6d3`), against
whose committed run the reference numbers above were produced
(jaxerobulk rev `a750051`). The NEMO 4.2 conventions cited above were
read from the release sources (forge.nemo-ocean.eu, tag `4.2.0`:
`sbcblk.F90`, `sbc_phy.F90`, `phycst.F90`, `eosbn2.F90`,
`sbcblk_algo_ncar.F90`).
