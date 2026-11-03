"""eorca1_bulk.py — external-reference validation of the jaxerobulk CORE/NCAR
chain against the eORCA1 run of de Lavergne et al. (2026).

Gate A of the neutral-density benchmark ecosystem: the millennial eORCA1
record (JAMES 18(2), doi:10.1029/2024MS004824; Zenodo record 14041098,
CC-BY-4.0) was forced with the CORE.v2 normal year (CNYF) through NEMO
4.2's ``sbcblk`` with ``ln_NCAR``. Its TRA restart (year-1001 end state,
1000-12-31) carries the model's own ``qns_b`` / ``emp_b`` / ``utau_b`` /
``vtau_b``; this script recomputes them from the CNYF state fields with
the model's own SST (``sst_m``) through jaxerobulk, and compares.

Protocol facts read off the record's namelists and forcing files (the
run configuration is documented in validation/README.md): noleap
calendar (snapshot = forcing day 365, no phase margin); all time
interpolation OFF, so the model consumed literal records (Dec 31
03/09/15/21 UTC state fields, day-365 daily radiation, December
precipitation); ``qns_b`` includes evaporation/rain/snow heat-content
terms; ``emp_b`` additionally contains runoff, SSS-restoring and ice
freshwater, and the ``nn_fwb=1`` global-mean correction.

The comparison is a variant matrix: it climbs from the naive protocol
(day-mean inputs, three-term qns) to the matched protocol step by step,
so every residual term is attributed:

  V0  day-mean of the last four 6-h inputs, one flux call, three-term
      qns (lw + sen + lat);
  V1  mean of the per-record fluxes (Jensen-safe day mean);
  V2  last 6-h record only (Dec 31 21 UTC) — the inputs NEMO consumed;
  V3  V2 + the NEMO 4.2 precipitation/evaporation heat-content terms —
      the exact matched comparison (the validation target);
  V4  V3 with bicubic wind sampling (NEMO remaps the winds bicubically);
  V5  V3 on a strict ice-margin mask (``sst_m > 0.5 C``);
  E1  emp in anomaly form (``nn_fwb=1`` correction removed by
      area-weighted demeaning).

Verdict bar (see validation/README.md): V3 passes if
|qns bias| <= 5 W/m2 and corr >= 0.92, with no uniform regression
signature (Δqns regressed on {1, LWDN, sigma*T^4, |Qlat|, |Qsen|}).

Usage::

    python validation/eorca1_bulk.py --eorca1-dir <data_repository> \\
        --cnyf-dir <CNYF> [--out-dir runs/eorca1-validation]

Both data directories are external (never committed to this
repository):
  - eorca1-dir: the Zenodo record's ``data_repository`` (domain_cfg.nc
    under input_fields/, TRA_10001231_restart.nc under restart/);
  - cnyf-dir: the GFDL CORE.v2 normal-year forcing files
    (t_10/q_10/u_10/v_10/slp/ncar_rad/ncar_precip .15JUNE2009.nc).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxerobulk.api import aerobulk_model
from jaxerobulk.thermodynamics import qlw_net

SST_FILL = 288.0
LAT_COVERAGE = 88.5  # T62 Gaussian grid edge
N_RECORDS = 4  # Dec 31 03/09/15/21 UTC

# NEMO 4.2 constants of the qns heat-content terms (blk_oce_2, sbcblk.F90
# r4.2.0; rcp from eosbn2.F90, rLfus/rcpi from phycst.F90).
RCP_W = 3991.86795711963  # seawater specific heat [J/kg/K]
RCP_I = 2067.0  # fresh ice specific heat [J/kg/K]
R_LFUS = 0.334e6  # latent heat of fusion [J/kg]
SIGMA = 5.67e-8  # Stefan-Boltzmann [W/m2/K4]

VERDICT_QNS_BIAS = 5.0
VERDICT_QNS_CORR = 0.92


def normalize(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for dim in da.dims:
        lower = dim.lower()
        if lower in ("time", "lat", "lon"):
            rename[dim] = lower
    return da.rename(rename)


def _grid(da: xr.DataArray):
    da = normalize(da)
    da = da.assign_coords(lon=da.lon % 360.0).sortby("lon")
    lat1, lon1 = da.lat.values, da.lon.values
    values = da.values
    if values.ndim == 2:
        values = values[None]
    return lat1, lon1, values  # (nrec, lat, lon)


def sample_records(da: xr.DataArray, lat2: np.ndarray, lon2: np.ndarray
                   ) -> np.ndarray:
    """Bilinear source-grid records -> model points, time axis kept."""
    lat1, lon1, values = _grid(da)
    lon_ext = np.concatenate([lon1, [lon1[0] + 360.0]])
    val_ext = np.concatenate([values, values[..., :1]], axis=-1)
    q_lat = lat2.ravel()
    q_lon = np.mod(lon2.ravel(), 360.0)
    q_lon = np.where(q_lon < lon1[0], q_lon + 360.0, q_lon)
    out = np.empty((val_ext.shape[0], lat2.shape[0], lat2.shape[1]))
    for r in range(val_ext.shape[0]):
        interp = RegularGridInterpolator(
            (lat1, lon_ext), val_ext[r], bounds_error=False, fill_value=np.nan)
        out[r] = interp(np.column_stack([q_lat, q_lon])).reshape(lat2.shape)
    return out


def sample_records_bicubic(da: xr.DataArray, lat2: np.ndarray,
                           lon2: np.ndarray) -> np.ndarray:
    """Bicubic source-grid records -> model points (NEMO-style wind remap)."""
    lat1, lon1, values = _grid(da)
    if lat1[0] > lat1[-1]:
        lat1, values = lat1[::-1], values[:, ::-1]
    lon_ext = np.concatenate([lon1, [lon1[0] + 360.0]])
    val_ext = np.concatenate([values, values[..., :1]], axis=-1)
    q_lat = lat2.ravel()
    q_lon = np.mod(lon2.ravel(), 360.0)
    q_lon = np.where(q_lon < lon1[0], q_lon + 360.0, q_lon)
    out = np.empty((val_ext.shape[0], lat2.shape[0], lat2.shape[1]))
    for r in range(val_ext.shape[0]):
        spl = RectBivariateSpline(lat1, lon_ext, val_ext[r], kx=3, ky=3)
        out[r] = spl.ev(q_lat, q_lon).reshape(lat2.shape)
    return out


def stats(mask: np.ndarray, label: str, ours: np.ndarray,
          model: np.ndarray) -> dict:
    sel = mask & np.isfinite(ours) & np.isfinite(model)
    d = ours[sel] - model[sel]
    return {
        "label": label, "n": int(sel.sum()),
        "bias": float(d.mean()), "rms": float(np.sqrt((d**2).mean())),
        "corr": float(np.corrcoef(ours[sel], model[sel])[0, 1]),
    }


def area_demean(field: np.ndarray, wet: np.ndarray, area: np.ndarray
                ) -> np.ndarray:
    """Area-weighted global demean over the wet domain (nn_fwb=1 proxy)."""
    sel = wet & np.isfinite(field)
    mean = float((field[sel] * area[sel]).sum() / area[sel].sum())
    return field - mean


def _load_cnyf(var: str, path: Path, last: int) -> xr.DataArray:
    with xr.open_dataset(path, decode_times=False) as ds:
        da = ds[var]
        return da.isel({da.dims[0]: slice(-last, None)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eorca1-dir", required=True,
                    help="Zenodo data_repository (domain_cfg.nc, restart)")
    ap.add_argument("--cnyf-dir", required=True,
                    help="GFDL CORE.v2 normal-year forcing directory")
    ap.add_argument("--out-dir", default="eorca1-validation-out",
                    help="where to write maps and JSON (default: cwd)")
    args = ap.parse_args()
    eorca1 = Path(args.eorca1_dir)
    cnyf = Path(args.cnyf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dom = xr.open_dataset(eorca1 / "input_fields" / "domain_cfg.nc")
    lat2 = dom.gphit.squeeze().values
    lon2 = dom.glamt.squeeze().values
    wet2d = dom.bottom_level.squeeze().values > 0
    area = (dom.e1t.squeeze().values * dom.e2t.squeeze().values)
    dom.close()

    rs = xr.open_dataset(eorca1 / "restart" / "TRA_10001231_restart.nc")
    fields = {v: rs[v].squeeze().values for v in
              ("qns_b", "emp_b", "utau_b", "vtau_b", "sst_m")}
    rs.close()
    sst_m = fields["sst_m"]
    open_water = ((sst_m > -1.0) & wet2d & (np.abs(lat2) <= 64.0)
                  & (np.abs(lat2) <= LAT_COVERAGE))
    strict = ((sst_m > 0.5) & wet2d & (np.abs(lat2) <= 64.0)
              & (np.abs(lat2) <= LAT_COVERAGE))
    print(f"model grid {lat2.shape}; wet cells {int(wet2d.sum())}; "
          f"ice-free open-water cells compared: {int(open_water.sum())}; "
          f"strict (sst>0.5): {int(strict.sum())}")

    t10 = _load_cnyf("T_10_MOD", cnyf / "t_10.15JUNE2009.nc", N_RECORDS)
    q10 = _load_cnyf("Q_10_MOD", cnyf / "q_10.15JUNE2009.nc", N_RECORDS)
    u10 = _load_cnyf("U_10_MOD", cnyf / "u_10.15JUNE2009.nc", N_RECORDS)
    v10 = _load_cnyf("V_10_MOD", cnyf / "v_10.15JUNE2009.nc", N_RECORDS)
    slp = _load_cnyf("SLP", cnyf / "slp.15JUNE2009.nc", N_RECORDS)
    lwdn = _load_cnyf("LWDN_MOD", cnyf / "ncar_rad.15JUNE2009.nc", 1)
    prc = _load_cnyf("PRC_MOD", cnyf / "ncar_precip.15JUNE2009.nc", 1)
    snow = _load_cnyf("SNOW", cnyf / "ncar_precip.15JUNE2009.nc", 1)

    def at_model(da):
        return np.nan_to_num(sample_records(da, lat2, lon2), nan=0.0)

    t_s = at_model(t10)
    q_s = at_model(q10)
    u_s = at_model(u10)
    v_s = at_model(v10)
    p_s = at_model(slp)
    lwdn_s = np.nan_to_num(sample_records(lwdn, lat2, lon2)[0], nan=0.0)
    prc_s = np.nan_to_num(sample_records(prc, lat2, lon2)[0], nan=0.0)
    snow_s = np.nan_to_num(sample_records(snow, lat2, lon2)[0], nan=0.0)

    sst_k = np.where(np.isfinite(sst_m), sst_m, 0.0) + 273.15
    sst_k = np.where(open_water, sst_k, SST_FILL)  # land fill, masked after
    sst_k_j = jnp.asarray(sst_k)

    flux = jax.jit(lambda *a: aerobulk_model(10.0, 10.0, *a, algo="ncar",
                                             hum_type="sh"))
    qlw = np.asarray(qlw_net(jnp.asarray(lwdn_s), sst_k_j))
    rain_s = np.maximum(prc_s - snow_s, 0.0)

    def call(r, u_wind=None, v_wind=None):
        res = flux(sst_k_j,
                   jnp.asarray(t_s[r]), jnp.asarray(q_s[r]),
                   jnp.asarray(u_s[r] if u_wind is None else u_wind),
                   jnp.asarray(v_s[r] if v_wind is None else v_wind),
                   jnp.asarray(p_s[r]))
        return {k: np.asarray(res[k]) for k in
                ("Qsen", "Qlat", "tau_x", "tau_y", "Evap")}

    def qns_3term(res):
        return res["Qsen"] + res["Qlat"] + qlw

    def qns_full(res):
        # NEMO 4.2 blk_oce_2, all heat contents relative to 0 C
        evap_up = -res["Evap"]  # NEMO's pevp (positive upward)
        tair_c = t_s[-1] - 273.15
        return (qlw + res["Qsen"] + res["Qlat"]
                - snow_s * R_LFUS
                - evap_up * sst_m * RCP_W
                + rain_s * tair_c * RCP_W
                + snow_s * np.minimum(tair_c, 0.0) * RCP_I)

    # V0: day-mean of inputs, one call (the naive protocol)
    res_v0 = flux(sst_k_j, jnp.asarray(t_s.mean(axis=0)),
                  jnp.asarray(q_s.mean(axis=0)), jnp.asarray(u_s.mean(axis=0)),
                  jnp.asarray(v_s.mean(axis=0)), jnp.asarray(p_s.mean(axis=0)))
    res_v0 = {k: np.asarray(res_v0[k]) for k in
              ("Qsen", "Qlat", "tau_x", "tau_y", "Evap")}
    qns_v0 = qns_3term(res_v0)
    utau_v0, vtau_v0 = res_v0["tau_x"], res_v0["tau_y"]
    evap_up_v0 = -res_v0["Evap"]

    # per-record calls
    rec = [call(r) for r in range(N_RECORDS)]
    qns_v1 = np.mean([qns_3term(r) for r in rec], axis=0)
    utau_v1 = np.mean([r["tau_x"] for r in rec], axis=0)
    vtau_v1 = np.mean([r["tau_y"] for r in rec], axis=0)

    # V2/V3: last record (Dec 31 21 UTC) — what NEMO consumed at the end
    last = rec[-1]
    qns_v2 = qns_3term(last)
    qns_v3 = qns_full(last)
    utau_v2, vtau_v2 = last["tau_x"], last["tau_y"]
    evap_up_v2 = -last["Evap"]

    # V4: bicubic wind sampling, last record
    u_cub = np.nan_to_num(sample_records_bicubic(u10, lat2, lon2)[-1], nan=0.0)
    v_cub = np.nan_to_num(sample_records_bicubic(v10, lat2, lon2)[-1], nan=0.0)
    res_v4 = call(-1, u_wind=u_cub, v_wind=v_cub)
    qns_v4 = qns_full(res_v4)
    utau_v4, vtau_v4 = res_v4["tau_x"], res_v4["tau_y"]

    # emp variants (NEMO convention: emp = evap_up - precip [- runoff ...])
    emp_v2 = evap_up_v2 - prc_s
    emp_anom = area_demean(emp_v2, wet2d, area)
    emp_b_anom = area_demean(fields["emp_b"], wet2d, area)

    def three(mask, qns_ours, utau_ours, vtau_ours, evap_up):
        emp_ours = evap_up - prc_s
        return [
            stats(mask, "qns (W/m2)", qns_ours, fields["qns_b"]),
            stats(mask, "utau (N/m2)", utau_ours, fields["utau_b"]),
            stats(mask, "vtau (N/m2)", vtau_ours, fields["vtau_b"]),
            stats(mask, "emp (kg/m2/s, ours = E - P, model = E - P - R + erp"
                       " + ice - fwb)", emp_ours, fields["emp_b"]),
        ]

    table = {
        "V0 day-mean inputs, 3-term qns (naive protocol)": three(
            open_water, qns_v0, utau_v0, vtau_v0, evap_up_v0),
        "V1 mean of per-record fluxes, 3-term qns": three(
            open_water, qns_v1, utau_v1, vtau_v1, evap_up_v2),
        "V2 last 6h record, 3-term qns": three(
            open_water, qns_v2, utau_v2, vtau_v2, evap_up_v2),
        "V3 = V2 + NEMO heat-content terms (matched)": three(
            open_water, qns_v3, utau_v2, vtau_v2, evap_up_v2),
        "V4 = V3 + bicubic wind": three(
            open_water, qns_v4, utau_v4, vtau_v4, evap_up_v2),
        "V5 = V3, strict ice-margin mask (sst>0.5C)": three(
            strict, qns_v3, utau_v2, vtau_v2, evap_up_v2),
        "E1 = emp anomaly-form (fwb correction proxy)": [
            stats(open_water, "emp anomaly (kg/m2/s)", emp_anom, emp_b_anom)],
    }

    print("\neORCA1 validation — jaxerobulk NCAR on CNYF vs restart:")
    for variant, rows in table.items():
        print(f"  {variant}:")
        for r in rows:
            print(f"     {r['label']}: n={r['n']}, bias={r['bias']:+.4g}, "
                  f"rms={r['rms']:.4g}, corr={r['corr']:.4f}")

    # regression of the matched Δqns (V3)
    sel = open_water & np.isfinite(qns_v3)
    delta = (fields["qns_b"] - qns_v3)[sel]
    X = np.column_stack([np.ones(sel.sum()), lwdn_s[sel],
                         SIGMA * sst_k[sel] ** 4,
                         np.abs(last["Qlat"][sel]), np.abs(last["Qsen"][sel])])
    names = ["1 (intercept)", "LWDN", "sigma*SST^4", "|Qlat|", "|Qsen|"]
    beta, *_ = np.linalg.lstsq(X, delta, rcond=None)
    resid = delta - X @ beta
    dof = sel.sum() - len(beta)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    err = np.sqrt(np.diag(cov))
    r2 = 1.0 - float(resid @ resid) / float(
        (delta - delta.mean()) @ (delta - delta.mean()))
    regression = {
        "target": "qns_b - qns_V3 (matched)",
        "n": int(sel.sum()), "r2": r2, "sigma_est": float(np.sqrt(sigma2)),
        "terms": {nm: {"coef": float(b), "stderr": float(e),
                       "t": float(b / e)}
                  for nm, b, e in zip(names, beta, err)},
    }
    print(f"\n  regression of matched Δqns (qns_b − qns_V3) on flux amplitudes"
          f" (n={regression['n']}, R2={r2:.3f},"
          f" resid sigma={regression['sigma_est']:.1f} W/m2):")
    for nm in names:
        t = regression["terms"][nm]
        print(f"     {nm:>15s}: {t['coef']:+9.5f} +/- {t['stderr']:.5f}"
              f"  (t = {t['t']:+.1f})")

    v3_qns = table["V3 = V2 + NEMO heat-content terms (matched)"][0]
    verdict = {
        "bar_qns_bias_max_abs": VERDICT_QNS_BIAS,
        "bar_qns_corr_min": VERDICT_QNS_CORR,
        "qns_bias": v3_qns["bias"], "qns_corr": v3_qns["corr"],
        "pass_bias": abs(v3_qns["bias"]) <= VERDICT_QNS_BIAS,
        "pass_corr": v3_qns["corr"] >= VERDICT_QNS_CORR,
    }
    print(f"\n  verdict bar (V3): |bias| <= {VERDICT_QNS_BIAS} W/m2 -> "
          f"{v3_qns['bias']:+.2f} ({'PASS' if verdict['pass_bias'] else 'FAIL'}); "
          f"corr >= {VERDICT_QNS_CORR} -> {v3_qns['corr']:.4f} "
          f"({'PASS' if verdict['pass_corr'] else 'FAIL'}); "
          "uniform-signature: read the regression above")

    # maps
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    reg_map = np.full(fields["qns_b"].shape, np.nan)
    reg_map[sel] = resid
    panels = [
        ("qns W/m2", fields["qns_b"], qns_v3, qns_v3 - fields["qns_b"],
         400.0, 100.0),
        ("qns W/m2: V0, protocol artifact (V0-V3), regression resid",
         qns_v0, qns_v0 - qns_v3, reg_map, 400.0, 40.0),
        ("utau N/m2", fields["utau_b"], utau_v2, utau_v2 - fields["utau_b"],
         0.3, 0.1),
        ("emp anomaly kg/m2/s", emp_b_anom, emp_anom, emp_anom - emp_b_anom,
         3e-4, 2e-4),
    ]
    for row, (label, f_model, f_ours, f_diff, vmax, vmax_d) in enumerate(panels):
        base = label.split(" ")[0]
        for col, (title, field, v) in enumerate([
                (f"restart {base}", f_model, vmax),
                (f"jaxerobulk {base}", f_ours, vmax),
                ("difference", f_diff, vmax_d)]):
            ax = axes[row, col]
            show = np.where(open_water, field, np.nan)
            pcm = ax.pcolormesh(lon2, lat2, show, cmap="RdBu_r",
                                vmin=-v, vmax=v, shading="auto")
            ax.set_title(title)
            ax.set_ylim(-80, 88)
            fig.colorbar(pcm, ax=ax, shrink=0.7)
    out_png = out_dir / "eorca1-maps.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nmaps -> {out_png}")

    payload = {
        "validation": "jaxerobulk CORE/NCAR chain vs eORCA1 TRA year-1001 end",
        "run": "eorca1 TRA year-1001 end state (1000-12-31, noleap calendar)",
        "mask": {"open_water_sst_gt_m1_lat64": int(open_water.sum()),
                 "strict_sst_gt_0.5": int(strict.sum())},
        "variants": table,
        "regression": regression,
        "verdict": verdict,
    }
    out_json = out_dir / "eorca1-validation.json"
    out_json.write_text(json.dumps(payload, indent=1))
    print(f"json  -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
