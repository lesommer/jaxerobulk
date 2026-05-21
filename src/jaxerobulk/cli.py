"""CLI for JaxeroBulk.

Three subcommands matching the FORTRAN test suite surface:
  aerobulk-toy     — interactive single-point exploration
  aerobulk-compute — batch computation from CSV/NetCDF forcing
  aerobulk-compare — cross-algorithm comparison at a single point
"""

import argparse
import sys
import json

import jax
import jax.numpy as jnp

from jaxerobulk.api import aerobulk_model, Algorithm


jax.config.update("jax_enable_x64", True)


_ALGO_NAMES = [a.value for a in Algorithm]


def _parse_algo(name):
    return Algorithm(name.lower())


def _toy(args):
    zt = args.zt
    zu = args.zu
    sst = jnp.float64(args.sst)
    t_zt = jnp.float64(args.t_air)
    q_zt = jnp.float64(args.humidity)
    U_zu = jnp.float64(args.u_wind)
    V_zu = jnp.float64(args.v_wind)
    slp = jnp.float64(args.slp)

    kwargs = dict(
        zt=zt, zu=zu, sst=sst, t_zt=t_zt, hum_zt=q_zt,
        U_zu=U_zu, V_zu=V_zu, slp=slp,
        algo=args.algorithm, nb_iter=args.niter,
    )

    if args.skin:
        kwargs["l_use_skin"] = True
        kwargs["rad_sw"] = jnp.float64(args.rad_sw)
        kwargs["rad_lw"] = jnp.float64(args.rad_lw)

    result = aerobulk_model(**kwargs)

    print(f"Algorithm: {args.algorithm}")
    print(f"Input:  SST={args.sst}K  T_air={args.t_air}K  q={args.humidity}  "
          f"U={args.u_wind}m/s  V={args.v_wind}m/s  SLP={args.slp}Pa  zt={zt}m  zu={zu}m")
    print(f"Output: Cd={float(result['Cd']):.6e}  Ch={float(result['Ch']):.6e}  Ce={float(result['Ce']):.6e}")
    print(f"        Qlat={float(result['Qlat']):.2f} W/m^2  Qsen={float(result['Qsen']):.2f} W/m^2")
    print(f"        tau_x={float(result['tau_x']):.4f} N/m^2  tau_y={float(result['tau_y']):.4f} N/m^2")
    print(f"        u*={float(result['u_star']):.6f} m/s  L={float(result['L']):.2f} m")
    if args.skin and "dT_cs" in result:
        print(f"        dT_cs={float(result['dT_cs']):.4f} K  dT_wl={float(result.get('dT_wl', 0)):.4f} K")
        print(f"        T_skin={float(result['T_s']):.4f} K")


def _compare(args):
    zt = args.zt
    zu = args.zu
    sst = jnp.float64(args.sst)
    t_zt = jnp.float64(args.t_air)
    q_zt = jnp.float64(args.humidity)
    U_zu = jnp.float64(args.u_wind)
    V_zu = jnp.float64(args.v_wind)
    slp = jnp.float64(args.slp)

    algos = args.algorithms if args.algorithms else _ALGO_NAMES

    print(f"Input:  SST={args.sst}K  T_air={args.t_air}K  q={args.humidity}  "
          f"U={args.u_wind}m/s  V={args.v_wind}m/s  SLP={args.slp}Pa  zt={zt}m  zu={zu}m")
    print()
    header = f"{'Algorithm':>12s}  {'Cd':>12s}  {'Ch':>12s}  {'Ce':>12s}  {'Qlat':>10s}  {'Qsen':>10s}  {'u*':>10s}  {'L':>10s}"
    print(header)
    print("-" * len(header))

    for algo_name in algos:
        try:
            result = aerobulk_model(
                zt=zt, zu=zu, sst=sst, t_zt=t_zt, hum_zt=q_zt,
                U_zu=U_zu, V_zu=V_zu, slp=slp,
                algo=algo_name, nb_iter=args.niter,
            )
            print(f"{algo_name:>12s}  {float(result['Cd']):12.6e}  {float(result['Ch']):12.6e}  "
                  f"{float(result['Ce']):12.6e}  {float(result['Qlat']):10.2f}  {float(result['Qsen']):10.2f}  "
                  f"{float(result['u_star']):10.6f}  {float(result['L']):10.2f}")
        except Exception as e:
            print(f"{algo_name:>12s}  ERROR: {e}")


def _compute(args):
    if not args.input_csv:
        print("Error: --input-csv is required for compute. Provide a CSV file with forcing data.", file=sys.stderr)
        sys.exit(1)
    _compute_csv(args)


def _compute_csv(args):
    import csv

    zt = args.zt
    zu = args.zu
    algo = args.algorithm
    niter = args.niter

    with open(args.input_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.output_csv:
        out_fields = ["row", "Cd", "Ch", "Ce", "Qlat", "Qsen", "tau_x", "tau_y", "u_star", "Evap"]
        out_file = open(args.output_csv, "w", newline="")
        writer = csv.DictWriter(out_file, fieldnames=out_fields)
        writer.writeheader()

    for i, row in enumerate(rows):
        sst = jnp.float64(float(row["sst"]))
        t_zt = jnp.float64(float(row["t_air"]))
        q_zt = jnp.float64(float(row.get("q", row.get("humidity", "0.018"))))
        U_zu = jnp.float64(float(row.get("u_wind", row.get("U", "0.0"))))
        V_zu = jnp.float64(float(row.get("v_wind", row.get("V", "0.0"))))
        slp = jnp.float64(float(row.get("slp", "101000.0")))

        kwargs = dict(
            zt=zt, zu=zu, sst=sst, t_zt=t_zt, hum_zt=q_zt,
            U_zu=U_zu, V_zu=V_zu, slp=slp,
            algo=algo, nb_iter=niter,
        )

        if "rad_sw" in row and "rad_lw" in row:
            kwargs["l_use_skin"] = True
            kwargs["rad_sw"] = jnp.float64(float(row["rad_sw"]))
            kwargs["rad_lw"] = jnp.float64(float(row["rad_lw"]))

        result = aerobulk_model(**kwargs)

        out_row = {
            "row": i,
            "Cd": float(result["Cd"]),
            "Ch": float(result["Ch"]),
            "Ce": float(result["Ce"]),
            "Qlat": float(result["Qlat"]),
            "Qsen": float(result["Qsen"]),
            "tau_x": float(result["tau_x"]),
            "tau_y": float(result["tau_y"]),
            "u_star": float(result["u_star"]),
            "Evap": float(result["Evap"]),
        }

        if args.output_csv:
            writer.writerow(out_row)
        else:
            print(json.dumps(out_row))

    if args.output_csv:
        out_file.close()
        print(f"Wrote {len(rows)} rows to {args.output_csv}")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="jaxerobulk",
        description="JaxeroBulk: differentiable aerodynamic bulk formulae in JAX",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- toy ----
    toy = subparsers.add_parser("toy", help="Interactive single-point exploration")
    toy.add_argument("--algorithm", "-a", default="coare3p6", choices=_ALGO_NAMES, help="Bulk algorithm (default: coare3p6)")
    toy.add_argument("--sst", type=float, default=300.0, help="Sea surface temperature [K] (default: 300)")
    toy.add_argument("--t-air", type=float, default=298.0, help="Air temperature at zt [K] (default: 298)")
    toy.add_argument("--humidity", type=float, default=0.018, help="Specific humidity at zt [kg/kg] (default: 0.018)")
    toy.add_argument("--u-wind", type=float, default=5.0, help="Zonal wind at zu [m/s] (default: 5)")
    toy.add_argument("--v-wind", type=float, default=0.0, help="Meridional wind at zu [m/s] (default: 0)")
    toy.add_argument("--slp", type=float, default=101000.0, help="Sea-level pressure [Pa] (default: 101000)")
    toy.add_argument("--zt", type=float, default=2.0, help="Height for T/q [m] (default: 2)")
    toy.add_argument("--zu", type=float, default=10.0, help="Height for wind [m] (default: 10)")
    toy.add_argument("--niter", type=int, default=5, help="Number of iterations (default: 5)")
    toy.add_argument("--skin", action="store_true", help="Enable cool-skin/warm-layer")
    toy.add_argument("--rad-sw", type=float, default=200.0, help="Downwelling shortwave [W/m^2] (default: 200)")
    toy.add_argument("--rad-lw", type=float, default=350.0, help="Downwelling longwave [W/m^2] (default: 350)")
    toy.set_defaults(func=_toy)

    # ---- compare ----
    compare = subparsers.add_parser("compare", help="Cross-algorithm comparison at a single point")
    compare.add_argument("--algorithms", "-a", nargs="+", default=None, choices=_ALGO_NAMES, help="Algorithms to compare (default: all)")
    compare.add_argument("--sst", type=float, default=300.0, help="Sea surface temperature [K]")
    compare.add_argument("--t-air", type=float, default=298.0, help="Air temperature at zt [K]")
    compare.add_argument("--humidity", type=float, default=0.018, help="Specific humidity at zt [kg/kg]")
    compare.add_argument("--u-wind", type=float, default=5.0, help="Zonal wind at zu [m/s]")
    compare.add_argument("--v-wind", type=float, default=0.0, help="Meridional wind at zu [m/s]")
    compare.add_argument("--slp", type=float, default=101000.0, help="Sea-level pressure [Pa]")
    compare.add_argument("--zt", type=float, default=2.0, help="Height for T/q [m]")
    compare.add_argument("--zu", type=float, default=10.0, help="Height for wind [m]")
    compare.add_argument("--niter", type=int, default=5, help="Number of iterations")
    compare.set_defaults(func=_compare)

    # ---- compute ----
    compute = subparsers.add_parser("compute", help="Batch computation from CSV forcing")
    compute.add_argument("--algorithm", "-a", default="coare3p6", choices=_ALGO_NAMES, help="Bulk algorithm")
    compute.add_argument("--input-csv", type=str, help="Input CSV file with forcing data")
    compute.add_argument("--output-csv", type=str, default=None, help="Output CSV file (default: stdout as JSON)")
    compute.add_argument("--zt", type=float, default=2.0, help="Height for T/q [m]")
    compute.add_argument("--zu", type=float, default=10.0, help="Height for wind [m]")
    compute.add_argument("--niter", type=int, default=5, help="Number of iterations")
    compute.set_defaults(func=_compute)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


def aerobulk_toy_main():
    sys.argv = ["aerobulk-toy", "toy"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()
