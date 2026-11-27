"""Command-line entry point: ``python -m flexo2d``."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flexo2d",
        description="Flexoelectric coefficients of a 2D material from a "
                    "series of rippled supercells.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="native dataset written by flexo2d.write_npz")
    src.add_argument("--poscar", nargs="+",
                     help="one POSCAR/CONTCAR per amplitude, flat one first")
    src.add_argument("--extxyz", help="extended-XYZ trajectory, one frame per amplitude")

    p.add_argument("--center", required=True, metavar="EL",
                   help="element acting as the centre of each dipole unit (e.g. Mo)")
    p.add_argument("--shell", nargs="+", metavar="EL",
                   help="elements allowed in the shell (default: all others)")
    p.add_argument("--cutoff", type=float, default=0.30,
                   help="bond cutoff in nm (default: %(default)s)")
    p.add_argument("--charge", action="append", default=[], metavar="EL=Q",
                   help="fixed charge per element, repeatable (e.g. --charge Mo=1.083)")
    p.add_argument("--charge-set", help="named set from flexo2d.species.REFERENCE_CHARGE_SETS")
    p.add_argument("--use-file-charges", action="store_true",
                   help="use the per-atom charges stored in the input file")
    p.add_argument("--axis", type=int, default=0, help="ripple direction (default: x)")
    p.add_argument("--periods", type=int, default=1,
                   help="ripple periods per supercell (default: %(default)s)")
    p.add_argument("--degree", type=int, default=2,
                   help="degree of the zero-deformation extrapolation")
    p.add_argument("--amplitudes", nargs="+", type=float,
                   help="deformation amplitude of each configuration")
    p.add_argument("--no-in-plane", action="store_true",
                   help="skip the in-plane (f1111, e111) block")
    p.add_argument("--json", metavar="PATH", help="also write results as JSON")
    p.add_argument("--csv", metavar="PATH", help="also write the per-config table as CSV")
    p.add_argument("--figure", metavar="PATH", help="write a summary figure")
    p.add_argument("--shells", action="store_true",
                   help="only list neighbour distances, to help pick --cutoff")
    return p


def main(argv=None) -> int:
    from . import (
        FixedCharges, FlexoCalculation, NeighbourShell, TabulatedCharges,
        charges_from_reference_set, coefficient_table, read_extxyz_series,
        read_npz, read_poscar_series, sinusoidal_bending,
    )
    from .charges import neutralize

    args = build_parser().parse_args(argv)

    table = None
    if args.npz:
        series, table = read_npz(args.npz)
    elif args.poscar:
        series = read_poscar_series(args.poscar, amplitudes=args.amplitudes)
    else:
        series, table = read_extxyz_series(args.extxyz, amplitudes=args.amplitudes)
    if args.amplitudes is not None:
        if len(args.amplitudes) != len(series):
            print(f"error: {len(args.amplitudes)} amplitudes for "
                  f"{len(series)} configurations", file=sys.stderr)
            return 2
        series.amplitudes = np.asarray(args.amplitudes, dtype=float)

    scheme = NeighbourShell(args.center, args.shell, cutoff=args.cutoff)

    if args.shells:
        print(f"coordination shells around {args.center} in the reference "
              f"configuration:")
        print(f"  {'dist/nm':>9s}  {'per centre':>10s}   pair")
        for dist, mult, a, b in scheme.report_shells(series.reference, 0.6, 12):
            print(f"  {dist:9.5f}  {mult:10.3g}   {a}-{b}")
        print("\npick --cutoff inside the gap after the shell(s) you want")
        return 0

    if args.use_file_charges:
        if table is None:
            print("error: --use-file-charges but the input has no charges",
                  file=sys.stderr)
            return 2
        charge_model = TabulatedCharges(neutralize(table))
    elif args.charge_set:
        charge_model = charges_from_reference_set(args.charge_set)
    elif args.charge:
        by = {}
        for item in args.charge:
            if "=" not in item:
                print(f"error: --charge expects EL=Q, got {item!r}", file=sys.stderr)
                return 2
            el, q = item.split("=", 1)
            by[el] = float(q)
        charge_model = FixedCharges(by)
    else:
        print("error: give charges via --charge, --charge-set or "
              "--use-file-charges", file=sys.stderr)
        return 2

    calc = FlexoCalculation(
        series,
        charge_model=charge_model,
        cluster_scheme=scheme,
        mode=sinusoidal_bending(in_plane=not args.no_in_plane),
        axis=args.axis,
        n_periods=args.periods,
        extrapolation_degree=args.degree,
    )
    result = calc.run()
    print(result.report())

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(coefficient_table(result) + "\n")
        print(f"wrote {args.csv}")
    if args.json:
        payload = {
            "series": result.series_name,
            "mode": result.mode.name,
            "cluster_area_nm2": result.cluster_area,
            "closed_forms": {k: str(v) for k, v in result.closed_forms.items()},
            "amplitudes": result.fields.amplitudes.tolist(),
            "per_configuration": {
                k: [None if not np.isfinite(v) else v for v in vals]
                for k, vals in result.coefficients.items()
            },
            "zero_deformation_limit": {
                k: {
                    "value": None if not np.isfinite(ex.value) else ex.value,
                    "stderr": None if not np.isfinite(ex.stderr) else ex.stderr,
                    "n_used": ex.n_used,
                    "degree": ex.degree,
                    "note": ex.note,
                }
                for k, ex in result.extrapolations.items()
            },
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")
    if args.figure:
        from .plotting import summary_figure
        summary_figure(result, args.figure)
        print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
