#!/usr/bin/env python3
"""Generate a rippled deformation series and write it out for DFT.

The usual workflow for a new material:

1. relax the flat unit cell in your DFT code,
2. run this to build a flat supercell and bend it to several amplitudes,
3. relax each rippled cell (ions only -- keep the cell fixed, it is already
   the right length),
4. read the relaxed geometries back and analyse them.

The bend is **isometric**: the neutral surface keeps its arc length and the
cell shortens to match. Simply adding ``A cos(kx)`` to z instead would stretch
the sheet -- 2.2% of spurious tensile strain at the largest amplitude used
here, which is around twenty times the real membrane strain and the wrong sign.

Output is written as ``POSCAR_<nn>.vasp``, with the sheet centred in the
vacuum so it renders as one continuous sheet rather than two halves.

    python examples/generate_ripples.py --out ripples/
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexo2d import (                                            # noqa: E402
    arc_length, read_npz, read_poscar, ripple_series, supercell,
    write_extxyz, write_poscar_series,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mos2_ripple.npz",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unitcell", help="flat POSCAR to start from "
                                       "(default: the flat MoS2 reference)")
    ap.add_argument("--nx", type=int, default=1,
                    help="supercell repeats along the ripple direction")
    ap.add_argument("--ny", type=int, default=1)
    ap.add_argument("--amplitudes", nargs="+", type=float,
                    default=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20],
                    help="ripple amplitudes in nm, starting at 0")
    ap.add_argument("--periods", type=int, default=1,
                    help="ripple periods in the supercell")
    ap.add_argument("--out", default="ripples",
                    help="directory for the POSCAR_<nn>.vasp files")
    ap.add_argument("--xyz", help="also write an extended-XYZ trajectory here")
    args = ap.parse_args()

    if args.unitcell:
        flat = read_poscar(args.unitcell)
    else:
        flat = read_npz(DATA)[0][0]
        print("starting from the flat MoS2 reference in data/")

    if args.nx > 1 or args.ny > 1:
        flat = supercell(flat, args.nx, args.ny)
    print(f"flat cell: {np.round(np.diag(flat.cell), 5)} nm, "
          f"{flat.n_atoms} atoms")
    print()

    series = ripple_series(
        flat, args.amplitudes, n_periods=args.periods,
        name="rippled sheet",
    )

    l_flat = flat.cell[0, 0]
    print(f"{'A (nm)':>8s} {'cell L (nm)':>12s} {'contraction':>12s} "
          f"{'curvature max':>14s} {'naive strain':>13s}")
    for i, a in enumerate(series.amplitudes):
        L = series[i].cell[0, 0]
        k = 2 * np.pi * args.periods / L
        print(f"{a:8.4f} {L:12.5f} {100*(L/l_flat-1):11.3f}% "
              f"{a*k*k:13.4f}  {100*(arc_length(a, l_flat)/l_flat - 1):12.2f}%")
    print()
    print("'naive strain' is what you would have injected by adding A*cos(kx)")
    print("to z without shortening the cell -- this is what is avoided.")
    print()

    paths = write_poscar_series(args.out, series)
    print(f"wrote {len(paths)} files to {args.out}/ "
          f"({os.path.basename(paths[0])} ... {os.path.basename(paths[-1])})")
    print("  the sheet is centred in the vacuum, so it opens cleanly in VESTA")
    print("  relax the IONS ONLY in each; the cell length is already correct")
    if args.xyz:
        write_extxyz(args.xyz, series)
        print(f"wrote {args.xyz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
