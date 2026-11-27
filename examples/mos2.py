#!/usr/bin/env python3
"""Flexoelectric coefficients of monolayer MoS2 from the dataset in data/.

Run from the repository root::

    python examples/mos2.py              # text report
    python examples/mos2.py --figure out.png

Equivalent command line::

    python -m flexo2d --npz data/mos2_ripple.npz --center Mo --shell S \
        --cutoff 0.30 --charge Mo=1.083 --charge S=-0.5415
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexo2d import (                                            # noqa: E402
    FixedCharges, FlexoCalculation, NeighbourShell, TabulatedCharges,
    coefficient_table, fourier_smooth, neutralize, read_npz,
)

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mos2_ripple.npz",
)

# Bader charges of monolayer MoS2, averaged over the sheet. Neutral by
# construction: 1.083 - 2 * 0.5415 = 0.
CHARGES = {"Mo": 1.083, "S": -0.5415}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", help="write a summary figure here")
    ap.add_argument("--csv", help="write the per-configuration table here")
    ap.add_argument("--bader", action="store_true",
                    help="use the per-atom Bader charges instead of the "
                         "averaged ones (noisier; see the README)")
    args = ap.parse_args()

    series, bader_table = read_npz(DATA)

    print(f"dataset   : {series.name}")
    print(f"            {len(series)} configurations, "
          f"{series.reference.n_atoms} atoms each")
    print(f"cell       : {np.round(np.diag(series.reference.cell), 4)} nm")
    print(f"ripple k   : {np.round(series.wavevectors(), 4)} 1/nm")
    print()

    # 1. Choose the bond cutoff by looking at the actual neighbour distances.
    scheme = NeighbourShell("Mo", "S", cutoff=0.30)
    print("Mo coordination shells in the flat reference:")
    print(f"  {'dist/nm':>9s}  {'per Mo':>7s}   pair")
    for dist, mult, a, b in scheme.report_shells(series.reference, 0.45, 6):
        print(f"  {dist:9.5f}  {mult:7.3g}   {a}-{b}")
    print("  -> 0.30 nm sits in the gap after the first shell (6 S per Mo)\n")

    clusters = scheme.clusters(series.reference)
    print(f"clusters   : {len(clusters)} (one per Mo), "
          f"{len(clusters[0].members)} members each, "
          f"weights {np.round(clusters[0].weights, 4)}")
    print("             the 1/3 weights are found automatically: each S bonds")
    print("             to three Mo, so the units tile the sheet exactly once\n")

    # 2. Charges.
    if args.bader:
        charge_model = TabulatedCharges(neutralize(bader_table))
        print("charges    : per-atom Bader populations, net charge removed")
    else:
        charge_model = FixedCharges(CHARGES)
        print(f"charges    : fixed, {CHARGES}")

    # 3. Run.
    result = FlexoCalculation(
        series,
        charge_model=charge_model,
        cluster_scheme=scheme,
        neutrality_tol=1.0 if args.bader else 1e-6,
    ).run()

    print()
    print(result.report())

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write(coefficient_table(result) + "\n")
        print(f"wrote {args.csv}")

    if args.figure:
        from flexo2d.plotting import summary_figure

        summary_figure(result, args.figure)
        print(f"wrote {args.figure}")

    # Where a noisy per-atom charge profile can be smoothed before use.
    if args.bader:
        mo = neutralize(bader_table)[:, :16]
        print("\nMo charge profile, configuration 10:")
        print("  raw     :", np.round(mo[10][:8], 4))
        print("  smoothed:", np.round(fourier_smooth(mo[10], 3)[:8], 4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
