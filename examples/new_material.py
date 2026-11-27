#!/usr/bin/env python3
"""Template: apply flexo2d to a 2D material that is not MoS2.

The only material-specific inputs are

1. the rippled structures,
2. the charges,
3. the bond cutoff that defines a local dipole unit.

This script shows the three realistic ways to supply (1), then runs the same
pipeline used for MoS2. Edit the CONFIG block and go.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flexo2d import (                                            # noqa: E402
    FixedCharges, FlexoCalculation, NeighbourShell, TabulatedCharges,
    neutralize, read_extxyz_series, read_npz, read_poscar_series,
    sinusoidal_bending,
)

# --------------------------------------------------------------------- CONFIG
#: Element that sits at the centre of one local dipole unit (the metal in a
#: TMD, boron in h-BN, ...).
CENTER = "Mo"
#: Elements allowed in its shell. None means "every other element".
SHELL = ["S"]
#: Bond cutoff in nm. Run with SHOW_SHELLS = True first to choose it.
CUTOFF = 0.30
#: Charges in units of e. They must make each local unit neutral:
#: q_center + coordination * q_shell / bonds_per_shell_atom == 0.
CHARGES = {"Mo": 1.083, "S": -0.5415}
#: Number of ripple periods inside the supercell.
N_PERIODS = 1
#: Ripple direction: 0 = a1, 1 = a2.
AXIS = 0
#: Print neighbour distances and stop, to help pick CUTOFF.
SHOW_SHELLS = False
# ---------------------------------------------------------------------------


def load_series():
    """Pick one of these three routes and delete the others."""

    # (a) VASP: one relaxed structure per ripple amplitude, flat one first.
    #     Amplitudes can be anything monotonic -- an index is fine.
    #     This is what examples/generate_ripples.py writes.
    vasp = sorted(p for p in os.listdir(".")
                  if p.startswith("POSCAR_") and p.endswith(".vasp"))
    if vasp:
        return read_poscar_series(vasp, name="my material"), None

    # (b) An extended-XYZ trajectory, optionally carrying per-atom charges in
    #     a `charges` column.
    if os.path.exists("ripples.xyz"):
        return read_extxyz_series("ripples.xyz", name="my material")

    # (c) A cached native dataset.
    if os.path.exists("my_material.npz"):
        return read_npz("my_material.npz")

    raise SystemExit(
        "No input found. Put your rippled structures next to this script as\n"
        "  POSCAR_00.vasp, POSCAR_01.vasp, ...     (VASP), or\n"
        "  ripples.xyz                             (extended XYZ), or\n"
        "  my_material.npz                         (flexo2d native),\n"
        "then edit the CONFIG block at the top of this file."
    )


def main() -> int:
    series, charge_table = load_series()

    print(f"{len(series)} configurations, {series.reference.n_atoms} atoms")
    print(f"species: {series.reference.unique_species()}")
    print(f"cell   : {np.round(np.diag(series.reference.cell), 4)} nm")
    print(f"k      : {np.round(series.wavevectors(AXIS, N_PERIODS), 4)} 1/nm")
    print()

    scheme = NeighbourShell(CENTER, SHELL, cutoff=CUTOFF)

    if SHOW_SHELLS:
        print(f"{CENTER}-shell distances in the reference configuration (nm):")
        for dist, a, b in scheme.report_shells(series.reference, 0.6, 24):
            print(f"  {dist:.5f}   {a}-{b}")
        print("\nSet CUTOFF inside the gap after the first group, then set "
              "SHOW_SHELLS = False.")
        return 0

    clusters = scheme.clusters(series.reference)
    print(f"{len(clusters)} clusters, {len(clusters[0].members)} members each,")
    print(f"weights {np.round(clusters[0].weights, 4)}")
    net = sum(
        CHARGES[series.reference.species[m]] * w
        for m, w in zip(clusters[0].members, clusters[0].weights)
    )
    print(f"net charge of one cluster: {net:+.2e} e")
    if abs(net) > 1e-6:
        print("  -> adjust CHARGES so a unit is neutral, or the dipole will be")
        print("     taken about each cluster centroid (inter-cluster charge")
        print("     transfer is then not counted as polarization)")
    print()

    if charge_table is not None:
        charge_model = TabulatedCharges(neutralize(charge_table))
        print("using per-atom charges from the input file")
    else:
        charge_model = FixedCharges(CHARGES)
        print("using the fixed charges from CONFIG")

    result = FlexoCalculation(
        series,
        charge_model=charge_model,
        cluster_scheme=scheme,
        mode=sinusoidal_bending(),
        axis=AXIS,
        n_periods=N_PERIODS,
        neutrality_tol=1e-6 if charge_table is None else 1.0,
    ).run()

    print()
    print(result.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
