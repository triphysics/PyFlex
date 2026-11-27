"""flexo2d -- flexoelectric and piezoelectric coefficients of 2D materials.

The package extracts flexoelectric coefficients from a series of rippled
(bent) supercells plus a set of atomic charges, using a local point-charge
dipole model. Nothing in the pipeline is specific to any one material: a new
material is a new set of structures, charges and a bond cutoff.

Quick start
-----------
.. code-block:: python

    from flexo2d import (
        FlexoCalculation, FixedCharges, NeighbourShell, read_poscar_series,
    )

    series = read_poscar_series(
        [f"POSCAR_{i:02d}.vasp" for i in range(11)], name="MoS2"
    )
    result = FlexoCalculation(
        series,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
    ).run()
    print(result.report())

The pipeline, step by step
--------------------------
0. :mod:`flexo2d.build` generates the rippled supercells to feed to DFT
   (isometric bending, so the sheet is not stretched).
1. :mod:`flexo2d.io` loads the deformation series back.
2. :mod:`flexo2d.charges` supplies per-atom charges.
3. :mod:`flexo2d.clusters` cuts the sheet into neutral local dipole units.
4. :mod:`flexo2d.fields` builds cluster dipoles, centroids and displacements.
5. :mod:`flexo2d.modes` declares the deformation ansatz and the constitutive
   law, and solves symbolically for each coefficient.
6. :mod:`flexo2d.fitting` fits the ansatz (linear least squares).
7. :mod:`flexo2d.extrapolate` takes the zero-deformation limit.
8. :mod:`flexo2d.pipeline` runs the whole thing; :mod:`flexo2d.report` and
   :mod:`flexo2d.plotting` present it.
"""

from .build import (
    arc_length,
    isometric_ripple,
    max_amplitude,
    projected_length,
    ripple_series,
    supercell,
)
from .charges import (
    ChargeModel,
    FixedCharges,
    TabulatedCharges,
    charges_from_reference_set,
    fourier_smooth,
    neutralize,
)
from .clusters import Cluster, ClusterScheme, ExplicitGroups, NeighbourShell
from .extrapolate import Extrapolation, extrapolate, resolvable_mask
from .fields import FieldSet, build_fields
from .fitting import Ansatz, FitResult, fit_linear
from .io import (
    read_extxyz_series,
    read_npz,
    read_poscar,
    read_poscar_series,
    series_from_layer_arrays,
    write_extxyz,
    write_npz,
    write_poscar,
    write_poscar_series,
)
from .modes import DeformationMode, Relation, sinusoidal_bending
from .pipeline import FlexoCalculation, FlexoResult
from .report import coefficient_table, format_result
from .species import ATOMIC_MASSES, REFERENCE_CHARGE_SETS, mass
from .structure import ConfigSeries, Structure
from .units import per_area, to_si

__version__ = "0.1.0"

__all__ = [
    "ATOMIC_MASSES", "arc_length", "isometric_ripple", "max_amplitude",
    "projected_length",
    "ripple_series", "supercell", "write_extxyz", "write_poscar",
    "write_poscar_series", "Ansatz", "ChargeModel", "Cluster", "ClusterScheme",
    "ConfigSeries", "DeformationMode", "ExplicitGroups", "Extrapolation",
    "FieldSet", "FitResult", "FixedCharges", "FlexoCalculation", "FlexoResult",
    "NeighbourShell", "REFERENCE_CHARGE_SETS", "Relation", "Structure",
    "TabulatedCharges", "build_fields", "charges_from_reference_set",
    "coefficient_table", "extrapolate", "fit_linear", "format_result",
    "fourier_smooth", "mass", "neutralize", "per_area", "read_extxyz_series",
    "read_npz", "read_poscar", "read_poscar_series", "resolvable_mask",
    "series_from_layer_arrays", "sinusoidal_bending", "to_si", "write_npz",
]
