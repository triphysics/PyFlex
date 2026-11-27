"""Unit conventions and conversions.

Internally **everything** is kept in a single consistent set of units:

======================  =========================
quantity                internal unit
======================  =========================
length                  nanometre (nm)
charge                  elementary charge (e)
mass                    atomic mass unit (u)
dipole moment           e.nm
wavevector              1/nm
======================  =========================

Flexoelectric and piezoelectric coefficients therefore come out *per cluster*
(per formula unit) as

* ``f_ijkl``  ->  e.nm^2
* ``e_ijk``   ->  e.nm

Dividing by the in-plane area of one cluster turns these into the intensive 2D
coefficients usually quoted in the literature; :func:`to_si` and
:func:`per_area` do that for you.
"""

from __future__ import annotations

ELEMENTARY_CHARGE = 1.602176634e-19   # C
NM = 1e-9                             # m

#: ``value_in_SI = value_internal * CONVERSIONS[name]``
CONVERSIONS = {
    # dipole moment: e.nm -> C.m
    "dipole": ELEMENTARY_CHARGE * NM,
    # piezoelectric-like, per cluster: e.nm -> C.m
    "piezo_per_cluster": ELEMENTARY_CHARGE * NM,
    # flexoelectric, per cluster: e.nm^2 -> C.m^2
    "flexo_per_cluster": ELEMENTARY_CHARGE * NM**2,
    # 2D polarization: e/nm -> C/m
    "polarization_2d": ELEMENTARY_CHARGE / NM,
    # 2D piezoelectric coefficient: e/nm * nm = e -> C
    "piezo_2d": ELEMENTARY_CHARGE,
    # 2D flexoelectric coefficient: e.nm -> C.m  (often quoted in nC/m * nm)
    "flexo_2d": ELEMENTARY_CHARGE * NM,
}


def to_si(value, kind: str):
    """Convert an internal-unit `value` of the given `kind` to SI."""
    try:
        return value * CONVERSIONS[kind]
    except KeyError as exc:  # pragma: no cover - programming error
        raise KeyError(
            f"unknown unit kind {kind!r}; known: {sorted(CONVERSIONS)}"
        ) from exc


def per_area(value, area_nm2: float):
    """Make a per-cluster coefficient intensive by dividing by a cluster area.

    Parameters
    ----------
    value
        Coefficient per cluster (``e.nm^2`` for flexoelectric, ``e.nm`` for
        piezoelectric).
    area_nm2
        In-plane area belonging to one cluster, in nm^2.
    """
    if area_nm2 <= 0:
        raise ValueError("area_nm2 must be positive")
    return value / area_nm2


def flexo_2d_to_nC_per_m(value_e_nm: float) -> float:
    """``e.nm`` (2D flexoelectric coefficient) -> ``nC/m`` times nm.

    This is the combination most papers tabulate for 2D flexoelectricity, i.e.
    a transverse flexoelectric coefficient with dimensions of
    polarization/curvature.
    """
    return value_e_nm * ELEMENTARY_CHARGE / NM * 1e9 * NM  # = value * e/nm * nm


# ---------------------------------------------------------------- dimensions
#: Dimension of every quantity a fitted ansatz can produce, as
#: ``(length_exponent, charge_exponent)`` in the internal nm / e units.
#: ``modes.dimension_of`` uses this to work out the units of a coefficient
#: from its closed form, so a new DeformationMode gets correct units for free.
SYMBOL_DIMENSIONS = {
    "k": (-1, 0),        # wavevector, 1/nm
    "U0": (1, 0),        # displacement amplitude, nm
    "V0": (0, 0),        # uniform strain, dimensionless
    "W0": (1, 0),        # displacement amplitude, nm
    "Us": (1, 0),        # quadrature displacement amplitude, nm
    "Ws": (1, 0),
    "cU": (1, 0),        # displacement offset, nm
    "dzs": (1, 1),       # dipole amplitude, e.nm
    "dzc": (1, 1),
    "dxs": (1, 1),
    "dxc": (1, 1),
    "cd": (1, 1),        # uniform dipole, e.nm
}


def format_dimension(length: int, charge: int, per_area: bool = False) -> str:
    """Internal-unit label for a quantity of the given dimension."""
    if per_area:
        length -= 2
    parts = []
    if charge:
        parts.append("e" if charge == 1 else f"e^{charge}")
    if length:
        parts.append("nm" if length == 1 else f"nm^{length}")
    return ".".join(parts) if parts else "1"


def si_factor_and_label(length: int, charge: int, per_area: bool = False):
    """SI conversion factor and label for a quantity of the given dimension."""
    if per_area:
        length -= 2
    factor = (ELEMENTARY_CHARGE**charge) * (NM**length)
    parts = []
    if charge:
        parts.append("C" if charge == 1 else f"C^{charge}")
    if length:
        parts.append("m" if length == 1 else f"m^{length}")
    return factor, (".".join(parts) if parts else "1")
