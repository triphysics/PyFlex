"""Per-element data: masses and a few convenience charge sets.

Nothing here is specific to a material -- add an entry and the rest of the
package picks it up.
"""

from __future__ import annotations

#: Standard atomic weights in atomic mass units.
ATOMIC_MASSES = {
    "H": 1.008, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.065, "Cl": 35.45,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922,
    "Se": 78.971, "Br": 79.904, "Rb": 85.468, "Sr": 87.62, "Y": 88.906,
    "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41, "In": 114.82,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Cs": 132.91,
    "Ba": 137.33, "La": 138.91, "Hf": 178.49, "Ta": 180.95, "W": 183.84,
    "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98,
}


def mass(symbol: str) -> float:
    """Atomic mass of `symbol` in u."""
    try:
        return ATOMIC_MASSES[symbol]
    except KeyError as exc:
        raise KeyError(
            f"no mass for element {symbol!r}; add it to "
            "flexo2d.species.ATOMIC_MASSES"
        ) from exc


def masses(symbols) -> list[float]:
    """Atomic masses for an iterable of element symbols."""
    return [mass(s) for s in symbols]


#: Charge sets used in the literature for a few 2D materials, in units of e.
#: These are *starting points* -- for quantitative work supply charges from
#: your own Bader/DDEC/Born-charge analysis via
#: :class:`flexo2d.charges.TabulatedCharges`.
REFERENCE_CHARGE_SETS = {
    # Sheet-averaged Bader charges of monolayer MoS2.
    "MoS2-bader": {"Mo": 1.083, "S": -0.5415},
    "MoSe2-bader": {"Mo": 1.02, "Se": -0.51},
    "WS2-bader": {"W": 1.12, "S": -0.56},
    # h-BN, formal-ish
    "BN-bader": {"B": 1.10, "N": -1.10},
}
