"""Charge models.

The point-charge dipole model needs one charge per atom per configuration.
Two sources are supported:

* :class:`FixedCharges` -- one charge per element, the same in every
  configuration. This is the rigid-ion model: the dipole then responds only to
  how the atoms move.
* :class:`TabulatedCharges` -- a charge for every atom in every configuration,
  e.g. from a Bader / DDEC / Hirshfeld analysis. This adds the charge-transfer
  contribution to the response.

Both can be post-processed with :func:`neutralize` (remove the small net charge
that population analyses leave behind) and :func:`fourier_smooth` (suppress
numerical noise in a per-atom charge profile while keeping its leading
harmonics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np

from .structure import ConfigSeries, Structure


class ChargeModel(Protocol):
    """Anything that can hand out one charge per atom for a configuration."""

    def charges(self, structure: Structure, index: int) -> np.ndarray:
        ...


@dataclass
class FixedCharges:
    """One charge per element, constant across the series (rigid-ion model).

    Parameters
    ----------
    by_species
        ``{element_symbol: charge_in_e}``.
    """

    by_species: Mapping[str, float]

    def charges(self, structure: Structure, index: int = 0) -> np.ndarray:
        try:
            return np.array(
                [float(self.by_species[s]) for s in structure.species], dtype=float
            )
        except KeyError as exc:
            missing = sorted(set(structure.species) - set(self.by_species))
            raise KeyError(
                f"no charge given for {missing}; species present: "
                f"{structure.unique_species()}"
            ) from exc

    def net_charge(self, structure: Structure) -> float:
        return float(self.charges(structure).sum())


@dataclass
class TabulatedCharges:
    """Per-atom, per-configuration charges, shape ``(n_config, n_atoms)``."""

    table: np.ndarray

    def __post_init__(self) -> None:
        self.table = np.asarray(self.table, dtype=float)
        if self.table.ndim != 2:
            raise ValueError(
                f"charge table must be (n_config, n_atoms), got {self.table.shape}"
            )

    def charges(self, structure: Structure, index: int) -> np.ndarray:
        if index >= len(self.table):
            raise IndexError(
                f"configuration {index} requested but the charge table has "
                f"{len(self.table)} rows"
            )
        row = self.table[index]
        if len(row) != structure.n_atoms:
            raise ValueError(
                f"charge row {index} has {len(row)} entries but the structure "
                f"has {structure.n_atoms} atoms"
            )
        return row.copy()


# --------------------------------------------------------------- processing
def neutralize(table: np.ndarray, groups: Sequence[Sequence[int]] | None = None
               ) -> np.ndarray:
    """Remove the residual net charge of each configuration.

    Population analyses do not sum to exactly zero. Spreading the residual
    evenly over the atoms keeps every local unit neutral, which a dipole model
    requires -- otherwise the "dipole" depends on where you put the origin.

    Parameters
    ----------
    table
        ``(n_config, n_atoms)`` charges.
    groups
        Optional atom groups to spread the residual over separately (e.g. one
        group per element). ``None`` spreads it over all atoms equally.
    """
    table = np.asarray(table, dtype=float)
    out = table.copy()
    if groups is None:
        out -= table.sum(axis=1, keepdims=True) / table.shape[1]
        return out
    residual = table.sum(axis=1)
    total = sum(len(g) for g in groups)
    if total == 0:
        raise ValueError("groups are empty")
    for g in groups:
        g = np.asarray(g, dtype=int)
        out[:, g] -= (residual / total)[:, None]
    return out


def fourier_smooth(profile: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """Keep only the `n_harmonics` strongest Fourier components of a profile.

    Take the real part of the DFT, keep the `n_harmonics` largest-magnitude
    components, zero the rest, and transform back. Use it on a per-atom charge
    profile along the ripple to remove sampling noise without touching the
    physical harmonics.

    Parameters
    ----------
    profile
        1D profile, or ``(n_config, n_points)`` to smooth each row.
    n_harmonics
        How many components to keep.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 2:
        return np.array([fourier_smooth(row, n_harmonics) for row in arr])
    if arr.ndim != 1:
        raise ValueError(f"profile must be 1D or 2D, got shape {arr.shape}")
    if n_harmonics < 1:
        raise ValueError("n_harmonics must be >= 1")

    spectrum = np.real(np.fft.fft(arr) / np.sqrt(len(arr)))
    keep = np.argsort(np.abs(spectrum))[::-1][:n_harmonics]
    filtered = np.zeros_like(spectrum)
    filtered[keep] = spectrum[keep]
    return np.real(np.fft.ifft(filtered * np.sqrt(len(arr))))


def charges_from_reference_set(name: str) -> FixedCharges:
    """A :class:`FixedCharges` model from :data:`flexo2d.species.REFERENCE_CHARGE_SETS`."""
    from .species import REFERENCE_CHARGE_SETS

    try:
        return FixedCharges(dict(REFERENCE_CHARGE_SETS[name]))
    except KeyError as exc:
        raise KeyError(
            f"unknown charge set {name!r}; known: "
            f"{sorted(REFERENCE_CHARGE_SETS)}"
        ) from exc


def check_series(model: ChargeModel, series: ConfigSeries, tol: float = 1e-6
                 ) -> np.ndarray:
    """Net charge of every configuration under `model`; warns via the result.

    Returns the array of net charges so callers can assert on it.
    """
    net = np.array(
        [model.charges(s, i).sum() for i, s in enumerate(series)], dtype=float
    )
    return net
