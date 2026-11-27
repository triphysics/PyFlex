"""Geometry containers: a single configuration and a deformation series."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .species import masses as _masses


@dataclass
class Structure:
    """One periodic configuration of a 2D material.

    Parameters
    ----------
    species
        Element symbol per atom, shape ``(N,)``.
    positions
        Cartesian coordinates in nm, shape ``(N, 3)``.
    cell
        Row-wise lattice vectors in nm, shape ``(3, 3)``. For a 2D material the
        third vector is the (large) vacuum direction.
    pbc
        Periodicity per lattice vector. Defaults to periodic in-plane, open
        out-of-plane.
    label
        Free-form identifier, carried through into results.
    atom_masses
        Optional per-atom masses in u, overriding the lookup in
        :data:`flexo2d.species.ATOMIC_MASSES`. Use it for model species, for
        isotopes, or for any symbol not in the table.
    """

    species: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple[bool, bool, bool] = (True, True, False)
    label: str = ""
    atom_masses: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.species = np.asarray(self.species, dtype=object)
        self.positions = np.asarray(self.positions, dtype=float)
        self.cell = np.asarray(self.cell, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if self.cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3), got {self.cell.shape}")
        if len(self.species) != len(self.positions):
            raise ValueError(
                f"{len(self.species)} species but {len(self.positions)} positions"
            )
        self.pbc = tuple(bool(p) for p in self.pbc)
        if self.atom_masses is not None:
            self.atom_masses = np.asarray(self.atom_masses, dtype=float)
            if self.atom_masses.shape != (len(self.positions),):
                raise ValueError(
                    f"atom_masses must be ({len(self.positions)},), got "
                    f"{self.atom_masses.shape}"
                )
            if np.any(self.atom_masses <= 0):
                raise ValueError("atom_masses must all be positive")

    # ---------------------------------------------------------------- basics
    @property
    def n_atoms(self) -> int:
        return len(self.positions)

    @property
    def masses(self) -> np.ndarray:
        """Atomic masses in u, shape ``(N,)``.

        Taken from :attr:`atom_masses` when given, otherwise looked up by
        element symbol.
        """
        if self.atom_masses is not None:
            return self.atom_masses
        return np.asarray(_masses(self.species), dtype=float)

    @property
    def lengths(self) -> np.ndarray:
        """Lengths of the three lattice vectors, in nm."""
        return np.linalg.norm(self.cell, axis=1)

    @property
    def in_plane_area(self) -> float:
        """Area of the in-plane cell, in nm^2."""
        return float(np.linalg.norm(np.cross(self.cell[0], self.cell[1])))

    def indices_of(self, species) -> np.ndarray:
        """Indices of atoms whose symbol is in `species`."""
        if isinstance(species, str):
            species = (species,)
        wanted = set(species)
        return np.array(
            [i for i, s in enumerate(self.species) if s in wanted], dtype=int
        )

    def unique_species(self) -> list[str]:
        seen: list[str] = []
        for s in self.species:
            if s not in seen:
                seen.append(s)
        return seen

    # ------------------------------------------------------------ transforms
    def copy(self) -> "Structure":
        return Structure(
            species=self.species.copy(),
            positions=self.positions.copy(),
            cell=self.cell.copy(),
            pbc=self.pbc,
            label=self.label,
            atom_masses=(
                None if self.atom_masses is None else self.atom_masses.copy()
            ),
        )

    def translated(self, shift) -> "Structure":
        out = self.copy()
        out.positions = out.positions + np.asarray(shift, dtype=float)
        return out

    def with_z_origin(self, z0: float) -> "Structure":
        """Shift so that `z0` becomes z = 0.

        Flexoelectric bookkeeping is referred to the neutral plane of the
        sheet; use this to put that plane at the origin.
        """
        return self.translated((0.0, 0.0, -z0))

    def centered_on_mass_plane(self, species=None) -> "Structure":
        """Shift z so the mass-weighted mean plane of `species` sits at z = 0."""
        idx = (
            np.arange(self.n_atoms)
            if species is None
            else self.indices_of(species)
        )
        w = self.masses[idx]
        z0 = float(np.sum(w * self.positions[idx, 2]) / np.sum(w))
        return self.with_z_origin(z0)


@dataclass
class ConfigSeries:
    """A series of configurations of the same material at growing deformation.

    The first entry is the undeformed reference by default; displacements are
    measured against ``self.reference``.

    Parameters
    ----------
    structures
        The configurations, ordered by increasing deformation.
    amplitudes
        The control parameter for each configuration (corrugation amplitude,
        bending index, ...). Defaults to ``0, 1, 2, ...``.
    reference_index
        Which configuration is the undeformed one.
    name
        Material / dataset name.
    """

    structures: list[Structure]
    amplitudes: np.ndarray | None = None
    reference_index: int = 0
    name: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.structures = list(self.structures)
        if not self.structures:
            raise ValueError("ConfigSeries needs at least one Structure")
        if self.amplitudes is None:
            self.amplitudes = np.arange(len(self.structures), dtype=float)
        self.amplitudes = np.asarray(self.amplitudes, dtype=float)
        if len(self.amplitudes) != len(self.structures):
            raise ValueError(
                f"{len(self.amplitudes)} amplitudes for "
                f"{len(self.structures)} structures"
            )
        n_at = {s.n_atoms for s in self.structures}
        if len(n_at) != 1:
            raise ValueError(
                "all configurations must have the same number of atoms, got "
                f"{sorted(n_at)}"
            )
        first = list(self.structures[self.reference_index].species)
        for i, s in enumerate(self.structures):
            if list(s.species) != first:
                raise ValueError(
                    f"configuration {i} has a different species order than the "
                    "reference; reorder the atoms so they correspond one-to-one"
                )

    def __len__(self) -> int:
        return len(self.structures)

    def __iter__(self) -> Iterable[Structure]:
        return iter(self.structures)

    def __getitem__(self, i):
        return self.structures[i]

    @property
    def reference(self) -> Structure:
        return self.structures[self.reference_index]

    def wavevectors(self, axis: int = 0, n_periods: int = 1) -> np.ndarray:
        """Fundamental wavevector ``2*pi*n_periods / L_axis`` per configuration.

        The corrugation wavelength is set by the supercell, so this is fixed by
        geometry and needs no fitting. Pass `n_periods` if the supercell holds
        more than one period of the ripple.
        """
        return np.array(
            [2.0 * np.pi * n_periods / s.lengths[axis] for s in self.structures]
        )

    def with_z_origins(self, z0: Sequence[float]) -> "ConfigSeries":
        """Return a copy with a per-configuration z shift applied."""
        z0 = np.asarray(z0, dtype=float)
        if len(z0) != len(self):
            raise ValueError(f"{len(z0)} offsets for {len(self)} configurations")
        return ConfigSeries(
            [s.with_z_origin(z) for s, z in zip(self.structures, z0)],
            amplitudes=self.amplitudes,
            reference_index=self.reference_index,
            name=self.name,
            metadata=dict(self.metadata),
        )

    def centered_on_mass_plane(self, species=None) -> "ConfigSeries":
        """Put each configuration's mass mean-plane at z = 0."""
        return ConfigSeries(
            [s.centered_on_mass_plane(species) for s in self.structures],
            amplitudes=self.amplitudes,
            reference_index=self.reference_index,
            name=self.name,
            metadata=dict(self.metadata),
        )
