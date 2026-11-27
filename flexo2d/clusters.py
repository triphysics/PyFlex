"""Partitioning a structure into local dipole units ("clusters").

A flexoelectric calculation from point charges needs the sheet cut into
charge-neutral local units whose dipole moment is well defined. For a TMD each
unit is one metal atom plus its chalcogen shell; for h-BN a B with its three N,
and so on.

:class:`NeighbourShell` does this generically from a distance cutoff, including
periodic images, and assigns each shared neighbour a weight of 1/(number of
bonds it makes) so the units tile the sheet exactly once. :class:`ExplicitGroups`
is the escape hatch for structures where you want to dictate the grouping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .structure import Structure


@dataclass(frozen=True)
class Cluster:
    """One local dipole unit.

    Attributes
    ----------
    center
        Index of the central atom.
    members
        Atom indices contributing to this unit, the centre included. An atom
        may appear more than once when several of its periodic images bond to
        the same centre.
    weights
        Weight of each member, shape ``(M,)``.
    offsets
        Cartesian lattice-image offset in nm to add to each member's position,
        shape ``(M, 3)``. Needed because a dipole must be built from unwrapped
        coordinates.
    """

    center: int
    members: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray

    def positions(self, structure: Structure) -> np.ndarray:
        """Unwrapped member coordinates, shape ``(M, 3)``."""
        return structure.positions[self.members] + self.offsets


class ClusterScheme(Protocol):
    """Anything that can cut a structure into local dipole units."""

    def clusters(self, structure: Structure) -> list[Cluster]:
        ...


def _image_range(structure: Structure, cutoff: float) -> list[np.ndarray]:
    """Lattice translation vectors to search, given a distance `cutoff`."""
    recip = np.linalg.inv(structure.cell).T       # rows: b_i with b_i.a_j = delta_ij
    spans = []
    for i in range(3):
        if not structure.pbc[i]:
            spans.append(np.array([0]))
            continue
        spacing = 1.0 / np.linalg.norm(recip[i])  # interplanar spacing along a_i
        n = int(np.ceil(cutoff / spacing))
        spans.append(np.arange(-n, n + 1))
    return [
        np.array([i, j, k])
        for i in spans[0]
        for j in spans[1]
        for k in spans[2]
    ]


@dataclass
class NeighbourShell:
    """Build one cluster per `center_species` atom from a distance cutoff.

    Parameters
    ----------
    center_species
        Element symbol(s) acting as cluster centres (the metal in a TMD).
    shell_species
        Element symbol(s) allowed in the shell. ``None`` means "everything
        that is not a centre".
    cutoff
        Bond cutoff in nm. Pick it in the gap between the first and second
        neighbour shell; :meth:`report_shells` helps.
    share_weights
        Divide each neighbour's weight by the number of bonds it makes, so the
        clusters tile the sheet exactly once. Leave this on unless you know
        you want overlapping units.
    require_neutral
        Tolerance for the net charge of a cluster. Checked by
        :func:`flexo2d.fields.dipole_field`, not here.
    """

    center_species: Sequence[str] | str
    shell_species: Sequence[str] | str | None = None
    cutoff: float = 0.30
    share_weights: bool = True
    min_coordination: int = 1

    def _shell_indices(self, structure: Structure) -> np.ndarray:
        """Indices of the atoms allowed in a shell."""
        if self.shell_species is None:
            centre_set = (
                {self.center_species}
                if isinstance(self.center_species, str)
                else set(self.center_species)
            )
            return structure.indices_of(
                [s for s in structure.unique_species() if s not in centre_set]
            )
        return structure.indices_of(self.shell_species)

    def clusters(self, structure: Structure) -> list[Cluster]:
        centers = structure.indices_of(self.center_species)
        if len(centers) == 0:
            raise ValueError(
                f"no atoms of species {self.center_species!r} in the structure "
                f"(present: {structure.unique_species()})"
            )
        shell = self._shell_indices(structure)

        images = _image_range(structure, self.cutoff)
        shifts = np.array([img @ structure.cell for img in images])  # (I, 3)

        # bonds[c] = list of (shell_atom_index, cartesian_offset)
        bonds: dict[int, list[tuple[int, np.ndarray]]] = {}
        bond_count = np.zeros(structure.n_atoms, dtype=float)
        cutoff2 = self.cutoff**2
        for c in centers:
            rc = structure.positions[c]
            found: list[tuple[int, np.ndarray]] = []
            for j in shell:
                d0 = structure.positions[j] - rc
                d = d0[None, :] + shifts                      # (I, 3)
                dist2 = np.einsum("ij,ij->i", d, d)
                hit = np.nonzero(dist2 <= cutoff2)[0]
                for h in hit:
                    # an atom is never its own neighbour, but its periodic
                    # images may be (matters when centre and shell species
                    # are the same element)
                    if int(j) == int(c) and dist2[h] < 1e-12:
                        continue
                    found.append((int(j), shifts[h].copy()))
                    bond_count[j] += 1.0
            if len(found) < self.min_coordination:
                raise ValueError(
                    f"atom {c} ({structure.species[c]}) has only {len(found)} "
                    f"neighbour(s) within {self.cutoff} nm; the cutoff is too "
                    "small or the structure is broken"
                )
            bonds[int(c)] = found

        out: list[Cluster] = []
        for c in centers:
            found = bonds[int(c)]
            members = [int(c)] + [j for j, _ in found]
            offsets = [np.zeros(3)] + [o for _, o in found]
            if self.share_weights:
                weights = [1.0] + [1.0 / bond_count[j] for j, _ in found]
            else:
                weights = [1.0] * len(members)
            out.append(
                Cluster(
                    center=int(c),
                    members=np.array(members, dtype=int),
                    weights=np.array(weights, dtype=float),
                    offsets=np.array(offsets, dtype=float),
                )
            )
        return out

    # ------------------------------------------------------------ diagnostics
    def report_shells(
        self, structure: Structure, max_distance: float = 0.6, n: int = 12,
        group_tol: float = 5e-3,
    ) -> list[tuple[float, float, str, str]]:
        """Coordination shells around the centres, to help choose :attr:`cutoff`.

        Distances within `group_tol` of each other are reported as one shell.
        Look for the gap in the distance column and put :attr:`cutoff` in it.

        Returns
        -------
        list of (distance_nm, neighbours_per_centre, centre_symbol, shell_symbol)
            Sorted by distance, at most `n` entries. ``neighbours_per_centre``
            counts periodic images, so it is the multiplicity a cluster would
            actually pick up.
        """
        centers = structure.indices_of(self.center_species)
        if len(centers) == 0:
            raise ValueError(
                f"no atoms of species {self.center_species!r} in the structure "
                f"(present: {structure.unique_species()})"
            )
        shell = self._shell_indices(structure)
        images = _image_range(structure, max_distance)
        shifts = np.array([img @ structure.cell for img in images])

        rows: list[tuple[float, str, str]] = []
        for c in centers:
            rc = structure.positions[c]
            for j in shell:
                d = (structure.positions[j] - rc)[None, :] + shifts
                for dist in np.sqrt(np.einsum("ij,ij->i", d, d)):
                    if 1e-6 < dist <= max_distance:
                        rows.append(
                            (float(dist), str(structure.species[c]),
                             str(structure.species[j]))
                        )
        rows.sort()

        grouped: list[tuple[float, float, str, str]] = []
        i = 0
        n_centers = len(centers)
        while i < len(rows) and len(grouped) < n:
            j = i
            while j < len(rows) and rows[j][0] - rows[i][0] <= group_tol:
                j += 1
            chunk = rows[i:j]
            pairs = sorted({(a, b) for _, a, b in chunk})
            grouped.append((
                float(np.mean([r[0] for r in chunk])),
                len(chunk) / n_centers,
                "/".join(sorted({a for a, _ in pairs})),
                "/".join(sorted({b for _, b in pairs})),
            ))
            i = j
        return grouped


@dataclass
class ExplicitGroups:
    """Clusters given by hand.

    Parameters
    ----------
    groups
        One entry per cluster: ``(center_index, member_indices, weights)``.
        Offsets are taken to be zero, so supply coordinates that are already
        unwrapped.
    """

    groups: Sequence[tuple[int, Sequence[int], Sequence[float]]]

    def clusters(self, structure: Structure) -> list[Cluster]:
        out = []
        for center, members, weights in self.groups:
            members = np.asarray(members, dtype=int)
            weights = np.asarray(weights, dtype=float)
            if len(members) != len(weights):
                raise ValueError(
                    f"cluster at {center}: {len(members)} members but "
                    f"{len(weights)} weights"
                )
            if members.max(initial=0) >= structure.n_atoms:
                raise ValueError(
                    f"cluster at {center} refers to atom "
                    f"{members.max()} but the structure has "
                    f"{structure.n_atoms} atoms"
                )
            out.append(
                Cluster(
                    center=int(center),
                    members=members,
                    weights=weights,
                    offsets=np.zeros((len(members), 3)),
                )
            )
        return out
