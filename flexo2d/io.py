"""Readers that turn external data into a :class:`~flexo2d.structure.ConfigSeries`.

Supported inputs
----------------
``read_npz`` / ``write_npz``
    The native format: one ``.npz`` holding every configuration. Use this to
    cache a dataset once it has been assembled.
``read_poscar_series``
    A list of VASP POSCAR/CONTCAR files, one per deformation amplitude. This is
    the normal route for a new material: relax the rippled supercells, then
    point the reader at the CONTCARs.
``read_extxyz_series``
    Extended-XYZ frames, which can also carry per-atom charges in a
    ``charges`` column.
``series_from_layer_arrays``
    One array per atomic layer, shaped ``(n_config, n_cell, 3)``, for data that
    is organised by crystallographic layer rather than by configuration.

Lengths are nanometres everywhere inside the package; the POSCAR and XYZ
readers convert from Angstrom for you.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

from .structure import ConfigSeries, Structure

ANGSTROM_TO_NM = 0.1


# --------------------------------------------------------------- native npz
def write_npz(path: str, series: ConfigSeries,
              charges: np.ndarray | None = None) -> None:
    """Save a series (and optionally a charge table) to one ``.npz``."""
    payload = {
        "species": np.array([str(s) for s in series.reference.species]),
        "positions": np.array([s.positions for s in series]),
        "cells": np.array([s.cell for s in series]),
        "pbc": np.array(series.reference.pbc),
        "amplitudes": series.amplitudes,
        "reference_index": np.array(series.reference_index),
        "name": np.array(series.name),
        "labels": np.array([s.label for s in series]),
    }
    if charges is not None:
        payload["charges"] = np.asarray(charges, dtype=float)
    np.savez_compressed(path, **payload)


def read_npz(path: str) -> tuple[ConfigSeries, np.ndarray | None]:
    """Load a series written by :func:`write_npz`.

    Returns ``(series, charges_or_None)``.
    """
    d = np.load(path, allow_pickle=False)
    species = [str(s) for s in d["species"]]
    pbc = tuple(bool(p) for p in d["pbc"])
    labels = [str(x) for x in d["labels"]] if "labels" in d else None
    structures = [
        Structure(
            species=species,
            positions=pos,
            cell=cell,
            pbc=pbc,
            label=(labels[i] if labels else f"config {i}"),
        )
        for i, (pos, cell) in enumerate(zip(d["positions"], d["cells"]))
    ]
    series = ConfigSeries(
        structures,
        amplitudes=d["amplitudes"],
        reference_index=int(d["reference_index"]),
        name=str(d["name"]),
    )
    return series, (d["charges"] if "charges" in d else None)


# ------------------------------------------------------------------- POSCAR
def read_poscar(path: str) -> Structure:
    """Read one VASP POSCAR / CONTCAR. Lengths converted to nm."""
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    if len(lines) < 8:
        raise ValueError(f"{path}: too short to be a POSCAR")

    comment = lines[0].strip()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(v) for v in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                      # negative scale means target volume
        vol = abs(np.linalg.det(cell))
        scale = (abs(scale) / vol) ** (1.0 / 3.0)
    cell *= scale

    tok5 = lines[5].split()
    if all(t.lstrip("+-").isdigit() for t in tok5):
        raise ValueError(
            f"{path}: no species-name line. flexo2d needs element symbols; "
            "add a VASP-5 style line above the atom counts."
        )
    names = tok5
    counts = [int(v) for v in lines[6].split()]
    if len(names) != len(counts):
        raise ValueError(
            f"{path}: {len(names)} species names but {len(counts)} counts"
        )
    species = [n for n, c in zip(names, counts) for _ in range(c)]
    n_atoms = sum(counts)

    i = 7
    if lines[i].strip()[:1].upper() in ("S",):     # Selective dynamics
        i += 1
    mode = lines[i].strip()[:1].upper()
    i += 1
    raw = np.array(
        [[float(v) for v in lines[i + j].split()[:3]] for j in range(n_atoms)]
    )
    if mode in ("C", "K"):                          # Cartesian
        pos = raw * scale
    else:                                           # Direct / fractional
        pos = raw @ cell

    return Structure(
        species=species,
        positions=pos * ANGSTROM_TO_NM,
        cell=cell * ANGSTROM_TO_NM,
        label=comment or os.path.basename(path),
    )


def read_poscar_series(paths: Sequence[str], amplitudes=None, name: str = "",
                       reference_index: int = 0) -> ConfigSeries:
    """Read one POSCAR per deformation amplitude, in order."""
    if len(paths) == 0:
        raise ValueError("no POSCAR paths given")
    return ConfigSeries(
        [read_poscar(p) for p in paths],
        amplitudes=amplitudes,
        reference_index=reference_index,
        name=name or "poscar series",
    )


# ------------------------------------------------------------------ ext-xyz
def read_extxyz_series(path: str, amplitudes=None, name: str = "",
                       reference_index: int = 0,
                       charge_column: str | None = "charges"
                       ) -> tuple[ConfigSeries, np.ndarray | None]:
    """Read every frame of an extended-XYZ file. Lengths converted to nm.

    Returns ``(series, charges_or_None)``. The charge column is picked up when
    the frame's ``Properties`` string declares `charge_column`.
    """
    structures: list[Structure] = []
    charge_rows: list[list[float]] = []
    with open(path) as fh:
        while True:
            head = fh.readline()
            if not head.strip():
                break
            n = int(head)
            comment = fh.readline()
            cell, props = _parse_extxyz_comment(comment)
            fields = _extxyz_fields(props)
            species, pos, charges = [], [], []
            for _ in range(n):
                tok = fh.readline().split()
                rec = _extxyz_record(tok, fields)
                species.append(rec["species"])
                pos.append(rec["pos"])
                if charge_column and charge_column in rec:
                    charges.append(rec[charge_column])
            structures.append(
                Structure(
                    species=species,
                    positions=np.array(pos) * ANGSTROM_TO_NM,
                    cell=cell * ANGSTROM_TO_NM,
                    label=f"frame {len(structures)}",
                )
            )
            if charges:
                charge_rows.append(charges)
    if not structures:
        raise ValueError(f"{path}: no frames found")
    series = ConfigSeries(
        structures, amplitudes=amplitudes, reference_index=reference_index,
        name=name or os.path.basename(path),
    )
    table = np.array(charge_rows) if len(charge_rows) == len(structures) else None
    return series, table


def _parse_extxyz_comment(comment: str) -> tuple[np.ndarray, str]:
    lat = _extxyz_quoted(comment, "Lattice")
    if lat is None:
        raise ValueError("extxyz frame has no Lattice=\"...\" in its comment line")
    vals = [float(v) for v in lat.split()]
    if len(vals) != 9:
        raise ValueError(f"Lattice needs 9 numbers, got {len(vals)}")
    props = _extxyz_quoted(comment, "Properties") or "species:S:1:pos:R:3"
    return np.array(vals).reshape(3, 3), props


def _extxyz_quoted(text: str, key: str) -> str | None:
    i = text.find(key + "=")
    if i < 0:
        return None
    j = i + len(key) + 1
    if text[j] == '"':
        k = text.index('"', j + 1)
        return text[j + 1:k]
    k = text.find(" ", j)
    return text[j:] if k < 0 else text[j:k]


def _extxyz_fields(props: str) -> list[tuple[str, str, int]]:
    parts = props.split(":")
    out = []
    for i in range(0, len(parts) - 2, 3):
        out.append((parts[i], parts[i + 1], int(parts[i + 2])))
    return out


def _extxyz_record(tokens: Sequence[str], fields) -> dict:
    rec: dict = {}
    at = 0
    for name, kind, count in fields:
        chunk = tokens[at:at + count]
        at += count
        if name == "species":
            rec["species"] = chunk[0]
        elif name == "pos":
            rec["pos"] = [float(v) for v in chunk]
        elif kind in ("R", "I"):
            rec[name] = float(chunk[0]) if count == 1 else [float(v) for v in chunk]
        else:
            rec[name] = chunk[0] if count == 1 else list(chunk)
    return rec


# ---------------------------------------------- layer-organised arrays
def series_from_layer_arrays(
    layers: Mapping[str, np.ndarray],
    cell_lengths: np.ndarray,
    *,
    amplitudes=None,
    species_of: Mapping[str, str] | None = None,
    charges: Mapping[str, np.ndarray] | None = None,
    z_origins=None,
    name: str = "",
    reference_index: int = 0,
    order: Sequence[str] | None = None,
) -> tuple[ConfigSeries, np.ndarray | None]:
    """Assemble a series from one coordinate array per atomic layer.

    For data organised as a separate array per crystallographic layer
    (``S1``, ``S2``, ``Mo``), each shaped ``(n_config, n_cell, 3)``.

    Parameters
    ----------
    layers
        ``{layer_name: array}`` with every array shaped
        ``(n_config, n_cell, 3)``, in nm.
    cell_lengths
        ``(n_config, 3)`` orthorhombic cell lengths in nm, or ``(n_config,)``
        giving only the ripple direction -- then `transverse` and `vacuum`
        must be supplied through a ``(n_config, 3)`` array instead.
    species_of
        ``{layer_name: element_symbol}``. Defaults to stripping trailing
        digits off the layer name (``"S1" -> "S"``).
    charges
        Optional ``{layer_name: (n_config, n_cell)}`` per-atom charges;
        returned as a single ``(n_config, n_atoms)`` table in the same atom
        order as the structures.
    z_origins
        Optional ``(n_config,)`` z offsets subtracted from every atom, used to
        put the sheet's neutral plane at z = 0.
    order
        Layer order in the assembled atom list. Defaults to ``sorted(layers)``.

    Returns
    -------
    (series, charge_table_or_None)
    """
    order = list(order) if order is not None else sorted(layers)
    missing = [k for k in order if k not in layers]
    if missing:
        raise KeyError(f"layers {missing} not in the given arrays")

    arrays = {k: np.asarray(layers[k], dtype=float) for k in order}
    shapes = {k: a.shape for k, a in arrays.items()}
    for k, a in arrays.items():
        if a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(
                f"layer {k!r} must be (n_config, n_cell, 3), got {a.shape}"
            )
    n_config = {s[0] for s in shapes.values()}
    if len(n_config) != 1:
        raise ValueError(f"layers disagree on the configuration count: {shapes}")
    n_config = n_config.pop()

    cell_lengths = np.asarray(cell_lengths, dtype=float)
    if cell_lengths.shape != (n_config, 3):
        raise ValueError(
            f"cell_lengths must be ({n_config}, 3), got {cell_lengths.shape}"
        )

    def symbol(layer: str) -> str:
        if species_of and layer in species_of:
            return species_of[layer]
        return layer.rstrip("0123456789") or layer

    species: list[str] = []
    for k in order:
        species += [symbol(k)] * arrays[k].shape[1]

    z0 = np.zeros(n_config) if z_origins is None else np.asarray(z_origins, float)
    if z0.shape != (n_config,):
        raise ValueError(f"z_origins must be ({n_config},), got {z0.shape}")

    structures = []
    for i in range(n_config):
        pos = np.vstack([arrays[k][i] for k in order]).copy()
        pos[:, 2] -= z0[i]
        structures.append(
            Structure(
                species=species,
                positions=pos,
                cell=np.diag(cell_lengths[i]),
                label=f"config {i}",
            )
        )

    series = ConfigSeries(
        structures, amplitudes=amplitudes, reference_index=reference_index,
        name=name or "layer series",
    )

    table = None
    if charges is not None:
        miss = [k for k in order if k not in charges]
        if miss:
            raise KeyError(f"no charges given for layers {miss}")
        table = np.hstack([np.asarray(charges[k], dtype=float) for k in order])
        if table.shape != (n_config, len(species)):
            raise ValueError(
                f"charge table came out {table.shape}, expected "
                f"{(n_config, len(species))}"
            )
    return series, table


# ------------------------------------------------------------------- writing
NM_TO_ANGSTROM = 10.0


def write_poscar(path: str, structure: Structure, *, direct: bool = True,
                 comment: str | None = None) -> None:
    """Write one VASP POSCAR. Lengths converted from nm to Angstrom.

    Atoms are grouped by species, as VASP requires.
    """
    order: list[str] = []
    for s in structure.species:
        if s not in order:
            order.append(str(s))
    index = [i for sp in order for i in structure.indices_of(sp)]
    counts = [int(np.sum(structure.species == sp)) for sp in order]

    cell = structure.cell * NM_TO_ANGSTROM
    pos = structure.positions[index] * NM_TO_ANGSTROM

    lines = [comment or (structure.label or "generated by flexo2d"), "   1.0"]
    for row in cell:
        lines.append("  {:>21.16f} {:>21.16f} {:>21.16f}".format(*row))
    lines.append("   " + "  ".join(f"{s:>3s}" for s in order))
    lines.append("   " + "  ".join(f"{c:>3d}" for c in counts))
    if direct:
        lines.append("Direct")
        frac = pos @ np.linalg.inv(cell)
        rows = frac
    else:
        lines.append("Cartesian")
        rows = pos
    for row in rows:
        lines.append("  {:>19.16f} {:>19.16f} {:>19.16f}".format(*row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_poscar_series(directory: str, series: ConfigSeries, *,
                        prefix: str = "POSCAR", suffix: str = ".vasp",
                        direct: bool = True) -> list[str]:
    """Write one POSCAR per configuration into `directory`.

    Files are named ``<prefix>_<index><suffix>``, e.g. ``POSCAR_00.vasp``. The
    ``.vasp`` extension is what VESTA and most viewers recognise. Index zero is
    the flat reference, and the index is zero padded so that sorting the
    filenames keeps the series in amplitude order -- which matters, because
    reading them back out of order puts the reference in the wrong place.

    Returns the paths written.
    """
    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, s in enumerate(series):
        p = os.path.join(directory, f"{prefix}_{i:02d}{suffix}")
        write_poscar(
            p, s, direct=direct,
            comment=f"{series.name} | amplitude {series.amplitudes[i]:g}",
        )
        paths.append(p)
    return paths


def write_extxyz(path: str, series: ConfigSeries,
                 charges: np.ndarray | None = None) -> None:
    """Write a series as an extended-XYZ trajectory, nm converted to Angstrom.

    `charges`, shape ``(n_config, n_atoms)``, is written as a ``charges``
    column that :func:`read_extxyz_series` reads back.
    """
    if charges is not None:
        charges = np.asarray(charges, dtype=float)
        if charges.shape != (len(series), series.reference.n_atoms):
            raise ValueError(
                f"charges must be {(len(series), series.reference.n_atoms)}, "
                f"got {charges.shape}"
            )
    with open(path, "w") as fh:
        for i, s in enumerate(series):
            lat = " ".join(f"{v:.10f}" for v in (s.cell * NM_TO_ANGSTROM).ravel())
            props = "species:S:1:pos:R:3"
            if charges is not None:
                props += ":charges:R:1"
            fh.write(f"{s.n_atoms}\n")
            fh.write(
                f'Lattice="{lat}" Properties={props} '
                f'amplitude={series.amplitudes[i]:g}\n'
            )
            for j in range(s.n_atoms):
                x, y, z = s.positions[j] * NM_TO_ANGSTROM
                line = f"{str(s.species[j]):<3s} {x:>18.10f} {y:>18.10f} {z:>18.10f}"
                if charges is not None:
                    line += f" {charges[i, j]:>14.8f}"
                fh.write(line + "\n")
