"""Building the input structures: supercells and rippled (bent) sheets.

Generating the deformation series matters as much as analysing it. The ripple
has to be an **isometric** bend -- one that preserves the arc length of the
sheet's neutral surface -- because anything else stretches the sheet and mixes
a membrane (piezoelectric) response into the bending signal you are trying to
measure.

The naive construction ``z += A*cos(k*x)`` does exactly that wrong thing: it
leaves x alone, so the sheet gets longer by a factor
``(1/2pi) * integral sqrt(1 + (A k sin u)^2) du`` and acquires a uniform
tensile strain that grows as ``(A k)^2 / 4``. For the MoS2 dataset shipped in
``data/`` that would be 2.3% of spurious strain at the largest amplitude -- far
larger than the real response.

:func:`isometric_ripple` instead maps the flat coordinate onto *arc length*
along the bent mid-surface, shrinks the cell accordingly, and places each atom
along the local surface normal at its original height. Layers above and below
the neutral plane then stretch and compress as ``+- h * curvature``, which is
the physics of bending.

The MoS2 dataset shipped in ``data/`` came from relaxed DFT, and conserves its
neutral surface arc length to within 0.1% across the whole series.

Typical use: build a flat supercell, ripple it at several amplitudes, write the
result out for DFT relaxation, then read the relaxed geometries back.

.. code-block:: python

    flat = supercell(read_poscar("POSCAR.unitcell"), 8, 1)
    series = ripple_series(flat, amplitudes=[0.0, 0.02, 0.04, 0.06])
    write_poscar_series("ripples/", series)
"""

from __future__ import annotations

import warnings

import numpy as np

from .structure import ConfigSeries, Structure

__all__ = [
    "supercell",
    "isometric_ripple",
    "ripple_series",
    "arc_length",
    "projected_length",
    "max_amplitude",
]


def _check_axis_is_cartesian(structure: Structure, axis: int) -> None:
    v = structure.cell[axis]
    off = np.delete(v, axis)
    if np.linalg.norm(off) > 1e-8 * max(np.linalg.norm(v), 1.0):
        raise ValueError(
            f"the ripple axis (lattice vector {axis} = {v}) must be parallel to "
            "a Cartesian axis; reorient the cell first"
        )


def supercell(structure: Structure, nx: int, ny: int = 1, nz: int = 1
              ) -> Structure:
    """Tile `structure` ``nx x ny x nz`` times.

    Parameters
    ----------
    structure
        The unit cell.
    nx, ny, nz
        Repeat counts along each lattice vector.
    """
    reps = (int(nx), int(ny), int(nz))
    if any(r < 1 for r in reps):
        raise ValueError(f"repeat counts must be >= 1, got {reps}")

    species, positions = [], []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                shift = i * structure.cell[0] + j * structure.cell[1] + k * structure.cell[2]
                species.extend(list(structure.species))
                positions.append(structure.positions + shift)

    cell = structure.cell * np.array(reps)[:, None]
    masses = structure.atom_masses
    return Structure(
        species=species,
        positions=np.vstack(positions),
        cell=cell,
        pbc=structure.pbc,
        label=f"{structure.label} {reps[0]}x{reps[1]}x{reps[2]}".strip(),
        atom_masses=(None if masses is None else np.tile(masses, int(np.prod(reps)))),
    )


def arc_length(amplitude: float, projected_length: float, n_periods: int = 1
               ) -> float:
    """Arc length of ``z = A cos(k x)`` over one cell of the projected length.

    ``k = 2 pi n_periods / projected_length``.
    """
    if projected_length <= 0:
        raise ValueError("projected_length must be positive")
    k = 2.0 * np.pi * n_periods / projected_length
    # integral over a whole number of periods, so average over [0, 2 pi)
    u = np.linspace(0.0, 2.0 * np.pi, 20001)
    integrand = np.sqrt(1.0 + (amplitude * k * np.sin(u)) ** 2)
    mean = float(np.trapz(integrand, u) / (2.0 * np.pi))
    return projected_length * mean


def max_amplitude(flat_length: float, n_periods: int = 1) -> float:
    """Largest amplitude an isometric ripple can have in a cell of this length.

    One period of ``A cos(k x)`` has arc length of at least its total
    variation, ``4 A``, however much the cell is squeezed. So a sheet of length
    ``flat_length`` can only be folded into `n_periods` periods while keeping
    its length if ``4 * n_periods * A < flat_length``. At that bound the
    projected cell length goes to zero.

    In practice the usable amplitude is well below this: see
    :func:`isometric_ripple`, which also refuses ripples whose offset surface
    would self-intersect.
    """
    return float(flat_length) / (4.0 * int(n_periods))


def projected_length(amplitude: float, flat_length: float, n_periods: int = 1,
                     tol: float = 1e-12) -> float:
    """Cell length after bending a sheet of length `flat_length` isometrically.

    Solves ``arc_length(A, L) == flat_length`` for ``L``, i.e. finds how much
    the cell must shrink so that the neutral surface keeps its length.

    Raises
    ------
    ValueError
        If no solution exists, i.e. ``4 * n_periods * amplitude >=
        flat_length``. Without this check the search collapses to ``L = 0`` and
        silently returns a degenerate cell.
    """
    amplitude = abs(float(amplitude))
    flat_length = float(flat_length)
    if flat_length <= 0:
        raise ValueError("flat_length must be positive")
    if amplitude == 0.0:
        return flat_length

    limit = max_amplitude(flat_length, n_periods)
    if amplitude >= limit:
        raise ValueError(
            f"amplitude {amplitude:g} nm cannot be reached isometrically in a "
            f"cell of length {flat_length:g} nm with {n_periods} period(s): "
            f"the arc length of the ripple ({4 * n_periods * amplitude:g} nm at "
            f"minimum) already exceeds the sheet. The hard bound is "
            f"A < {limit:g} nm; use a longer cell, a smaller amplitude, or "
            "fewer periods."
        )

    lo, hi = 1e-12 * flat_length, flat_length
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if arc_length(amplitude, mid, n_periods) < flat_length:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * flat_length:
            break
    return 0.5 * (lo + hi)


def isometric_ripple(
    flat: Structure,
    amplitude: float,
    *,
    n_periods: int = 1,
    axis: int = 0,
    neutral_plane: float | None = None,
    preserve_arc_length: bool = True,
    phase: float = 0.0,
    wrap: bool = True,
    center: bool = True,
    max_bending_strain: float = 0.3,
    n_samples: int = 20001,
    label: str | None = None,
) -> Structure:
    """Bend a flat sheet into a sinusoidal ripple without stretching it.

    Parameters
    ----------
    flat
        The flat starting structure, periodic along `axis`.
    amplitude
        Out-of-plane amplitude ``A`` of the neutral surface, in nm. The peak to
        trough height is ``2 A``.
    n_periods
        Ripple periods in the cell.
    axis
        Ripple propagation direction (0 = first lattice vector).
    neutral_plane
        z of the surface that keeps its length. Defaults to the mass-weighted
        mean plane of the sheet, which is the natural bending neutral surface.
    preserve_arc_length
        Shrink the cell so the neutral surface keeps its length (the physical
        choice). Setting this ``False`` keeps the cell length and therefore
        stretches the sheet -- only useful for demonstrating what that does.
    phase
        Phase offset of the ripple in radians, i.e. ``A cos(k x + phase)``.
        Changing it slides the ripple relative to the atomic lattice. Results
        that move with it are registry artifacts, not material response --
        a cheap and strong diagnostic.
    wrap
        Fold the result back into the cell. Bending displaces atoms along the
        surface normal, which can push one just outside; the structure is
        periodic either way, but written files are tidier wrapped.
    center
        Put the sheet in the middle of the vacuum, by shifting its neutral
        surface to ``cell_z / 2``. Without this a sheet built around ``z = 0``
        straddles the cell boundary: half of it has negative z and viewers draw
        it as two half-sheets at the top and bottom of the box. The shift is the
        same constant for every amplitude, so it cancels out of the
        displacement field and does not affect any coefficient.
    max_bending_strain
        Warn when the outermost layer's bending strain ``|h| * curvature``
        exceeds this. Values approaching 1 mean the sheet is folding through
        itself and are refused outright.
    n_samples
        Grid points used to invert arc length; raise for very large amplitudes.
    label
        Overrides the structure label.

    Returns
    -------
    Structure
        The rippled sheet, with the cell shortened along `axis` when
        `preserve_arc_length` is set.

    Raises
    ------
    ValueError
        If the amplitude cannot be reached isometrically (see
        :func:`max_amplitude`), or if the offset surface would self-intersect,
        or if the result would not fit in its own cell.
    """
    _check_axis_is_cartesian(flat, axis)
    if not flat.pbc[axis]:
        raise ValueError(
            f"axis {axis} is not periodic (pbc={flat.pbc}); a ripple has to "
            "close on itself, so the ripple direction must be periodic"
        )
    l_flat = float(flat.cell[axis, axis])
    if l_flat <= 0:
        raise ValueError(f"cell length along axis {axis} must be positive")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")

    if neutral_plane is None:
        w = flat.masses
        neutral_plane = float(np.sum(w * flat.positions[:, 2]) / np.sum(w))

    out = flat.copy()
    out.label = label if label is not None else f"{flat.label} ripple A={amplitude:g}"

    def finish(structure: Structure) -> Structure:
        """Centre the sheet in the vacuum; see the `center` argument."""
        if center:
            shift = 0.5 * structure.cell[2, 2] - neutral_plane
            structure.positions = structure.positions + np.array([0.0, 0.0, shift])
        return structure

    if amplitude == 0.0:
        return finish(out)

    l_proj = (
        projected_length(amplitude, l_flat, n_periods)
        if preserve_arc_length else l_flat
    )
    k = 2.0 * np.pi * n_periods / l_proj

    # An offset surface a distance h from a curve of curvature kappa is only
    # regular while |h * kappa| < 1; past that the inner side folds through
    # itself and the "structure" is nonsense.
    height_all = flat.positions[:, 2] - neutral_plane
    h_max = float(np.abs(height_all).max())
    curvature = abs(amplitude) * k * k
    strain = h_max * curvature
    if strain >= 1.0:
        raise ValueError(
            f"amplitude {amplitude:g} nm gives curvature {curvature:g} /nm, so "
            f"the outermost layer (h = {h_max:g} nm) would be bent through "
            f"|h*kappa| = {strain:.3g} >= 1: the offset surface self-intersects "
            "and the sheet folds through itself. Reduce the amplitude or use a "
            "longer cell."
        )
    if strain > max_bending_strain:
        warnings.warn(
            f"bending strain in the outermost layer is {strain:.1%} "
            f"(|h*kappa| with h = {h_max:g} nm, curvature {curvature:g} /nm). "
            "That is far outside linear response; the flexoelectric "
            "coefficient is defined in the small-curvature limit.",
            RuntimeWarning,
            stacklevel=2,
        )

    # s(X): arc length along the bent neutral surface, from 0 to the flat length
    x_grid = np.linspace(0.0, l_proj, n_samples)
    slope_grid = -amplitude * k * np.sin(k * x_grid + phase)
    ds = np.sqrt(1.0 + slope_grid**2)
    s_grid = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(x_grid))])

    # flat coordinate along the axis plays the role of arc length
    s = np.mod(flat.positions[:, axis], l_flat)
    if preserve_arc_length:
        # guard against the tiny mismatch between s_grid[-1] and l_flat
        s = s * (s_grid[-1] / l_flat)
    x_on_curve = np.interp(s, s_grid, x_grid)

    height = height_all
    z_mid = amplitude * np.cos(k * x_on_curve + phase)
    slope = -amplitude * k * np.sin(k * x_on_curve + phase)
    norm = np.sqrt(1.0 + slope**2)

    positions = flat.positions.copy()
    positions[:, axis] = x_on_curve - height * slope / norm
    positions[:, 2] = neutral_plane + z_mid + height / norm

    cell = flat.cell.copy()
    cell[axis] = cell[axis] * (l_proj / l_flat)

    # The construction is periodic by design, but check it: atoms must not
    # span more than one lattice vector along the ripple.
    spread = float(positions[:, axis].ptp())
    if spread >= l_proj:
        raise ValueError(
            f"the rippled structure spans {spread:g} nm along axis {axis} but "
            f"its cell is only {l_proj:g} nm long, so it is no longer periodic. "
            "This should not happen for a valid amplitude -- please report it."
        )
    if wrap:
        positions[:, axis] = np.mod(positions[:, axis], l_proj)

    out.positions = positions
    out.cell = cell
    return finish(out)


def ripple_series(
    flat: Structure,
    amplitudes,
    *,
    n_periods: int = 1,
    axis: int = 0,
    neutral_plane: float | None = None,
    preserve_arc_length: bool = True,
    phase: float = 0.0,
    wrap: bool = True,
    center: bool = True,
    name: str = "",
) -> ConfigSeries:
    """Build a whole deformation series from one flat structure.

    The first amplitude should be 0 -- that configuration is the undeformed
    reference the displacement field is measured against.

    Parameters
    ----------
    flat
        Flat starting structure.
    amplitudes
        Ripple amplitudes in nm, increasing, starting at 0. Every one must be
        below :func:`max_amplitude` for this cell, and small enough that the
        sheet does not fold through itself.
    name
        Series name.

    Returns
    -------
    ConfigSeries
        Ready for :class:`~flexo2d.pipeline.FlexoCalculation`, or for writing
        out as DFT input with :func:`flexo2d.io.write_poscar_series`.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) == 0:
        raise ValueError("amplitudes must be a non-empty 1D sequence")
    if amplitudes[0] != 0.0:
        raise ValueError(
            f"the first amplitude must be 0 (the flat reference), got "
            f"{amplitudes[0]}"
        )
    if np.any(np.diff(amplitudes) <= 0):
        raise ValueError("amplitudes must be strictly increasing")

    structures = [
        isometric_ripple(
            flat, float(a), n_periods=n_periods, axis=axis,
            neutral_plane=neutral_plane,
            preserve_arc_length=preserve_arc_length, phase=phase, wrap=wrap,
            center=center, label=f"ripple A={a:g} nm",
        )
        for a in amplitudes
    ]
    return ConfigSeries(
        structures, amplitudes=amplitudes, reference_index=0,
        name=name or f"{flat.label} ripple series".strip(),
    )
