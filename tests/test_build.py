"""Generating supercells and rippled sheets, and writing them out."""

import os

import numpy as np
import pytest

from conftest import DATA


# ---------------------------------------------------------------- supercell
def test_supercell_counts_and_cell():
    from flexo2d import Structure, supercell

    unit = Structure(
        ["Mo", "S"], np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.15]]),
        np.diag([0.5, 0.3, 2.0]),
    )
    sc = supercell(unit, 4, 2)
    assert sc.n_atoms == 16
    np.testing.assert_allclose(np.diag(sc.cell), [2.0, 0.6, 2.0])
    assert sorted(set(sc.species)) == ["Mo", "S"]
    # the original atoms are present at the origin image
    assert any(np.allclose(p, [0.0, 0.0, 0.0]) for p in sc.positions)
    assert any(np.allclose(p, [1.5, 0.3, 0.0]) for p in sc.positions)


def test_supercell_carries_custom_masses():
    from flexo2d import Structure, supercell

    unit = Structure(["A"], np.zeros((1, 3)), np.eye(3),
                     atom_masses=np.array([7.0]))
    sc = supercell(unit, 3)
    np.testing.assert_allclose(sc.masses, 7.0)
    assert sc.n_atoms == 3


def test_supercell_rejects_bad_counts():
    from flexo2d import Structure, supercell

    with pytest.raises(ValueError, match="repeat counts"):
        supercell(Structure(["A"], np.zeros((1, 3)), np.eye(3)), 0)


# ------------------------------------------------------------- arc geometry
def test_arc_length_of_a_flat_sheet_is_its_length():
    from flexo2d import arc_length

    assert arc_length(0.0, 4.0) == pytest.approx(4.0)


def test_arc_length_grows_with_amplitude():
    from flexo2d import arc_length

    lengths = [arc_length(a, 4.0) for a in (0.0, 0.1, 0.2, 0.3)]
    assert lengths == sorted(lengths)
    assert lengths[0] == pytest.approx(4.0)


def test_projected_length_inverts_arc_length():
    from flexo2d import arc_length, projected_length

    flat = 4.36477
    for a in (0.05, 0.1, 0.2):
        proj = projected_length(a, flat)
        assert proj < flat
        assert arc_length(a, proj) == pytest.approx(flat, rel=1e-9)


# ----------------------------------------------------------------- rippling
def test_isometric_ripple_preserves_neutral_surface_length(mos2):
    from flexo2d import arc_length, isometric_ripple

    series, _ = mos2
    flat = series[0]
    l_flat = flat.cell[0, 0]
    for a in (0.05, 0.1, 0.2):
        bent = isometric_ripple(flat, a)
        assert bent.cell[0, 0] < l_flat
        assert arc_length(a, bent.cell[0, 0]) == pytest.approx(l_flat, rel=1e-8)


def test_isometric_ripple_barely_changes_bond_lengths(mos2):
    """The point of bending isometrically: bonds are not stretched."""
    from flexo2d import NeighbourShell, isometric_ripple

    series, _ = mos2
    flat = series[0]
    scheme = NeighbourShell("Mo", "S", cutoff=0.30)

    def bonds(s):
        out = []
        for c in scheme.clusters(s):
            pos = c.positions(s)
            out += list(np.linalg.norm(pos[1:] - pos[0], axis=1))
        return np.array(out)

    b0 = bonds(flat)
    b1 = bonds(isometric_ripple(flat, 0.10))
    # bending strains the off-plane layers by ~h*curvature; a few percent
    assert np.abs(b1 / b0 - 1).max() < 0.05


def test_naive_ripple_stretches_the_sheet(mos2):
    """preserve_arc_length=False is the wrong thing, and measurably so."""
    from flexo2d import arc_length, isometric_ripple

    series, _ = mos2
    flat = series[0]
    l_flat = flat.cell[0, 0]
    naive = isometric_ripple(flat, 0.20, preserve_arc_length=False)
    assert naive.cell[0, 0] == pytest.approx(l_flat)
    # the sheet now has to cover more arc length in the same cell
    assert arc_length(0.20, l_flat) / l_flat - 1 > 0.02


def test_ripple_series_reproduces_the_dft_cell_lengths(mos2):
    """Validation against the relaxed DFT geometries in data/.

    The reference dataset conserves its neutral-surface arc length, so the
    generator should land on the same cell lengths. What is left over is the
    genuine membrane relaxation DFT adds, and it is small.
    """
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
    )

    series, _ = mos2
    result = FlexoCalculation(
        series,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
    ).run()
    amps = np.array([f["Uz"].values["U0"] for f in result.fits])
    amps[0] = 0.0

    generated = ripple_series(series[0], np.abs(amps))
    l_gen = np.array([s.cell[0, 0] for s in generated])
    l_dft = np.array([s.cell[0, 0] for s in series])
    assert np.abs(l_gen / l_dft - 1).max() < 2e-3


def test_ripple_series_validates_amplitudes(mos2):
    from flexo2d import ripple_series

    series, _ = mos2
    with pytest.raises(ValueError, match="first amplitude must be 0"):
        ripple_series(series[0], [0.1, 0.2])
    with pytest.raises(ValueError, match="strictly increasing"):
        ripple_series(series[0], [0.0, 0.2, 0.1])


def test_ripple_rejects_a_non_cartesian_axis():
    from flexo2d import Structure, isometric_ripple

    skew = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    s = Structure(["Mo"], np.zeros((1, 3)), skew)
    with pytest.raises(ValueError, match="parallel to a Cartesian axis"):
        isometric_ripple(s, 0.1)


# -------------------------------------------------------------- phase-free
def test_free_phase_mode_is_phase_invariant(mos2):
    """Sliding the ripple across the lattice must not move the answer."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
    )

    series, _ = mos2
    amps = np.array([0.0, 0.05, 0.10, 0.15, 0.20])
    values = []
    for phase in (0.0, 0.6, np.pi / 2, 2.0):
        gen = ripple_series(series[0], amps, phase=phase)
        res = FlexoCalculation(
            gen,
            charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
            cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
            extrapolation_degree=1,
        ).run()
        values.append(res.extrapolations["f3131"].value)
    values = np.array(values)
    assert np.all(np.isfinite(values))
    assert abs(values.std() / values.mean()) < 0.01


def test_phase_locked_mode_breaks_off_phase(mos2):
    """Why free_phase is the default: the locked ansatz cannot fit a shifted ripple."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
        sinusoidal_bending,
    )

    series, _ = mos2
    gen = ripple_series(series[0], [0.0, 0.05, 0.10, 0.15], phase=np.pi / 2)
    res = FlexoCalculation(
        gen,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
        mode=sinusoidal_bending(free_phase=False),
        extrapolation_degree=1,
    ).run()
    assert not np.isfinite(res.extrapolations["f3131"].value)


def test_rigid_ion_bending_gives_a_smaller_coefficient(mos2):
    """Clamped-vs-relaxed decomposition, now possible because we can generate.

    An ideal rigid-ion isometric bend of the flat cell carries only the
    geometric part of the response. The DFT-relaxed series adds internal
    relaxation on top, and it is a sizeable, steady fraction.
    """
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
    )

    series, _ = mos2
    charges = FixedCharges({"Mo": 1.083, "S": -0.5415})
    scheme = NeighbourShell("Mo", "S", cutoff=0.30)

    relaxed = FlexoCalculation(series, charge_model=charges,
                               cluster_scheme=scheme).run()
    amps = np.array([f["Uz"].values["U0"] for f in relaxed.fits])
    amps[0] = 0.0
    rigid = FlexoCalculation(ripple_series(series[0], np.abs(amps)),
                             charge_model=charges, cluster_scheme=scheme).run()

    f_rigid = rigid.extrapolations["f3131"].value
    f_relax = relaxed.extrapolations["f3131"].value
    assert f_rigid < 0 and f_relax < 0
    assert abs(f_rigid) < abs(f_relax)
    share = (f_relax - f_rigid) / f_relax
    assert 0.10 < share < 0.30


def test_e331_is_nonzero_even_for_an_ideal_ripple(mos2):
    """e331 is symmetry-forbidden, so whatever comes out is systematic error.

    An ideal, noiseless, isometric ripple still produces a sizeable e331 --
    confirming it measures the method's error floor rather than a material
    property. See TODO.md P0.
    """
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
    )

    series, _ = mos2
    res = FlexoCalculation(
        ripple_series(series[0], [0.0, 0.05, 0.10, 0.15, 0.20]),
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
        extrapolation_degree=1,
    ).run()
    assert abs(res.extrapolations["e331"].value) > 1e-3


# ------------------------------------------------------------------ writing
def test_poscar_round_trip(tmp_path, mos2):
    from flexo2d import read_poscar, write_poscar

    series, _ = mos2
    p = str(tmp_path / "POSCAR")
    write_poscar(p, series[3])
    back = read_poscar(p)
    assert back.n_atoms == series[3].n_atoms
    np.testing.assert_allclose(back.cell, series[3].cell, atol=1e-9)
    # species are grouped on write, so compare as multisets of (species, pos)
    assert sorted(back.species) == sorted(series[3].species)
    for sp in set(series[3].species):
        a = np.sort(series[3].positions[series[3].indices_of(sp)], axis=0)
        b = np.sort(back.positions[back.indices_of(sp)], axis=0)
        np.testing.assert_allclose(b, a, atol=1e-8)


def test_poscar_cartesian_round_trip(tmp_path, mos2):
    from flexo2d import read_poscar, write_poscar

    series, _ = mos2
    p = str(tmp_path / "POSCAR")
    write_poscar(p, series[0], direct=False)
    back = read_poscar(p)
    np.testing.assert_allclose(np.diag(back.cell), np.diag(series[0].cell),
                               atol=1e-9)


def test_poscar_series_written_in_order(tmp_path, mos2):
    from flexo2d import read_poscar_series, write_poscar_series

    series, _ = mos2
    paths = write_poscar_series(str(tmp_path / "ripples"), series)
    assert len(paths) == len(series)
    assert [os.path.basename(p) for p in paths] == [
        f"POSCAR_{i:02d}.vasp" for i in range(11)
    ]
    back = read_poscar_series(paths)
    lengths_in = [s.cell[0, 0] for s in series]
    lengths_out = [s.cell[0, 0] for s in back]
    np.testing.assert_allclose(lengths_out, lengths_in, atol=1e-9)


def test_extxyz_round_trip_with_charges(tmp_path, mos2):
    from flexo2d import read_extxyz_series, write_extxyz

    series, charges = mos2
    p = str(tmp_path / "traj.xyz")
    write_extxyz(p, series, charges=charges)
    back, q = read_extxyz_series(p)
    assert len(back) == len(series)
    np.testing.assert_allclose(q, charges, atol=1e-7)
    np.testing.assert_allclose(back[5].positions, series[5].positions, atol=1e-9)
    assert list(back.reference.species) == list(series.reference.species)


def test_extxyz_rejects_mismatched_charges(tmp_path, mos2):
    from flexo2d import write_extxyz

    series, _ = mos2
    with pytest.raises(ValueError, match="charges must be"):
        write_extxyz(str(tmp_path / "t.xyz"), series, charges=np.zeros((2, 2)))


def test_generated_series_runs_end_to_end(tmp_path, mos2):
    """Generate -> write -> read -> analyse, the whole loop."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, read_poscar_series,
        ripple_series, write_poscar_series,
    )

    series, _ = mos2
    gen = ripple_series(series[0], [0.0, 0.06, 0.12, 0.18], name="generated")
    paths = write_poscar_series(str(tmp_path / "out"), gen)
    back = read_poscar_series(paths, amplitudes=gen.amplitudes)
    res = FlexoCalculation(
        back,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
        extrapolation_degree=1,
    ).run()
    f = res.coefficients["f3131"]
    assert np.isfinite(f[1:]).all()
    assert np.all(f[1:] < 0)


# ------------------------------------------- periodicity and validity guards
def test_tiling_consistency(mos2):
    """ripple(1x, n=1) tiled twice must equal ripple(2x, n=2) exactly.

    The strongest statement that the bending map is periodic: if the mapping
    were not commensurate with the cell, these would differ at the seam.
    """
    from flexo2d import isometric_ripple, supercell

    series, _ = mos2
    flat = series[0]
    one = isometric_ripple(flat, 0.15, n_periods=1)
    two = isometric_ripple(supercell(flat, 2, 1), 0.15, n_periods=2)
    assert two.cell[0, 0] == pytest.approx(2 * one.cell[0, 0], rel=1e-12)

    tiled = np.vstack([one.positions,
                       one.positions + np.array([one.cell[0, 0], 0.0, 0.0])])
    species2 = list(one.species) * 2
    for sp in ("Mo", "S"):
        a = np.array(sorted(map(tuple,
                                tiled[[i for i, s in enumerate(species2) if s == sp]])))
        b = np.array(sorted(map(tuple, two.positions[two.indices_of(sp)])))
        np.testing.assert_allclose(a, b, atol=1e-8)


def test_neutral_surface_spacing_survives_the_boundary(mos2):
    """Arc spacing between neutral-plane atoms, including across the wrap."""
    from flexo2d import isometric_ripple

    series, _ = mos2
    flat = series[0]
    amp = 0.15
    bent = isometric_ripple(flat, amp)
    L = bent.cell[0, 0]
    k = 2 * np.pi / L

    mo = bent.positions[bent.indices_of("Mo")]
    mo = mo[np.argsort(mo[:, 0])]
    # Mo lies on the neutral plane, which centring puts at cell_z / 2, so it
    # must sit exactly on the curve measured from there
    z0 = bent.cell[2, 2] / 2
    np.testing.assert_allclose(mo[:, 2] - z0, amp * np.cos(k * mo[:, 0]),
                               atol=1e-9)

    x = np.concatenate([mo[:, 0], [mo[0, 0] + L]])
    spacing = []
    for i in range(len(x) - 1):
        xs = np.linspace(x[i], x[i + 1], 2001)
        spacing.append(np.trapz(np.sqrt(1 + (amp * k * np.sin(k * xs)) ** 2), xs))
    spacing = np.array(spacing)
    # every gap, the one across the boundary included, equals the flat spacing
    assert np.abs(spacing - spacing.mean()).max() < 1e-5


def test_atoms_stay_inside_the_cell(mos2):
    """Bending displaces atoms along the normal, which can push one outside."""
    from flexo2d import isometric_ripple

    series, _ = mos2
    flat = series[0]
    for phase in (0.0, 1.0, np.pi / 2):
        bent = isometric_ripple(flat, 0.20, phase=phase)
        L = bent.cell[0, 0]
        x = bent.positions[:, 0]
        assert x.min() >= -1e-12 and x.max() < L, f"phase {phase}"
    # and without wrapping, at least one escapes -- which is what wrap fixes
    loose = isometric_ripple(flat, 0.20, phase=1.0, wrap=False)
    assert loose.positions[:, 0].min() < 0.0


def test_amplitude_beyond_the_isometric_limit_is_refused(mos2):
    """A sinusoid needs arc length >= 4*n*A, so large A has no solution.

    Before this check the bisection collapsed to L = 0 and silently returned a
    degenerate cell with every atom on top of every other.
    """
    from flexo2d import isometric_ripple, max_amplitude, projected_length

    series, _ = mos2
    flat = series[0]
    l_flat = flat.cell[0, 0]
    limit = max_amplitude(l_flat)
    assert limit == pytest.approx(l_flat / 4.0)

    # just inside the bound still works
    assert projected_length(0.9 * limit, l_flat) > 0.0
    for bad in (limit, 1.05 * limit, 10 * limit):
        with pytest.raises(ValueError, match="cannot be reached isometrically"):
            projected_length(bad, l_flat)
        with pytest.raises(ValueError, match="cannot be reached isometrically"):
            isometric_ripple(flat, bad)


def test_limit_scales_with_cell_length_and_periods(mos2):
    from flexo2d import isometric_ripple, max_amplitude

    series, _ = mos2
    flat = series[0]
    l_flat = flat.cell[0, 0]
    assert max_amplitude(l_flat, 2) == pytest.approx(max_amplitude(l_flat) / 2)
    # an amplitude fine for one period is impossible for four
    amp = 0.4 * max_amplitude(l_flat)
    isometric_ripple(flat, amp, n_periods=1)
    with pytest.raises(ValueError, match="cannot be reached isometrically"):
        isometric_ripple(flat, amp, n_periods=4)


def test_small_cell_refuses_a_modest_amplitude(mos2):
    """The case a user hits first: a short cell cannot take much ripple."""
    from flexo2d import Structure, isometric_ripple, max_amplitude

    series, _ = mos2
    flat = series[0]
    short = Structure(
        flat.species[:6], flat.positions[:6].copy(),
        np.diag([flat.cell[0, 0] / 8, flat.cell[1, 1], flat.cell[2, 2]]),
    )
    assert max_amplitude(short.cell[0, 0]) < 0.14
    with pytest.raises(ValueError, match="cannot be reached isometrically"):
        isometric_ripple(short, 0.15)


def test_self_intersecting_offset_surface_is_refused(mos2):
    """|h * curvature| >= 1 folds the sheet through itself."""
    from flexo2d import isometric_ripple, max_amplitude

    series, _ = mos2
    flat = series[0]
    # inside the arc-length bound, but the offset surface is degenerate
    amp = 0.9
    assert amp < max_amplitude(flat.cell[0, 0])
    with pytest.raises(ValueError, match="self-intersects"):
        isometric_ripple(flat, amp)


def test_large_but_valid_bending_strain_warns(mos2):
    """Geometrically valid, but far outside the linear-response regime."""
    from flexo2d import isometric_ripple

    series, _ = mos2
    # A = 0.7 nm gives |h*kappa| ~ 0.42: a real structure, but a useless one
    with pytest.warns(RuntimeWarning, match="bending strain"):
        isometric_ripple(series[0], 0.70, max_bending_strain=0.3)

    # the amplitudes anyone would actually use stay silent
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        isometric_ripple(series[0], 0.20)


def test_ripple_needs_a_periodic_axis(mos2):
    from flexo2d import Structure, isometric_ripple

    series, _ = mos2
    flat = series[0]
    open_x = Structure(flat.species, flat.positions, flat.cell,
                       pbc=(False, True, False))
    with pytest.raises(ValueError, match="not periodic"):
        isometric_ripple(open_x, 0.1)


def test_no_atom_pair_collapses_across_a_parameter_sweep(mos2):
    """Nothing in the allowed range may produce overlapping atoms."""
    from flexo2d import isometric_ripple, supercell

    series, _ = mos2
    flat = series[0]
    for nx, periods in ((1, 1), (2, 2), (3, 3)):
        base = supercell(flat, nx, 1) if nx > 1 else flat
        for amp in (0.05, 0.15, 0.25):
            bent = isometric_ripple(base, amp, n_periods=periods)
            P, C = bent.positions, bent.cell
            shifts = np.array([[i * C[0, 0], j * C[1, 1], 0.0]
                               for i in (-1, 0, 1) for j in (-1, 0, 1)])
            worst = np.inf
            for i in range(len(P)):
                d = (P - P[i])[None, :, :] + shifts[:, None, :]
                r = np.sqrt((d ** 2).sum(-1))
                r[r < 1e-9] = np.inf
                worst = min(worst, r.min())
            assert worst > 0.20, f"nx={nx} periods={periods} amp={amp}: {worst}"


# ------------------------------------------------- centring and file naming
def test_sheet_is_centred_in_the_vacuum(mos2):
    """Without centring the sheet straddles z = 0 and viewers split it in two."""
    from flexo2d import isometric_ripple

    series, _ = mos2
    flat = series[0]
    for amp in (0.0, 0.10, 0.20):
        bent = isometric_ripple(flat, amp)
        z = bent.positions[:, 2]
        lz = bent.cell[2, 2]
        assert z.min() > 0.0 and z.max() < lz
        # equal vacuum above and below
        assert z.min() == pytest.approx(lz - z.max(), abs=1e-9)

    loose = isometric_ripple(flat, 0.10, center=False)
    assert loose.positions[:, 2].min() < 0.0


def test_centring_is_a_pure_translation_and_cancels(mos2):
    """The shift is the same constant at every amplitude, so no coefficient moves."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, ripple_series,
    )

    series, _ = mos2
    amps = [0.0, 0.05, 0.10, 0.15, 0.20]
    out = {}
    for center in (True, False):
        out[center] = FlexoCalculation(
            ripple_series(series[0], amps, center=center),
            charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
            cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
            extrapolation_degree=1,
        ).run().extrapolations
    for name in ("f3131", "e331"):
        assert out[True][name].value == pytest.approx(
            out[False][name].value, rel=1e-12
        )

    # and the shift really is uniform across the series
    gen = ripple_series(series[0], amps)
    raw = ripple_series(series[0], amps, center=False)
    shifts = [
        (g.positions[:, 2] - r.positions[:, 2])
        for g, r in zip(gen, raw)
    ]
    for sh in shifts:
        np.testing.assert_allclose(sh, shifts[0][0], atol=1e-12)


def test_series_filenames_are_vesta_friendly(tmp_path, mos2):
    from flexo2d import write_poscar_series

    series, _ = mos2
    paths = write_poscar_series(str(tmp_path / "out"), series)
    names = [os.path.basename(p) for p in paths]
    assert names[0] == "POSCAR_00.vasp"
    assert names[10] == "POSCAR_10.vasp"
    # zero padding keeps plain sorting in amplitude order
    assert names == sorted(names)
