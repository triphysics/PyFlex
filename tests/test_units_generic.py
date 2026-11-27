"""Unit tests for the generic machinery, with no reference to MoS2."""

import numpy as np
import pytest
import sympy as sp

from conftest import square_sheet


# ----------------------------------------------------------------- structure
def test_structure_validates_shapes():
    from flexo2d import Structure

    with pytest.raises(ValueError, match=r"positions must be"):
        Structure(["A"], np.zeros((1, 2)), np.eye(3))
    with pytest.raises(ValueError, match=r"cell must be"):
        Structure(["A"], np.zeros((1, 3)), np.eye(2))
    with pytest.raises(ValueError, match="species but"):
        Structure(["A", "B"], np.zeros((1, 3)), np.eye(3))


def test_series_rejects_inconsistent_configurations():
    from flexo2d import ConfigSeries, Structure

    a = Structure(["A", "B"], np.zeros((2, 3)), np.eye(3))
    b = Structure(["B", "A"], np.zeros((2, 3)), np.eye(3))
    c = Structure(["A"], np.zeros((1, 3)), np.eye(3))
    with pytest.raises(ValueError, match="different species order"):
        ConfigSeries([a, b])
    with pytest.raises(ValueError, match="same number of atoms"):
        ConfigSeries([a, c])
    with pytest.raises(ValueError, match="amplitudes for"):
        ConfigSeries([a, a], amplitudes=[0.0])


def test_wavevector_follows_cell_length():
    from flexo2d import ConfigSeries, Structure

    s = [
        Structure(["A"], np.zeros((1, 3)), np.diag([L, 1.0, 1.0]))
        for L in (2.0, 4.0)
    ]
    k = ConfigSeries(s).wavevectors()
    np.testing.assert_allclose(k, [np.pi, np.pi / 2])
    k2 = ConfigSeries(s).wavevectors(n_periods=3)
    np.testing.assert_allclose(k2, 3 * k)


def test_in_plane_area_and_mass_plane():
    from flexo2d import Structure

    s = Structure(
        ["C", "C"], np.array([[0.0, 0, 0.0], [0.0, 0, 2.0]]),
        np.diag([3.0, 4.0, 10.0]),
    )
    assert s.in_plane_area == pytest.approx(12.0)
    assert s.centered_on_mass_plane().positions[:, 2].tolist() == [-1.0, 1.0]


# ------------------------------------------------------------------ clusters
def test_neighbour_shell_shares_weights_and_tiles_once():
    from flexo2d import NeighbourShell

    sheet = square_sheet(n=6, a=0.25)
    cl = NeighbourShell("A", "B", cutoff=0.2).clusters(sheet)
    assert len(cl) == 6
    for c in cl:
        assert len(c.members) == 3                 # centre + two B neighbours
        np.testing.assert_allclose(c.weights, [1.0, 0.5, 0.5])
    # every B atom is accounted for exactly once across all clusters
    total = np.zeros(sheet.n_atoms)
    for c in cl:
        np.testing.assert_array_equal(
            c.offsets[0], np.zeros(3)
        )
        for m, w in zip(c.members, c.weights):
            total[m] += w
    np.testing.assert_allclose(total, 1.0)


def test_neighbour_shell_uses_periodic_images():
    from flexo2d import NeighbourShell

    sheet = square_sheet(n=4, a=0.25)
    cl = NeighbourShell("A", "B", cutoff=0.2).clusters(sheet)
    # the last A must bond across the x boundary, i.e. use a non-zero offset
    assert any(np.abs(c.offsets).max() > 0 for c in cl)


def test_cutoff_too_small_is_an_error():
    from flexo2d import NeighbourShell

    with pytest.raises(ValueError, match="cutoff is too small"):
        NeighbourShell("A", "B", cutoff=0.01).clusters(square_sheet())


def test_missing_centre_species_is_an_error():
    from flexo2d import NeighbourShell

    with pytest.raises(ValueError, match="no atoms of species"):
        NeighbourShell("Zz", "B", cutoff=0.2).clusters(square_sheet())


def test_report_shells_groups_and_counts():
    from flexo2d import NeighbourShell

    rows = NeighbourShell("A", "B", cutoff=0.2).report_shells(
        square_sheet(n=6, a=0.25), max_distance=0.5, n=6
    )
    d = [r[0] for r in rows]
    assert d == sorted(d)
    # first shell: two B at half the A spacing, per A centre
    assert rows[0][0] == pytest.approx(0.125)
    assert rows[0][1] == pytest.approx(2.0)
    assert rows[0][2:] == ("A", "B")


# ------------------------------------------------------------------- charges
def test_fixed_charges_reports_missing_species():
    from flexo2d import FixedCharges

    with pytest.raises(KeyError, match="no charge given"):
        FixedCharges({"A": 1.0}).charges(square_sheet())


def test_neutralize_removes_net_charge():
    from flexo2d import neutralize

    table = np.array([[1.0, 1.0, -1.0], [0.5, 0.5, 0.5]])
    out = neutralize(table)
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-14)
    # relative differences within a configuration are preserved
    np.testing.assert_allclose(np.diff(out[0]), np.diff(table[0]))


def test_neutralize_with_groups():
    from flexo2d import neutralize

    table = np.array([[1.0, 1.0, 1.0, 1.0]])
    out = neutralize(table, groups=[[0, 1], [2, 3]])
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-14)


def test_fourier_smooth_keeps_the_leading_harmonic():
    from flexo2d import fourier_smooth

    n = 64
    x = np.arange(n) * 2 * np.pi / n
    clean = 0.7 * np.cos(x)
    rng = np.random.default_rng(0)
    noisy = clean + 0.05 * rng.standard_normal(n)
    out = fourier_smooth(noisy, n_harmonics=3)
    assert np.std(out - clean) < np.std(noisy - clean)


def test_fourier_smooth_rejects_bad_input():
    from flexo2d import fourier_smooth

    with pytest.raises(ValueError, match="n_harmonics"):
        fourier_smooth(np.zeros(8), n_harmonics=0)
    with pytest.raises(ValueError, match="1D or 2D"):
        fourier_smooth(np.zeros((2, 2, 2)))


# ------------------------------------------------------------------- fitting
def test_ansatz_rejects_nonlinear_parameters():
    from flexo2d import Ansatz

    x, a = sp.symbols("x a")
    with pytest.raises(ValueError, match="not linear"):
        Ansatz(sp.sin(a * x), (a,), x)


def test_ansatz_rejects_parameter_free_term():
    from flexo2d import Ansatz

    x, a = sp.symbols("x a")
    with pytest.raises(ValueError, match="parameter-free term"):
        Ansatz(a * x + 3, (a,), x)


def test_linear_fit_recovers_exact_parameters():
    from flexo2d import Ansatz, fit_linear
    from flexo2d.modes import K, X

    a, b = sp.symbols("a b")
    ansatz = Ansatz(a * sp.sin(K * X) + b * sp.cos(K * X), (a, b), X)
    k = 1.37
    x = np.linspace(0, 2 * np.pi / k, 23, endpoint=False)
    y = 0.3 * np.sin(k * x) - 1.2 * np.cos(k * x)
    fit = fit_linear(ansatz, x, y, {K: k}, name="t")
    assert fit["a"] == pytest.approx(0.3, abs=1e-12)
    assert fit["b"] == pytest.approx(-1.2, abs=1e-12)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.rmse < 1e-12


def test_linear_fit_needs_enough_points():
    from flexo2d import Ansatz, fit_linear
    from flexo2d.modes import K, X

    a, b = sp.symbols("a b")
    ansatz = Ansatz(a * sp.sin(K * X) + b * sp.cos(K * X), (a, b), X)
    with pytest.raises(ValueError, match="data points for"):
        fit_linear(ansatz, [0.0], [0.0], {K: 1.0})


# --------------------------------------------------------------------- modes
def test_mode_detects_an_unexcited_coefficient():
    from flexo2d import Ansatz, DeformationMode, Relation
    from flexo2d.modes import K, X

    U0, dzc, f = sp.symbols("U0 dzc f", real=True)
    disp = {"Uz": Ansatz(U0 * sp.cos(K * X), (U0,), X)}
    pol = {"Pz": Ansatz(dzc * sp.cos(K * X), (dzc,), X)}
    # a relation whose coefficient multiplies something identically zero
    mode = DeformationMode(
        "bad", disp, pol,
        [Relation("Pz", f * sp.diff(disp["Uz"].expression, X, 4) * 0, (f,))],
    )
    with pytest.raises(ValueError, match="could not determine"):
        mode.solve()


def test_mode_rejects_a_nonperiodic_relation():
    from flexo2d import Ansatz, DeformationMode, Relation
    from flexo2d.modes import K, X

    U0, dzc, f = sp.symbols("U0 dzc f", real=True)
    disp = {"Uz": Ansatz(U0 * X, (U0,), X)}
    pol = {"Pz": Ansatz(dzc * sp.cos(K * X), (dzc,), X)}
    mode = DeformationMode(
        "bad", disp, pol, [Relation("Pz", f * disp["Uz"].expression, (f,))]
    )
    with pytest.raises(ValueError, match="not .*captured by harmonics"):
        mode.solve()


def test_mode_can_switch_off_a_block():
    from flexo2d import sinusoidal_bending

    only_z = sinusoidal_bending(in_plane=False)
    assert set(only_z.solve()) == {"f3131", "e331"}
    only_x = sinusoidal_bending(out_of_plane=False)
    assert set(only_x.solve()) == {"f1111", "e111"}
    with pytest.raises(ValueError, match="at least one"):
        sinusoidal_bending(in_plane=False, out_of_plane=False)


def test_mode_records_over_determination_as_a_check():
    """The uniform term is kept as a cross-check, never as the estimator."""
    from flexo2d import sinusoidal_bending

    for free_phase in (True, False):
        det = sinusoidal_bending(free_phase=free_phase).solve()["e111"]
        assert "const" not in det.from_harmonic
        assert [t for t, _ in det.alternatives] == ["Px:const"]

    # the phase-locked solve determines each coefficient from one harmonic
    assert sinusoidal_bending(free_phase=False).solve()[
        "e111"].from_harmonic == "Px:cos(2kx)"
    # the phase-free solve needs them jointly
    assert sinusoidal_bending().solve()["e111"].from_harmonic.startswith("joint")


# ------------------------------------------------------------- extrapolation
def test_extrapolation_recovers_a_known_polynomial():
    from flexo2d import extrapolate

    amp = np.arange(1.0, 9.0)
    vals = 3.0 - 0.5 * amp + 0.25 * amp**2
    ex = extrapolate("c", amp, vals, degree=2)
    assert ex.value == pytest.approx(3.0, abs=1e-10)
    assert ex.r_squared == pytest.approx(1.0, abs=1e-12)
    assert ex.n_used == 8


def test_extrapolation_degrades_gracefully():
    from flexo2d import extrapolate

    amp = np.array([1.0, 2.0])
    ex = extrapolate("c", amp, np.array([1.0, 2.0]), degree=2)
    assert ex.degree == 1
    assert "degree reduced" in ex.note
    assert ex.value == pytest.approx(0.0, abs=1e-12)

    none = extrapolate("c", amp, np.array([np.nan, np.nan]))
    assert not np.isfinite(none.value)
    assert "no configuration" in none.note


def test_resolvable_mask_drops_small_denominators():
    from flexo2d import resolvable_mask

    vals = np.array([1.0, 1.0, 1.0, 1.0])
    den = np.array([1e-6, 0.01, 0.5, 1.0])
    mask = resolvable_mask(vals, den, rel_floor=0.05)
    assert mask.tolist() == [False, False, True, True]


def test_resolvable_mask_uses_signal_to_noise():
    from flexo2d import resolvable_mask

    vals = np.ones(3)
    den = np.array([1.0, 1.0, 1.0])
    se = np.array([0.5, 0.01, 0.001])
    mask = resolvable_mask(vals, den, se, min_snr=10.0, rel_floor=0.0)
    assert mask.tolist() == [False, True, True]


def test_long_extrapolation_is_flagged():
    from flexo2d import extrapolate

    amp = np.arange(11.0)
    vals = np.where(amp >= 4, 1.0 + 0.01 * amp, np.nan)
    ex = extrapolate("c", amp, vals, degree=2)
    assert ex.n_used == 7
    assert ex.reach == pytest.approx(4.0 / 6.0)
    assert ex.is_long_extrapolation
    assert "degree-dependent" in ex.summary()


def test_short_extrapolation_is_not_flagged():
    from flexo2d import extrapolate

    amp = np.arange(1.0, 11.0)
    ex = extrapolate("c", amp, 1.0 + 0.01 * amp, degree=2)
    assert ex.reach == pytest.approx(1.0 / 9.0)
    assert not ex.is_long_extrapolation
    assert "degree-dependent" not in ex.summary()


def test_atom_is_not_its_own_neighbour():
    """Centre and shell may be the same element (homonuclear sheets)."""
    import numpy as np

    from flexo2d import NeighbourShell, Structure

    # a 1D chain of one element: each atom has exactly two neighbours
    a = 0.142
    n = 8
    s = Structure(
        ["C"] * n,
        np.column_stack([np.arange(n) * a, np.zeros(n), np.zeros(n)]),
        np.diag([n * a, 10.0, 10.0]),
        pbc=(True, False, False),
    )
    cl = NeighbourShell("C", "C", cutoff=0.15).clusters(s)
    assert len(cl) == n
    for c in cl:
        assert len(c.members) == 3          # centre + two neighbours
        assert c.members[0] == c.center
        # the centre must not appear again at zero offset
        repeats = [
            i for i in range(1, len(c.members))
            if c.members[i] == c.center and not c.offsets[i].any()
        ]
        assert not repeats
