"""Validate the package against an independent implementation of the model.

``independent_coefficients`` below is written from scratch -- explicit loops
over sublattices, no flexo2d code -- so agreement is a real cross-check of
:mod:`flexo2d.clusters`, :mod:`flexo2d.fields`, :mod:`flexo2d.fitting` and
:mod:`flexo2d.modes` rather than a tautology.

The MoS2 ripple dataset in ``data/`` is the reference case throughout.
"""

import numpy as np
import pytest

from conftest import DATA

QMO, QS = 1.083, -0.5415
MMO, MS = 95.95, 32.065


def independent_coefficients(series):
    """f3131, e331 per configuration, computed without using flexo2d."""
    n_cfg = len(series)
    mo = np.array([s.positions[s.indices_of("Mo")] for s in series])
    s_all = np.array([s.positions[s.indices_of("S")] for s in series])
    s1, s2 = s_all[:, :16, :], s_all[:, 16:, :]
    ly = series.reference.cell[1, 1]
    ey = np.array([0.0, ly, 0.0])

    f3131, e331 = [], []
    for c in range(n_cfg):
        p_list, r_list = [], []
        for j in range(16):
            # Each Mo bonds to the S at the same index once, and to the S at
            # index j+1 twice (its two periodic images along y).
            jn = (j + 1) % 16
            wrap = np.array([series[c].cell[0, 0], 0.0, 0.0]) if jn == 0 else 0.0
            shell = [
                (s1[c, j], 0.0), (s2[c, j], 0.0),
                (s1[c, jn] + wrap, +ey), (s2[c, jn] + wrap, +ey),
                (s1[c, jn] + wrap, -ey), (s2[c, jn] + wrap, -ey),
            ]
            pos = [mo[c, j]] + [base + off for base, off in shell]
            qs = [QMO] + [QS / 3.0] * 6
            ms = [MMO] + [MS / 3.0] * 6
            pos = np.array(pos)
            p_list.append(np.array(qs) @ pos)
            r_list.append((np.array(ms) @ pos) / sum(ms))
        p_list = np.array(p_list)
        r_list = np.array(r_list)
        if c == 0:
            r_ref = r_list.copy()
            f3131.append(np.nan)
            e331.append(np.nan)
            continue
        uz = (r_list - r_ref)[:, 2]
        x = r_list[:, 0]
        k = 2 * np.pi / series[c].cell[0, 0]

        A = np.cos(k * x)[:, None]
        u0 = float(np.linalg.lstsq(A, uz, rcond=None)[0][0])
        B = np.column_stack([np.sin(k * x), np.cos(k * x)])
        dzs, dzc = np.linalg.lstsq(B, p_list[:, 2], rcond=None)[0]
        f3131.append(-2 * dzc / (u0 * k**2))
        e331.append(-2 * dzs / (u0 * k))
    return np.array(f3131), np.array(e331)


def test_matches_independent_implementation(mos2):
    """Cross-check the phase-locked mode against a from-scratch implementation."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, sinusoidal_bending,
    )

    series, _ = mos2
    locked = FlexoCalculation(
        series,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
        mode=sinusoidal_bending(free_phase=False),
    ).run()
    f_ind, e_ind = independent_coefficients(series)
    f_pkg = locked.coefficients["f3131"]
    e_pkg = locked.coefficients["e331"]
    ok = np.isfinite(f_ind) & np.isfinite(f_pkg)
    assert ok.sum() == 10
    np.testing.assert_allclose(f_pkg[ok], f_ind[ok], rtol=1e-10)
    np.testing.assert_allclose(e_pkg[ok], e_ind[ok], rtol=1e-10)


def test_closed_forms_match_hand_derivation():
    """The phase-locked solve must reproduce the closed forms derived by hand."""
    import sympy as sp

    from flexo2d import sinusoidal_bending

    def sym(expr, name):
        """The symbol called `name` inside `expr`, with its own assumptions."""
        for s in expr.free_symbols:
            if str(s) == name:
                return s
        raise AssertionError(f"{name} not in {expr}")

    cf = {
        n: d.expression
        for n, d in sinusoidal_bending(free_phase=False).solve().items()
    }

    def check(name, builder):
        got = cf[name]
        want = builder(lambda n: sym(got, n))
        assert sp.simplify(got - want) == 0, f"{name}: {got} != {want}"

    check("f3131", lambda g: -2 * g("dzc") / (g("U0") * g("k") ** 2))
    check("e331", lambda g: -2 * g("dzs") / (g("U0") * g("k")))
    check("f1111", lambda g: -g("dxs") / (4 * g("W0") * g("k") ** 2))
    check("e111", lambda g: g("dxc") / (2 * g("W0") * g("k")))


def test_transverse_coefficient_is_stable(mos2_result):
    """f3131 is the well-conditioned coefficient: flat to a few percent."""
    f = mos2_result.coefficients["f3131"]
    f = f[np.isfinite(f)]
    assert len(f) == 10
    assert np.all(f < 0)
    assert abs(f.std() / f.mean()) < 0.02
    ex = mos2_result.extrapolations["f3131"]
    assert ex.value == pytest.approx(-0.02205, rel=0.02)
    assert ex.n_used == 10


def test_out_of_plane_fits_are_excellent(mos2_result):
    q = mos2_result.fit_quality()
    assert np.nanmin(q["Uz"][1:]) > 0.999
    assert np.nanmin(q["Pz"][1:]) > 0.998


def test_in_plane_block_drops_unresolvable_configs(mos2_result):
    """W0 is in the noise for small ripples, so those configs must be dropped."""
    ex = mos2_result.extrapolations["f1111"]
    assert ex.n_used == 7, "expected the 7 largest-ripple configurations"
    assert not ex.used[0], "the flat reference must never be used"
    assert not ex.used[1:4].any(), "small-ripple configs are not resolvable"


def test_clusters_are_neutral(mos2_result):
    assert np.abs(mos2_result.fields.net_charges).max() < 1e-12


def test_wavevector_is_geometric(mos2):
    series, _ = mos2
    k = series.wavevectors()
    np.testing.assert_allclose(
        k, [2 * np.pi / s.cell[0, 0] for s in series], rtol=1e-14
    )
    # the dataset's own tabulated wavevectors agree to 4e-4
    tabulated = [1.4399, 1.43991, 1.44131, 1.44331, 1.44583, 1.44905,
                 1.45278, 1.45746, 1.46252, 1.46845, 1.47493]
    np.testing.assert_allclose(k, tabulated, atol=7e-4)


def test_raw_bader_charges_are_noisier_than_the_rigid_ion_model():
    """Document the model's sensitivity to the charge input.

    Raw per-atom Bader populations carry numerical noise comparable to the
    dipole signal, and leave each cluster slightly non-neutral, so the
    coefficient drifts strongly with amplitude. Sheet-averaged (rigid-ion)
    charges are the stable choice. This test pins that behaviour down so a
    future change to the charge handling cannot pass unnoticed.
    """
    import warnings

    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, TabulatedCharges,
        read_npz,
    )

    series, table = read_npz(DATA)
    assert table is not None
    scheme = NeighbourShell("Mo", "S", cutoff=0.30)

    fixed = FlexoCalculation(
        series,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=scheme,
    ).run().coefficients["f3131"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bader = FlexoCalculation(
            series,
            charge_model=TabulatedCharges(table),
            cluster_scheme=scheme,
            neutrality_tol=1.0,
        ).run().coefficients["f3131"]

    fixed, bader = fixed[1:], bader[1:]
    # same sign and order of magnitude ...
    assert np.all(bader < 0)
    assert 0.2 < abs(np.median(bader) / np.median(fixed)) < 5.0
    # ... but far less stable across the series
    spread = lambda a: abs(a.std() / a.mean())
    assert spread(fixed) < 0.02
    assert spread(bader) > 5 * spread(fixed)


def test_non_neutral_clusters_warn_but_stay_well_defined(mos2):
    from flexo2d import FlexoCalculation, NeighbourShell, TabulatedCharges

    series, table = mos2
    calc = FlexoCalculation(
        series,
        charge_model=TabulatedCharges(table),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
        neutrality_tol=1e-6,
    )
    with pytest.warns(RuntimeWarning, match="net charge"):
        res = calc.run()
    assert np.isfinite(res.coefficients["f3131"][1:]).all()

    # with dipole_origin='cell' the same input is an error, not a warning
    calc.dipole_origin = "cell"
    with pytest.raises(ValueError, match="origin-dependent"):
        calc.run()


def test_free_phase_agrees_with_the_locked_form_on_this_data(mos2):
    """The MoS2 ripple is phase-aligned, so both modes must agree here."""
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, sinusoidal_bending,
    )

    series, _ = mos2
    out = {}
    for tag, mode in (("free", sinusoidal_bending()),
                      ("locked", sinusoidal_bending(free_phase=False))):
        out[tag] = FlexoCalculation(
            series,
            charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
            cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
            mode=mode,
        ).run().extrapolations
    for name in ("f3131", "e331"):
        assert out["free"][name].value == pytest.approx(
            out["locked"][name].value, rel=1e-3
        )


def test_extrapolation_stability_across_degree(mos2):
    """f3131 must not depend on the extrapolation degree; f1111 does.

    This is the quantitative basis for the warning the report prints: the
    transverse coefficient is determined by configurations that reach down to
    the smallest ripple, the in-plane one is not.
    """
    from flexo2d import FixedCharges, FlexoCalculation, NeighbourShell

    series, _ = mos2
    out = {}
    for degree in (0, 1, 2):
        res = FlexoCalculation(
            series,
            charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
            cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
            extrapolation_degree=degree,
        ).run()
        out[degree] = res.extrapolations

    f3131 = [out[d]["f3131"].value for d in (0, 1, 2)]
    assert max(f3131) - min(f3131) < 0.03 * abs(np.mean(f3131))
    assert not out[2]["f3131"].is_long_extrapolation

    f1111 = [out[d]["f1111"].value for d in (0, 1, 2)]
    assert max(f1111) - min(f1111) > 0.5 * abs(np.mean(f1111))
    assert out[2]["f1111"].is_long_extrapolation
