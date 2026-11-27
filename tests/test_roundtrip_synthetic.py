"""End-to-end round trips on a synthetic material with planted coefficients.

A two-species sheet is built so that its cluster dipole and displacement
fields are exactly the ansatz of :func:`flexo2d.modes.sinusoidal_bending`, with
chosen values of ``f3131``, ``e331``, ``f1111`` and ``e111``. The pipeline then
has to return those numbers. Nothing about MoS2 is involved, so this is the
test that the package really is material-agnostic.

Construction
------------
Each cluster is one ``A`` (charge ``+q``) paired with one ``B`` (charge
``-q``), equal masses. With the dipole taken about the cluster centroid,

    centroid = R + d/2,        dipole = -q * d

where ``d`` is the A->B separation. Planting a dipole profile ``P(x)`` needs
``d = -P(x)/q``; planting a displacement ``U(x)`` needs
``R = x0 + U(x) - d(x)/2``.

Two regimes are tested separately, because they are not equally exact:

* **Pure bending** (no in-plane displacement). The cluster abscissa is then
  undisplaced, the planted profile is sampled at exactly the coordinate it was
  built on, and ``f3131``/``e331`` come back to machine precision.
* **With in-plane displacement**, which is what ``f1111``/``e111`` need. The
  displacement is a Lagrangian field but is fitted against the deformed
  (Eulerian) coordinate, so there is an intrinsic O(W0 k) mismatch; here it is
  pinned at the percent level so a regression would show up.
"""

import numpy as np
import pytest

Q = 0.8
A_SPACING = 0.30          # nm between neighbouring clusters
N_CELL = 24               # clusters along the ripple
N_CONFIG = 7

F3131, E331 = -0.0231, -0.00417
F1111, E111 = -0.0125, 0.00052


def build_series(*, in_plane: bool):
    """Synthetic series; `in_plane` switches the U_x displacement on."""
    from flexo2d import ConfigSeries, Structure

    lx = N_CELL * A_SPACING
    k = 2.0 * np.pi / lx
    cell = np.diag([lx, 0.40, 3.0])
    x0 = np.arange(N_CELL) * A_SPACING

    structures, amps = [], []
    for c in range(N_CONFIG):
        amp = float(c)
        u0 = 0.02 * amp                             # out-of-plane amplitude
        w0 = (0.004 * amp) if in_plane else 0.0     # in-plane 2nd harmonic
        v0 = (-0.004 * amp) if in_plane else 0.0    # uniform in-plane strain

        dzc = -F3131 * u0 * k**2 / 2.0
        dzs = -E331 * u0 * k / 2.0
        dxs = -F1111 * 4.0 * w0 * k**2
        dxc = E111 * 2.0 * w0 * k
        cd = E111 * v0                              # over-determined constant

        uz = u0 * np.cos(k * x0)
        ux = v0 * x0 + w0 * np.sin(2 * k * x0)
        pz = dzs * np.sin(k * x0) + dzc * np.cos(k * x0)
        px = dxs * np.sin(2 * k * x0) + dxc * np.cos(2 * k * x0) + cd

        d = np.zeros((N_CELL, 3))
        d[:, 0] = -px / Q
        d[:, 2] = -pz / Q
        d[:, 1] = 0.05                              # keep A and B apart

        r_a = np.zeros((N_CELL, 3))
        r_a[:, 0] = x0 + ux - d[:, 0] / 2.0
        r_a[:, 1] = -d[:, 1] / 2.0
        r_a[:, 2] = uz - d[:, 2] / 2.0
        r_b = r_a + d

        # Model species with explicit masses: no element table involved, so
        # this also exercises Structure(atom_masses=...).
        structures.append(
            Structure(
                ["A"] * N_CELL + ["B"] * N_CELL,
                np.vstack([r_a, r_b]),
                cell,
                label=f"amp {amp:g}",
                atom_masses=np.full(2 * N_CELL, 20.0),
            )
        )
        amps.append(amp)

    return ConfigSeries(structures, amplitudes=np.array(amps),
                        name="synthetic AB sheet"), k


def run(series):
    from flexo2d import FixedCharges, FlexoCalculation, NeighbourShell

    return FlexoCalculation(
        series,
        charge_model=FixedCharges({"A": Q, "B": -Q}),
        cluster_scheme=NeighbourShell("A", "B", cutoff=0.12),
        rel_floor=0.0,
        min_snr=0.0,
    ).run()


@pytest.fixture(scope="module")
def bending():
    series, k = build_series(in_plane=False)
    return run(series), k


@pytest.fixture(scope="module")
def full():
    series, k = build_series(in_plane=True)
    return run(series), k


# ------------------------------------------------------- pure bending: exact
def test_clusters_are_one_to_one(bending):
    result, _ = bending
    assert result.fields.n_cluster == N_CELL
    assert np.abs(result.fields.net_charges).max() < 1e-14


def test_wavevector_matches_construction(bending):
    result, k = bending
    np.testing.assert_allclose(result.fields.wavevectors, k, rtol=1e-14)


def test_out_of_plane_fits_are_exact(bending):
    result, _ = bending
    for key in ("Uz", "Pz"):
        r2 = result.fit_quality()[key]
        finite = r2[np.isfinite(r2)]
        assert np.all(finite > 1.0 - 1e-12), f"{key}: {finite}"


@pytest.mark.parametrize("name,planted", [("f3131", F3131), ("e331", E331)])
def test_transverse_coefficients_recovered_exactly(bending, name, planted):
    result, _ = bending
    vals = result.coefficients[name]
    finite = vals[np.isfinite(vals)]
    assert len(finite) == N_CONFIG - 1
    np.testing.assert_allclose(finite, planted, rtol=1e-10)
    assert result.extrapolations[name].value == pytest.approx(planted, rel=1e-9)


def test_in_plane_block_is_absent_without_in_plane_motion(bending):
    """With W0 == 0 the in-plane coefficients must not be reported as numbers."""
    result, _ = bending
    for name in ("f1111", "e111"):
        ex = result.extrapolations[name]
        assert ex.n_used == 0 or not np.isfinite(ex.value)


# --------------------------- with in-plane motion: the extrapolation is what
# --------------------------- recovers the planted values
@pytest.mark.parametrize(
    "name,planted",
    [("f3131", F3131), ("e331", E331), ("f1111", F1111), ("e111", E111)],
)
def test_extrapolation_recovers_planted_values(full, name, planted):
    """The zero-deformation limit is accurate even where single configs are not.

    This is the whole point of extrapolating. With in-plane motion present, the
    Lagrangian/Eulerian mismatch biases each individual configuration, by up to
    a factor of several for the small ``e`` coefficients. The bias vanishes
    linearly as the ripple flattens, so a quadratic extrapolation to zero
    deformation removes it and returns the planted value to well under a
    percent.
    """
    result, _ = full
    assert result.extrapolations[name].value == pytest.approx(planted, rel=0.01)


def test_free_phase_removes_most_of_the_per_configuration_bias(full):
    """The phase-free ansatz largely cures the Lagrangian/Eulerian bias.

    With in-plane motion present, the displacement field is Lagrangian but is
    sampled at the deformed (Eulerian) coordinate. That mismatch shows up
    mainly as a *phase* error, so carrying both quadratures of the ansatz
    absorbs it. Under the phase-locked ansatz single configurations are off by
    tens of percent -- up to a factor of several for ``e111``; under the
    phase-free one they are within a few percent.
    """
    from flexo2d import (
        FixedCharges, FlexoCalculation, NeighbourShell, sinusoidal_bending,
    )

    result, _ = full
    series, _ = build_series(in_plane=True)
    locked = FlexoCalculation(
        series,
        charge_model=FixedCharges({"A": Q, "B": -Q}),
        cluster_scheme=NeighbourShell("A", "B", cutoff=0.12),
        mode=sinusoidal_bending(free_phase=False),
        rel_floor=0.0,
        min_snr=0.0,
    ).run()

    def drift(res, name, planted):
        v = res.coefficients[name]
        return float(np.abs(v[np.isfinite(v)] / planted - 1.0).max())

    for name, planted in (("f3131", F3131), ("e331", E331),
                          ("f1111", F1111), ("e111", E111)):
        free_drift = drift(result, name, planted)
        locked_drift = drift(locked, name, planted)
        assert free_drift < 0.05, f"{name}: free-phase drift {free_drift:.3f}"
        assert free_drift <= locked_drift + 1e-9, (
            f"{name}: free-phase ({free_drift:.3f}) should not be worse than "
            f"phase-locked ({locked_drift:.3f})"
        )

    # the small coefficients are where it matters most
    assert drift(locked, "e111", E111) > 0.5
    assert drift(result, "e111", E111) < 0.05


def test_consistency_check_tracks_e111(full):
    """The over-determined constant term gives an independent e111 estimate."""
    result, _ = full
    alt = result.consistency["e111"]["Px:const"]
    finite = alt[np.isfinite(alt)]
    assert len(finite) == N_CONFIG - 1
    # same order of magnitude and sign as planted
    assert np.all(np.sign(finite) == np.sign(E111))
    assert 0.2 < abs(np.median(finite) / E111) < 5.0


def test_intensive_conversion_divides_by_cluster_area(full):
    result, _ = full
    area = result.cluster_area
    assert area == pytest.approx(N_CELL * A_SPACING * 0.40 / N_CELL)
    assert result.intensive("f3131") == pytest.approx(
        result.extrapolations["f3131"].value / area
    )


def test_report_and_csv_run(full):
    from flexo2d import coefficient_table

    result, _ = full
    text = result.report()
    assert "f3131" in text and "Zero-deformation limit" in text
    csv = coefficient_table(result)
    assert csv.splitlines()[0].startswith("amplitude,")
    assert len(csv.splitlines()) == N_CONFIG + 1
