"""Readers and writers."""

import numpy as np
import pytest

POSCAR = """MoS2 ripple test
   1.0
     4.3647709    0.0000000    0.0000000
     0.0000000    0.3150002    0.0000000
     0.0000000    0.0000000    2.8150002
   Mo   S
    1    2
Cartesian
   0.1812540  0.3150000  1.5650000
   0.0003080  0.3150000  1.4145300
   0.0003078  0.3150000  1.7154700
"""

POSCAR_DIRECT = """direct test
   10.0
     1.0 0.0 0.0
     0.0 1.0 0.0
     0.0 0.0 1.0
   Si
    2
Direct
   0.00 0.00 0.00
   0.50 0.50 0.50
"""

EXTXYZ = """2
Lattice="4.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:charges:R:1
Ga 0.0 0.0 0.0 0.75
N  1.0 0.0 0.0 -0.75
2
Lattice="4.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:charges:R:1
Ga 0.0 0.0 0.1 0.76
N  1.0 0.0 0.0 -0.76
"""


def test_poscar_cartesian(tmp_path):
    from flexo2d import read_poscar

    p = tmp_path / "POSCAR"
    p.write_text(POSCAR)
    s = read_poscar(str(p))
    assert list(s.species) == ["Mo", "S", "S"]
    # Angstrom in the file, nm inside the package
    np.testing.assert_allclose(np.diag(s.cell), [0.43647709, 0.03150002, 0.28150002])
    np.testing.assert_allclose(s.positions[0], [0.01812540, 0.03150000, 0.15650000])
    assert s.label == "MoS2 ripple test"


def test_poscar_direct_and_scaling(tmp_path):
    from flexo2d import read_poscar

    p = tmp_path / "POSCAR"
    p.write_text(POSCAR_DIRECT)
    s = read_poscar(str(p))
    np.testing.assert_allclose(np.diag(s.cell), [1.0, 1.0, 1.0])   # 10 A = 1 nm
    np.testing.assert_allclose(s.positions[1], [0.5, 0.5, 0.5])


def test_poscar_without_species_names_is_rejected(tmp_path):
    from flexo2d import read_poscar

    bad = POSCAR.replace("   Mo   S\n", "")
    p = tmp_path / "POSCAR"
    p.write_text(bad)
    with pytest.raises(ValueError, match="species-name line"):
        read_poscar(str(p))


def test_poscar_series(tmp_path):
    from flexo2d import read_poscar_series

    paths = []
    for i in range(3):
        p = tmp_path / f"POSCAR.{i}"
        p.write_text(POSCAR)
        paths.append(str(p))
    series = read_poscar_series(paths, amplitudes=[0.0, 1.0, 2.0], name="t")
    assert len(series) == 3
    np.testing.assert_allclose(series.amplitudes, [0, 1, 2])


def test_extxyz_with_charges(tmp_path):
    from flexo2d import read_extxyz_series

    p = tmp_path / "traj.xyz"
    p.write_text(EXTXYZ)
    series, charges = read_extxyz_series(str(p))
    assert len(series) == 2
    assert list(series.reference.species) == ["Ga", "N"]
    np.testing.assert_allclose(np.diag(series.reference.cell), [0.4, 0.2, 2.0])
    assert charges.shape == (2, 2)
    np.testing.assert_allclose(charges[1], [0.76, -0.76])


def test_npz_round_trip(tmp_path):
    from flexo2d import ConfigSeries, Structure, read_npz, write_npz

    s = [
        Structure(["Mo", "S"], np.array([[0.0, 0, 0], [0.1, 0, 0.2]]),
                  np.diag([1.0, 2.0, 3.0]), label=f"c{i}")
        for i in range(2)
    ]
    series = ConfigSeries(s, amplitudes=[0.0, 1.5], name="rt")
    table = np.array([[1.0, -1.0], [1.1, -1.1]])
    path = str(tmp_path / "d.npz")
    write_npz(path, series, charges=table)

    back, q = read_npz(path)
    assert back.name == "rt"
    assert list(back.reference.species) == ["Mo", "S"]
    np.testing.assert_allclose(back.amplitudes, [0.0, 1.5])
    np.testing.assert_allclose(back[1].positions, series[1].positions)
    np.testing.assert_allclose(q, table)
    assert back[1].label == "c1"


def test_layer_arrays_assemble_in_order():
    from flexo2d import series_from_layer_arrays

    n_cfg, n_cell = 3, 4
    layers = {
        "Mo": np.zeros((n_cfg, n_cell, 3)),
        "S1": np.ones((n_cfg, n_cell, 3)),
        "S2": 2 * np.ones((n_cfg, n_cell, 3)),
    }
    charges = {
        "Mo": np.full((n_cfg, n_cell), 1.0),
        "S1": np.full((n_cfg, n_cell), -0.5),
        "S2": np.full((n_cfg, n_cell), -0.5),
    }
    cells = np.tile([4.0, 0.3, 2.8], (n_cfg, 1))
    series, table = series_from_layer_arrays(
        layers, cells, charges=charges,
        species_of={"Mo": "Mo", "S1": "S", "S2": "S"},
        order=["Mo", "S1", "S2"],
        z_origins=np.array([0.0, 1.0, 2.0]),
    )
    assert list(series.reference.species) == ["Mo"] * 4 + ["S"] * 8
    assert table.shape == (3, 12)
    np.testing.assert_allclose(table[0, :4], 1.0)
    # z_origins subtracted
    np.testing.assert_allclose(series[1].positions[0, 2], -1.0)
    np.testing.assert_allclose(series[0].positions[4, 2], 1.0)


def test_layer_arrays_validate_inputs():
    from flexo2d import series_from_layer_arrays

    good = {"Mo": np.zeros((2, 3, 3))}
    cells = np.tile([1.0, 1.0, 1.0], (2, 1))
    with pytest.raises(KeyError, match="not in the given arrays"):
        series_from_layer_arrays(good, cells, order=["Nope"])
    with pytest.raises(ValueError, match=r"must be \(n_config, n_cell, 3\)"):
        series_from_layer_arrays({"Mo": np.zeros((2, 3))}, cells)
    with pytest.raises(ValueError, match="cell_lengths must be"):
        series_from_layer_arrays(good, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="disagree on the configuration count"):
        series_from_layer_arrays(
            {"Mo": np.zeros((2, 3, 3)), "S": np.zeros((3, 3, 3))}, cells
        )
    with pytest.raises(KeyError, match="no charges given"):
        series_from_layer_arrays(good, cells, charges={})


def test_species_inferred_from_layer_name():
    from flexo2d import series_from_layer_arrays

    series, _ = series_from_layer_arrays(
        {"S1": np.zeros((1, 2, 3)), "S2": np.zeros((1, 2, 3))},
        np.array([[1.0, 1.0, 1.0]]),
        order=["S1", "S2"],
    )
    assert list(series.reference.species) == ["S"] * 4
