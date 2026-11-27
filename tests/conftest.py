import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "mos2_ripple.npz",
)


@pytest.fixture(scope="session")
def mos2():
    """The MoS2 ripple series and its Bader charge table."""
    from flexo2d import read_npz

    if not os.path.exists(DATA):
        pytest.skip(f"dataset missing: {DATA}")
    return read_npz(DATA)


@pytest.fixture(scope="session")
def mos2_result(mos2):
    from flexo2d import FixedCharges, FlexoCalculation, NeighbourShell

    series, _ = mos2
    return FlexoCalculation(
        series,
        charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
        cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
    ).run()


def square_sheet(n=6, a=0.25, charge=1.0):
    """A toy two-species square sheet, for cluster/charge tests."""
    from flexo2d import Structure

    species, pos = [], []
    for i in range(n):
        species.append("A")
        pos.append([i * a, 0.0, 0.0])
        species.append("B")
        pos.append([(i + 0.5) * a, 0.0, 0.0])
    return Structure(
        species=species,
        positions=np.array(pos),
        cell=np.diag([n * a, 10.0, 10.0]),
        pbc=(True, False, False),
    )
