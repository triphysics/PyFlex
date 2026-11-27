"""The command-line interface."""

import json
import os

import pytest

from conftest import DATA


def test_cli_runs_on_the_dataset(tmp_path, capsys):
    from flexo2d.cli import main

    if not os.path.exists(DATA):
        pytest.skip("dataset missing")
    out_json = str(tmp_path / "r.json")
    out_csv = str(tmp_path / "r.csv")
    code = main([
        "--npz", DATA, "--center", "Mo", "--shell", "S", "--cutoff", "0.30",
        "--charge", "Mo=1.083", "--charge", "S=-0.5415",
        "--json", out_json, "--csv", out_csv,
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "f3131" in text

    with open(out_json) as fh:
        payload = json.load(fh)
    assert payload["zero_deformation_limit"]["f3131"]["value"] == pytest.approx(
        -0.02205, rel=0.02
    )
    form = payload["closed_forms"]["f3131"]
    assert "dzc" in form and "U0" in form and "k**2" in form
    assert len(open(out_csv).read().splitlines()) == 12


def test_cli_lists_shells(capsys):
    from flexo2d.cli import main

    if not os.path.exists(DATA):
        pytest.skip("dataset missing")
    assert main(["--npz", DATA, "--center", "Mo", "--shells"]) == 0
    out = capsys.readouterr().out
    assert "pick --cutoff" in out
    # first shell: 6 S per Mo at ~0.236 nm, then a clear gap to ~0.394 nm
    rows = [ln.split() for ln in out.splitlines() if ln.strip().startswith("0.")]
    assert float(rows[0][0]) == pytest.approx(0.236, abs=2e-3)
    assert float(rows[0][1]) == pytest.approx(6.0)
    assert float(rows[1][0]) > 0.35


def test_cli_requires_charges(capsys):
    from flexo2d.cli import main

    if not os.path.exists(DATA):
        pytest.skip("dataset missing")
    assert main(["--npz", DATA, "--center", "Mo"]) == 2
    assert "give charges" in capsys.readouterr().err


def test_cli_rejects_bad_charge_syntax(capsys):
    from flexo2d.cli import main

    if not os.path.exists(DATA):
        pytest.skip("dataset missing")
    assert main(["--npz", DATA, "--center", "Mo", "--charge", "Mo"]) == 2
    assert "EL=Q" in capsys.readouterr().err


def test_cli_charge_set_and_no_in_plane(capsys):
    from flexo2d.cli import main

    if not os.path.exists(DATA):
        pytest.skip("dataset missing")
    code = main([
        "--npz", DATA, "--center", "Mo", "--charge-set", "MoS2-bader",
        "--no-in-plane",
    ])
    assert code == 0
    text = capsys.readouterr().out
    assert "f3131" in text and "f1111" not in text
