"""Plain-text reporting of a :class:`~flexo2d.pipeline.FlexoResult`."""

from __future__ import annotations

import numpy as np

from .modes import dimension_of
from .units import format_dimension, si_factor_and_label


def _row(values, fmt="{:>10.4g}") -> str:
    return "".join(fmt.format(v) if np.isfinite(v) else f"{'--':>10s}"
                    for v in values)


def format_result(result, intensive: bool = True) -> str:
    """A readable summary of a whole calculation."""
    out: list[str] = []
    w = 78
    out.append("=" * w)
    title = result.series_name or "flexoelectric calculation"
    out.append(f"{title}   [mode: {result.mode.name}]")
    out.append("=" * w)

    f = result.fields
    out.append(
        f"{f.n_config} configurations, {f.n_cluster} clusters per configuration, "
        f"{f.cluster_area:.5f} nm^2 per cluster"
    )
    out.append(
        f"worst cluster net charge: {np.abs(f.net_charges).max():.2e} e "
        "(must be ~0 for a well-defined dipole)"
    )
    out.append("")

    out.append("Closed forms obtained from the constitutive relations")
    out.append("-" * w)
    for name, expr in result.closed_forms.items():
        out.append(f"  {name:>7s} = {expr}")
    out.append("")

    out.append("Fit quality (R^2 per configuration)")
    out.append("-" * w)
    out.append(f"{'ansatz':>8s}" + _row(f.amplitudes, "{:>10.3g}"))
    for key, r2 in result.fit_quality().items():
        out.append(f"{key:>8s}" + _row(r2, "{:>10.5f}"))
    out.append("")

    dims = {n: dimension_of(e) for n, e in result.closed_forms.items()}
    per_cluster_units = ", ".join(
        f"{n} in {format_dimension(*dims[n])}" for n in result.coefficients
    )
    out.append(f"Coefficient per configuration (per cluster: {per_cluster_units})")
    out.append("-" * w)
    out.append(f"{'amp':>8s}" + _row(f.amplitudes, "{:>10.3g}"))
    for name, vals in result.coefficients.items():
        out.append(f"{name:>8s}" + _row(vals))
        for tag, alt in result.consistency.get(name, {}).items():
            out.append(f"{'  check':>8s}" + _row(alt) + f"   <- {tag}")
    out.append("")

    out.append("Zero-deformation limit")
    out.append("-" * w)
    for name, ex in result.extrapolations.items():
        out.append("  " + ex.summary())
    out.append("")

    if intensive:
        out.append("Intensive 2D coefficients (per cluster value / cluster area)")
        out.append("-" * w)
        for name in result.extrapolations:
            v = result.intensive(name)
            le, qe = dims[name]
            label = format_dimension(le, qe, per_area=True)
            factor, si_label = si_factor_and_label(le, qe, per_area=True)
            if not np.isfinite(v):
                out.append(f"  {name:>7s} ->         --")
                continue
            out.append(
                f"  {name:>7s} -> {v:+.6g} {label:<8s} = {v * factor:+.4e} {si_label}"
            )
        out.append("")

    notes: list[str] = []
    present = set(result.coefficients)
    if {"f3131", "e331"} & present:
        notes += [
            "f3131, e331 come from the out-of-plane (transverse) response; they are",
            "  the robust pair, since U_z is large and clean in every rippled cell.",
        ]
    if {"f1111", "e111"} & present:
        notes += [
            "f1111, e111 come from the in-plane second harmonic, which vanishes as the",
            "  ripple flattens -- configurations where W0 is still in the noise are",
            "  dropped automatically (see the 'configs' count above).",
        ]
    if any(ex.is_long_extrapolation for ex in result.extrapolations.values()):
        flagged = [
            n for n, ex in result.extrapolations.items()
            if ex.is_long_extrapolation
        ]
        notes += [
            f"{', '.join(flagged)}: the usable configurations start well above zero",
            "  deformation, so the limit is a long extrapolation. Re-run with a",
            "  different extrapolation_degree to see how much it moves before",
            "  quoting it, and prefer more small-ripple configurations.",
        ]
    if {"e331", "e111"} & present:
        notes += [
            "The e coefficients are the quadrature partners of the dominant harmonic,",
            "  so single configurations are far more biased for them than for the f",
            "  coefficients. Trust the extrapolated limit, not individual rows.",
        ]
    if result.consistency:
        notes.append(
            "'check' rows are independent determinations from an over-determined"
        )
        notes.append(
            "  harmonic; large disagreement means the ansatz does not fit the data."
        )
    if notes:
        out.append("Notes")
        out.append("-" * w)
        out += ["  " + n for n in notes]
    out.append("=" * w)
    return "\n".join(out)


def coefficient_table(result) -> str:
    """CSV of the per-configuration coefficients."""
    names = list(result.coefficients)
    lines = ["amplitude," + ",".join(names)]
    for i, a in enumerate(result.fields.amplitudes):
        vals = [result.coefficients[n][i] for n in names]
        lines.append(
            f"{a:g}," + ",".join("" if not np.isfinite(v) else f"{v:.10g}"
                                 for v in vals)
        )
    return "\n".join(lines)
