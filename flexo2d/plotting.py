"""Matplotlib figures for a :class:`~flexo2d.pipeline.FlexoResult`.

Every function takes an optional ``ax``/``axes`` so the panels can be dropped
into a larger figure, and returns the axes it drew on. matplotlib is imported
lazily, so the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotting needs matplotlib; install it or use flexo2d.report instead"
        ) from exc
    return plt


def plot_profiles(result, ansatz_key: str, configs=None, ax=None):
    """Data points and fitted curve for one ansatz, over several configurations.

    Parameters
    ----------
    result
        A :class:`~flexo2d.pipeline.FlexoResult`.
    ansatz_key
        ``"Uz"``, ``"Ux"``, ``"Pz"`` or ``"Px"``.
    configs
        Which configurations to draw; default is all with a finished fit.
    """
    plt = _plt()
    ax = ax or plt.subplots(figsize=(4.2, 4.0))[1]
    if configs is None:
        configs = [i for i, f in enumerate(result.fits) if ansatz_key in f]
    cmap = plt.get_cmap("viridis")
    for i in configs:
        fit = result.fits[i].get(ansatz_key)
        if fit is None:
            continue
        c = cmap(i / max(len(result.fits) - 1, 1))
        ax.plot(fit.x, fit.y, "o", ms=4, color=c, alpha=0.85)
        xs = np.linspace(fit.x.min(), fit.x.max(), 400)
        ansatz = (
            result.mode.displacement.get(ansatz_key)
            or result.mode.polarization[ansatz_key]
        )
        from .modes import K

        ax.plot(
            xs,
            ansatz.evaluate(xs, fit.values, {K: float(result.fields.wavevectors[i])}),
            "-", lw=1.2, color=c,
            label=f"cfg {i}" if len(configs) <= 6 else None,
        )
    unit = "nm" if ansatz_key.startswith("U") else "e.nm"
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(f"{ansatz_key} ({unit})")
    ax.set_title(f"{ansatz_key}: data and fit")
    if len(configs) <= 6:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_coefficient(result, name: str, ax=None):
    """One coefficient against amplitude, with the extrapolation overlaid."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(4.2, 4.0))[1]
    ex = result.extrapolations[name]
    vals = result.coefficients[name]
    amps = result.fields.amplitudes

    used = ex.used
    ax.plot(amps[used], vals[used], "o", ms=6, label="used in fit")
    dropped = (~used) & np.isfinite(vals)
    if dropped.any():
        ax.plot(amps[dropped], vals[dropped], "x", ms=7, color="0.6",
                label="not resolvable")
    if np.isfinite(ex.value):
        xs = np.linspace(0.0, amps.max(), 200)
        ax.plot(xs, ex.evaluate(xs), "-", lw=1.3,
                label=f"deg-{ex.degree} extrapolation")
        ax.plot([0.0], [ex.value], "*", ms=14,
                label=f"limit = {ex.value:+.4g}")
    unit = "e.nm$^2$" if name.startswith("f") else "e.nm"
    ax.set_xlabel("deformation amplitude")
    ax.set_ylabel(f"{name} ({unit} per cluster)")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_fit_quality(result, ax=None):
    """R^2 of every ansatz against amplitude -- the first thing to check."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(4.6, 3.6))[1]
    amps = result.fields.amplitudes
    for key, r2 in result.fit_quality().items():
        ax.plot(amps, r2, "o-", ms=4, label=key)
    ax.axhline(0.99, ls="--", lw=1, color="0.5")
    ax.set_xlabel("deformation amplitude")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(min(0.9, np.nanmin([r.min() for r in result.fit_quality().values()])), 1.001)
    ax.set_title("fit quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def summary_figure(result, path: str | None = None):
    """A one-page overview: profiles, fit quality and every coefficient."""
    plt = _plt()
    keys = list(result.fits[-1]) if result.fits else []
    names = list(result.coefficients)
    n_cols = max(len(keys), len(names), 1)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.6 * n_cols, 10.5),
                             squeeze=False)
    for j in range(n_cols):
        for i in range(3):
            axes[i][j].axis("off")
    for j, key in enumerate(keys):
        axes[0][j].axis("on")
        plot_profiles(result, key, ax=axes[0][j])
    axes[1][0].axis("on")
    plot_fit_quality(result, ax=axes[1][0])
    for j, name in enumerate(names):
        axes[2][j].axis("on")
        plot_coefficient(result, name, ax=axes[2][j])
    fig.suptitle(result.series_name or "flexo2d", fontsize=13)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
