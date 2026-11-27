"""Extrapolating a coefficient to the zero-deformation limit.

Each configuration in the series gives one estimate of every coefficient. The
flexoelectric coefficient is defined in the linear-response limit, so the
physical value is the extrapolation of that series to zero deformation.

Two things make this delicate and are handled explicitly here:

* **Resolvability.** A coefficient whose closed form divides by a fitted
  amplitude (``f1111 = -dxs/(4 W0 k^2)``) is numerically meaningless while that
  amplitude is still in the noise. Small-ripple configurations must be dropped,
  and :func:`resolvable_mask` decides which ones from the fit uncertainties
  instead of by hand.
* **Curvature.** The estimates drift with amplitude through anharmonicity, so a
  quadratic in the amplitude is the default extrapolant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Extrapolation:
    """A coefficient extrapolated to zero deformation."""

    coefficient: str
    amplitudes: np.ndarray
    values: np.ndarray
    used: np.ndarray
    degree: int
    coefficients: np.ndarray
    value: float
    stderr: float
    r_squared: float
    note: str = ""
    #: How far the extrapolation reaches beyond the usable data, as a fraction
    #: of the span of that data. Above ~0.3 the answer depends strongly on
    #: :attr:`degree` and should not be quoted without saying so.
    reach: float = 0.0

    @property
    def n_used(self) -> int:
        return int(np.count_nonzero(self.used))

    def evaluate(self, amplitude) -> np.ndarray:
        return np.polyval(self.coefficients, amplitude)

    @property
    def is_long_extrapolation(self) -> bool:
        """True when the limit sits far outside the data that determined it."""
        return self.reach > 0.3

    def summary(self) -> str:
        txt = (
            f"{self.coefficient:>7s} -> {self.value:+.6g} +- {self.stderr:.2g}  "
            f"(deg {self.degree} fit of {self.n_used}/{len(self.values)} "
            f"configs, R2={self.r_squared:.4f})"
        )
        notes = [self.note] if self.note else []
        if self.is_long_extrapolation:
            notes.append(
                f"reaches {self.reach:.1f}x the data span beyond it - strongly "
                "degree-dependent, treat as indicative only"
            )
        return txt + (("  [" + "; ".join(notes) + "]") if notes else "")


def resolvable_mask(
    values: np.ndarray,
    denominator: np.ndarray | None = None,
    denominator_stderr: np.ndarray | None = None,
    *,
    min_snr: float = 10.0,
    rel_floor: float = 0.05,
) -> np.ndarray:
    """Which configurations actually resolve a coefficient.

    Parameters
    ----------
    values
        The per-configuration coefficient estimates.
    denominator
        The fitted amplitude appearing in the denominator of the closed form
        (``U0`` for ``f3131``, ``W0`` for ``f1111``, ...). ``None`` keeps every
        finite value.
    denominator_stderr
        Standard error of `denominator` from the fit. When given, a
        configuration is kept only if ``|denominator| > min_snr * stderr``.
    min_snr
        Required signal-to-noise on the denominator.
    rel_floor
        Also require ``|denominator|`` to exceed this fraction of its largest
        value across the series. This catches the small-ripple configurations
        where the second-harmonic in-plane response has not developed yet.
    """
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if denominator is None:
        return mask
    den = np.abs(np.asarray(denominator, dtype=float))
    mask &= np.isfinite(den)
    peak = den[np.isfinite(den)].max(initial=0.0)
    if peak > 0:
        mask &= den > rel_floor * peak
    if denominator_stderr is not None:
        se = np.asarray(denominator_stderr, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mask &= den > min_snr * se
    return mask


def extrapolate(
    coefficient: str,
    amplitudes,
    values,
    *,
    mask=None,
    degree: int = 2,
    target: float = 0.0,
) -> Extrapolation:
    """Fit a polynomial in the amplitude and evaluate it at `target`.

    Parameters
    ----------
    coefficient
        Name, carried into the result.
    amplitudes
        Series control parameter per configuration.
    values
        Coefficient estimate per configuration.
    mask
        Which configurations to use; see :func:`resolvable_mask`.
    degree
        Polynomial degree. Reduced automatically if there are too few points.
    target
        Where to evaluate -- zero deformation by default.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    values = np.asarray(values, dtype=float)
    if amplitudes.shape != values.shape:
        raise ValueError(
            f"{len(amplitudes)} amplitudes but {len(values)} values"
        )
    if mask is None:
        mask = np.isfinite(values)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)

    x, y = amplitudes[mask], values[mask]
    note = ""
    if len(x):
        span = float(x.max() - x.min())
        gap = max(float(x.min()) - target, target - float(x.max()), 0.0)
        reach = gap / span if span > 0 else float("inf")
    else:
        reach = 0.0
    if len(x) == 0:
        return Extrapolation(
            coefficient, amplitudes, values, mask, 0, np.array([np.nan]),
            float("nan"), float("nan"), float("nan"),
            "no configuration resolves this coefficient", 0.0,
        )

    deg = min(degree, len(x) - 1)
    if deg < degree:
        note = f"degree reduced {degree}->{deg} ({len(x)} usable points)"
    if deg < 0:
        return Extrapolation(
            coefficient, amplitudes, values, mask, 0, np.array([y[0]]),
            float(y[0]), float("nan"), float("nan"),
            "single usable configuration; no extrapolation possible", reach,
        )
    if deg == 0:
        c = np.array([y.mean()])
        se = float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else float("nan")
        return Extrapolation(
            coefficient, amplitudes, values, mask, 0, c, float(c[0]), se,
            float("nan"), note or "constant fit", reach,
        )

    # polyfit covariance needs more points than parameters
    if len(x) > deg + 1:
        c, cov = np.polyfit(x, y, deg, cov=True)
        powers = np.array([target**(deg - i) for i in range(deg + 1)])
        var = float(powers @ cov @ powers)
        se = float(np.sqrt(max(var, 0.0)))
    else:
        c = np.polyfit(x, y, deg)
        se = float("nan")
        note = (note + "; " if note else "") + "no uncertainty (exactly determined)"

    resid = y - np.polyval(c, x)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")

    return Extrapolation(
        coefficient, amplitudes, values, mask, deg, c,
        float(np.polyval(c, target)), se, r2, note, reach,
    )
