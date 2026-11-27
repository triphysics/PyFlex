"""The end-to-end calculation.

.. code-block:: python

    calc = FlexoCalculation(series, charge_model=..., cluster_scheme=...)
    result = calc.run()
    print(result.report())

Everything material-specific lives in the three inputs (`series`,
`charge_model`, `cluster_scheme`); the mode and the algebra are generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import sympy as sp

from .charges import ChargeModel
from .clusters import ClusterScheme
from .extrapolate import Extrapolation, extrapolate, resolvable_mask
from .fields import FieldSet, build_fields
from .fitting import FitResult, fit_linear
from .modes import DENOMINATOR_PARAMETERS, K, DeformationMode, sinusoidal_bending
from .structure import ConfigSeries
from .units import per_area


@dataclass
class FlexoResult:
    """Everything a run produces."""

    series_name: str
    mode: DeformationMode
    fields: FieldSet
    #: ``fits[config_index][ansatz_key]`` -> :class:`FitResult`
    fits: list[dict[str, FitResult]]
    #: ``coefficients[name]`` -> per-configuration values, shape ``(n_config,)``
    coefficients: dict[str, np.ndarray]
    #: ``consistency[name]`` -> alternative per-configuration values
    consistency: dict[str, dict[str, np.ndarray]]
    extrapolations: dict[str, Extrapolation]
    closed_forms: dict[str, sp.Expr]
    cluster_area: float

    # ------------------------------------------------------------------ views
    @property
    def amplitudes(self) -> np.ndarray:
        return self.fields.amplitudes

    def fit_quality(self) -> dict[str, np.ndarray]:
        """R^2 per ansatz per configuration."""
        keys = list(self.fits[0]) if self.fits else []
        return {
            k: np.array([f[k].r_squared if k in f else np.nan for f in self.fits])
            for k in keys
        }

    def intensive(self, name: str) -> float:
        """A zero-deformation coefficient divided by the area per cluster.

        Turns the per-formula-unit value into the intensive 2D coefficient.
        """
        return per_area(self.extrapolations[name].value, self.cluster_area)

    def report(self, intensive: bool = True) -> str:
        from .report import format_result

        return format_result(self, intensive=intensive)


@dataclass
class FlexoCalculation:
    """Configure and run a flexoelectric calculation.

    Parameters
    ----------
    series
        Deformation series; configuration ``reference_index`` is the flat one.
    charge_model
        Per-atom charges (see :mod:`flexo2d.charges`).
    cluster_scheme
        Local dipole units (see :mod:`flexo2d.clusters`).
    mode
        Deformation ansatz and constitutive law. Defaults to
        :func:`flexo2d.modes.sinusoidal_bending`.
    axis
        Ripple direction, 0 = x.
    n_periods
        Ripple periods per supercell.
    skip_reference
        Exclude the flat configuration from the coefficient series. Its
        displacement field is identically zero, so every coefficient is 0/0
        there. Leave this on.
    extrapolation_degree
        Degree of the polynomial used for the zero-deformation limit.
    min_snr, rel_floor
        Resolvability thresholds, see :func:`flexo2d.extrapolate.resolvable_mask`.
    centroid_weights
        ``"mass"`` (default) or ``"uniform"``.
    dipole_origin
        ``"centroid"`` (default) or ``"cell"``; see
        :func:`flexo2d.fields.build_fields`.
    """

    series: ConfigSeries
    charge_model: ChargeModel
    cluster_scheme: ClusterScheme
    mode: DeformationMode = field(default_factory=sinusoidal_bending)
    axis: int = 0
    n_periods: int = 1
    skip_reference: bool = True
    extrapolation_degree: int = 2
    min_snr: float = 10.0
    rel_floor: float = 0.05
    centroid_weights: str = "mass"
    dipole_origin: str = "centroid"
    neutrality_tol: float = 1e-6

    # ---------------------------------------------------------------- running
    def build_fields(self) -> FieldSet:
        return build_fields(
            self.series,
            self.charge_model,
            self.cluster_scheme,
            axis=self.axis,
            n_periods=self.n_periods,
            centroid_weights=self.centroid_weights,
            neutrality_tol=self.neutrality_tol,
            dipole_origin=self.dipole_origin,
        )

    def run(self) -> FlexoResult:
        fields = self.build_fields()
        determinations = self.mode.solve()
        closed = {n: d.expression for n, d in determinations.items()}

        targets = self.mode.fit_targets()
        fits: list[dict[str, FitResult]] = []
        for i in range(fields.n_config):
            subs = {K: float(fields.wavevectors[i])}
            per_config: dict[str, FitResult] = {}
            for key, kind, quantity, component in targets:
                ansatz = (
                    self.mode.displacement[key]
                    if kind == "displacement"
                    else self.mode.polarization[key]
                )
                x, y = fields.profile(quantity, component, i, axis=self.axis)
                per_config[key] = fit_linear(
                    ansatz, x, y, subs, name=f"{key}[cfg {i}]"
                )
            fits.append(per_config)

        # Flatten all fitted parameters per configuration.
        params: list[dict[str, float]] = []
        errors: list[dict[str, float]] = []
        for per_config in fits:
            p, e = {}, {}
            for fr in per_config.values():
                p.update(fr.values)
                e.update(fr.stderr)
            params.append(p)
            errors.append(e)

        skip = (
            {self.series.reference_index} if self.skip_reference else set()
        )

        def evaluate(expr: sp.Expr) -> np.ndarray:
            fn_syms = sorted(expr.free_symbols, key=str)
            fn = sp.lambdify(fn_syms, expr, "numpy")
            out = np.full(fields.n_config, np.nan)
            for i in range(fields.n_config):
                if i in skip:
                    continue
                vals = []
                ok = True
                for s in fn_syms:
                    if s == K:
                        vals.append(float(fields.wavevectors[i]))
                    elif str(s) in params[i]:
                        vals.append(float(params[i][str(s)]))
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    try:
                        out[i] = float(fn(*vals))
                    except (ZeroDivisionError, FloatingPointError, ValueError):
                        out[i] = np.nan
            return out

        coefficients = {n: evaluate(e) for n, e in closed.items()}
        consistency = {
            n: {tag: evaluate(alt) for tag, alt in d.alternatives}
            for n, d in determinations.items()
            if d.alternatives
        }

        extrapolations: dict[str, Extrapolation] = {}
        for name, values in coefficients.items():
            den_name = DENOMINATOR_PARAMETERS.get(name)
            den = den_se = None
            if den_name is not None:
                den = np.array(
                    [p.get(den_name, np.nan) for p in params], dtype=float
                )
                den_se = np.array(
                    [e.get(den_name, np.nan) for e in errors], dtype=float
                )
                for i in skip:
                    den[i] = np.nan
            mask = resolvable_mask(
                values, den, den_se,
                min_snr=self.min_snr, rel_floor=self.rel_floor,
            )
            extrapolations[name] = extrapolate(
                name, fields.amplitudes, values,
                mask=mask, degree=self.extrapolation_degree,
            )

        return FlexoResult(
            series_name=self.series.name,
            mode=self.mode,
            fields=fields,
            fits=fits,
            coefficients=coefficients,
            consistency=consistency,
            extrapolations=extrapolations,
            closed_forms=closed,
            cluster_area=fields.cluster_area,
        )
