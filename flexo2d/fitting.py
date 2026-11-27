"""Fitting profiles to an ansatz that is linear in its parameters.

The ripple wavelength is set by the supercell, not by the fit, so every ansatz
in :mod:`flexo2d.modes` is *linear* in its unknowns. That makes the fit an
ordinary linear least-squares problem: one exact solve, no initial guesses and
no convergence failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import sympy as sp


@dataclass
class Ansatz:
    """A model expression that is linear in its free parameters.

    Parameters
    ----------
    expression
        A sympy expression in the abscissa symbol, the wavevector symbol and
        the free parameters.
    parameters
        The free parameter symbols, in the order the fit reports them.
    variable
        The abscissa symbol.
    """

    expression: sp.Expr
    parameters: tuple[sp.Symbol, ...]
    variable: sp.Symbol

    def __post_init__(self) -> None:
        self.parameters = tuple(self.parameters)
        for p in self.parameters:
            if sp.simplify(sp.diff(self.expression, p, 2)) != 0:
                raise ValueError(
                    f"ansatz {self.expression} is not linear in {p}; "
                    "flexo2d fits linear-in-parameter models only"
                )
        const = self.expression
        for p in self.parameters:
            const = const - p * sp.diff(self.expression, p)
        if sp.simplify(const) != 0:
            raise ValueError(
                f"ansatz {self.expression} has a parameter-free term "
                f"({sp.simplify(const)}); add it as an explicit parameter"
            )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(str(p) for p in self.parameters)

    def basis(self, substitutions: Mapping[sp.Symbol, float]):
        """Numeric basis functions, one per parameter."""
        return [
            sp.lambdify(
                self.variable,
                sp.diff(self.expression, p).subs(substitutions),
                "numpy",
            )
            for p in self.parameters
        ]

    def design_matrix(self, x: np.ndarray,
                      substitutions: Mapping[sp.Symbol, float]) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        cols = []
        for fn in self.basis(substitutions):
            col = np.asarray(fn(x), dtype=float)
            if col.ndim == 0:                     # a constant basis function
                col = np.full_like(x, float(col))
            cols.append(col)
        return np.column_stack(cols)

    def evaluate(self, x: np.ndarray, values: Mapping[str, float],
                 substitutions: Mapping[sp.Symbol, float]) -> np.ndarray:
        subs = dict(substitutions)
        for p in self.parameters:
            subs[p] = float(values[str(p)])
        fn = sp.lambdify(self.variable, self.expression.subs(subs), "numpy")
        out = np.asarray(fn(np.asarray(x, dtype=float)), dtype=float)
        return out if out.ndim else np.full_like(np.asarray(x, float), float(out))


@dataclass
class FitResult:
    """Outcome of one least-squares fit."""

    name: str
    values: dict[str, float]
    stderr: dict[str, float]
    r_squared: float
    rmse: float
    n_points: int
    x: np.ndarray = field(repr=False, default=None)
    y: np.ndarray = field(repr=False, default=None)
    y_model: np.ndarray = field(repr=False, default=None)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def summary(self) -> str:
        pars = ", ".join(
            f"{k}={v:+.6g}+-{self.stderr[k]:.2g}" for k, v in self.values.items()
        )
        return f"{self.name}: R2={self.r_squared:.6f} rmse={self.rmse:.3e}  {pars}"


def fit_linear(ansatz: Ansatz, x, y, substitutions: Mapping[sp.Symbol, float],
               name: str = "") -> FitResult:
    """Least-squares fit of a linear-in-parameters `ansatz` to ``(x, y)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x has shape {x.shape} but y has shape {y.shape}")
    n_par = len(ansatz.parameters)
    if len(x) < n_par:
        raise ValueError(
            f"{name or 'fit'}: {len(x)} data points for {n_par} parameters"
        )

    A = ansatz.design_matrix(x, substitutions)
    coef, residues, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank < n_par:
        raise ValueError(
            f"{name or 'fit'}: design matrix is rank deficient "
            f"({rank} < {n_par}); the basis functions are not independent on "
            "this sampling grid"
        )

    model = A @ coef
    resid = y - model
    dof = max(len(x) - n_par, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan

    names = ansatz.parameter_names
    return FitResult(
        name=name,
        values={n: float(c) for n, c in zip(names, coef)},
        stderr={n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)},
        r_squared=r2,
        rmse=float(np.sqrt(float(resid @ resid) / len(x))),
        n_points=len(x),
        x=x, y=y, y_model=model,
    )
