"""Deformation modes: the ansatz for the fields and the constitutive matching.

A *mode* says three things:

1. how the displacement field of the sheet is parameterised
   (``U_z = U0 cos(k x)`` for a sinusoidal ripple, ...),
2. how the polarization field is parameterised
   (``P_z = dzs sin(k x) + dzc cos(k x)``, ...),
3. the constitutive law linking them, e.g.
   ``P_z = e331 * u_zx + f3131 * du_zx/dx``.

Given those, :meth:`DeformationMode.solve` matches harmonics symbolically and
returns a closed-form expression for every piezoelectric and flexoelectric
coefficient in terms of the fitted amplitudes -- plus any *extra* relations the
ansatz over-determines, which are free consistency checks on the model.

Adding a new mode (biaxial bending, torsion, a different symmetry class) means
writing one more :class:`DeformationMode`; nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import sympy as sp

from .fitting import Ansatz

#: Shared symbols: abscissa along the ripple, and the ripple wavevector.
X = sp.Symbol("x", real=True)
K = sp.Symbol("k", real=True, positive=True)


@dataclass
class Relation:
    """One constitutive equation ``field == rhs``.

    Parameters
    ----------
    field
        Key into the mode's :attr:`DeformationMode.polarization` ansatz dict.
    rhs
        Right-hand side: a sympy expression built from the displacement
        ansatz, the coefficient symbols and derivatives.
    unknowns
        The coefficient symbols this relation determines.
    """

    field: str
    rhs: sp.Expr
    unknowns: tuple[sp.Symbol, ...]


@dataclass
class Determination:
    """A closed-form expression for one coefficient."""

    coefficient: str
    expression: sp.Expr
    from_harmonic: str
    #: Extra expressions for the same coefficient from over-determined
    #: harmonics; they should agree, and disagreement means the ansatz does not
    #: describe the data.
    alternatives: list[tuple[str, sp.Expr]] = field(default_factory=list)


@dataclass
class DeformationMode:
    """A parameterised deformation plus its constitutive law."""

    name: str
    displacement: dict[str, Ansatz]
    polarization: dict[str, Ansatz]
    relations: list[Relation]
    max_harmonic: int = 4
    description: str = ""

    # ------------------------------------------------------------- internals
    def _harmonics(self, expr: sp.Expr) -> dict[str, sp.Expr]:
        """Split `expr` into coefficients of 1, sin(n k x) and cos(n k x).

        Done by projection onto the harmonic basis over one period, so it works
        for any trigonometric form of the ansatz (products and powers of sines
        included), not only for expressions already written as a harmonic sum.
        """
        t = sp.Symbol("_t", real=True)
        e = sp.expand(expr.subs(X, t / K))
        two_pi = 2 * sp.pi

        def project(basis, norm):
            return sp.simplify(sp.integrate(sp.expand(e * basis), (t, 0, two_pi)) / norm)

        out: dict[str, sp.Expr] = {}
        recon = sp.Integer(0)
        c0 = project(sp.Integer(1), two_pi)
        if c0 != 0:
            out["const"] = c0
        recon += c0
        for n in range(1, self.max_harmonic + 1):
            for fn, tag in ((sp.sin, "sin"), (sp.cos, "cos")):
                c = project(fn(n * t), sp.pi)
                if c != 0:
                    out[f"{tag}({n}kx)"] = c
                recon += c * fn(n * t)

        residual = sp.simplify(e - recon)
        if residual != 0:
            raise ValueError(
                f"mode {self.name!r}: the expression {sp.simplify(expr)} is not "
                f"captured by harmonics up to n={self.max_harmonic} "
                f"(residual {residual}). Raise max_harmonic, or check that the "
                "constitutive relation is periodic in the ripple."
            )
        return out

    def solve(self) -> dict[str, Determination]:
        """Closed-form coefficients in terms of the fitted amplitudes."""
        equations: list[tuple[str, sp.Expr, set[sp.Symbol]]] = []
        all_unknowns: list[sp.Symbol] = []
        for rel in self.relations:
            try:
                lhs = self.polarization[rel.field].expression
            except KeyError as exc:
                raise KeyError(
                    f"relation refers to polarization field {rel.field!r}, "
                    f"which is not in {sorted(self.polarization)}"
                ) from exc
            all_unknowns.extend(rel.unknowns)
            for tag, coef in self._harmonics(lhs - rel.rhs).items():
                involved = {u for u in rel.unknowns if coef.has(u)}
                equations.append((f"{rel.field}:{tag}", coef, involved))

        solved: dict[str, Determination] = {}
        known: dict[sp.Symbol, sp.Expr] = {}

        def resolve(pending):
            """Greedy single-unknown passes, then a joint solve of the rest."""
            progress = True
            while progress:
                progress = False
                still = []
                for tag, coef, involved in pending:
                    free = [u for u in involved if u not in known]
                    if len(free) == 1:
                        sol = sp.solve(sp.Eq(coef.subs(known), 0), free[0],
                                       dict=True)
                        if sol:
                            expr = sp.simplify(sol[0][free[0]])
                            known[free[0]] = expr
                            solved[str(free[0])] = Determination(
                                str(free[0]), expr, tag
                            )
                            progress = True
                            continue
                    still.append((tag, coef, involved))
                pending = still

            # coupled system: solve what is left together. A phase-free ansatz
            # needs this, since there every harmonic involves every unknown.
            outstanding = {u for _, _, inv in pending for u in inv} - set(known)
            if outstanding:
                free = sorted(outstanding, key=str)
                eqs, tags = [], []
                for tag, coef, involved in pending:
                    if involved & outstanding:
                        eqs.append(sp.Eq(coef.subs(known), 0))
                        tags.append(tag)
                sol = sp.solve(eqs, free, dict=True)
                if sol:
                    label = "joint(" + ", ".join(tags) + ")"
                    for u in free:
                        if u in sol[0]:
                            expr = sp.simplify(sol[0][u])
                            known[u] = expr
                            solved[str(u)] = Determination(str(u), expr, label)
                    return []
            return pending

        # The uniform term of a polarization profile also absorbs any rigid
        # offset of the dipole origin and any uniform background, so it is the
        # weakest estimator. Determine everything possible from the oscillatory
        # harmonics first, and fall back to the constant term only if something
        # is still undetermined; whatever is left over becomes a cross-check.
        oscillatory = [e for e in equations if not e[0].endswith(":const")]
        constant = [e for e in equations if e[0].endswith(":const")]

        leftover = resolve(oscillatory)
        if any(str(u) not in solved for u in dict.fromkeys(all_unknowns)):
            leftover = resolve(leftover + constant)
        else:
            leftover = leftover + constant

        # Record over-determinations as alternatives / consistency relations.
        for tag, coef, involved in leftover:
            for u in involved:
                if str(u) in solved:
                    sol = sp.solve(sp.Eq(coef, 0), u, dict=True)
                    if sol:
                        solved[str(u)].alternatives.append(
                            (tag, sp.simplify(sol[0][u]))
                        )

        missing = [
            str(u) for u in dict.fromkeys(all_unknowns) if str(u) not in solved
        ]
        if missing:
            raise ValueError(
                f"mode {self.name!r}: could not determine {missing} from the "
                "declared relations; the ansatz does not excite them"
            )
        return solved

    # ---------------------------------------------------------------- public
    @property
    def coefficient_names(self) -> list[str]:
        names: list[str] = []
        for rel in self.relations:
            for u in rel.unknowns:
                if str(u) not in names:
                    names.append(str(u))
        return names

    def fit_targets(self) -> list[tuple[str, str, str, int]]:
        """``(ansatz_key, kind, quantity, component)`` for each field to fit."""
        comp = {"x": 0, "y": 1, "z": 2}
        out = []
        for key in self.displacement:
            out.append((key, "displacement", "displacement", comp[key[-1].lower()]))
        for key in self.polarization:
            out.append((key, "polarization", "dipole", comp[key[-1].lower()]))
        return out


# --------------------------------------------------------------------------
# The built-in mode: a one-dimensional sinusoidal ripple along x.
# --------------------------------------------------------------------------
def sinusoidal_bending(
    *,
    name: str | None = None,
    in_plane: bool = True,
    out_of_plane: bool = True,
    free_phase: bool = True,
) -> DeformationMode:
    """A sheet rippled sinusoidally along x -- the standard bending test.

    Out-of-plane block (transverse flexoelectricity):

    .. code-block:: text

        U_z(x) = U0 cos(k x)
        P_z(x) = dzs sin(k x) + dzc cos(k x)
        P_z    = e331 * u_zx + f3131 * d(u_zx)/dx,     u_zx = (1/2) dU_z/dx

    In-plane block (longitudinal flexoelectricity). Bending stretches the two
    chalcogen layers oppositely, which drives the second harmonic:

    .. code-block:: text

        U_x(x) = V0 x + W0 sin(2k x)
        P_x(x) = dxs sin(2k x) + dxc cos(2k x) + cd
        P_x    = e111 * dU_x/dx + f1111 * d2U_x/dx2

    Solving gives the closed forms

    .. code-block:: text

        f3131 = -2 dzc / (U0 k^2)      e331 = -2 dzs / (U0 k)
        f1111 =   -dxs / (4 W0 k^2)    e111 =  dxc / (2 W0 k)

    and, as a bonus, the over-determination ``cd = e111 V0`` -- a consistency
    check between the uniform part of ``P_x`` and the uniform strain ``V0``.

    Parameters
    ----------
    in_plane, out_of_plane
        Switch either block off if your data only resolves one of them.
    free_phase
        Let the ripple sit at an arbitrary phase relative to the cell origin,
        by carrying both quadratures of the displacement
        (``U_z = Uc cos(kx) + Us sin(kx)``) and solving the resulting coupled
        system. The extracted coefficients are then independent of where the
        ripple happens to sit in the cell. Switch it off only to reproduce the
        original phase-locked formulation; with ``U_s = 0`` the two agree
        exactly, but a relaxed structure whose ripple has drifted in phase
        gives nonsense without it.
    """
    U0, V0, W0 = sp.symbols("U0 V0 W0", real=True)
    Us, Ws = sp.symbols("Us Ws", real=True)
    dzs, dzc = sp.symbols("dzs dzc", real=True)
    dxs, dxc, cd = sp.symbols("dxs dxc cd", real=True)
    e331, f3131 = sp.symbols("e331 f3131", real=True)
    e111, f1111 = sp.symbols("e111 f1111", real=True)

    displacement: dict[str, Ansatz] = {}
    polarization: dict[str, Ansatz] = {}
    relations: list[Relation] = []

    if out_of_plane:
        if free_phase:
            u_z = Ansatz(
                U0 * sp.cos(K * X) + Us * sp.sin(K * X), (U0, Us), X
            )
        else:
            u_z = Ansatz(U0 * sp.cos(K * X), (U0,), X)
        p_z = Ansatz(dzs * sp.sin(K * X) + dzc * sp.cos(K * X), (dzs, dzc), X)
        displacement["Uz"] = u_z
        polarization["Pz"] = p_z
        # u_zx is the symmetrised gradient of the out-of-plane displacement
        u_zx = sp.diff(u_z.expression, X) / 2
        relations.append(
            Relation("Pz", e331 * u_zx + f3131 * sp.diff(u_zx, X), (e331, f3131))
        )

    if in_plane:
        if free_phase:
            u_x = Ansatz(
                V0 * X + W0 * sp.sin(2 * K * X) + Ws * sp.cos(2 * K * X),
                (V0, W0, Ws), X,
            )
        else:
            u_x = Ansatz(V0 * X + W0 * sp.sin(2 * K * X), (V0, W0), X)
        p_x = Ansatz(
            dxs * sp.sin(2 * K * X) + dxc * sp.cos(2 * K * X) + cd * sp.Integer(1),
            (dxs, dxc, cd),
            X,
        )
        displacement["Ux"] = u_x
        polarization["Px"] = p_x
        relations.append(
            Relation(
                "Px",
                e111 * sp.diff(u_x.expression, X)
                + f1111 * sp.diff(u_x.expression, X, 2),
                (e111, f1111),
            )
        )

    if not relations:
        raise ValueError("enable at least one of in_plane / out_of_plane")

    if name is None:
        name = "sinusoidal-bending-x" + ("" if free_phase else "-phase-locked")
    return DeformationMode(
        name=name,
        displacement=displacement,
        polarization=polarization,
        relations=relations,
        description=sinusoidal_bending.__doc__.strip().splitlines()[0],
    )


def dimension_of(expr: sp.Expr) -> tuple[int, int]:
    """``(length_exponent, charge_exponent)`` of a coefficient's closed form.

    Derived from the expression itself rather than from the coefficient's
    name, so a new :class:`DeformationMode` reports correct units with no
    extra bookkeeping.
    """
    from .units import SYMBOL_DIMENSIONS

    length, charge = sp.symbols("_L _Q", positive=True)
    sub = {}
    for sym in expr.free_symbols:
        try:
            le, qe = SYMBOL_DIMENSIONS[str(sym)]
        except KeyError as exc:
            raise KeyError(
                f"no dimension registered for fitted parameter {sym!r}; add it "
                "to flexo2d.units.SYMBOL_DIMENSIONS"
            ) from exc
        sub[sym] = length**le * charge**qe
    powers = sp.Poly(sp.simplify(expr.subs(sub) * sp.Symbol("_c")),
                     length, charge).monoms()
    if len(powers) != 1:
        raise ValueError(
            f"closed form {expr} is not dimensionally homogeneous: {powers}"
        )
    return int(powers[0][0]), int(powers[0][1])


#: Coefficients that need a non-vanishing fitted amplitude in the denominator.
#: Used to decide which configurations can actually resolve a coefficient.
DENOMINATOR_PARAMETERS = {
    "f3131": "U0",
    "e331": "U0",
    "f1111": "W0",
    "e111": "W0",
}
