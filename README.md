# flexo2d

Flexoelectric and piezoelectric coefficients of 2D materials, from a series of
rippled supercells plus a set of atomic charges.

Nothing in the pipeline is specific to one material: a new material is a new set
of structures, a charge model and a bond cutoff.

```python
from flexo2d import FixedCharges, FlexoCalculation, NeighbourShell, read_poscar_series

series = read_poscar_series([f"POSCAR_{i:02d}.vasp" for i in range(11)], name="MoS2")
result = FlexoCalculation(
    series,
    charge_model=FixedCharges({"Mo": 1.083, "S": -0.5415}),
    cluster_scheme=NeighbourShell("Mo", "S", cutoff=0.30),
).run()
print(result.report())
```

or from the shell:

```
python -m flexo2d --npz data/mos2_ripple.npz --center Mo --shell S \
    --cutoff 0.30 --charge Mo=1.083 --charge S=-0.5415
```

## The method

A flat sheet is rippled sinusoidally along one in-plane direction with growing
amplitude. For each configuration the sheet is cut into charge-neutral local
units (one metal plus its chalcogen shell, for a TMD), and each unit gives a
dipole moment **p** and a centroid **r**. Fitting the resulting profiles against
the bending ansatz and matching harmonics against the constitutive law gives the
coefficients; extrapolating the series to zero amplitude gives the
linear-response values.

```
            U_z(x) = U0 cos(kx)                P_z(x) = dzs sin(kx) + dzc cos(kx)
            U_x(x) = V0 x + W0 sin(2kx)         P_x(x) = dxs sin(2kx) + dxc cos(2kx) + cd

  out of plane   P_z = e331 u_zx + f3131 d(u_zx)/dx,   u_zx = (1/2) dU_z/dx
  in plane       P_x = e111 dU_x/dx + f1111 d2U_x/dx2

  =>   f3131 = -2 dzc / (U0 k^2)        e331 = -2 dzs / (U0 k)
       f1111 =   -dxs / (4 W0 k^2)      e111 =  dxc / (2 W0 k)
```

Those closed forms are **not hard-coded**. `flexo2d.modes` declares the ansatz
and the constitutive law in sympy and derives them by harmonic matching, so a new
deformation mode or symmetry class only needs a new `DeformationMode`. The
derivation also finds one relation the ansatz over-determines, `cd = e111 V0`,
which the code reports as a free consistency check.

## Pipeline

| step | module | what it does |
|---|---|---|
| 0 | `build` | **generate** the rippled supercells to feed to DFT (isometric bending) |
| 1 | `io` | read the deformation series (POSCAR, extended XYZ, native `.npz`) |
| 2 | `charges` | per-atom charges; neutralisation and Fourier smoothing |
| 3 | `clusters` | cut the sheet into neutral local dipole units, with periodic images |
| 4 | `fields` | cluster dipoles, centroids and the displacement field |
| 5 | `modes` | declare the ansatz + constitutive law, solve symbolically |
| 6 | `fitting` | fit the ansatz (exact linear least squares) |
| 7 | `extrapolate` | take the zero-deformation limit |
| 8 | `pipeline`, `report`, `plotting` | orchestrate and present |

Each layer is swappable: `ClusterScheme`, `ChargeModel` and `DeformationMode`
are the three extension points.

## Generating the input structures

`flexo2d.build` makes the deformation series for you:

```python
from flexo2d import read_poscar, supercell, ripple_series, write_poscar_series

flat = supercell(read_poscar("POSCAR.unitcell"), 8, 1)
series = ripple_series(flat, amplitudes=[0.0, 0.04, 0.08, 0.12, 0.16, 0.20])
write_poscar_series("ripples/", series)     # POSCAR_00.vasp ... then relax IONS ONLY
```

or `python examples/generate_ripples.py --out ripples/`.

Files are written as `POSCAR_<nn>.vasp` — the extension viewers such as VESTA
recognise, with the index zero padded so sorting keeps the series in amplitude
order. The sheet is centred in the vacuum, so it does not straddle the cell
boundary.

The bend is **isometric**: the neutral surface keeps its arc length and the
cell shortens to match. This matters more than it sounds. Adding `A·cos(kx)` to
z without shortening the cell stretches the sheet by
`⟨√(1+(Ak sin u)²)⟩ − 1`, which for the amplitudes used here reaches **+2.2%**
of spurious uniform tensile strain — roughly twenty times the real membrane
strain (−0.11%) and of the opposite sign. It would contaminate the bending
signal with a piezoelectric response.

Periodicity is checked directly: rippling a 1× cell with one period and tiling it
twice reproduces a 2× cell rippled with two periods to 1e-9, and the
neutral-surface arc spacing is uniform to 1e-5 nm *including across the cell
boundary*.

### Amplitude limits

One period of `A·cos(kx)` has arc length of at least its total variation, `4A`,
no matter how far the cell is squeezed. So an isometric ripple only exists while

```
4 · n_periods · A  <  L_flat          (flexo2d.max_amplitude)
```

A short cell hits this fast — a 0.55 nm cell caps out at A < 0.14 nm. Past the
bound there is no solution and the cell length collapses to zero; `flexo2d`
refuses with the bound in the message rather than returning a degenerate cell.

A second, usually tighter limit: an offset surface a distance `h` from a curve
of curvature `κ` is only regular while `|h·κ| < 1`. Beyond that the inner
chalcogen layer folds through itself. That is refused too, and a bending strain
above 30% warns — it is far outside the linear response the coefficient is
defined in.

## Applying it to another 2D material

1. **Build the structures.** `ripple_series` above, then relax each rippled
   cell's ions in DFT with the cell held fixed. Keep the atom ordering
   identical across configurations.
2. **Find the cutoff.** `python -m flexo2d --npz ... --center B --shells` prints
   the coordination shells with their multiplicities; put the cutoff in the gap
   after the first one.
3. **Pick charges** so that one local unit is neutral. `NeighbourShell` divides
   each shared neighbour's weight by its bond count, so for h-BN (B with 3 N,
   each N bonded to 3 B) a unit is `1 B + 3 × (1/3) N` and neutrality means
   `q_B + q_N = 0`.
4. **Run it.** `examples/new_material.py` is a template with a CONFIG block.

The built-in mode assumes a sheet rippled along one axis, with the out-of-plane
response at the fundamental and the in-plane response at the second harmonic.
That holds for any centrosymmetric-in-plane 2D material bent this way. For
something else — biaxial bending, torsion, a polar monolayer with a non-zero
intrinsic dipole — write a new `DeformationMode`; nothing downstream changes.

## Units

Internally: nm, elementary charge, u. Coefficients come out **per local unit**,
so `f` is in `e·nm²` and `e` in `e·nm`. `result.intensive(name)` divides by the
in-plane area per unit to give the intensive 2D coefficient — `f` in `e`
(SI: C), `e` in `e/nm` (SI: C/m). Units are derived from each coefficient's
closed form by `modes.dimension_of`, so a new `DeformationMode` gets them right
with no extra bookkeeping. The report prints both.

For MoS₂ the intensive transverse coefficient is
`f3131 = -0.257 e = -4.11·10⁻²⁰ C`. Note the factor-of-2 convention: `f3131`
multiplies `∂ₓu_zx = ½w''`, so a coefficient defined directly against `w''` is
half this.

## Design notes

* **The fits are linear.** Because `k` is fixed by the supercell, every ansatz
  is linear in its parameters, so fitting is one exact least-squares solve —
  no initial guesses, no convergence failures.
* **The wavevector is geometric,** `k = 2π·n_periods/L_x`, read off the cell
  rather than supplied.
* **Configuration selection is automatic.** `f1111` and `e111` divide by the
  fitted in-plane amplitude `W0`, which is in the noise for small ripples;
  `extrapolate.resolvable_mask` drops those configurations on a
  signal-to-noise criterion instead of by hand.
* **Local units tile the sheet exactly once.** `NeighbourShell` divides each
  shared neighbour's weight by its bond count, found from the structure, so no
  per-material bookkeeping is needed.

## Results for monolayer MoS₂

`python examples/mos2.py`, with sheet-averaged Bader charges
(Mo +1.083, S −0.5415):

| coefficient | zero-deformation limit | per unit | spread across configs |
|---|---|---|---|
| `f3131` | −2.205·10⁻² | e·nm² | 1.6% — well determined |
| `e331`  | −4.08·10⁻³  | e·nm  | 8.8% |
| `f1111` | −1.1·10⁻² *(see below)* | e·nm² | resolvable only for the 7 largest ripples |
| `e111`  | −0.7 to −4.9·10⁻² *(see below)* | e·nm | poorly determined |

`f3131`, the transverse flexoelectric coefficient, is the robust output: the
`U_z` and `P_z` fits have R² > 0.9986 everywhere and the per-configuration
values vary by under 2%. The `e` coefficients are the quadrature partners of the
same harmonics, so they are far more sensitive — see the caveats below.

## Caveats worth knowing

* **`e331` is symmetry-forbidden.** Monolayer 1H-MoS₂ is D₃ₕ (6̄m2), and the
  whole third row of the piezoelectric matrix vanishes: P_z cannot respond
  linearly to any strain. Independently, `u_zx = ½∂U_z/∂x` is not a shear strain
  — in Kirchhoff plate kinematics ε_xz ≡ 0, and what is left is half the local
  rotation, which frame indifference forbids polarization from depending on.
  Whatever comes out is therefore systematic error, and an ideal noiseless
  ripple still gives −3.2·10⁻³, 77% of the value found on real data. Use it as
  the method's error floor.
* **`f1111` and `e111` rest on a long extrapolation here.** They are only
  resolvable for ripple amplitudes 4–10, so reaching back to zero covers 0.7×
  the span of the data that determines them, and the answer depends on the
  extrapolation degree:

  | degree | `f3131` | `e331` | `f1111` | `e111` |
  |---|---|---|---|---|
  | 0 (mean) | −2.154·10⁻² | −4.28·10⁻³ | −1.11·10⁻² | −0.74·10⁻² |
  | 1 (linear) | −2.218·10⁻² | −4.26·10⁻³ | −0.89·10⁻² | −1.51·10⁻² |
  | 2 (quadratic) | −2.205·10⁻² | −4.08·10⁻³ | +0.003·10⁻² | −4.93·10⁻² |

  `f3131` moves by under 3% and `e331` by under 5%, so both are safe to quote.
  The in-plane pair is not: the report flags them automatically. To pin them
  down, add configurations at small ripple amplitude.
* **The in-plane fit parameters are largely kinematic.** For an inextensible
  sheet `w = U₀cos(kx)`, the exact projected in-plane displacement is
  `u(x) = −(U₀²k²/4)x + (U₀²k/8)sin(2kx)`, and the fitted `V0`, `W0` match that
  geometry to a few percent. So dividing by them does not give a strain-response
  coefficient. See TODO.md P2.
* **The ripple's phase is fitted, not assumed.** The displacement field is
  Lagrangian but is sampled at the deformed (Eulerian) coordinate, and that
  mismatch appears mainly as a phase error. The default mode therefore carries
  both quadratures (`U_z = U0 cos kx + Us sin kx`) and solves the resulting
  coupled system, so the answer does not depend on where the ripple sits in the
  cell. On the synthetic benchmark this cuts the per-configuration error for
  `e111` from a factor of 6 down to 1%. `sinusoidal_bending(free_phase=False)`
  gives the phase-locked formulation, which returns NaN on a ripple that has
  drifted in phase — as a relaxed structure easily can.
* **Raw population-analysis charges are noisy.** Per-atom Bader charges leave
  each local unit slightly non-neutral and carry noise comparable to the dipole
  signal; on this dataset they make `f3131` drift by a factor of 5 across the
  series. Neutralise them (`charges.neutralize`) and smooth them
  (`charges.fourier_smooth`) before use, or use averaged charges. Non-neutral
  units are not silently accepted: their dipoles are referred to the unit's own
  centroid and a warning says so.
* **This is a rigid point-charge model.** With `FixedCharges` the polarization
  responds only to ionic displacement; electronic polarizability and charge
  transfer between units are not included. Applied to graphene it returns
  identically zero, because all Bader charges vanish by symmetry — while
  graphene's real flexoelectric response comes entirely from π–σ
  rehybridization. See TODO.md P1/P3.

## Next steps

See [TODO.md](TODO.md). The short version: `f3131` is the one usable output and
still needs the electronic contribution and a q→0 study; `e331` is
symmetry-forbidden and serves as the error floor; the in-plane block divides by
a kinematic artifact and needs reformulating.

## Layout

```
flexo2d/        units species structure build io charges clusters fields
                modes fitting extrapolate pipeline report plotting cli
data/           mos2_ripple.npz        rippled MoS2 monolayer, 11 amplitudes
examples/       mos2.py                coefficients from the shipped dataset
                generate_ripples.py    build a series and write it for DFT
                new_material.py        template for a new material
tests/          100 tests, including a synthetic round trip with planted
                coefficients, periodicity checks on the generator, and a
                cross-check against an independent implementation
```

## Install and test

Needs Python 3.9+, numpy and sympy; matplotlib only for plotting.

```
pip install -e .
pytest tests -q
```
