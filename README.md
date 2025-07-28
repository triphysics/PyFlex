# PyFlex

**Python implementation of flexoelectric and piezoelectric coupling in a bended 2D material**

It computes how sinusoidal bending (corrugation) of a MoS₂ monolayer induces both flexoelectric and piezoelectric polarization via strain and strain‐gradient couplings.

---

## 🔬 Physical Background

1. **Strain tensor** \(u_{ij}(x)\) arises from atomic displacements \(U_i(x)\).  
2. **Flexoelectricity**: polarization \(P_i\) ≃ \(f_{ijkl}\,\partial_j u_{kl}\) — coupling of strain gradients \(\partial_j u_{kl}\).  
3. **Piezoelectricity**: polarization \(P_i\) ≃ \(e^s_{ijk}\,u_{jk}\) — coupling of strains \(u_{jk}\) (surface‐induced for centrosymmetric monolayers).  
4. **Total polarization** (neglecting electrostatic potential gradient term):
   \[
   P_i(x) \;=\; 
     f_{ijkl}\,\frac{\partial u_{kl}}{\partial x_j}
   + e^s_{ijk}\,u_{jk}
   \]

---

## 🧮 Key Equations

### 1. Displacement fields
We impose a sinusoidal corrugation along \(x\):
\[
\begin{aligned}
u_z(x) &= U_0 \,\cos(k x), \\
u_x(x) &= V_0\,x \;+\; W_0\,\cos(2 k x).
\end{aligned}
\]

- \(U_0, V_0, W_0\): amplitudes determined by relaxation  
- \(k = 2\pi/\lambda\): wavevector of corrugation

### 2. Strain and strain gradients
From \(u_i(x)\), the nonzero strain components are:
\[
\begin{aligned}
u_{zx}(x) &= \tfrac12\!\Bigl(\partial_x u_z+\partial_z u_x\Bigr)
            = -\frac{U_0\,k}{2}\,\sin(kx), \\[6pt]
u_{xx}(x) &= \partial_x u_x
            = V_0 \;+\; 2\,W_0\,k\,\cos(2kx).
\end{aligned}
\]
Their derivatives (strain gradients) are
\[
\begin{aligned}
\partial_x u_{zx}(x)
  &= -\frac{U_0\,k^2}{2}\,\cos(kx), 
&
\partial_x u_{xx}(x)
  &= -4\,W_0\,k^2\,\sin(2kx).
\end{aligned}
\]

### 3. Local dipole moments
Each unit cell acquires an effective dipole \(\mathbf{d}(x)\) with projections
\[
\begin{aligned}
d_z(x)
&= e_{zzx}\,u_{zx}(x)
 + e_{zxx}\,u_{xx}(x)
 + f_{zxzx}\,\frac{\partial u_{zx}}{\partial x}
 + f_{zxxx}\,\frac{\partial u_{xx}}{\partial x},
\\[4pt]
d_x(x)
&= e_{xzx}\,u_{zx}(x)
 + e_{xxx}\,u_{xx}(x)
 + f_{xxxz}\,\frac{\partial u_{xz}}{\partial x}
 + f_{xxxx}\,\frac{\partial u_{xx}}{\partial x}.
\end{aligned}
\]
- \(e_{ijk}\): surface‐induced piezoelectric coefficients  
- \(f_{ijkl}\): static flexoelectric coefficients  

### 4. Extraction of tensor components
Fitting the computed \(d_i(x)\) vs. \(u_{jk}\), \(\partial u_{jk}/\partial x\) yields:
\[
\begin{aligned}
f_{3131} &= -\frac{2\,d^{(z)}_\mathrm{cos}}{U_0\,k^2},
& e_{331} &= -\frac{2\,d^{(z)}_\mathrm{sin}}{U_0\,k}, \\
f_{1111} &= -\frac{d^{(x)}_\mathrm{sin}}{4\,W_0\,k^2},
& e_{111} &= \frac{d^{(x)}_\mathrm{cos}}{2\,W_0\,k}.
\end{aligned}
\]
Where \(d^{(z)}_{\cos}, d^{(z)}_{\sin}, d^{(x)}_{\cos}, d^{(x)}_{\sin}\) are the sinusoidal fit amplitudes.

---

## 📊 Features

- **Automated dipole‐moment calculation** from atomic positions + Bader charges  
- **Sinusoidal and harmonic fitting** of displacement, strain, and dipole data  
- **Extraction** of \(\{f_{ijkl}, e_{ijk}\}\) vs. corrugation (1%–10%)  
- **Unit conversion** to mC/m² and pC/m  
- **Visualization** of:
  - Displacement profiles \(u_i(x)\)  
  - Strain and strain‐gradient fields  
  - Dipole moment distributions  
  - Tensor‐component vs. corrugation plots  

---

## 📈 Sample Results

- **Flexoelectric vs. piezoelectric**: \(\lvert f\rvert/\lvert e\rvert\sim5\!-\!30\,\mathrm{nm}^{-1}\)  
- **Typical values** (at 5% corrugation):
  - \(f_{3131}\approx -3.53\,\mathrm{mC/m}^2\)  
  - \(f_{1111}\approx -1.77\,\mathrm{mC/m}^2\)  
  - \(e_{331}\approx -0.731\,\mathrm{pC/m}\)  
  - \(e_{111}\approx -1.65\,\mathrm{pC/m}\)  

---

## 📦 Dependencies

```text
numpy      # array math
scipy      # curve_fit, derivatives
sympy      # symbolic expressions
matplotlib # plotting
pandas     # tables & CSV I/O
jupyter    # interactive exploration

