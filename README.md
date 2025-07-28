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
$$
u_z(x) = U_0 \,\cos(k\,x), 
\qquad
u_x(x) = V_0\,x + W_0\,\cos(2\,k\,x).
$$

### 2. Strain and strain gradients
From \(u_i(x)\), the nonzero strain components are:
$$
u_{zx}(x)
= \tfrac12\bigl(\partial_x u_z + \partial_z u_x\bigr)
= -\frac{U_0\,k}{2}\,\sin(k\,x),
$$
$$
u_{xx}(x)
= \partial_x\,u_x
= V_0 + 2\,W_0\,k\,\cos(2\,k\,x).
$$
Their derivatives (strain gradients) are
$$
\frac{\partial u_{zx}}{\partial x}
= -\frac{U_0\,k^2}{2}\,\cos(k\,x),
\qquad
\frac{\partial u_{xx}}{\partial x}
= -4\,W_0\,k^2\,\sin(2\,k\,x).
$$

### 3. Local dipole moments
Each unit cell acquires an effective dipole \(\mathbf{d}(x)\) with projections
$$
d_z(x)
= e_{zzx}\,u_{zx}(x)
+ e_{zxx}\,u_{xx}(x)
+ f_{zxzx}\,\frac{\partial u_{zx}}{\partial x}
+ f_{zxxx}\,\frac{\partial u_{xx}}{\partial x},
$$
$$
d_x(x)
= e_{xzx}\,u_{zx}(x)
+ e_{xxx}\,u_{xx}(x)
+ f_{xxxz}\,\frac{\partial u_{xz}}{\partial x}
+ f_{xxxx}\,\frac{\partial u_{xx}}{\partial x}.
$$

### 4. Extraction of tensor components
Fitting the computed \(d_i(x)\) against the known sinusoidal forms yields:
$$
\begin{aligned}
f_{3131} &= -\frac{2\,d^{(z)}_\mathrm{cos}}{U_0\,k^2}, 
&\quad
e_{331} &= -\frac{2\,d^{(z)}_\mathrm{sin}}{U_0\,k}, \\[4pt]
f_{1111} &= -\frac{d^{(x)}_\mathrm{sin}}{4\,W_0\,k^2}, 
&\quad
e_{111} &= \frac{d^{(x)}_\mathrm{cos}}{2\,W_0\,k}.
\end{aligned}
$$

