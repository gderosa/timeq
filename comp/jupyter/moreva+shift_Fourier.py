from sympy import *
from sympy.matrices.expressions.fourier import DFT
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


init_printing()

# $\hbar = \omega = 0$, hence $\hat{H}_T = \hat{\Omega}$.

# **Polarization eigensates (our computational or fiducial basis) in $\mathcal{H}_T$**

H_hv = Matrix([
    [1],
    [0]
])
V_hv = Matrix([
    [0],
    [1]
])

Psi_hv = (1 / sqrt(2)) * (TensorProduct(H_hv, V_hv) - TensorProduct(V_hv, H_hv))

HT_hv = Matrix([
    [ 0, I],
    [-I, 0]
])
HS_hv = Matrix([
    [ 0, I],
    [-I, 0]
])

J_hv = TensorProduct(HT_hv, eye(2)) + TensorProduct(eye(2), HS_hv)

J_hv @ Psi_hv

F = DFT(2).as_explicit().as_mutable()

F

Dagger(F)

F @ Dagger(F)

eigensys_H_T = HT_hv.eigenvects()

eigensys_H_T

eigenvalues_H_T = list(map(lambda el: el[0], eigensys_H_T))

# **Change to H_T representation**

U = Matrix([eigensys_H_T[0][2][0].T, eigensys_H_T[1][2][0].T]).T / sqrt(2)

U

U @ Dagger(U)

HT_HT = Dagger(U) @ HT_hv @ U

HT_HT

# **Clock frequency (or energy) resolution**

delta_HT = abs(HT_HT[1,1] - HT_HT[0,0])

delta_HT

# **Time operator in clock frequency (or energy) eigenbasis**

T_HT = (pi / (delta_HT**2)) * F @ HT_HT @ Dagger(F) 

T_HT

eigenvals_T = list(T_HT.eigenvals().keys())

eigenvals_T.sort()

eigenvals_T

# **Time resolution of the clock (difference between contiguous eigenvalues)**

delta_T = eigenvals_T[1] - eigenvals_T[0]

delta_T

# **(Double) change of basis:**
#
# Now, the $U^{\dagger}$ matrix translates components **from Polarization into Energy/Frequency eigenbasis** representation in the clock space.
#
# The (inverse) Fourier $F^{\dagger}$ does the same **from Energy/Frequency into Time**.
#
# As we are operating in the product space, we need the tensor product by the Identity in the "system" space $\mathcal{H}_S$.
#
#

# **Correction term due to $\omega_0 \neq 0$ (freq. shift of Fourier transform)**
#
# See `{eq:IDFT:chrepr:tshift}` in the thesis, "Non-zero initial values":
# \begin{equation}
#   \langle{t_{m}}|{\psi}\rangle = e^{i\omega_{0}t_m} \sum_n F^{\dagger}_{mn} \langle{\omega_n}|{\psi}\rangle \text{.}
# \end{equation}
#
# Shift term: $e^{i\omega_{0}t_m} \text{,} \; \forall m = 0, 1$.

omega_0 = eigenvalues_H_T[0]

shift = list(map(lambda t_m: exp(t_m * omega_0 * I), eigenvals_T))


shift

# matrix form
Shift = diag(*shift)

Shift

UU = TensorProduct(Shift @ Dagger(F) @ Dagger(U), eye(2))

UU

# **Page--Wootters history vector in time representation (or time $\otimes$ polarization, more correctly)**

Psi_t = (UU @ Psi_hv)

Psi_t

psi_0 = Matrix(Psi_t[0:2])
psi_1 = Matrix(Psi_t[2:])

psi_0 = (psi_0 / psi_0.norm())
psi_0

psi_1 = (psi_1 / psi_1.norm())
psi_1

# **Time evolution in standard quantum mechanics (for comparison)**

U_evol = exp(-I*HS_hv*delta_T)

U_evol

evolved_Schrod = U_evol @ psi_0

evolved_PW = psi_1

evolved_Schrod

evolved_PW

# With the shift term
# (and simply recalling that $i=e^{\frac{i\pi}{2}}$),
# it is seen that
# results from the two theories coincide.

simplify(evolved_Schrod - evolved_PW)

# &square;
