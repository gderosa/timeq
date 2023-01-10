from sympy import *
from sympy.matrices.expressions.fourier import DFT
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


init_printing()

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

eigensys = HT_hv.eigenvects()

eigensys

U = Matrix([eigensys[0][2][0].T, eigensys[1][2][0].T]).T / sqrt(2)

U

U @ Dagger(U)

HT_HT = Dagger(U) @ HT_hv @ U

HT_HT

delta_HT = abs(HT_HT[1,1] - HT_HT[0,0])

delta_HT

T_HT = (pi / (delta_HT**2)) * F @ HT_HT @ Dagger(F) 

T_HT

# interestingly, it is diagonal, but in reverse time order...
T_hv = U @ T_HT @ Dagger(U)
T_hv

psi_i = -H_hv

psi_f_PW = V_hv

delta_T = pi/2







U_evol = exp(-I*HS_hv*delta_T)

U_evol

psi_f_Schrod = U_evol @ psi_i

psi_f_Schrod

psi_f_PW




