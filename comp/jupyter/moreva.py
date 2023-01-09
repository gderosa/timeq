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

F.__class__

eigensys = HT_hv.eigenvects()

eigensys

U = Matrix([eigensys[0][2][0].T, eigensys[1][2][0].T]).T / sqrt(2)

Dagger(U) @ HS_hv @ U

TensorProduct(F, eye(2))

TensorProduct(U, eye(2))

UU = TensorProduct(F, eye(2)) @ TensorProduct(U, eye(2)) 

UU

Psi_t = UU @ Psi_hv

Psi_t

psi_0 = Matrix(Psi_t[0:2])
psi_1 = Matrix(Psi_t[2:])

psi_0

psi_1

t0 = eigensys[0][0]
t1 = eigensys[1][0]

t0, t1

U_evol = exp(-I*HS_hv*(t1-t0))

U_evol

evolved_Schrod = U_evol @ psi_0

evolved_PW = psi_1

evolved_Schrod.evalf()

evolved_PW.evalf()




