from sympy import *
from sympy.matrices.expressions.fourier import DFT
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger


init_printing()

F = DFT(2).as_explicit()

F

F @ Dagger(F)

HT_hv = Matrix([
    [ 0, I],
    [-I, 0]
]) 

HS_hv = Matrix([
    [ 0, I],
    [-I, 0]
])

eigensys = HT_hv.eigenvects()

eigensys

U = Matrix([eigensys[0][2][0].T, eigensys[1][2][0].T]).T / sqrt(2)

U

 Dagger(U) @ U

Dagger(U) @ HS_hv @ U

UU = TensorProduct(F@U, eye(2))

UU


