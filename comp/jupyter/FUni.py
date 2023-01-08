from sympy import *
from sympy.physics.quantum.dagger import Dagger
from sympy.matrices.expressions.fourier import DFT
init_printing(use_unicode=True)

F = DFT(3).as_explicit()

U = Matrix([
    [0, 1, 0], 
    [I, 0, 0], 
    [0, 0,-I]
])

U @ Dagger(U)

Dagger(U) @ U

 nhy67U @ F @ Dagger(U)

F


