'''
PW 1=1 qubit
'''

from sympy import *
from sympy.physics.matrices import mdft
from sympy.abc import omega
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import TensorProduct

init_printing(use_unicode=True)

F = mdft(2)

T = (1/omega) * Matrix([
    [-pi/4, 0],
    [0, +pi/4]
])

Omega = (4*omega**2/pi)*F*T*(F.adjoint())

Hs = I*omega*hbar*Matrix([
    [0, 1],
    [-1,0]
])

J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)

pprint(J)
