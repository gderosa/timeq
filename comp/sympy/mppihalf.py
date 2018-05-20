'''
PW 1=1 qubit
'''

from sympy import *
from sympy.physics.matrices import mdft
from sympy.abc import omega
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum import TensorProduct

init_printing(use_unicode=True)

#Ta, Tb = symbols('Ta Tb', real=True)

Ta, Tb = -Rational(1, 4), Rational(1, 4)

F = mdft(2)

T = (pi/omega) * Matrix([
    [Ta, 0],
    [0, Tb]
])

Omega = (4*omega**2/pi)*F*T*(F.adjoint())

Hs = I*omega*hbar*Matrix([
    [0, 1],
    [-1,0]
])

J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)

pprint(J.eigenvects())
