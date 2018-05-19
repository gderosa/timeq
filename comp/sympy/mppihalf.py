'''
PW 1=1 qubit
'''

from sympy import *
from sympy.physics.matrices import mdft
from sympy.abc import omega
from sympy.physics.quantum.constants import hbar

init_printing(use_unicode=True)

F = mdft(2)

T = (1/omega) * Matrix([
    [-pi/4, 0],
    [0, +pi/4]
])

Omega = (4*omega**2/pi)*F*T*F.adjoint()



pprint(hbar * Omega)
