'''
PW adim
'''

from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct

F = mdft(2)

T = Matrix([
    [-pi/4, 0],
    [0, pi/4]
])

Omega = F*T*(F.adjoint())

Hs = I*Matrix([
    [0, 1],
    [-1,0]
])

J = TensorProduct(Omega, eye(2)) + TensorProduct(eye(2), Hs)

eigensystem = J.eigenvects()

eigensystem_s = Hs.eigenvects()