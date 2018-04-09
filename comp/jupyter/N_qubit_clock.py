from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum.dagger import Dagger

init_printing() 

N = 4

T = diag(*map(lambda k: 2*pi*k/N, range(N)))

F = mdft(N)

Omega = F * T * Dagger(F) * 2/pi

pprint(Omega)
