# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.constants import hbar

# %%
# Remeber this to have LaTeX rendered output in Jupyter
init_printing()

# %%
Delta = Symbol(r'\Delta', real=True)
delta = Symbol(r'\delta', real=True)
Omega = Symbol(r'\Omega')
omega = Symbol(r'\omega', real=True)

# %%
hbar = 1

# %%
Delta = Rational(1,2) # 1/N, N=dimension of H_T

# %%
delta = 0.0001

# %%
F = mdft(2)

# %%
T = (pi/omega) * Matrix([
    [delta, 0],
    [0, Delta+delta]
])

# %%
T

# %%
Omega = (omega**2/(pi*Delta**2))*F*T*(F.adjoint())

# %%
Omega

# %%
Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])

# %%
J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)

# %%
J

# %%
J.eigenvects()


# %% [markdown]
# ## Comparison with ordinary QM 

# %%
def evolve_psi(t, t0, psi0):
    return exp(-I*Hs*(t-t0)/hbar)*psi0


# %%
def evolve_psi_correction(t, t0, eigenJ):
    return exp(eigenJ*I*(t-t0)/hbar)


# %%
evolve_psi(t=0, t0=0, psi0=Matrix([I,0]))

# %%
evolve_psi(pi/(2*omega), 0, Matrix([I,0])) * evolve_psi_correction(pi/(2*omega), 0, hbar*omega)

# %% [markdown]
# With the phase correction due to the non-zero eigenvalue of J, there is agreement between PW and ordinary QM.
