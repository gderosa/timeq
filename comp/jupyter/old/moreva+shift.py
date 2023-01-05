# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # Analysys of the Moreva et al. experiment
#

# %% [markdown]
#
#
# ## Preliminaries

# %%
# Symbolic computation
from sympy import *
#from sympy.physics.matrices import mdft
from sympy.matrices.expressions.fourier import DFT
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.constants import hbar

# %%
# Remeber this to have LaTeX rendered output in Jupyter
init_printing()

# %% [markdown]
# ## Computation

# %%
Omega = Symbol(r'\Omega')
omega = Symbol(r'\omega', real=True)

# %%
F = DFT(2).as_mutable()

# %%
Omega = I*omega*Matrix([
    [0, 1],
    [-1,0]
])

# %%
Omega.eigenvects()

# %%
T = (pi / (2*omega)**2) * F.adjoint()*Omega*F

# %%
T

# %%
T.eigenvects()

# %%
T_d = diag(-pi/(4*omega), pi/(4*omega))

# %%
T_d

# %%
tshift = -T_d[0, 0]

# %%
T_prime_d =  tshift*eye(2) + T_d

# %%
T_prime_d

# %%
Omega_prime_T_d = (pi/((pi/(2*omega))**2))*F*T_prime_d*F.adjoint()

# %%
Omega_prime_T_d

# %%
Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])

# %%
J = TensorProduct(hbar*Omega_prime_T_d, eye(2)) + TensorProduct(eye(2), Hs)

# %%
J

# %%
J.eigenvects()

# %% [markdown]
# ## Ordinary quantum theory

# %%
t = Symbol('t')
t0 = Symbol('t_0')
psi0 = Matrix([-I, 1])

# %%
exp(-I*Hs*(t-t0)/hbar)

# %%
psi = exp(-I*Hs*(t-t0)/hbar) * psi0 * exp(-I*omega*(t-t0))  # correction term due to non-zero eigevalue of J

# %%
psi

# %%
psi.subs({t: pi/(4*omega), t0: -pi/(4*omega)})

# %% [markdown]
# With an initial time of 0, and the correction term due to non-zero eigenvalue, there is perfect accordance betweem P-W and QM.

# %%
