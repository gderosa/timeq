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
from sympy.physics.matrices import mdft
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
F = mdft(2)

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

# %% [markdown]
# Check: this is what we would obtain with matric of cols egeinv

# %%
R = (1/sqrt(2)) * Matrix([
    [I, -I],
    [1, 1]
])

# %%
R.adjoint()*T*R

# %%
Omega_T_d = (pi/((pi/(2*omega))**2))*F*T_d*F.adjoint()

# %%
Omega_T_d

# %%
Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])

# %%
J = TensorProduct(hbar*Omega_T_d, eye(2)) + TensorProduct(eye(2), Hs)

# %%
J

# %%
J.eigenvects()

# %% [markdown]
# ## Ordinary quantum theory

# %%
t = Symbol('t')
t0 = Symbol('t_0')

# %%
exp(-I*Hs*(t-t0)/hbar)

# %%
exp(-I*Hs*(t-t0)/hbar) * Matrix([0, -I])

# %%
(exp(-I*Hs*(t-t0)/hbar) * Matrix([0, -I])).subs({t: pi/(4*omega), t0: -pi/(4*omega)})

# %% [markdown]
# There is consistency in predicting the probability (square modulus), but not probability amplitute: at $t=\frac{\pi}{4\omega}$ P-W finds $(1, 0)$ instead of $(-i, 0)$. But the Rabi oscillation in terms of probability from 100% $\left|V\right>$ at $t=t_0=-\frac{\pi}{4\omega}$, to 100% $\left|H\right>$ at $t=\frac{\pi}{4\omega}$ is correctly predicted.
