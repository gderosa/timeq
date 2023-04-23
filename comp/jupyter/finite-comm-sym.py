# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from sympy import *
from sympy.matrices.expressions.fourier import DFT
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger

# %%
# Remeber this to have LaTeX rendered output in Jupyter
init_printing()

# %%
N      = Symbol(r'N', real=True)
deltaT = Symbol(r'\delta{T}', real=True)
DeltaT = Symbol(r'\Delta{T}', real=True)
T      = Symbol(r'T')
F      = Symbol(r'F')
Omega  = Symbol(r'\Omega')

# %%
N = Integer(4)

# %%
deltaT = Rational(1, 1)

# %%
DeltaT = N * deltaT

# %%
T = diag(*Range(N))

# %%
T

# %%
F = DFT(N)

# %%
F.as_explicit()

# %%
Dagger(F).as_explicit()

# %%
Omega = (Dagger(F) @ T @ F).as_explicit()

# %%
Omega

# %%
T @ Omega - Omega @ T

# %% [markdown]
# Zero-valued diagonal, as argued in Weyl, H. (1950), "The theory of groups and quantum mechanics".
#
# The commutator is not a constant.
