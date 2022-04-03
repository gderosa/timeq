# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from scipy.linalg import dft, norm, expm, det, inv


# %%
N = 4

# %%
T = np.diag(np.arange(N))

# %%
F = dft(N, scale='sqrtn').conj()

# %%
F_dagger = F.conj().T

# %%
Omega = F @ T @ F_dagger * 2*np.pi * N / N**2

# %%
comm = (T @ Omega) - (Omega @ T)

# %%
comm

# %%
np.round(comm, 2)

# %% [markdown]
# The commutator is *not* a constant in finite-domensional Hilbert spaces, where &ldquo;canonically conjugate&rdquo; operators are obtaineded via DFT instead of differentiation.

# %%
