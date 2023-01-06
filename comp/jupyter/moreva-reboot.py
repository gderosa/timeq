# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np

# Definitions after Eq. (2) in https://arxiv.org/abs/1310.4691

H_T = np.array([
    [ 0.0 , 1.0j],
    [-1.0j, 0.0 ]
])

H_S = np.array([
    [ 0.0 , 1.0j],
    [-1.0j, 0.0 ]
])

np.linalg.eig(H_T)


