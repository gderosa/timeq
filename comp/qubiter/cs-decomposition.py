# You should invoke this from the same directory as qubiter!

# Based on https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/quantum_CSD_compiler_intro.ipynb

import os
import sys

sys.path.insert(0,os.getcwd())

import numpy as np
import cuncsd_sq as csd
import math
from quantum_CSD_compiler.Tree import *
from quantum_CSD_compiler.DiagUnitaryExpander import *
from quantum_CSD_compiler.MultiplexorExpander import *

import pandas as pd

# https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/gate-expansions.ipynb
from CGateExpander import *

num_bits = 3
num_bits = 2
file_prefix = os.path.dirname(__file__) + '/io_folder/csd'


def expand_with_identity(ml, n):
    # on the right
    for i in range(len(ml)):
        ml[i].extend([0]*(n-len(ml)))
    for i in range(len(ml),n):
        row = [0]*n
        row[i] = 1
        ml.append(row)
        

init_unitary_mat_l = [
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0,-1, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [-1,0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0,-1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0,-1, 0, 0, 0]
]

init_unitary_mat_l = [
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [1,0,0,0]
]

#expand_with_identity(init_unitary_mat_l, 2**num_bits)

init_unitary_mat = np.array(init_unitary_mat_l, dtype=np.complex_)

emb = CktEmbedder(num_bits, num_bits)
t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
t.close_files()

"""
The above code automatically creates an expansion of $U$ into DIAG and MP_Y lines.
"""

# Gate expansion


DiagUnitaryExpander(file_prefix, num_bits, 'exact')


MultiplexorExpander(file_prefix + '_X1', num_bits, 'exact')

# https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/gate-expansions.ipynb
# Seems redundant
CGateExpander(file_prefix + '_X2', num_bits)


#  IBM

from for_IBM_devices.Qubiter_to_IBMqasm2 import *
from ForbiddenCNotExpander import *

import for_IBM_devices.ibm_chip_couplings as ibm

#num_bits =5
c_to_t = None

#expand_with_identity(init_unitary_mat_l, 2**num_bits)

init_unitary_mat = np.array(init_unitary_mat_l, dtype=np.complex_)

emb = CktEmbedder(num_bits, num_bits)
t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
t.close_files()

DiagUnitaryExpander(file_prefix, num_bits, 'exact')
MultiplexorExpander(file_prefix + '_X1', num_bits, 'exact')
#CGateExpander(file_prefix + '_X2', num_bits)

# IBM-specific
#ForbiddenCNotExpander(file_prefix + '_X3', num_bits, c_to_t)
q2i = Qubiter_to_IBMqasm2(file_prefix + '_X2', num_bits, c_to_t, write_qubiter_files=True)
