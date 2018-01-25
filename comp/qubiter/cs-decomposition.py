# You should invoke this from the same directory as qubiter!

# Based on https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/quantum_CSD_compiler_intro.ipynb

import os
import sys

sys.path.insert(0,os.getcwd())

import numpy as np
import cuncsd_sq as csd
import math
from FouSEO_writer import *
from quantum_CSD_compiler.Tree import *
from quantum_CSD_compiler.DiagUnitarySEO_writer import *
from quantum_CSD_compiler.MultiplexorSEO_writer import *
import pandas as pd

# https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/gate-expansions.ipynb
from CGateExpander import *
from SEO_writer import *

num_bits = 3 # 5
#init_unitary_mat = FouSEO_writer.fourier_trans_mat(1 << num_bits)

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

#expand_with_identity(init_unitary_mat_l, 2**num_bits)

init_unitary_mat = np.array(init_unitary_mat_l, dtype=np.complex_)

print(init_unitary_mat)

print(init_unitary_mat.shape)

emb = CktEmbedder(num_bits, num_bits)
file_prefix = os.path.dirname(__file__) + '/io_folder/csd_test'
t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
t.close_files()

"""
The above code automatically creates an expansion of $U$ into DIAG and MP_Y lines.
"""



# Gate expansion

# some expansions may be redundant...


'''
# DiagUnitary expander
num_angles = (1 << num_bits)
emb = CktEmbedder(num_bits, num_bits)
rad_angles = list(np.random.rand(num_angles)*2*np.pi)
wr = DiagUnitarySEO_writer(file_prefix, emb, 'exact', rad_angles)
wr.write()
wr.close_files()
# Check
matpro = SEO_MatrixProduct(file_prefix, num_bits)
exact_mat = DiagUnitarySEO_writer.du_mat(rad_angles)
err = np.linalg.norm(matpro.prod_arr - exact_mat)
print("diag unitary error=", err)
'''

exit()

# Multiplexor expander
file_prefix = file_prefix+'__blah_'

num_angles = (1 << num_bits-1)
emb = CktEmbedder(num_bits, num_bits)
rad_angles = list(np.random.rand(num_angles)*2*np.pi)
rad_angles = [np.pi/4] * num_angles
rad_angles = [np.pi/6, np.pi/3, np.pi/2, (2/3)*np.pi]
wr = MultiplexorSEO_writer(file_prefix, emb, 'exact', rad_angles)
wr.write()
wr.close_files()
# Check
matpro = SEO_MatrixProduct(file_prefix, num_bits)
exact_mat = MultiplexorSEO_writer.mp_mat(rad_angles)
err = np.linalg.norm(matpro.prod_arr - exact_mat)
print("multiplexor error=", err)

# https://github.com/artiste-qb-net/qubiter/blob/master/jupyter-notebooks/gate-expansions.ipynb
#CGateExpander(file_prefix, num_bits)


#  IBM
'''
from for_IBM_devices.Qubiter_to_IBMqasm2 import *
from ForbiddenCNotExpander import *

import for_IBM_devices.ibm_chip_couplings as ibm

c_to_t = ibm.ibmqx4_edges
ForbiddenCNotExpander(file_prefix, num_bits, c_to_t)
q2i = Qubiter_to_IBMqasm2(file_prefix + '_X1', num_bits, c_to_t, write_qubiter_files=True)
'''