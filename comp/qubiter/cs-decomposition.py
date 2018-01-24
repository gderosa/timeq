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

num_bits = 3
#init_unitary_mat = FouSEO_writer.fourier_trans_mat(1 << num_bits)

init_unitary_mat = np.array([
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0,-1, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [-1,0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0,-1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0,-1, 0, 0, 0]
], dtype=np.complex_)


init_unitary_mat = np.array([
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.complex_)


emb = CktEmbedder(num_bits, num_bits)
file_prefix = os.path.dirname(__file__) + '/io_folder/csd_test'
t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
t.close_files()

"""
The above code automatically creates an expansion of $U$ into DIAG and MP_Y lines.
"""



# Gate expansion

# commenting out what seems unnecessary for now...

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

# Multiplexor expander
num_angles = (1 << num_bits-1)
emb = CktEmbedder(num_bits, num_bits)
rad_angles = list(np.random.rand(num_angles)*2*np.pi)
wr = MultiplexorSEO_writer(file_prefix, emb, 'exact', rad_angles)
wr.write()
wr.close_files()
# Check
matpro = SEO_MatrixProduct(file_prefix, num_bits)
exact_mat = MultiplexorSEO_writer.mp_mat(rad_angles)
err = np.linalg.norm(matpro.prod_arr - exact_mat)
print("multiplexor error=", err)


#CGateExpander(file_prefix, num_bits)