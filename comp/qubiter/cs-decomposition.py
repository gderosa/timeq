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

num_bits = 3
#init_unitary_mat = FouSEO_writer.fourier_trans_mat(1 << num_bits)

init_unitary_mat = np.array([
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0,-1, 0],
  [0, 1, 0, 0, 0, 0, 0, 0],
  [-1,0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0,-1, 0, 0, 0, 0, 0],
  [0, 0,-1, 0, 0, 0, 0, 0],
  [0, 0, 0,-1, 0, 0, 0, 0]
], dtype=np.complex_)

emb = CktEmbedder(num_bits, num_bits)
file_prefix = os.path.dirname(__file__) + '/io_folder/csd_test'
t = Tree(True, file_prefix, emb, init_unitary_mat, verbose=False)
t.close_files()

"""
The above code automatically creates an expansion of $U$ into DIAG and MP_Y lines. Next we print the Picture file that was created.
"""

file = file_prefix + '_3_ZLpic.txt'
df = pd.read_csv(file, delim_whitespace=True, header=None)
print(df)

file = file_prefix + '_3_eng.txt'
df = pd.read_csv(file, delim_whitespace=True, header=None)
print(df)

#file_prefix = "io_folder/d_unitary_exact_check"
num_bits = 4
num_angles = (1 << num_bits)
emb = CktEmbedder(num_bits, num_bits)
rad_angles = list(np.random.rand(num_angles)*2*np.pi)
wr = DiagUnitarySEO_writer(file_prefix, emb, 'exact', rad_angles)
wr.write()
wr.close_files()
file = file_prefix + '_4_ZLpic.txt'
with open(file) as f:
    print(f.read())

"""

  |   |   |   Ph  
  |   |   |   Rz  
  |   |   @---X   
  |   |   |   Rz  
  |   |   @---X   
  |   |   Rz  |   
  |   @---X   |   

Etc. etc.

We can check that our exact expansion is correct as follows.
We can multiply the gates of the expansion using the class SEO_MatrixProduct.
Call the gate product matpro.prod_arr. Using the angles rad_angles that we stored,
we can construct the exact diagonal unitary, call it exact_mat.
Call err the norm of matpro.prod_arr - exact_mat, and print err.

"""

matpro = SEO_MatrixProduct(file_prefix, num_bits)
exact_mat = DiagUnitarySEO_writer.du_mat(rad_angles)
err = np.linalg.norm(matpro.prod_arr - exact_mat)
print("diag unitary error=", err)

"""

diag unitary error= 5.48660383707e-12

Next, we create English and Picture files containing an expansion of the 4 qubit gate
  Ry--%---%---%
This represents a multiplexor matrix. The angles are chosen at random and stored in the variable rad_angles. We then print the Picture file.

"""

#file_prefix = "io_folder/plexor_exact_check"
num_bits = 4
num_angles = (1 << (num_bits-1))
emb = CktEmbedder(num_bits, num_bits)
rad_angles = list(np.random.rand(num_angles)*2*np.pi)
wr = MultiplexorSEO_writer(file_prefix, emb, 'exact', rad_angles)
wr.write()
wr.close_files()
file = file_prefix + '_4_ZLpic.txt'
with open(file) as f:
    print(f.read())

"""
  Ry  |   |   |   
  X---+---+---@   
  Ry  |   |   |   
  X---+---@   |   
  Ry  |   |   |   
  X---+---+---@   
  Ry  |   |   |   
  X---@   |   |   
  Ry  |   |   |   
  X---+---+---@   
  Ry  |   |   |   
  X---+---@   |   
  Ry  |   |   |   
  X---+---+---@   
  Ry  |   |   |   
  X---@   |   |   

We can check that our exact expansion is correct as follows.
We can multiply the gates of the expansion using the class SEO_MatrixProduct.
Call the gate product matpro.prod_arr.
Using the angles rad_angles that we stored, we can construct the exact multiplexor matrix,
call it exact_mat. Call err the norm of matpro.prod_arr - exact_mat, and print err.
"""



matpro = SEO_MatrixProduct(file_prefix, num_bits)
exact_mat = MultiplexorSEO_writer.mp_mat(rad_angles)
err = np.linalg.norm(matpro.prod_arr - exact_mat)
print("multiplexor error=", err)

"""
multiplexor error= 4.20573242068e-12
"""

"""
A moral of the above calculations is that using CSD quantum compiling blindly
will give a SEO for a quantum Fourier Transform QFT that is exponential
in the number of qubits $n$.
And yet we know that Coppersmith came up with an expansion for the QFT that
is polynomial in $n$. But there is hope: CSD is not a unique decomposition.
Ref.3 explains how one can coax a CSD compiler to yield Coppersmith's decompostion.
"""

"""
Refs.:

    1. R.R. Tucci, A Rudimentary Quantum Compiler(2cnd Ed.) https://arxiv.org/abs/quant-ph/9902062

    2. Qubiter 1.11, a C++ program whose first version was released together with Ref.1 above. Qubiter 1.11 is included in the quantum_CSD_compiler/LEGACY folder of this newer, pythonic version of Qubiter

    3. R.R. Tucci, Quantum Fast Fourier Transform Viewed as a Special Case of Recursive Application of Cosine-Sine Decomposition, https://arxiv.org/abs/quant-ph/0411097
"""