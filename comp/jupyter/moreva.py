from sympy import *

init_printing()

HT_hv = Matrix([
    [ 0, I],
    [-I, 0]
]) 

HS_hv = Matrix([
    [ 0, I],
    [-I, 0]
])

eigensys = HT_hv.eigenvects()


