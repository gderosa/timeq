# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Detector model: Kiukas / Ruschhaupt / Schmidt / Werner

# %%
from sympy import *
#from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct
from sympy.functions.special.delta_functions import Heaviside
from sympy.physics.quantum.dagger import Dagger

from sympy.stats import ContinuousRV, variance, std

from sympy.plotting import plot, plot3d_parametric_line

import numpy as np

import scipy.integrate

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.rcParams['text.usetex'] = False

# https://matplotlib.org/gallery/mplot3d/lines3d.html?highlight=parametric
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# %%
init_printing ()

# %%
gamma = Symbol('gamma', real=True, positive=True)
t = Symbol('t', real=True)
tprime = Symbol('t\'', real=True)
omega = Symbol('omega', real=True)
nu = Symbol('nu', real=True)

# %%
GAMMA = Rational(1, 1000)
GAMMA_n = float(GAMMA)


# %%
def D(_gamma):
    return Rational(1, 2) * Matrix([
        [0, 0],
        [0, _gamma]
    ])


# %%
H = Matrix ([
[0, 1] ,
[1, 0]
])

# %%
H.eigenvects()


# %% [markdown]
# It's manually seen that $\langle H \rangle = 0$ and $\langle H^2 \rangle = 1$, therefore $\sigma_{H} = 1$.

# %%
def K(_gamma):
    return H - I*D(_gamma)


# %%
def B(_gamma):
    return lambda t: exp(-I*K(_gamma)*t)


# %%
def U():
    return lambda t: exp(-I*H*t)


# %%
def non_unitary_psi(_t, _gamma=GAMMA):
    return B(_gamma)(_t) * Matrix([1,0])


# %%
def unitary_psi(_t):
    return U()(_t) * Matrix([1,0])


# %%
non_unitary_psi(t, gamma)

# %%
plot(re(non_unitary_psi(t)[0]), (t, 0, 100), line_color='r')

# %%
plot(im(non_unitary_psi(t)[1]), (t, 0, 100), line_color='b')


# %%
def lossy_norm(_t, _gamma=GAMMA):
    psi = B(_gamma)(_t) * Matrix([1,0])
    return abs(psi[0])**2 + abs(psi[1])**2


# %%
lossy_norm(t, gamma)

# %%
_non_unitary_psi_n = lambdify(t, non_unitary_psi(t), "numpy")


# %%
def non_unitary_psi_n(_t):
    return _non_unitary_psi_n(_t).T[0]


# %%
N(non_unitary_psi(1.0))

# %%
non_unitary_psi_n(1.0)

# %%
_lossy_norm_n = lambdify(t, lossy_norm(t), "numpy")
def lossy_norm_n(_t):
    # prevent a warning, even if we know it's real
    return np.real(_lossy_norm_n(_t))


# %%
N(lossy_norm(40))

# %%
lossy_norm_n(40)

# %%
plot(lossy_norm(t),(t, 0, 2*pi), line_color='g')


# %%
def prob_0_unitary(t):
    return abs(unitary_psi(t)[0]**2)


# %%
def prob_1_unitary(t):
    return abs(unitary_psi(t)[1]**2)


# %%
X = np.linspace(1e-6, 2*np.pi, 1000)  # avoid singularity in t=0

# %%
Y = lossy_norm_n(X)

# %%
plt.plot(X, -np.gradient(Y, X), 'm')


# %%
# we have set gamma = 2*sqrt(2)
def hatpsi(_t, _gamma=GAMMA):
    return \
        Heaviside(_t) * \
        Matrix([
            [0, 0],
            [0, sqrt(_gamma)]
        ]) * \
        non_unitary_psi(_t, _gamma)
        
def hatpsi_n(_t):
    return \
        np.heaviside(_t, 0) * \
        np.array([
            [0, 0],
            [0, np.sqrt(GAMMA_n)]
        ]) @ \
        non_unitary_psi_n(_t)
        
        
    

# %%
def hatpsisquarednorm(_t, _gamma=GAMMA):
    return simplify(
        abs(hatpsi(_t, _gamma)[0])**sympify(2) + abs(hatpsi(_t, _gamma)[1])**sympify(2)
    )

def hatpsisquarednorm_n(_t):
    return abs(hatpsi_n(_t)[0]**2) + abs(hatpsi_n(_t)[1]**2)


# %%
non_unitary_psi_n(1.234)

# %%
hatpsisquarednorm(t, _gamma=gamma)


# %%
def prob_0_hatpsi(_t):
    return abs(hatpsi(_t)[0]**2) / (abs(hatpsi(_t)[0]**2) + abs(hatpsi(_t)[1]**2))


# %%
def prob_1_hatpsi(_t):
    return abs(hatpsi(_t)[1]**2) / (abs(hatpsi(_t)[0]**2) + abs(hatpsi(_t)[1]**2))


# %%
plot( abs(hatpsi(t)[1]**2), (t, -2, 2*pi), line_color='b')

# %%
hatpsisquarednorm(1.0)

# %%
hatpsisquarednorm_n(1.0)

# %%
# Need to integrate numerically
bayesian_denominator_nonpw = scipy.integrate.quad(hatpsisquarednorm_n, 0, 2*np.pi)[0]

# %%
plot( abs(hatpsi(t)[1]**2)/bayesian_denominator_nonpw, (t, -2, 2*pi), line_color='y')

# %%


#### TODO: Fourier transform...


# TODO: switch to numeric and use FFT 
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.fft.html#real-and-hermitian-transforms
# The below takes ages to complete


#def fhatpsi1(_nu):
#    return fourier_transform(hatpsi(t)[1], t, _nu)

# %%
#plot(abs(fhatpsi1(nu))**2, (nu, -1, 1), line_color='#bbbbbb')

# %% [markdown]
# > The above Fourier transform is defined in frequency (\nu) not angular frequency (\omega),
# therefore needs rescaling.

# %%
#def fhatpsiomega(_omega):
#    return fhatpsi1(_omega/(2*pi)) / sqrt((2*pi))

# %%
#plot(abs(fhatpsiomega(omega))**2, (omega, -2*pi, 2*pi), line_color='magenta')

# %%
# graphical comparison with a normalized gaussian
#sigma = 1.0
#plot((1/(sqrt(2*pi)*sigma)) * exp(-omega**2/(2*(sigma)**2)), (omega, -2*pi, 2*pi), line_color='magenta')

# %% [markdown]
# ## (Discrete) Page-Wootters model

# %%
from scipy.linalg import dft, norm, expm
from scipy import stats

# %%
T = np.diag(np.arange(0,32)) * np.pi / 16

# %%
F = dft(32, scale='sqrtn')

# %%
F_dagger = F.conj().T

# %%
Omega = F_dagger @ T @ F * 16 / np.pi

# %%
oeigenvalues, oeigenvectors = np.linalg.eig(Omega)

# %%
np.round(oeigenvalues)

# %%
H = np.array([
    [0, 1],
    [1, 0]
])

# %%
J = np.kron(Omega, np.eye(2)) + np.kron(np.eye(32), H)

# %%
eigenvalues, eigenvectors = np.linalg.eig(J)

# %%
EnergyCorrectionMatrices = np.zeros((64, 64, 64), complex)
for n in range(64):
    EnergyCorrectionMatrices[n] = np.kron(
        expm(-1j*eigenvalues[n]*T),
        np.eye(2)
    )
# TODO: DRY
EnergyCorrectionMatricesT = np.zeros((64, 32, 32), complex)
for n in range(64):
    EnergyCorrectionMatricesT[n] = expm(-1j*eigenvalues[n]*T)


# %%
def history_vector(eigenindex):
    # Needs matrix transposition ".T" (different convention as opposed to Mathematica)
    eigenvector = eigenvectors.T[eigenindex]
    return EnergyCorrectionMatrices[eigenindex] @ eigenvector

# "unflatten" the history_vector v into a a sequence of qubit component pairs
def reshape(v):
    return np.reshape(v, (-1,2))

# also make the first component real
def normalize_initial(v):
    vout = np.zeros(64, complex)
    # A phase factor to make it real
    vout = v * np.exp(-1j * np.angle(v[0]))
    # And a factor to normalize the initial state
    vout = vout / sqrt(
        np.abs(vout[0]**2) + np.abs(vout[1]**2)
    )
    return vout


# %%
# Find the best linear combination to obtain |0> as initial state
def find_best():
    max_prob0 = 0
    max_prob0_i = 0
    max_prob0_j = 0
    for i in range(32):
        for j in range(32):
            qbi = reshape(history_vector(i))
            qbj = reshape(history_vector(j))
            qbit_hist = qbi + qbj
            prob0 = np.abs(qbit_hist[0][0]**2) / (
                np.abs(qbit_hist[0][0]**2) + np.abs(qbit_hist[0][1]**2)
            )
            if prob0 > max_prob0:
                max_prob0 = prob0
                max_prob0_i = i
                max_prob0_j = j
    print (max_prob0_i, max_prob0_j, max_prob0)
    return (max_prob0_i, max_prob0_j)
    


# %%
# start with |0> as close as possible
i, j = find_best()
qbhistvec = normalize_initial(history_vector(i) + history_vector(j))
qbhist = reshape(qbhistvec) 

# %%
qbhist = qbhist.astype(complex)

# %% [markdown]
# Consitently with "odinary QM" findings, the component along |0> stays purely real, and the component along |1> stays purely imaginary.

# %%
# Fill data for plotting
times = np.arange(0, 2*np.pi, np.pi/16)
norms = np.zeros(32)
probs0 = np.zeros(32)
probs1 = np.zeros(32)
# Components 0 are pure real, componets 1 are pure imag
real_parts0 = np.real(qbhist.T[0])
imag_parts0 = np.imag(qbhist.T[0])
real_parts1 = np.real(qbhist.T[1])
imag_parts1 = np.imag(qbhist.T[1])

for i in range(0, 32):
    norms[i] = (np.abs(qbhist[i][0]**2) + np.abs(qbhist[i][1]**2))
    probs0[i] = np.abs(qbhist[i][0]**2) / (
        np.abs(qbhist[i][0]**2) + np.abs(qbhist[i][1]**2) )
    probs1[i] = np.abs(qbhist[i][1]**2) / (
        np.abs(qbhist[i][0]**2) + np.abs(qbhist[i][1]**2) )

# %%
plt.plot(times, norms/norms[0], 'g^')

# %%
plt.plot(times, probs0, 'rs')

# %%
plt.plot(times, probs1, 'bs')

# %%
plt.plot(times, real_parts0, 'rs')

# %%
plt.scatter(times, imag_parts0, color='0.75', marker='s')

# %%
plt.scatter(times, real_parts1, color='0.75', marker='s')

# %%
plt.plot(times, imag_parts1, 'bs')

# %% [markdown]
# ## TOA prob as in Maccone/Sacha arXiv:1810.12869
# _Adapted from $\S$ "Time of arbitrary event"._

# %% [markdown]
# \begin{equation}
#     p(t|1) = \frac{\left|\psi(1|t)\right|^2}{\int_0^\mathcal{T} dt \left|\psi(1|t)\right|^2} = 
#     \frac{
#         \left| {}_{T}\langle n | \otimes {}_{S}\langle 1 | \Psi \rangle\rangle \right|^2
#     }{
#         %\frac{\mathcal{T}}{N}
#         \sum_{n'=0}^{N-1} \left| {}_{T}\langle n' | \otimes {}_{S}\langle 1 | \Psi \rangle\rangle \right|^2
#     }
# \end{equation}

# %% [markdown]
# Where:
#
# * $\psi(1|t) = \langle 1 | \psi(t) \rangle = {}_{T}\langle n | \otimes {}_{S}\langle 1 | \Psi \rangle\rangle$
# * $t = \frac{2\pi}{N} n$ and ${}_{T}\langle n |$ is the corresponding time eigenstate.
# * We fix period $\mathcal{T}=2\pi$
#     i.e. period of a Rabi oscillation for $H = \hbar\omega\begin{pmatrix}0&1\\1&0\end{pmatrix}$
#     and $\hbar = \omega = 1$.

# %%
qbhistvec =  qbhistvec.astype(complex)
qbhistvec_normalized = qbhistvec / np.linalg.norm(qbhistvec)


# %%
def t_eigenstate(n):
    v = np.zeros(32, dtype=complex)
    v[n] = 1
    return v


# %%
t_eigenstate(n=2)

# %%
qubit1 = np.array([0, 1])


# %%
def tn_ox_1(n):
    return np.kron(t_eigenstate(n), qubit1)


# %%
def joint_prob(n):
    return np.abs(tn_ox_1(n) @ qbhistvec_normalized)**2


# %%
X = np.arange(32)
iterable = (joint_prob(n) for n in X)
Y = np.fromiter(iterable, float)

# %%
# A "time bin" is large as 2*pi/N
X = X * (2*np.pi/32) # real time
Y = Y / (2*np.pi/32) # probability _density_

# %%
bayes_denominator = np.sum(Y * (2*np.pi/32))
Y = Y / bayes_denominator

# %%
plt.plot(X, Y, 'ys')

# %%
np.sum(Y) * 2*np.pi/32

# %%
