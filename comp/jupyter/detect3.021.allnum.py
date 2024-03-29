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
# ## Detector model: 3-level system

# %% [markdown]
# Three-level "$\Lambda$" system, of interest for
# * detector models (decay into a metastable state),
# * STIRAP
# * EIT

# %%
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import expm, norm

# matplotlib.rcParams['text.usetex'] = False

# https://matplotlib.org/gallery/mplot3d/lines3d.html?highlight=parametric
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# %%
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['axes.labelpad']  = 14

# %%
#matplotlib.rcParams

# %%
from IPython.display import display, Latex #, Math

# %% language="javascript"
#     // do not generate scroll areas, expand figures instead
#     IPython.OutputArea.auto_scroll_threshold = 9999

# %%
H = np.array([
    [-2,     0,      32],
    [0,      2,       8],
    [32,     8,       3]
], np.complex_) / 96


# %%
def U(t):
    return expm(-1j*H*t)


# %%
psi_0 = np.array([1, 0, 0], np.complex_)


# %%
def unitary_psi(t):
    return U(t) @ psi_0


# %%
def prob(t):
    probabilities = [0, 0, 0]
    for i in 0, 1, 2:
        probabilities[i] = norm(unitary_psi(t)[i])**2
    return probabilities


# %%
TMIN, TMAX, TMAX_EXTENDED = 0, 40, 150
TMIN_N, TMAX_N = float(TMIN), float(TMAX)

# %%
NPLOTPOINTS = 3200

# %%
times = np.linspace(TMIN_N, TMAX_N, num=NPLOTPOINTS)
times_extended = np.linspace(TMIN_N, TMAX_EXTENDED, num=NPLOTPOINTS)

# %%
probs = [None, None, None]
for i in 0, 1, 2:
    probs[i] = np.fromiter((prob(t)[i] for t in times), float)
#     fig, ax = plt.subplots(figsize=(12, 6*np.max(probs[i])))
#     ax.plot(times, probs[i])
#     plt.show()

# %%
# Avoid *tiny* negative numbers, just out of numeric approximation, which will cause problems later,
# when their value is in fact juzt zero.
probs = np.round(probs, decimals=12)

# %%
UNISYM = {
    'psi': u'\u03C8',
    '^2' : u'\u00B2'
}
PROB_LABELS        = ['', '', '']
PROB_AMP_LABELS    = ['', '', '']
PROB_AMP_LABELS_PW = ['', '', '']

for i in 0, 1, 2:
    PROB_AMP_LABELS[i] = '<' + str(i) + '|' + UNISYM['psi'] + '(t)' + '>'
    PROB_LABELS[i]     = '|' + PROB_AMP_LABELS[i] + '|' + UNISYM['^2']

    PROB_AMP_LABELS_PW[i] = '<t|\u2297<%d|\u03A8>>' % i

# %% [markdown]
# https://matplotlib.org/gallery/lines_bars_and_markers/stackplot_demo.html#sphx-glr-gallery-lines-bars-and-markers-stackplot-demo-py

# %%
prob_stack = np.vstack(probs)

# %%
colors  =  ['#d44', '#4d4', '#44d']
colors  =  ['r', 'g', 'b']
hatches =  ['\\\\\\', '////////', '...']

linewidths =  [1, 1, 1]

labels     =  PROB_LABELS

fig, ax = plt.subplots(figsize=(12, 6))

stacks = ax.stackplot(times, (probs[0], probs[1], probs[2]))
ax.set_xlabel('t')

for stack, hatch, color, label, linewidth in zip(stacks, hatches, colors, labels, linewidths):
    stack.set_facecolor('w')
    stack.set_edgecolor(color)
    stack.set_label(label)
    stack.set_hatch(hatch)
    stack.set_linewidth(linewidth)

ax.plot(times, probs[0] + probs[1] + probs[2], c='m', label='||\u03C8(t)||\u00B2', linewidth=1.5)

desired_order   = [0, 3, 2, 1]
_handles, _labels = ax.get_legend_handles_labels()
handles         = [ _handles[i] for i in desired_order ]
labels          = [ _labels[i]  for i in desired_order ]
ax.legend(handles, labels, loc='center right', framealpha=1)

plt.savefig('_img/detect3.021/hermitian3color.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
rgbs = []
for i in range(NPLOTPOINTS):
    rgbs.append(
        (
            probs[0][i],
            probs[1][i],
            probs[2][i]
        )
    )

# %%
fig, ax = plt.subplots(figsize=(12,6))
ax.set_xlabel('t')
ax.scatter(times, np.zeros(NPLOTPOINTS)-0.1,
            c=rgbs, marker='|', s=400)

# "virtual", don't really want to show, only for legend
_c = ['r', '#0a0', 'b']
for i in 0, 1, 2:
    ax.plot(
        times, probs[i],
        c=_c[i],
        linewidth=2,
    )

ax.legend(
    PROB_LABELS,
    loc='upper right'
)

plt.savefig('_img/detect3.021/hermitian3lines.pdf', bbox_inches='tight', pad_inches=0)


# %%
unitary_psis = [np.zeros(NPLOTPOINTS)] * 3
for i in 0, 1, 2:
    unitary_psis[i] = np.fromiter( (unitary_psi(t)[i] for t in times), complex )

# %%
PROB_LABELS

# %%
# 3D parametric plot
for (vertical_angle,horizontal_angle,height,width,lloc,view) in (10,-70,12,12,'center left','side'), (80,-120,12,12,'right','top'):
    #fig = plt.figure(figsize=(width, height))
    #ax = fig.gca(projection='3d')
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='3d'))

    ax.view_init(vertical_angle, horizontal_angle) # rotate 3d point of view

    ax.set_xlabel('Re <0,1,2|\u03C8>')
    ax.set_ylabel('Im <0,1,2|\u03C8>')
    ax.set_zlabel('t')

    ax.scatter(
        np.zeros(NPLOTPOINTS, dtype=float),
        np.zeros(NPLOTPOINTS, dtype=float),
        times,
        c = rgbs,
        s = 24
    )
    for i in 0, 1, 2:
        ax.plot(
            np.real(unitary_psis[i]),
            np.imag(unitary_psis[i]),
            times,
            label = PROB_AMP_LABELS[i],
            #depthshade=False,
            c = _c[i]
        )
        ax.legend(loc=lloc)
    plt.savefig('_img/detect3.021/hermitianSpaceTime_%s.pdf' % view, bbox_inches=0, pad_inches=0)


# %% [markdown]
# ## Complex potential (detection by absorption)

# %%
H_n = H

# %%
GAMMA = 0.1


# %%
def D(_gamma=GAMMA):
    # no 1/2 factor, absorbed in the _gamma in the matrix here
    return np.array([
        [0, 0,      0],
        [0, 0,      0],
        [0, 0, _gamma]
    ], dtype=complex)


# %%
D()


# %%
def K(_gamma=GAMMA):
    return H_n - 1j*D(_gamma)


# %%
K()


# %%
def B(_t, _gamma=GAMMA):
    return expm(-1j*K(_gamma)*_t)


# %%
B(0)


# %%
def non_unitary_psi(_t, _gamma=GAMMA):
    return B(_t, _gamma) @ psi_0


# %%
evolution = np.zeros((3, NPLOTPOINTS), dtype=complex)
evolution_extended = np.zeros((3, NPLOTPOINTS), dtype=complex)

for i in 0, 1, 2:
    _iter = (non_unitary_psi(_t)[i] for _t in times)
    _iter_extended = (non_unitary_psi(_t)[i] for _t in times_extended)

    evolution[i] = np.fromiter(_iter, complex)
    evolution_extended[i] = np.fromiter(_iter_extended, complex)

_iter_norm = (norm(non_unitary_psi(_t)) for _t in times)
norms = np.fromiter(_iter_norm, float)

_iter_norm_extended = (norm(non_unitary_psi(_t)) for _t in times_extended)
norms_extended = np.fromiter(_iter_norm_extended, float)

# %%
colors  =  ['#f66', '#6d6', '#88f']
colors  =  ['r', 'g', 'b']
hatches =  ['\\\\\\', '////////', '...']
labels     =  PROB_LABELS

fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(times, np.ones(NPLOTPOINTS), c='grey', linestyle='dashed', label='Initial norm', linewidth=2.5)

ax.plot(times, norms**2, c='m', linewidth=1.5, label='||\u03C8(t)||\u00B2')

stacks = ax.stackplot(times, (
    np.abs(evolution[0])**2,
    np.abs(evolution[1])**2,
    np.abs(evolution[2])**2
))

ax.set_xlabel('t')

for stack, hatch, color, label in zip(stacks, hatches, colors, labels):
    stack.set_linewidth(.5)
    stack.set_facecolor('w')
    stack.set_edgecolor(color)
    stack.set_label(label)
    stack.set_hatch(hatch)

handles, labels = ax.get_legend_handles_labels()
desired_order = [0, 1, 4, 3, 2]
handles = [ handles[i] for i in desired_order ]
labels  = [  labels[i] for i in desired_order ]
ax.legend(handles, labels, loc='center')

plt.savefig('_img/detect3.021/loss3color.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
# loss of normalization, or integral of antiderivative...
bayesian_denominator_nonpw = 1 - norm(evolution.T[NPLOTPOINTS-1])**2

# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('t')
ax.set_ylabel('Detection probability density  =  - d||\u03C8(t)||\u00B2 / dt')
ax.plot(times, -np.gradient(norms**2, times), c='y', linewidth=3)
plt.savefig('_img/detect3.021/loss.pdf', bbox_inches='tight', pad_inches=0)

# %%
# Andreas: "CUT at 100"...
times_extended_cut = list((filter(lambda t: t<100, times_extended)))
NPLOTPOINTS_CUT = len(times_extended_cut)
norms_extdended_cut = norms_extended[:NPLOTPOINTS_CUT]
evolution_extended_cut = evolution_extended[:, :NPLOTPOINTS_CUT]
norms_extended_cut = norms_extended[:NPLOTPOINTS_CUT]

# %%
evolution_extended_cut[0].shape

# %%
labels = PROB_LABELS

colors  =  ['r', 'g', '#ccf']
hatches =  ['\\\\\\\\', '///////', '.....']

fig, ax = plt.subplots(figsize=(14, 7))

ax.set_xlabel('t')

ax.plot(times_extended_cut, np.ones(NPLOTPOINTS_CUT), c='grey', linestyle='dashed', label='Initial norm')

ax.plot(times_extended_cut, norms_extended_cut**2, c='m', linewidth=1, label='||\u03C8(t)||\u00B2')

stacks = ax.stackplot(times_extended_cut, (
    np.abs(evolution_extended_cut[0])**2,
    np.abs(evolution_extended_cut[1])**2,
    np.abs(evolution_extended_cut[2])**2
))

for stack, hatch, color, label in zip(stacks, hatches, colors, labels):
    stack.set_linewidth(1.25)
    stack.set_facecolor('w')
    stack.set_edgecolor(color)
    stack.set_label(label)
    stack.set_hatch(hatch)

handles, labels = ax.get_legend_handles_labels()
desired_order = [0, 1, 4, 3, 2]
handles = [ handles[i] for i in desired_order ]
labels  = [  labels[i] for i in desired_order ]
ax.legend(handles, labels, loc='center')

plt.savefig('_img/detect3.021/loss3color_ext_cut.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# %% [markdown]
# #### "Asymptotic" fidelity for |1>

# %%
times_extended[-1]

# %%
norms_extended[-1]**2

# %%
(np.abs(evolution_extended[1])**2)[-1] / norms_extended[-1]**2

# %% [markdown]
# ### Detection probability

# %%
_detection_prob = -np.gradient(norms_extended**2, times_extended)
_detection_prob_cut = _detection_prob[:NPLOTPOINTS_CUT]
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlabel('t')
ax.set_ylabel('Detection probability density  =  - d||\u03C8(t)||\u00B2 / dt')
ax.plot(times_extended_cut, _detection_prob_cut, c='y', linewidth=3)
plt.savefig('_img/detect3.021/loss_ext_cut.pdf', bbox_inches='tight', pad_inches=0)

# %% [markdown]
# #### "3D"

# %%
# 3D parametric plot

for (vertical_angle,horizontal_angle,height,width,lloc,view) in (15,-80,15,15,'center left','side'), (90, -80, 15, 15,'center left','top'):
    # fig = plt.figure(figsize=(width, height))
    # ax = fig.gca(projection='3d')
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='3d'))


    ax.view_init(vertical_angle, horizontal_angle) # rotate 3d point of view

    ax.set_xlabel('Re <0,1,2|\u03C8>')
    ax.set_ylabel('Im <0,1,2|\u03C8>')
    ax.set_zlabel('t')

    ax.scatter(
        np.zeros(NPLOTPOINTS, dtype=float),
        np.zeros(NPLOTPOINTS, dtype=float),
        times,

        c = np.array([
            np.abs(evolution[0])**2,
            np.abs(evolution[1])**2,
            np.abs(evolution[2])**2
        ]).T,
        s = 24,
        marker='o'
    )
    _c = ['r', 'g', 'b']

    for i in 0, 1, 2:
        ax.plot(
            np.real(evolution[i]),
            np.imag(evolution[i]),
            times,
            #depthshade=False,
            c = _c[i],
            linewidth=2,
            label = PROB_AMP_LABELS[i]
        )
        ax.legend(loc=lloc)
    plt.savefig('_img/detect3.021/NonHermitianSpaceTime_%s.pdf' % view, bbox_inches='tight', pad_inches=0)

# %% [markdown]
# ## Page-Wootters

# %%
from scipy.linalg import dft, norm, expm, det, inv

# %%
# Dimension of the system, or the spatial/"ordinary" Hilbert space
NS = 3
# Number of levels of the clock aka dimension of Time Hilbert space
NT = 60
# "Period"
DT = TMAX_N  # assume we start with time 0

T = DT * np.diag(np.arange(NT)) / NT

# %%

# %%
F = dft(NT, scale='sqrtn')
F_dagger = F.conj().T

# %%
Omega = F_dagger @ T @ F * 2*np.pi * NT / DT**2

# %%
J = np.kron(Omega, np.eye(3)) + np.kron(np.eye(NT), K())

# %%
eigenvalues, eigenvectors = np.linalg.eig(J)

# %%
eigenvectors = eigenvectors.T

# %%
eigenvectors_normalized_in_S = np.empty((NT*NS, NT*NS), dtype=complex)

for i in range(NT*NS):
    eigenvectors_normalized_in_S[i] = eigenvectors[i] / norm(eigenvectors[i][:3])


# %%
histories = np.empty((NT*NS, NT*NS), dtype=complex)

for i in range(NT*NS):
    histories[i] = \
        expm(np.kron( -1j*T*eigenvalues[i], np.eye(NS) )) @ \
        eigenvectors_normalized_in_S[i]


# %%
# Only implemented for NS=3
def find_linear_independent_initial(eigenvectors=eigenvectors_normalized_in_S):
    best_i, best_j, best_k = -1, -1, -1
    best_det = 0
    best_states = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    for i in range(NT*NS):
        for j in range(i, NT*NS):
            for k in range(j, NT*NS):
                # this normalization is not necessary if default
                # eigenvectors=eigenvectors_normalized_in_S
                # is given
                si = eigenvectors[i][:3]
                si = si / norm(si)
                sj = eigenvectors[j][:3]
                sj = sj / norm(sj)
                sk = eigenvectors[k][:3]
                sk = sk / norm(sk)
                states = np.array([
                    si,
                    sj,
                    sk
                ])
                _det = det(states)
                if abs(abs(_det)-1.0) < abs(abs(best_det) - 1.0):
                    best_det = _det
                    best_i, best_j, best_k = i, j, k
                    best_states = states
                if abs(abs(_det)-1.0) == 0:
                    return best_i, best_j, best_k, best_det
        percent = int(100 * (i + 1) / (NT*NS))
        print(
            str(percent) + '% scanned' + "\tabs(best_det) = " + str(abs(best_det)),
            end="\r", flush=True)

    return best_i, best_j, best_k, best_det



# %%
best_i, best_j, best_k, best_det = find_linear_independent_initial()

# %%
states = np.array([
    histories[best_i][:NS],
    histories[best_j][:NS],
    histories[best_k][:NS]
])
#assert(np.round(abs(det(states)), decimals=2) == 1.0)
# Find what linear combination would bring to the desired initial state psi_0_n
coeffs = inv(states.T) @ psi_0

# %%
history = coeffs.dot(np.array([
    histories[best_i],
    histories[best_j],
    histories[best_k]
]))

# %%
# 3D parametric plot

times_discrete = np.diag(T)

psi = history.reshape((-1,NS)).T

for (vertical_angle,horizontal_angle,height,width,lloc,view) in ((10,-120,15,25,'center right','side'), (80,-100,15,25,'center right','top')):
    #fig = plt.figure(figsize=(width, height))
    #ax = fig.gca(projection='3d')

    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='3d'))

    ax.view_init(vertical_angle, horizontal_angle) # rotate 3d point of view

    ax.set_xlabel('Re <t|\u2297<0,1,2|\u03A8>>')
    ax.set_ylabel('Im <t|\u2297<0,1,2|\u03A8>>')
    ax.set_zlabel('t')

    ax.scatter(
        np.zeros(NT, dtype=float),
        np.zeros(NT, dtype=float),
        times_discrete,

        c = np.round((abs(psi.T)**2), 8), # rounding, to avoid number instability causing out-of-range rgb vals
        s = 75,
        marker='o'
    )
    _c = ['r', 'g', 'b']

    for i in range(NS):
        ax.scatter(
            np.real(
                psi[i]
            ),
            np.imag(
                psi[i]
            ),
            times_discrete,

            marker = 's',
            #depthshade=False,
            #s = abs(_psi[i]**2)*60,
            s = 60,
            edgecolor = _c[i],
            facecolor = (0, 0, 0, 0),
            linewidth = 1,

            label = PROB_AMP_LABELS_PW[i]
        )
    ax.legend(loc=lloc)
    plt.savefig('_img/detect3.021/PWSpaceTime_%s.pdf' % view, bbox_inches='tight', pad_inches=0)


# %%
norm(psi)**2

# %%
norm(psi)

# %% [markdown]
# ### Overlapping  PW and QM continuous

# %%
# 3D parametric plot

times_discrete = np.diag(T)

psi = history.reshape((-1,NS)).T

for (vertical_angle,horizontal_angle,height,width,view) in (15,-80,15,15,'side'), (90,-80,15,15,'top'):
    #fig = plt.figure(figsize=(width, height))
    #ax = fig.gca(projection='3d')
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw=dict(projection='3d'))


    ax.view_init(vertical_angle, horizontal_angle) # rotate 3d point of view

    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    ax.set_zlabel('t')

    ax.scatter(
        np.zeros(NT, dtype=float),
        np.zeros(NT, dtype=float),
        times_discrete,

        c = np.round((abs(psi.T)**2), 8), # rounding, to prevent numeric instability from causing out-of-range RGB values
        s = 75,
        marker='o'
    )
    _c = ['r', 'g', 'b']
    for i in range(NS):
        ax.scatter(
            np.real(
                psi[i]
            ),
            np.imag(
                psi[i]
            ),
            times_discrete,

            marker = 's',
            #depthshade=False,
            #s = abs(_psi[i]**2)*60,
            s = 80,
            edgecolor = _c[i],
            facecolor = (0, 0, 0, 0),
            linewidth = 0.8,
            label = PROB_AMP_LABELS_PW[i]
        )

    # QM continuous
    for i in 0, 1, 2:
        ax.plot(
            np.real(evolution[i]),
            np.imag(evolution[i]),
            times,
            #depthshade=False,
            c = _c[i],
            linewidth=1.5,
            label = PROB_AMP_LABELS[i]
        )
    ax.legend(loc='center left')
    plt.savefig('_img/detect3.021/PWSpaceTimeFit_%s.pdf' % view, bbox_inches='tight', pad_inches=0)


# %% [markdown]
# ## TOA prob as in Maccone/Sacha arXiv:1810.12869
# _Adapted from $\S$ "Time of arbitrary event"._
#
# See also [`detect-gentle.ipynb`](detect-gentle.ipynb). Or [`detect-gentle.py`](detect-gentle.py) if you use Jupytext.
#

# %%
def t_eigenstate(n):
    v = np.zeros(NT, dtype=complex)
    v[n] = 1
    return v


# %%
arrived_state = np.array([0, 0, 1])


# %%
def tn_ox_arrived(n):
    return np.kron(t_eigenstate(n), arrived_state)


# %%
history_normalized = history / norm(history)  ## normalized in H_T \otimes H_S


# %%
def joint_prob(n):
    return np.abs(tn_ox_arrived(n) @ history_normalized)**2


# %%
X = np.arange(NT)

iterable = (joint_prob(n) for n in X)
Y = np.fromiter(iterable, float)


# %%
dT  = DT / (NT)

# %%
# A "time bin"
X = X * dT # real time
Y = Y / dT # probability _density_

# %%
bayes_denominator = np.sum(Y * dT)
Y = Y / bayes_denominator

# %%
# fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlabel('(t_n) t')
ax.set_ylabel('Detection probability density')
ax.plot(X, Y, 'bs', label='Discrete Page-Wootters')
ax.plot(times, -np.gradient(norms**2, times) / bayesian_denominator_nonpw,
        c='y', linewidth=2, label='Detector model (normalization loss)')
ax.legend()
plt.savefig('_img/detect3.021/conditionalProbFit.pdf', bbox_inches='tight', pad_inches=0)
plt.show()

# %%
nonpw_probs = -np.gradient(norms**2, times) / bayesian_denominator_nonpw

# %%
np.sum(nonpw_probs*times[1])

# %%
np.sum(Y*dT)

# %%
