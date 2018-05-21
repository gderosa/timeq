
# coding: utf-8

# In[183]:


from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.constants import hbar


# In[184]:


# Remeber this to have LaTeX rendered output in Jupyter
init_printing()


# In[196]:


Delta = Symbol(r'\Delta', real=True)
Omega = Symbol(r'\Omega', real=True)
omega = Symbol(r'\omega', real=True)
psi = Symbol(r'\psi')


# In[186]:


Delta = Rational(1,2) # 1/N, N=dimension of H_T


# In[187]:


F = mdft(2)


# In[188]:


T = (pi/omega) * Matrix([
    [0, 0],
    [0, Delta]
])


# In[189]:


T


# In[190]:


Omega = (omega**2/(pi*Delta**2))*F*T*(F.adjoint())


# In[191]:


Omega


# In[192]:


Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])


# In[193]:


J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)


# In[194]:


J


# In[195]:


J.eigenvects()


# ## Comparison with ordinary QM 

# In[207]:


def evolve_psi(t, t0, psi0):
    return exp(-I*Hs*(t-t0)/hbar)*psi0


# In[210]:


def evolve_psi_correction(t, t0, eigenJ):
    return exp(eigenJ*I*(t-t0)/hbar)


# In[208]:


evolve_psi(t=0, t0=0, psi0=Matrix([I,0]))


# In[212]:


evolve_psi(pi/(2*omega), 0, Matrix([I,0])) * evolve_psi_correction(pi/(2*omega), 0, hbar*omega)


# With the phase correction due to the non-zero eigenvalue of J, there is agreement between PW and ordinary QM.
