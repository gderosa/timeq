
# coding: utf-8

# In[186]:


from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.constants import hbar


# In[187]:


# Remeber this to have LaTeX rendered output in Jupyter
init_printing()


# In[188]:


Delta = Symbol(r'\Delta', real=True)
Omega = Symbol(r'\Omega', real=True)
omega = Symbol(r'\omega', real=True)


# In[189]:


Delta = Rational(1, 2)  # 1/N, N=dimension of H_T


# In[190]:


F = mdft(2)


# In[191]:


T = (pi/omega) * Matrix([
    [0, 0],
    [0, Delta]
])


# In[192]:


T


# In[193]:


Omega = (omega**2/(pi*Delta**2))*F*T*(F.adjoint())


# In[194]:


Omega


# In[195]:


Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])


# In[196]:


J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)


# In[197]:


J


# In[198]:


J.eigenvects()

