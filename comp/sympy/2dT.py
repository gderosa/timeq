
# coding: utf-8

# In[173]:


from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.constants import hbar


# In[174]:


# Remeber this to have LaTeX rendered output in Jupyter
init_printing()


# In[175]:


Delta = Symbol(r'\Delta', real=True)
Omega = Symbol(r'\Omega', real=True)
omega = Symbol(r'\omega', real=True)


# In[176]:


Delta = Rational(1, 11)


# In[177]:


F = mdft(2)


# In[178]:


T = (pi/omega) * Matrix([
    [0, 0],
    [0, Delta]
])


# In[179]:


T


# In[180]:


Omega = (omega**2/(pi*Delta**2))*F*T*(F.adjoint())


# In[181]:


Omega


# In[182]:


Hs = I*hbar*omega*Matrix([
    [0, 1],
    [-1,0]
])


# In[183]:


J = TensorProduct(hbar*Omega, eye(2)) + TensorProduct(eye(2), Hs)


# In[184]:


J


# In[185]:


J.eigenvects()

