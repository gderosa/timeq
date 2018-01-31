
# coding: utf-8

# In[1]:


from sympy import *
from sympy.physics.matrices import mdft
from sympy.physics.quantum.dagger import Dagger


# In[2]:


N = 4


# In[3]:


T = diag(*map(lambda k: 2*pi*k/N, range(N)))


# In[4]:


F = mdft(N)


# In[5]:


Omega = F * T * Dagger(F) * 2/pi


# In[6]:


Omega

