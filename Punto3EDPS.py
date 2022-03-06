#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sym
get_ipython().run_line_magic('matplotlib', 'inline')
import math
#Utilizando como guia el codigo de https://github.com/DavidReveloLuna/ProcesamientoDatos


# In[3]:





# In[36]:


n = sym.Symbol('n')
t=sym.Symbol('t')

Tmin = 0
Tmax = 1

T=Tmax-Tmin
w = 2*np.pi/T

f1=1
f2=-1


ft = sym.Piecewise((f1, ((t < 1/2) & (t > 0))), (f2, ((t >= 1/2) & (t <= 1))))
ft


f_integral = ft
a0 = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
print("a0 = ")
sym.pprint(a0)


f_integral = ft*sym.cos(n*w*t)
an = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
an = sym.simplify(an)
print("an = ")
sym.pprint(an)


f_integral = ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
print("bn = ")
bn = sym.simplify(bn)
sym.pprint(bn)


serie = 0
Armonicos = 100

for i in range(1,Armonicos+1):
    
    
    an_c = an.subs(n,i)
    bn_c = bn.subs(n,i)
    
    if abs(an_c) < 0.0001: an_c = 0
    if abs(bn_c) < 0.0001: bn_c = 0
        
    serie= serie + an_c*sym.cos(i*w*t) 
    serie = serie + bn_c*sym.sin(i*w*t) 

serie = a0/2+serie 

print('f(t)= ')
sym.pprint(serie)


fserie = sym.lambdify(t,serie)
f = sym.lambdify(t,ft)


v_tiempo = np.linspace(Tmin,Tmax,200)


fserieG = fserie(v_tiempo)
fG = f(v_tiempo)
 
plt.plot(v_tiempo,fG,label = 'f(t)')
plt.plot(v_tiempo,fserieG,label = 'Expansión')


plt.legend()

plt.show()


# In[35]:


n = sym.Symbol('n')
t=sym.Symbol('t')

pi = math.pi
Tmin = -pi
Tmax = pi

T=Tmax-Tmin
w = 2*np.pi/T

f1=0
f2=pi


ft = sym.Piecewise((f1, ((t < 0) & (t > -pi))), (f2, ((t >= 0) & (t < pi))))
ft


f_integral = ft
a0 = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
print("a0 = ")
sym.pprint(a0)


f_integral = ft*sym.cos(n*w*t)
an = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
an = sym.simplify(an)
print("an = ")
sym.pprint(an)


f_integral = ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral,(t,Tmin,Tmax))
print("bn = ")
bn = sym.simplify(bn)
sym.pprint(bn)


serie = 0
Armonicos = 1000

for i in range(1,Armonicos+1):
    
    
    an_c = an.subs(n,i)
    bn_c = bn.subs(n,i)
    
    if abs(an_c) < 0.0001: an_c = 0
    if abs(bn_c) < 0.0001: bn_c = 0
        
    serie= serie + an_c*sym.cos(i*w*t) 
    serie = serie + bn_c*sym.sin(i*w*t) 

serie = a0/2+serie 

print('f(t)= ')
sym.pprint(serie)


fserie = sym.lambdify(t,serie)
f = sym.lambdify(t,ft)


v_tiempo = np.linspace(Tmin,Tmax,200)


fserieG = fserie(v_tiempo)
fG = f(v_tiempo)
 
plt.plot(v_tiempo,fG,)
plt.plot(v_tiempo,fserieG)


plt.legend()

plt.show()

