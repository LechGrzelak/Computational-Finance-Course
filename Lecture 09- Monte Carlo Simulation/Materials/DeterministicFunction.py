#%%
"""
Created on Thu Nov 27 2018
Integrated Brownian motion
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def ComputeIntegral1(NoOfSamples,a,b,c,d,g):    
    x_i = np.random.uniform(a,b,NoOfSamples)
    y_i = np.random.uniform(c,d,NoOfSamples)
    
    p_i = g(x_i) > y_i
    p = np.sum(p_i) / NoOfSamples
    
    integral = p * (b-a)*(d-c) 
    
    plt.figure(1)
    plt.plot(x_i,y_i,'.r')
    plt.plot(x_i,g(x_i),'.b')
    return integral


def ComputeIntegral2(NoOfSamples,a,b,g): 
    x_i = np.random.uniform(a,b,NoOfSamples)
       
    p = (b-a)*np.mean(g(x_i))
    
    return p

NoOfSamples = 100000

a = 0.0
b = 1.0
c = 0.0
d = 3.0 

g = lambda x: np.exp(x)

output = ComputeIntegral1(NoOfSamples,a,b,c,d,g)

print('Integral from Monte Carlo 1 is {0}'.format(output))

output2 = ComputeIntegral2(NoOfSamples,a,b,g)

print('Integral from Monte Carlo 2 is {0}'.format(output2))
print('Integral computed analytically = {0}'.format(np.exp(b)-np.exp(a)))
