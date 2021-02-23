#%%
"""
Created on Thu Nov 27 2018
Integrated Brownian motion
@author: Lech A. Grzelak
"""
import numpy as np
import matplotlib.pyplot as plt

def ComputeIntegrals(NoOfPaths,NoOfSteps,T,g):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
    I1 = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])
    
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        I1[:,i+1] = I1[:,i] + g(W[:,i]) *(W[:,i+1]-W[:,i]) 
        time[i+1] = time[i] +dt
        
    paths = {"time":time,"W":W,"I1":I1}
    return paths


NoOfPaths = 100000
NoOfSteps = 1000
T = 2

g = lambda t: t

output = ComputeIntegrals(NoOfPaths,NoOfSteps,T , g)
timeGrid = output["time"]
G_T = output["I1"]

plt.figure(1)
plt.grid()
plt.hist(G_T[:,-1],50)
plt.xlabel("time")
plt.ylabel("value")
plt.title("Stochastic Integral")

EX = np.mean(G_T[:,-1])
Var = np.var(G_T[:,-1])
print('Mean = {0} and variance ={1}'.format(EX,Var))

