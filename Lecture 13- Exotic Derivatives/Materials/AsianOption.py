#%%
"""
Created on Thu Jan 16 2019
Pricing of Cash-or-Nothing options with the COS method
@author: Lech A. Grzelak
"""
import numpy as np

def PayoffValuation(S,T,r,payoff):
    # S is a vector of Monte Carlo samples at T
    return np.exp(-r*T) * np.mean(payoff(S))

def GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S_0):    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps+1])
   
    # Euler Approximation
    S1 = np.zeros([NoOfPaths, NoOfSteps+1])
    S1[:,0] =S_0
    
    time = np.zeros([NoOfSteps+1])
        
    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
        W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
        
        S1[:,i+1] = S1[:,i] + r * S1[:,i]* dt + sigma * S1[:,i] * (W[:,i+1] - W[:,i])
        time[i+1] = time[i] +dt
        
    # Retun S1 and S2
    paths = {"time":time,"S":S1}
    return paths


def mainCalculation():
    NoOfPaths = 5000
    NoOfSteps = 250
   
    S0    = 100.0
    r     = 0.05
    T    = 5
    sigma = 0.2
     
    paths = GeneratePathsGBMEuler(NoOfPaths,NoOfSteps,T,r,sigma,S0)
    S_paths= paths["S"]
    S_T = S_paths[:,-1]
    
    # Payoff setting    
    K  = 100.0
    
    # Payoff specification
    payoff = lambda S: np.maximum(S-K,0.0)  
        
    # Valuation
    val_t0 = PayoffValuation(S_T,T,r,payoff)
    print("Value of the contract at t0 ={0}".format(val_t0))
    
    
    A_T= np.mean(S_paths,1)
    valAsian_t0 = PayoffValuation(A_T,T,r,payoff)
    print("Value of the Asian option at t0 ={0}".format(valAsian_t0))
    
    print('variance of S(T) = {0}'.format(np.var(S_T)))
    print('variance of A(T) = {0}'.format(np.var(A_T)))
        
mainCalculation()