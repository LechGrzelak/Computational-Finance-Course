#%%
"""
Created on Wed Oct 24 20:37:32 2018
Black- Scholes implied volatility function

@author: Lech A. Grzelak
"""
import numpy as np
import scipy.stats as st
# Initial parameters and market quotes
V_market = 2    # market call option price
K        = 120  # strike
tau      = 1    # time-to-maturity
r        = 0.05 # interest rate
S_0      = 100  # today's stock price
sigmaInit    = 0.25  # Initial implied volatility
CP       ="c" #C is call and P is put

def ImpliedVolatility(CP,S_0,K,sigma,tau,r):
    error    = 1e10; # initial error
    #Handy lambda expressions
    optPrice = lambda sigma: BS_Call_Option_Price(CP,S_0,K,sigma,tau,r)
    vega= lambda sigma: dV_dsigma(S_0,K,sigma,tau,r)
    
    # While the difference between the model and the arket price is large
    # follow the iteration
    n = 1.0 
    while error>10e-10:
        g         = optPrice(sigma) - V_market
        g_prim    = vega(sigma)
        sigma_new = sigma - g / g_prim
    
        #error=abs(sigma_new-sigma)
        error=abs(g)
        sigma=sigma_new;
        
        print('iteration {0} with error = {1}'.format(n,error))
        
        n= n+1
    return sigma

# Vega, dV/dsigma
def dV_dsigma(S_0,K,sigma,tau,r):
    #parameters and value of Vega
    d2   = (np.log(S_0 / float(K)) + (r - 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    value = K * np.exp(-r * tau) * st.norm.pdf(d2) * np.sqrt(tau)
    return value

def BS_Call_Option_Price(CP,S_0,K,sigma,tau,r):
    #Black-Scholes Call option price
    d1    = (np.log(S_0 / float(K)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    d2    = d1 - sigma * np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower()=="1":
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower()=="-1":
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1)*S_0
    return value

sigma_imp = ImpliedVolatility(CP,S_0,K,sigmaInit,tau,r)
message = '''Implied volatility for CallPrice= {}, strike K={}, 
      maturity T= {}, interest rate r= {} and initial stock S_0={} 
      equals to sigma_imp = {:.7f}'''.format(V_market,K,tau,r,S_0,sigma_imp)
            
print(message)

# Check! 
val = BS_Call_Option_Price(CP,S_0,K,sigma_imp,tau,r)
print('Option Price for implied volatility of {0} is equal to {1}'.format(sigma_imp, val))
