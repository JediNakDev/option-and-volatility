"""
Heston model for Monte Carlo simulation

S: Current price of the underlying asset
v: Current variance of the underlying asset
dt: Time step (in years)
mu: Drift of the asset price
kappa: Mean reversion speed of the variance
theta: Long-term variance
rho: Correlation between the asset price and its variance
xi: Volatility of the variance
dW1: Standard Brownian motion for the asset price
dW2: Standard Brownian motion for the variance
"""

import numpy as np
from scipy.stats import norm


def calc_dW1_dW2(rho, dt):
    z1 = norm.rvs(size=1)
    z2 = norm.rvs(size=1)
    epsilon1 = z1
    epsilon2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    return epsilon1 * np.sqrt(dt), epsilon2 * np.sqrt(dt)


def calc_dv(v, dt, kappa, theta, xi, dW2):
    return kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2


def calc_dS(S, v, dt, mu, dW1):
    return mu * S * dt + np.sqrt(v) * S * dW1
