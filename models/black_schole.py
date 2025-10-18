"""
Black-Scholes model

S: Current price of the underlying asset
K: Strike price of the option
T: Time to maturity of the option (in years)
r: Risk-free interest rate (annualized)
sigma: Volatility of the underlying asset (annualized)
d1: First parameter in the Black-Scholes formula
d2: Second parameter in the Black-Scholes formula
"""

import numpy as np
from scipy.stats import norm


def calc_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def calc_d2(d1: float, T: float, sigma: float) -> float:
    return d1 - sigma * np.sqrt(T)


def find_d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    d1 = calc_d1(S, K, T, r, sigma)
    d2 = calc_d2(d1, T, sigma)
    return d1, d2


def call_option_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1, d2 = find_d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_option_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    c = call_option_price(S, K, T, r, sigma)
    return c - S + K * np.exp(-r * T)
