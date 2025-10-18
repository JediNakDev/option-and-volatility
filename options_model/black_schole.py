import numpy as np
from scipy.stats import norm


def calc_d1(S, K, T, r, vol):
    return (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))


def calc_d2(d1, T, vol):
    return d1 - vol * np.sqrt(T)


def find_d1_d2(S, K, T, r, vol):
    d1 = calc_d1(S, K, T, r, vol)
    d2 = calc_d2(d1, T, vol)
    return d1, d2


def call_option_price(S, K, T, r, vol):
    d1, d2 = find_d1_d2(S, K, T, r, vol)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_option_price(S, K, T, r, vol):
    c = call_option_price(S, K, T, r, vol)
    return c - S + K * np.exp(-r * T)
