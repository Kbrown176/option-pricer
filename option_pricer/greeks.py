import numpy as np
from scipy.stats import norm

from option_pricer.black_scholes import d1_d2


def _check_type(option_type):
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def delta(S, K, T, r, sigma, option_type="call"):
    """dV/dS."""
    _check_type(option_type)
    d1, _ = d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """d2V/dS2. Same for calls and puts."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """dV/dsigma, per 1.00 change in vol. Same for calls and puts."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    """dV/dt per year, where t is calendar time (so theta = -dV/dT)."""
    _check_type(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    vol_decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        return vol_decay - r * discounted_strike * norm.cdf(d2)
    return vol_decay + r * discounted_strike * norm.cdf(-d2)


def rho(S, K, T, r, sigma, option_type="call"):
    """dV/dr, per 1.00 change in the rate."""
    _check_type(option_type)
    _, d2 = d1_d2(S, K, T, r, sigma)
    discounted_strike = K * np.exp(-r * T)

    if option_type == "call":
        return T * discounted_strike * norm.cdf(d2)
    return -T * discounted_strike * norm.cdf(-d2)