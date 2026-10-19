import numpy as np
from option_pricer.black_scholes import black_scholes


def test_known_call_value():
    # Textbook case: S=100, K=100, T=1, r=5%, vol=20% gives about 10.4506
    price = black_scholes(100, 100, 1, 0.05, 0.2, "call")
    assert abs(price - 10.4506) < 1e-4


def test_put_call_parity():
    S, K, T, r, sigma = 100, 95, 0.5, 0.03, 0.25
    c = black_scholes(S, K, T, r, sigma, "call")
    p = black_scholes(S, K, T, r, sigma, "put")
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-10