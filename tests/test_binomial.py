import numpy as np
import pytest

from option_pricer.binomial import binomial_tree
from option_pricer.black_scholes import black_scholes

CASES = [
    (100, 100, 1.0, 0.05, 0.20),  # at the money
    (100, 80, 0.5, 0.03, 0.30),   # deep in the money call
    (100, 120, 2.0, 0.01, 0.15),  # out of the money call
]
KINDS = ["call", "put"]


def test_one_step_matches_hand_calculation():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    u = np.exp(sigma)
    d = 1 / u
    q = (np.exp(r) - d) / (u - d)
    by_hand = np.exp(-r) * q * (S * u - K)
    assert binomial_tree(S, K, T, r, sigma, N=1) == pytest.approx(by_hand, abs=1e-12)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_european_converges_to_black_scholes(S, K, T, r, sigma, kind):
    tree = binomial_tree(S, K, T, r, sigma, N=2000, option_type=kind)
    assert tree == pytest.approx(black_scholes(S, K, T, r, sigma, kind), abs=5e-3)


def test_error_shrinks_as_steps_increase():
    args = (100, 100, 1.0, 0.05, 0.20)
    exact = black_scholes(*args, "call")
    errors = [abs(binomial_tree(*args, N=n) - exact) for n in (10, 100, 1000)]
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_put_call_parity_holds_exactly_on_the_tree(S, K, T, r, sigma):
    c = binomial_tree(S, K, T, r, sigma, N=500, option_type="call")
    p = binomial_tree(S, K, T, r, sigma, N=500, option_type="put")
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-9)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_american_call_equals_european_call(S, K, T, r, sigma):
    # Without dividends, exercising a call early is never optimal
    euro = binomial_tree(S, K, T, r, sigma, N=500, option_type="call")
    amer = binomial_tree(S, K, T, r, sigma, N=500, option_type="call", american=True)
    assert amer == pytest.approx(euro, abs=1e-10)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_american_put_worth_at_least_european_put(S, K, T, r, sigma):
    euro = binomial_tree(S, K, T, r, sigma, N=500, option_type="put")
    amer = binomial_tree(S, K, T, r, sigma, N=500, option_type="put", american=True)
    assert amer >= euro


def test_american_put_known_value():
    price = binomial_tree(100, 100, 1.0, 0.05, 0.20, N=2000, option_type="put", american=True)
    assert price == pytest.approx(6.090, abs=2e-3)


def test_deep_in_the_money_american_put_is_exercised_now():
    price = binomial_tree(50, 100, 1.0, 0.05, 0.20, N=500, option_type="put", american=True)
    assert price == pytest.approx(50.0, abs=1e-10)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        binomial_tree(100, 100, 1.0, 0.05, 0.20, option_type="Call")


def test_arbitrage_tree_raises():
    # Tiny vol, big rate, one step: the bond beats the up move, so q > 1
    with pytest.raises(ValueError):
        binomial_tree(100, 100, 1.0, 0.50, 0.05, N=1)