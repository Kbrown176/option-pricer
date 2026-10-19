import numpy as np
import pytest

from option_pricer.black_scholes import black_scholes
from option_pricer.monte_carlo import (
    geometric_asian,
    mc_asian,
    mc_european,
    simulate_paths,
    simulate_terminal,
)

CASES = [
    (100, 100, 1.0, 0.05, 0.20),  # at the money
    (100, 80, 0.5, 0.03, 0.30),   # deep in the money call
    (100, 120, 2.0, 0.01, 0.15),  # out of the money call
]
KINDS = ["call", "put"]
ATM = CASES[0]

# Monte Carlo answers are random, so we compare against a tolerance measured
# in standard errors. Fixed seeds make every run identical.
TOL = 4


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("antithetic", [False, True])
def test_european_matches_black_scholes(S, K, T, r, sigma, kind, antithetic):
    res = mc_european(S, K, T, r, sigma, n_paths=200_000, option_type=kind,
                      antithetic=antithetic, seed=0)
    exact = black_scholes(S, K, T, r, sigma, kind)
    assert abs(res.price - exact) < TOL * res.std_error


def test_discounted_stock_is_a_martingale():
    # Under the risk-neutral measure, E[e^(-rT) S_T] = S. This fails if the
    # -sigma^2/2 drift correction is missing.
    rng = np.random.default_rng(0)
    discounted = np.exp(-0.05) * simulate_terminal(100, 1.0, 0.05, 0.20, 400_000, rng)
    se = discounted.std(ddof=1) / np.sqrt(len(discounted))
    assert abs(discounted.mean() - 100) < TOL * se


def test_std_error_shrinks_like_one_over_root_n():
    small = mc_european(*ATM, n_paths=100_000, seed=0)
    big = mc_european(*ATM, n_paths=400_000, seed=0)
    assert big.std_error / small.std_error == pytest.approx(0.5, rel=0.05)


def test_same_seed_gives_same_answer():
    assert mc_european(*ATM, seed=123) == mc_european(*ATM, seed=123)


def test_antithetic_reduces_std_error():
    plain = mc_european(*ATM, n_paths=100_000, seed=0)
    anti = mc_european(*ATM, n_paths=100_000, antithetic=True, seed=0)
    assert anti.std_error < plain.std_error


@pytest.mark.parametrize("kind", KINDS)
def test_geometric_asian_with_one_date_is_black_scholes(kind):
    geo = geometric_asian(*ATM, n_steps=1, option_type=kind)
    assert geo == pytest.approx(black_scholes(*ATM, kind), abs=1e-10)


def test_geometric_asian_formula_matches_simulation():
    rng = np.random.default_rng(0)
    paths = simulate_paths(100, 1.0, 0.05, 0.20, 12, 400_000, rng)
    geo_avg = np.exp(np.log(paths).mean(axis=1))
    discounted = np.exp(-0.05) * np.maximum(geo_avg - 100, 0)
    se = discounted.std(ddof=1) / np.sqrt(len(discounted))
    assert abs(discounted.mean() - geometric_asian(*ATM, n_steps=12)) < TOL * se


def test_asian_with_one_date_is_european():
    # An average over a single date (expiry) is just the price at expiry
    asian = mc_asian(*ATM, n_steps=1, n_paths=200_000, seed=0)
    assert abs(asian.price - black_scholes(*ATM)) < TOL * asian.std_error


@pytest.mark.parametrize("kind", KINDS)
def test_control_variate_agrees_with_plain_estimate(kind):
    plain = mc_asian(*ATM, option_type=kind, seed=0)
    cv = mc_asian(*ATM, option_type=kind, control_variate=True, seed=1)
    combined_se = np.hypot(plain.std_error, cv.std_error)
    assert abs(plain.price - cv.price) < TOL * combined_se


def test_control_variate_slashes_std_error():
    plain = mc_asian(*ATM, seed=0)
    cv = mc_asian(*ATM, control_variate=True, seed=0)
    assert cv.std_error < plain.std_error / 10


def test_arithmetic_asian_call_beats_geometric():
    # AM-GM: an arithmetic average is never below the geometric average
    cv = mc_asian(*ATM, control_variate=True, seed=0)
    assert cv.price > geometric_asian(*ATM, n_steps=12)


def test_asian_call_cheaper_than_european():
    # Averaging smooths out the price path, which acts like lower volatility
    cv = mc_asian(*ATM, control_variate=True, seed=0)
    assert cv.price < black_scholes(*ATM)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        mc_european(*ATM, option_type="Call")
        