import numpy as np
import pandas as pd
import pytest

from option_pricer.black_scholes import black_scholes
from option_pricer.implied_vol import implied_vol
from option_pricer.smile import build_smile, implied_forward

KINDS = ["call", "put"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("K", [70, 90, 100, 110, 130])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
def test_round_trip(kind, K, T, sigma):
    S, r = 100, 0.03
    price = black_scholes(S, K, T, r, sigma, kind)
    intrinsic = max(S - K * np.exp(-r * T), 0) if kind == "call" else max(K * np.exp(-r * T) - S, 0)
    if price - intrinsic < 1e-6:
        # Almost no time value, so the price barely depends on vol at all
        pytest.skip("No time value, so vol is not well defined")
    assert implied_vol(price, S, K, T, r, kind) == pytest.approx(sigma, abs=1e-7)


def test_case_that_breaks_plain_newton():
    # Plain Newton from sigma = 0.15 jumps to 46 and then to a huge negative number
    price = black_scholes(100, 140, 0.25, 0.05, 0.30)
    assert implied_vol(price, 100, 140, 0.25, 0.05) == pytest.approx(0.30, abs=1e-9)


def test_very_high_vol():
    price = black_scholes(100, 100, 0.5, 0.05, 3.0)
    assert implied_vol(price, 100, 100, 0.5, 0.05) == pytest.approx(3.0, abs=1e-8)


@pytest.mark.parametrize("bad_price", [-1.0, 0.5, 100.0, 150.0])
def test_prices_outside_arbitrage_bounds_raise(bad_price):
    # For S=K=100, r=5%, T=1, a call must cost between 4.88 and 100
    with pytest.raises(ValueError):
        implied_vol(bad_price, 100, 100, 1.0, 0.05, "call")


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        implied_vol(10.0, 100, 100, 1.0, 0.05, "Call")


def _synthetic_chain(spot, T, r, div_yield, strikes):
    """A fake option chain with a known skew and known dividend yield."""
    forward = spot * np.exp((r - div_yield) * T)
    spot_equiv = forward * np.exp(-r * T)
    true_vol = 0.20 - 0.4 * np.log(strikes / forward) + 0.8 * np.log(strikes / forward) ** 2
    chain = pd.DataFrame({
        "strike": strikes,
        "call_mid": [black_scholes(spot_equiv, k, T, r, v, "call") for k, v in zip(strikes, true_vol)],
        "put_mid": [black_scholes(spot_equiv, k, T, r, v, "put") for k, v in zip(strikes, true_vol)],
    })
    return chain, forward, true_vol


def test_implied_forward_recovers_dividend_adjusted_forward():
    strikes = np.arange(80.0, 121.0, 2.5)
    chain, forward, _ = _synthetic_chain(100, 0.5, 0.04, 0.015, strikes)
    assert implied_forward(chain, 0.5, 0.04, spot=100) == pytest.approx(forward, abs=1e-9)


def test_build_smile_recovers_true_vols():
    strikes = np.arange(80.0, 121.0, 2.5)
    chain, forward, true_vol = _synthetic_chain(100, 0.5, 0.04, 0.015, strikes)
    smile = build_smile(chain, 0.5, 0.04, forward)
    assert len(smile) == len(strikes)
    assert np.allclose(smile["implied_vol"], true_vol, atol=1e-7)
    assert set(smile.loc[smile["strike"] < forward, "type"]) == {"put"}
    assert set(smile.loc[smile["strike"] > forward, "type"]) == {"call"}


def test_build_smile_skips_impossible_quotes():
    strikes = np.arange(80.0, 121.0, 2.5)
    chain, forward, _ = _synthetic_chain(100, 0.5, 0.04, 0.015, strikes)
    chain.loc[chain["strike"] == 85.0, "put_mid"] = -0.01  # a broken quote
    smile = build_smile(chain, 0.5, 0.04, forward)
    assert 85.0 not in smile["strike"].values
    assert len(smile) == len(strikes) - 1