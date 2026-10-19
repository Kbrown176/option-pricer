import pytest

from option_pricer.black_scholes import black_scholes
from option_pricer.greeks import delta, gamma, vega, theta, rho

# Several market setups, so a lucky coincidence can't pass the tests
CASES = [
    (100, 100, 1.0, 0.05, 0.20),  # at the money
    (100, 80, 0.5, 0.03, 0.30),   # deep in the money call
    (100, 120, 2.0, 0.01, 0.15),  # out of the money call
]
KINDS = ["call", "put"]


def central_diff(f, x, h=1e-5):
    """First derivative, error of order h^2."""
    return (f(x + h) - f(x - h)) / (2 * h)


def second_diff(f, x, h=1e-2):
    """Second derivative, error of order h^2."""
    return (f(x + h) - 2 * f(x) + f(x - h)) / h**2


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_delta(S, K, T, r, sigma, kind):
    fd = central_diff(lambda s: black_scholes(s, K, T, r, sigma, kind), S)
    assert delta(S, K, T, r, sigma, kind) == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_gamma(S, K, T, r, sigma):
    fd = second_diff(lambda s: black_scholes(s, K, T, r, sigma, "call"), S)
    assert gamma(S, K, T, r, sigma) == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_vega(S, K, T, r, sigma):
    fd = central_diff(lambda v: black_scholes(S, K, T, r, v, "call"), sigma)
    assert vega(S, K, T, r, sigma) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_theta(S, K, T, r, sigma, kind):
    # Minus sign because T (time remaining) shrinks as calendar time passes
    fd = -central_diff(lambda t: black_scholes(S, K, t, r, sigma, kind), T)
    assert theta(S, K, T, r, sigma, kind) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_rho(S, K, T, r, sigma, kind):
    fd = central_diff(lambda rate: black_scholes(S, K, T, rate, sigma, kind), r)
    assert rho(S, K, T, r, sigma, kind) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_greeks_satisfy_black_scholes_pde(S, K, T, r, sigma, kind):
    V = black_scholes(S, K, T, r, sigma, kind)
    lhs = (
        theta(S, K, T, r, sigma, kind)
        + 0.5 * sigma**2 * S**2 * gamma(S, K, T, r, sigma)
        + r * S * delta(S, K, T, r, sigma, kind)
        - r * V
    )
    assert lhs == pytest.approx(0, abs=1e-9)


def test_textbook_values():
    args = (100, 100, 1.0, 0.05, 0.20)
    assert delta(*args, "call") == pytest.approx(0.63683, rel=1e-4)
    assert gamma(*args) == pytest.approx(0.018762, rel=1e-4)
    assert vega(*args) == pytest.approx(37.524, rel=1e-4)
    assert theta(*args, "call") == pytest.approx(-6.4140, rel=1e-4)
    assert rho(*args, "call") == pytest.approx(53.232, rel=1e-4)