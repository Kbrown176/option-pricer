from typing import NamedTuple

import numpy as np
from scipy.stats import norm


class MCResult(NamedTuple):
    """A Monte Carlo price together with its standard error."""

    price: float
    std_error: float

    def conf_interval(self, z=1.96):
        """95% confidence interval by default."""
        return self.price - z * self.std_error, self.price + z * self.std_error


def _payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def _normals(rng, shape, antithetic):
    """Standard normal draws. With antithetic=True, the second half mirrors the first."""
    if not antithetic:
        return rng.standard_normal(shape)
    half = rng.standard_normal((shape[0] // 2,) + tuple(shape[1:]))
    return np.concatenate([half, -half])


def _estimate(samples, antithetic):
    """Mean and standard error. Antithetic pairs are averaged first, because
    a draw and its mirror image are not independent."""
    if antithetic:
        n = len(samples) // 2
        samples = 0.5 * (samples[:n] + samples[n:])
    return MCResult(float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples))))


def simulate_terminal(S, T, r, sigma, n_paths, rng, antithetic=False):
    """Stock prices at T under the risk-neutral measure (exact, no time steps)."""
    Z = _normals(rng, (n_paths,), antithetic)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_paths(S, T, r, sigma, n_steps, n_paths, rng, antithetic=False):
    """Stock prices at n_steps equally spaced dates after today. Shape (n_paths, n_steps)."""
    dt = T / n_steps
    Z = _normals(rng, (n_paths, n_steps), antithetic)
    log_steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return S * np.exp(np.cumsum(log_steps, axis=1))


def mc_european(S, K, T, r, sigma, n_paths=100_000, option_type="call",
                antithetic=False, seed=None):
    """European option price by Monte Carlo."""
    rng = np.random.default_rng(seed)
    ST = simulate_terminal(S, T, r, sigma, n_paths, rng, antithetic)
    discounted = np.exp(-r * T) * _payoff(ST, K, option_type)
    return _estimate(discounted, antithetic)


def geometric_asian(S, K, T, r, sigma, n_steps, option_type="call"):
    """Closed-form price of a geometric-average Asian option with n_steps
    equally spaced averaging dates (the last one at expiry)."""
    mu = np.log(S) + (r - 0.5 * sigma**2) * T * (n_steps + 1) / (2 * n_steps)
    var = sigma**2 * T * (n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2)
    sd = np.sqrt(var)
    forward = np.exp(mu + 0.5 * var)  # expected value of the geometric average
    d1 = (mu - np.log(K) + var) / sd
    d2 = d1 - sd

    if option_type == "call":
        return np.exp(-r * T) * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return np.exp(-r * T) * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")


def mc_asian(S, K, T, r, sigma, n_steps=12, n_paths=100_000, option_type="call",
             control_variate=False, seed=None):
    """Arithmetic-average Asian option by Monte Carlo.

    With control_variate=True, the geometric Asian (known exactly) is used to
    cancel most of the simulation noise.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_paths(S, T, r, sigma, n_steps, n_paths, rng)
    disc = np.exp(-r * T)

    arithmetic = disc * _payoff(paths.mean(axis=1), K, option_type)
    if not control_variate:
        return _estimate(arithmetic, antithetic=False)

    geometric = disc * _payoff(np.exp(np.log(paths).mean(axis=1)), K, option_type)
    exact = geometric_asian(S, K, T, r, sigma, n_steps, option_type)

    cov = np.cov(arithmetic, geometric)
    b = cov[0, 1] / cov[1, 1]
    adjusted = arithmetic - b * (geometric - exact)
    return _estimate(adjusted, antithetic=False)
