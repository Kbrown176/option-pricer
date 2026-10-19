import numpy as np

from option_pricer.black_scholes import black_scholes
from option_pricer.greeks import vega


def implied_vol(price, S, K, T, r, option_type="call", tol=1e-10, max_iter=100):
    """Volatility that makes the Black-Scholes price equal the given price.

    Newton-Raphson, with a bisection fallback whenever a Newton step would
    leave the bracket known to contain the answer. That makes it fast in
    normal cases and impossible to send off to infinity in awkward ones.
    """
    discounted_strike = K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(S - discounted_strike, 0.0), S
    elif option_type == "put":
        lower, upper = max(discounted_strike - S, 0.0), discounted_strike
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if not lower < price < upper:
        raise ValueError(
            f"Price {price:.6f} is outside the no-arbitrage range "
            f"({lower:.6f}, {upper:.6f}), so no implied volatility exists."
        )

    def price_error(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - price

    # Bracket [lo, hi]: the price is too low at lo and too high at hi
    lo, hi = 1e-6, 1.0
    while price_error(hi) < 0:
        hi *= 2
        if hi > 100:
            raise ValueError("Implied volatility above 10,000%. Check the inputs.")

    # Start at the inflection point of price as a function of vol
    # (Manaster and Koehler, 1982), clipped into the bracket
    x = np.log(S / K) + r * T
    sigma = np.clip(np.sqrt(2 * abs(x) / T), 0.05, hi)

    for _ in range(max_iter):
        err = price_error(sigma)
        if abs(err) < tol:
            return float(sigma)

        # Price rises with vol, so the sign of the error tells us which side we are on
        if err > 0:
            hi = sigma
        else:
            lo = sigma

        v = vega(S, K, T, r, sigma)
        step = sigma - err / v if v > 0 else np.nan
        sigma = step if lo < step < hi else 0.5 * (lo + hi)

        if hi - lo < 1e-14:
            return float(sigma)

    raise RuntimeError("Implied volatility did not converge.")
