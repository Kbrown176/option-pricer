import numpy as np


def _payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def binomial_tree(S, K, T, r, sigma, N=500, option_type="call", american=False):
    """Cox-Ross-Rubinstein binomial tree price.

    N is the number of time steps. Set american=True to allow early exercise.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    growth = np.exp(r * dt)
    q = (growth - d) / (u - d)  # risk-neutral probability of an up move
    disc = 1 / growth

    if not 0 < q < 1:
        raise ValueError("No-arbitrage condition d < e^(r dt) < u fails. Increase N.")

    # Stock prices at expiry. Index j = number of up moves, from 0 to N
    j = np.arange(N + 1)
    stock = S * u**j * d**(N - j)
    value = _payoff(stock, K, option_type)

    # Step backwards through the tree, one time step at a time
    for step in range(N - 1, -1, -1):
        # Node j at this step leads to j+1 (up) or j (down) at the next step
        value = disc * (q * value[1:] + (1 - q) * value[:-1])

        if american:
            j = np.arange(step + 1)
            stock = S * u**j * d**(step - j)
            value = np.maximum(value, _payoff(stock, K, option_type))

    return value[0]