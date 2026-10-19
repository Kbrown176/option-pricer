"""Plot a Monte Carlo estimate and its 95% confidence band as paths are added.

Run from the project root:  python scripts/mc_convergence_plot.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from option_pricer.black_scholes import black_scholes  # noqa: E402
from option_pricer.monte_carlo import simulate_terminal  # noqa: E402

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
n_max = 1_000_000

rng = np.random.default_rng(42)
ST = simulate_terminal(S, T, r, sigma, n_max, rng)
payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)

# Running mean and standard error after the first n paths, for n from 100 to n_max
n = np.unique(np.logspace(2, np.log10(n_max), 200).astype(int))
csum = np.cumsum(payoffs)[n - 1]
csum_sq = np.cumsum(payoffs**2)[n - 1]
mean = csum / n
var = (csum_sq - n * mean**2) / (n - 1)
se = np.sqrt(var / n)
bs_price = black_scholes(S, K, T, r, sigma)

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.fill_between(n, mean - 1.96 * se, mean + 1.96 * se, color=BLUE, alpha=0.18,
                linewidth=0, label="95% confidence interval")
ax.plot(n, mean, color=BLUE, linewidth=2, label="Monte Carlo estimate")
ax.axhline(bs_price, color=MUTED, linewidth=1.5, linestyle="--", label="Black-Scholes")
ax.text(n_max, bs_price + 0.08, f"Black-Scholes {bs_price:.4f}", color=MUTED,
        ha="right", va="bottom", fontsize=9)

ax.set_xscale("log")
ax.set_title("Monte Carlo error shrinks like 1/\u221aN", color=INK, loc="left", fontsize=12)
ax.set_xlabel("Number of simulated paths (log scale)", color=MUTED)
ax.set_ylabel("European call price", color=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="lower right")

os.makedirs("images", exist_ok=True)
fig.tight_layout()
fig.savefig("images/mc_convergence.png")
print("Saved images/mc_convergence.png")
