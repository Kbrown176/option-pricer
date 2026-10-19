"""Plot how the binomial tree price converges to Black-Scholes as steps increase.

Run from the project root:  python scripts/convergence_plot.py
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from option_pricer.binomial import binomial_tree  # noqa: E402
from option_pricer.black_scholes import black_scholes  # noqa: E402

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
steps = range(1, 101)
tree_prices = [binomial_tree(S, K, T, r, sigma, N=n) for n in steps]
bs_price = black_scholes(S, K, T, r, sigma)

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
ax.plot(steps, tree_prices, color=BLUE, linewidth=2, label="Binomial tree (CRR)")
ax.axhline(bs_price, color=MUTED, linewidth=1.5, linestyle="--", label="Black-Scholes")
ax.text(100, bs_price + 0.06, f"Black-Scholes {bs_price:.4f}", color=MUTED,
        ha="right", va="bottom", fontsize=9)

ax.set_title("Binomial tree converges to Black-Scholes", color=INK, loc="left", fontsize=12)
ax.set_xlabel("Number of steps (N)", color=MUTED)
ax.set_ylabel("European call price", color=MUTED)
ax.set_ylim(9.4, 12.3)
ax.grid(axis="y", color=GRID, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="upper right")

os.makedirs("images", exist_ok=True)
fig.tight_layout()
fig.savefig("images/convergence.png")
print("Saved images/convergence.png")
