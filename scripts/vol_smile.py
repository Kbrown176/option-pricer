"""Fetch a live option chain and plot its implied volatility smile.

Run from the project root:
    python scripts/vol_smile.py          (S&P 500 index options)
    python scripts/vol_smile.py AAPL     (any ticker with listed options)
"""
import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from option_pricer.smile import build_smile, implied_forward  # noqa: E402

MIN_DAYS = 30            # use the first expiry at least this far away
MONEYNESS = (0.8, 1.2)   # keep strikes between 80% and 120% of spot
MAX_SPREAD = 0.5         # drop quotes whose bid-ask spread is over 50% of the mid


def risk_free_rate():
    """13-week US Treasury bill yield (^IRX), as a continuously compounded rate."""
    irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
    return float(np.log(1 + irx.iloc[-1] / 100))


def spot_price(tk):
    return float(tk.history(period="5d")["Close"].dropna().iloc[-1])


def pick_expiry(tk, min_days=MIN_DAYS):
    today = date.today()
    for expiry in tk.options:
        days = (date.fromisoformat(expiry) - today).days
        if days >= min_days:
            return expiry, days
    raise ValueError(f"No expiry at least {min_days} days away.")


def clean_side(df):
    """Keep sensible two-sided quotes and compute mid prices."""
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df = df[(df["ask"] - df["bid"]) / df["mid"] < MAX_SPREAD]
    # Some expiries list two contracts per strike (SPX and SPXW). Keep the busier one.
    df = df.sort_values("openInterest", ascending=False).drop_duplicates("strike")
    return df[["strike", "mid"]]


def load_chain(tk, expiry, spot):
    raw = tk.option_chain(expiry)
    calls = clean_side(raw.calls).rename(columns={"mid": "call_mid"})
    puts = clean_side(raw.puts).rename(columns={"mid": "put_mid"})
    chain = calls.merge(puts, on="strike").sort_values("strike")
    lo, hi = MONEYNESS
    keep = (chain["strike"] > lo * spot) & (chain["strike"] < hi * spot)
    return chain[keep].reset_index(drop=True)


def plot_smile(smile, forward, title, path):
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    for kind, colour, label in (("put", BLUE, "Out-of-the-money puts"),
                                ("call", ORANGE, "Out-of-the-money calls")):
        part = smile[smile["type"] == kind]
        ax.plot(part["strike"], 100 * part["implied_vol"], "o", color=colour,
                markersize=4.5, markeredgecolor="white", markeredgewidth=0.6, label=label)

    ax.axvline(forward, color=MUTED, linewidth=1, linestyle=":")
    ax.text(forward, ax.get_ylim()[1], f"  Forward {forward:,.0f}", color=MUTED,
            ha="left", va="top", fontsize=9)

    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.set_xlabel("Strike", color=MUTED)
    ax.set_ylabel("Implied volatility (%)", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="lower left")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "^SPX"
    tk = yf.Ticker(symbol)

    spot = spot_price(tk)
    r = risk_free_rate()
    expiry, days = pick_expiry(tk)
    T = days / 365

    chain = load_chain(tk, expiry, spot)
    forward = implied_forward(chain, T, r, spot)
    smile = build_smile(chain, T, r, forward)
    div_yield = r - np.log(forward / spot) / T
    atm_vol = np.interp(forward, smile["strike"], smile["implied_vol"])

    print(f"{symbol}  expiry {expiry} ({days} days)")
    print(f"Spot {spot:,.2f}   Forward {forward:,.2f}   Rate {r:.2%}   Implied dividend yield {div_yield:.2%}")
    print(f"ATM implied vol {atm_vol:.2%}   ({len(smile)} strikes)")

    name = symbol.replace("^", "")
    plot_smile(smile, forward, f"{name} implied volatility smile, expiry {expiry}",
               f"images/vol_smile_{name}.png")
    print(f"Saved images/vol_smile_{name}.png")


if __name__ == "__main__":
    main()
    