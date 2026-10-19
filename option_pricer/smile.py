import numpy as np
import pandas as pd

from option_pricer.implied_vol import implied_vol


def implied_forward(chain, T, r, spot, window=0.05):
    """Forward price implied by put-call parity: C - P = e^(-rT) (F - K).

    Uses strikes within `window` of spot, where both options trade actively,
    and takes the median to shrug off any bad quotes. This automatically
    accounts for dividends, which plain Black-Scholes ignores.
    """
    near = chain[(chain["strike"] / spot - 1).abs() < window]
    if near.empty:
        raise ValueError("No strikes near spot to infer the forward from.")
    forwards = near["strike"] + np.exp(r * T) * (near["call_mid"] - near["put_mid"])
    return float(forwards.median())


def build_smile(chain, T, r, forward):
    """Implied vol at each strike, using out-of-the-money options only:
    puts below the forward, calls above it.

    `chain` needs columns strike, call_mid and put_mid.
    """
    # Black-Scholes with spot set to the present value of the forward is the
    # same as pricing off the forward, so dividends are handled for free
    spot_equiv = forward * np.exp(-r * T)

    rows = []
    for strike, call_mid, put_mid in chain[["strike", "call_mid", "put_mid"]].itertuples(index=False):
        kind = "put" if strike < forward else "call"
        price = put_mid if kind == "put" else call_mid
        try:
            vol = implied_vol(price, spot_equiv, strike, T, r, kind)
        except (ValueError, RuntimeError):
            continue  # stale or crossed quote: skip it
        rows.append((strike, kind, vol))

    return pd.DataFrame(rows, columns=["strike", "type", "implied_vol"])
