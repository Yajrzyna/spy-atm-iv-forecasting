"""
Master Thesis — Predicting Short-Term Option Returns Using Machine Learning
Author  : Stanislaw Jarzyna
Step 3  : Feature Engineering
Purpose : Construct the full feature matrix used for model training.
          Features are grouped into three categories:
            1. Option-specific  — Greeks, implied volatility, moneyness, DTE, spread
            2. Underlying       — SPY log returns, realized volatility, momentum
            3. Macro            — VIX level and daily change

          SPY underlying features are derived from the raw (pre-filter) option
          chain so that the price series covers all intraday timestamps, not
          only those where at least one option passes the liquidity filters.

          VIX is the previous trading day's end-of-day level to avoid
          look-ahead bias (same-day EOD VIX is not observable intraday).

          Target variables are NOT computed here; they are computed per split
          in Step 4 to prevent boundary leakage across train/val/test periods.

Input   : 30 minutes/SPY Option Chain/parquet/spy_30min_option_chain.parquet
          30 minutes/SPY Option Chain/parquet/spy_30min_filtered.parquet
          End of Day/VIX Option Chains/parquet/vix_eod.parquet
Output  : 30 minutes/SPY Option Chain/parquet/spy_30min_features.parquet
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os
# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("OPTIONSDX_DIR", os.path.join(ROOT, "data"))

RAW_OPTIONS_PATH = f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_option_chain.parquet"
OPTIONS_PATH     = f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_filtered.parquet"
VIX_PATH         = f"{BASE}/End of Day/VIX Option Chains/parquet/vix_eod.parquet"
OUT_PATH         = f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_features.parquet"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

REALIZED_VOL_WINDOW = "6h30min"   # one trading session (9:30-16:00), time-based
MOMENTUM_WINDOW     = "6h30min"   # same span/rationale as above

# Both windows are time-based (not row-count) so they can never reach into the
# previous session: the overnight/weekend closure is >=17.5h, longer than
# either window, so a rolling window anchored anywhere in today's session is
# mathematically guaranteed to stop at today's open. The previous row-count
# implementation (window=13 / window=26 bars) did not have this guarantee —
# verified empirically to reach back 2-3 calendar days for early-session
# anchors, silently blending overnight/weekend price gaps into what was
# labelled an intraday feature.

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading filtered options data...")
df = pd.read_parquet(OPTIONS_PATH, engine="pyarrow")
df["QUOTE_READTIME"] = pd.to_datetime(df["QUOTE_READTIME"])
df["QUOTE_DATE"]     = pd.to_datetime(df["QUOTE_DATE"])
print(f"  Rows loaded : {len(df):,}")

# ---------------------------------------------------------------------------
# Section 1 — Option-specific features (already present from Step 2)
#
#   C_IV / P_IV          — implied volatility
#   STRIKE_DISTANCE_PCT  — moneyness (negative = OTM call / ITM put)
#   DTE                  — days to expiry
#   C_SPREAD_REL         — call relative bid-ask spread
#   P_SPREAD_REL         — put relative bid-ask spread
# ---------------------------------------------------------------------------

OPTION_FEATURES = [
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO",
    "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO",
    "C_IV", "P_IV", "STRIKE_DISTANCE_PCT", "DTE", "C_SPREAD_REL", "P_SPREAD_REL",
]

# ---------------------------------------------------------------------------
# Section 2 — Underlying features
#
# The raw (pre-filter) option chain is used here so that the SPY price
# series spans every intraday timestamp, not only those where at least one
# option survives the liquidity filters.  Reading just two columns from
# the large parquet keeps memory usage manageable.
# ---------------------------------------------------------------------------

print("Loading raw option chain for underlying price series...")
raw_underlying = pd.read_parquet(
    RAW_OPTIONS_PATH,
    engine="pyarrow",
    columns=["QUOTE_READTIME", "UNDERLYING_LAST"],
)
raw_underlying["QUOTE_READTIME"] = pd.to_datetime(raw_underlying["QUOTE_READTIME"])

underlying = (
    raw_underlying
    .drop_duplicates(subset="QUOTE_READTIME")
    .sort_values("QUOTE_READTIME")
    .set_index("QUOTE_READTIME")
)
del raw_underlying

underlying["SPY_RET"]          = np.log(underlying["UNDERLYING_LAST"] / underlying["UNDERLYING_LAST"].shift(1))
underlying["SPY_REALIZED_VOL"] = underlying["SPY_RET"].rolling(window=REALIZED_VOL_WINDOW, min_periods=2).std()
underlying["SPY_MOMENTUM"]     = underlying["SPY_RET"].rolling(window=MOMENTUM_WINDOW, min_periods=1).sum()

df = df.merge(
    underlying[["SPY_RET", "SPY_REALIZED_VOL", "SPY_MOMENTUM"]],
    left_on="QUOTE_READTIME",
    right_index=True,
    how="left",
)

# ---------------------------------------------------------------------------
# Section 3 — Macro features (previous trading day's end-of-day VIX)
# ---------------------------------------------------------------------------

print("Loading VIX data...")
vix         = pd.read_parquet(VIX_PATH, engine="pyarrow")
vix.columns = [col.strip().strip("[]") for col in vix.columns]
vix["QUOTE_DATE"] = pd.to_datetime(vix["QUOTE_DATE"])

vix_daily = (
    vix[["QUOTE_DATE", "UNDERLYING_LAST"]]
    .drop_duplicates(subset="QUOTE_DATE", keep="last")
    .rename(columns={"UNDERLYING_LAST": "VIX"})
    .sort_values("QUOTE_DATE")
    .set_index("QUOTE_DATE")
)

vix_daily["VIX_CHG"] = np.log(vix_daily["VIX"] / vix_daily["VIX"].shift(1))
vix_daily[["VIX", "VIX_CHG"]] = vix_daily[["VIX", "VIX_CHG"]].shift(1)

df = df.merge(
    vix_daily[["VIX", "VIX_CHG"]],
    left_on="QUOTE_DATE",
    right_index=True,
    how="left",
)

# ---------------------------------------------------------------------------
# Drop rows with NaN in features and save
#
# Targets are NOT computed here — they are computed per split in Step 4 to
# prevent boundary leakage across train / val / test periods.
# ---------------------------------------------------------------------------

FEATURE_COLS = OPTION_FEATURES + ["SPY_RET", "SPY_REALIZED_VOL", "SPY_MOMENTUM", "VIX", "VIX_CHG"]

df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
print(f"  Rows after dropping NaNs : {len(df):,}")

print(f"\nSaving feature dataset...")
df.to_parquet(OUT_PATH, index=False, engine="pyarrow", compression="snappy")
print(f"  Saved   : {OUT_PATH}")
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {list(df.columns)}")