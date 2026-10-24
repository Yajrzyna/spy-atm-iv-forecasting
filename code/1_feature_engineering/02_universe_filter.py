"""
Master Thesis — Predicting Short-Term Option Returns Using Machine Learning
Author  : Stanislaw Jarzyna
Step 2  : Universe Construction & Filtering
Purpose : Reduce the full SPY 30-minute option chain to the subset relevant
          for modelling: single ATM options with short maturities, after
          applying liquidity screens.

Filters applied (in order):
    1. Maturity      — keep options with 1 to 7 days to expiry
    2. Liquidity     — remove options with zero bid prices
    3. Spread        — remove options with abnormally wide bid-ask spreads
    4. IV floor      — remove options where C_IV or P_IV < 5% (near-zero IV
                       produces numerically unstable BSM Greeks)
    5. ATM selection — for each (QUOTE_DATE, EXPIRE_DATE), pick the single
                       strike closest to the underlying at the 10:00 ET
                       snapshot, then keep that same (EXPIRE_DATE, STRIKE)
                       contract for every bar of that trading day (9:30-16:00).
                       This is a day-level, fixed-K=1 selection: exactly one
                       option per expiry per day, no ambiguity about how many
                       strikes qualify, and no intraday moneyness churn. A
                       row-wise ±X% moneyness band was considered and rejected:
                       it re-selects the universe every 30 minutes, which (a)
                       yields a floating, unpredictable strike count per
                       snapshot (11 strikes/expiry in 2017 vs. 27 in 2023, as
                       SPY's fixed ~$1 strike spacing shrank relative to a
                       rising price), and (b) creates a selection bias where
                       forward-looking targets silently drop exactly the
                       largest underlying moves (the cases that push a strike
                       outside the band before the forecast horizon
                       completes) — inflating apparent forecastability.

Derived columns added:
    - C_MID        : call mid-price  = (C_BID + C_ASK) / 2
    - P_MID        : put mid-price   = (P_BID + P_ASK) / 2
    - C_SPREAD_REL : call relative spread = (C_ASK - C_BID) / C_MID
    - P_SPREAD_REL : put relative spread  = (P_ASK - P_BID) / P_MID

Input   : 30 minutes/SPY Option Chain/parquet/spy_30min_option_chain.parquet
Output  : 30 minutes/SPY Option Chain/parquet/spy_30min_filtered.parquet
"""

import pandas as pd
import pyarrow.parquet as pq
import time
from datetime import time as dtime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

import os
# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("OPTIONSDX_DIR", os.path.join(ROOT, "data"))

IN_PATH  = f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_option_chain.parquet"
OUT_PATH = f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_filtered.parquet"

# ---------------------------------------------------------------------------
# Filter thresholds — adjust here to experiment with different universes
# ---------------------------------------------------------------------------

DTE_MIN             = 1
DTE_MAX             = 7
MAX_RELATIVE_SPREAD = 0.50   # drop options where spread > 50% of mid-price
MIN_IV              = 0.05   # drop options where C_IV or P_IV < 5%
BATCH_SIZE          = 500_000
ATM_ANCHOR_TIME      = dtime(10, 0)   # snapshot used to pick the day's nearest strike

# ---------------------------------------------------------------------------
# Stream and filter in batches to avoid loading the full 5+ GB file at once
# ---------------------------------------------------------------------------

print("Loading SPY 30-minute option chain (batched)...")
start      = time.perf_counter()
pf         = pq.ParquetFile(IN_PATH)
total_raw  = 0
chunks     = []

for batch in pf.iter_batches(batch_size=BATCH_SIZE):
    chunk = batch.to_pandas()
    total_raw += len(chunk)

    # Step 1 — Maturity
    chunk = chunk[(chunk["DTE"] >= DTE_MIN) & (chunk["DTE"] <= DTE_MAX)]

    # Step 2 — Liquidity
    chunk = chunk[~((chunk["C_BID"] == 0) | (chunk["P_BID"] == 0))]

    # Step 3 — Mid-prices and relative spreads
    chunk["C_MID"] = (chunk["C_BID"] + chunk["C_ASK"]) / 2.0
    chunk["P_MID"] = (chunk["P_BID"] + chunk["P_ASK"]) / 2.0
    chunk["C_SPREAD_REL"] = (chunk["C_ASK"] - chunk["C_BID"]) / chunk["C_MID"]
    chunk["P_SPREAD_REL"] = (chunk["P_ASK"] - chunk["P_BID"]) / chunk["P_MID"]

    # Step 4 — Spread filter
    chunk = chunk[
        (chunk["C_SPREAD_REL"] <= MAX_RELATIVE_SPREAD) &
        (chunk["P_SPREAD_REL"] <= MAX_RELATIVE_SPREAD)
    ]

    # Step 5 — IV floor
    chunk = chunk[(chunk["C_IV"] >= MIN_IV) & (chunk["P_IV"] >= MIN_IV)]

    if len(chunk):
        chunks.append(chunk)

elapsed = time.perf_counter() - start
print(f"  Rows before filtering : {total_raw:>12,}")
print(f"  Time : {elapsed:.3f}s")

df = pd.concat(chunks, ignore_index=True)
print(f"  Rows after maturity/liquidity/spread/IV screens : {len(df):>12,}")

# ---------------------------------------------------------------------------
# Step 6 — ATM selection (day-level, fixed K=1)
#
# At the 10:00 snapshot of each QUOTE_DATE, pick the single strike closest
# to the underlying for every expiry alive that day, then keep that exact
# (EXPIRE_DATE, STRIKE) contract for all of that day's bars (9:30-16:00).
# ---------------------------------------------------------------------------

df["QUOTE_READTIME"] = pd.to_datetime(df["QUOTE_READTIME"])

anchor = df[df["QUOTE_READTIME"].dt.time == ATM_ANCHOR_TIME].copy()
anchor["ABS_MONEYNESS"] = anchor["STRIKE_DISTANCE_PCT"].abs()

selection = (
    anchor.sort_values("ABS_MONEYNESS")
    .drop_duplicates(subset=["QUOTE_DATE", "EXPIRE_DATE"], keep="first")
    [["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]]
)

df = df.merge(selection, on=["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"], how="inner")
print(f"  Rows after ATM (K=1) selection                  : {len(df):>12,}")
print(f"  Selected (day, expiry) contracts                : {len(selection):>12,}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print(f"\nSaving filtered dataset...")
df.to_parquet(OUT_PATH, index=False, engine="pyarrow", compression="snappy")
print(f"  Saved   : {OUT_PATH}")
print(f"  Rows    : {len(df):,}")
print(f"  Columns : {list(df.columns)}")
