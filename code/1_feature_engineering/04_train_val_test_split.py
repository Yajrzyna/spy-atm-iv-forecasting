"""
Master Thesis — Predicting Short-Term Option Returns Using Machine Learning
Author  : Stanislaw Jarzyna
Step 4  : Train / Validation / Test Split
Purpose : Partition the feature dataset into three non-overlapping temporal
          subsets. A strict chronological split is used — no shuffling —
          to prevent look-ahead bias and simulate realistic out-of-sample
          forecasting conditions.

          Targets and rolling-window lag features are computed independently
          within each split to prevent boundary leakage: contracts that span
          a split boundary cannot borrow future prices from the next period.

          Timestamp validity is enforced for every forward and backward bar:
          a bar is only accepted if its QUOTE_READTIME is exactly h × 30
          minutes from the current bar, nulling overnight and weekend gaps.

          Two sets of targets are produced for the two thesis lines:

          2a. Price Ret — log returns of the option mid-price at fixed horizons:
              C_RET_1h, C_RET_2h, C_RET_3h  (and P_ equivalents)

          2b. Rolling Window — forward ATM implied volatility for each of the
              twelve possible forecast horizons within a 13-bar trading day:
              C_IV_FWD_1 … C_IV_FWD_12  (and P_ equivalents)

              Observation-window IV lag features are also added:
              C_IV_LAG_1 … C_IV_LAG_12  (and P_ equivalents)

          dropna is applied on Price Ret targets only; Rolling Window NaNs
          (which vary by combination) are handled per-combination inside each
          model file in 2b.

Split:
    Train       : 2017 – 2021  (in-sample, model fitting)
    Validation  : 2022         (hyperparameter tuning, early stopping)
    Test        : 2023         (final out-of-sample evaluation, reported results)

Input   : 30 minutes/SPY Option Chain/parquet/spy_30min_features.parquet
Output  : 30 minutes/SPY Option Chain/parquet/spy_30min_train.parquet
          30 minutes/SPY Option Chain/parquet/spy_30min_val.parquet
          30 minutes/SPY Option Chain/parquet/spy_30min_test.parquet
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
DIR  = f"{BASE}/30 minutes/SPY Option Chain/parquet"

IN_PATH = f"{DIR}/spy_30min_features.parquet"

# 2a. Price Ret — rows with valid Price Ret targets only (dropna applied)
TRAIN_2A_PATH = f"{DIR}/spy_30min_train_2a.parquet"
VAL_2A_PATH   = f"{DIR}/spy_30min_val_2a.parquet"
TEST_2A_PATH  = f"{DIR}/spy_30min_test_2a.parquet"

# 2b. Rolling Window — all 13 bars per day preserved (NaN filtering per-combination in model files)
TRAIN_2B_PATH = f"{DIR}/spy_30min_train_2b.parquet"
VAL_2B_PATH   = f"{DIR}/spy_30min_val_2b.parquet"
TEST_2B_PATH  = f"{DIR}/spy_30min_test_2b.parquet"

# ---------------------------------------------------------------------------
# Split boundaries (end of each period, inclusive)
# ---------------------------------------------------------------------------

TRAIN_END = "2021-12-31"
VAL_END   = "2022-12-31"

# ---------------------------------------------------------------------------
# Target and lag definitions
# ---------------------------------------------------------------------------

RET_HORIZONS    = {"1h": 2, "2h": 4, "3h": 6}      # Price Ret: label → bars ahead
RET_TARGET_COLS = [f"{side}_RET_{h}" for h in RET_HORIZONS for side in ("C", "P")]

RW_BARS = 12    # Rolling Window: 13-bar trading day → lags/forwards 1 .. 12


def compute_targets_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-contract targets and observation-window lag features.

    Price Ret targets   : log return of option mid-price at fixed horizons.
    Rolling Window fwds : ATM IV level k bars ahead (k = 1 .. 12).
    Rolling Window lags : ATM IV level k bars in the past (k = 1 .. 12).

    All forward and backward bars are validated against wall-clock time;
    overnight and weekend gaps set the value to NaN.
    """
    df  = df.sort_values(["EXPIRE_DATE", "STRIKE", "QUOTE_READTIME"]).reset_index(drop=True)
    grp = df.groupby(["EXPIRE_DATE", "STRIKE"], sort=False)

    # --- Price Ret targets (2a) ---
    for label, h in RET_HORIZONS.items():
        future_c = grp["C_MID"].shift(-h)
        future_p = grp["P_MID"].shift(-h)
        future_t = grp["QUOTE_READTIME"].shift(-h)
        valid    = (future_t - df["QUOTE_READTIME"]) == pd.Timedelta(minutes=30 * h)
        df[f"C_RET_{label}"] = np.where(valid, np.log(future_c / df["C_MID"]), np.nan)
        df[f"P_RET_{label}"] = np.where(valid, np.log(future_p / df["P_MID"]), np.nan)

    # --- Rolling Window forward IV targets (2b) ---
    for k in range(1, RW_BARS + 1):
        future_c_iv = grp["C_IV"].shift(-k)
        future_p_iv = grp["P_IV"].shift(-k)
        future_t    = grp["QUOTE_READTIME"].shift(-k)
        valid_fwd   = (future_t - df["QUOTE_READTIME"]) == pd.Timedelta(minutes=30 * k)
        df[f"C_IV_FWD_{k}"] = np.where(valid_fwd, future_c_iv, np.nan)
        df[f"P_IV_FWD_{k}"] = np.where(valid_fwd, future_p_iv, np.nan)

    # --- Rolling Window IV lag features / observation window (2b) ---
    for k in range(1, RW_BARS + 1):
        past_c_iv = grp["C_IV"].shift(k)
        past_p_iv = grp["P_IV"].shift(k)
        past_t    = grp["QUOTE_READTIME"].shift(k)
        valid_lag = (df["QUOTE_READTIME"] - past_t) == pd.Timedelta(minutes=30 * k)
        df[f"C_IV_LAG_{k}"] = np.where(valid_lag, past_c_iv, np.nan)
        df[f"P_IV_LAG_{k}"] = np.where(valid_lag, past_p_iv, np.nan)

    return df

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading feature dataset...")
df = pd.read_parquet(IN_PATH, engine="pyarrow")
df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
print(f"  Total rows : {len(df):,}")
print(f"  Date range : {df['QUOTE_DATE'].min().date()}  →  {df['QUOTE_DATE'].max().date()}")

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

train = df[df["QUOTE_DATE"] <= TRAIN_END]
val   = df[(df["QUOTE_DATE"] > TRAIN_END) & (df["QUOTE_DATE"] <= VAL_END)]
test  = df[df["QUOTE_DATE"] > VAL_END]

# ---------------------------------------------------------------------------
# Compute targets and lags per split (prevents cross-period boundary leakage).
# dropna on Price Ret targets only; Rolling Window NaNs handled per-combination
# inside each 2b model file.
# ---------------------------------------------------------------------------

print("\nComputing per-split targets and lag features...")
train_full = compute_targets_and_lags(train).reset_index(drop=True)
val_full   = compute_targets_and_lags(val).reset_index(drop=True)
test_full  = compute_targets_and_lags(test).reset_index(drop=True)

# 2a: drop rows where any Price Ret target is NaN (last few bars of each day)
train_2a = train_full.dropna(subset=RET_TARGET_COLS).reset_index(drop=True)
val_2a   = val_full.dropna(subset=RET_TARGET_COLS).reset_index(drop=True)
test_2a  = test_full.dropna(subset=RET_TARGET_COLS).reset_index(drop=True)

# 2b: keep all 13 bars per day; per-combination NaN filtering happens inside model files
train_2b = train_full
val_2b   = val_full
test_2b  = test_full

# ---------------------------------------------------------------------------
# Verify — date ranges and row count compliance
# ---------------------------------------------------------------------------

print(f"\n  {'Line':<6} {'Split':<12} {'Start':<14} {'End':<14} {'Rows':>10}")
print(f"  {'-'*60}")
for line, splits in [("2a", [("Train", train_2a), ("Val", val_2a), ("Test", test_2a)]),
                     ("2b", [("Train", train_2b), ("Val", val_2b), ("Test", test_2b)])]:
    for name, subset in splits:
        print(f"  {line:<6} {name:<12} {str(subset['QUOTE_DATE'].min().date()):<14} "
              f"{str(subset['QUOTE_DATE'].max().date()):<14} {len(subset):>10,}")

print(f"\n  2a total rows : {len(train_2a) + len(val_2a) + len(test_2a):,}")
print(f"  2b total rows : {len(train_2b) + len(val_2b) + len(test_2b):,}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

print("\nSaving splits...")
train_2a.to_parquet(TRAIN_2A_PATH, index=False, engine="pyarrow", compression="snappy")
val_2a.to_parquet(VAL_2A_PATH,     index=False, engine="pyarrow", compression="snappy")
test_2a.to_parquet(TEST_2A_PATH,   index=False, engine="pyarrow", compression="snappy")

train_2b.to_parquet(TRAIN_2B_PATH, index=False, engine="pyarrow", compression="snappy")
val_2b.to_parquet(VAL_2B_PATH,     index=False, engine="pyarrow", compression="snappy")
test_2b.to_parquet(TEST_2B_PATH,   index=False, engine="pyarrow", compression="snappy")

print(f"  2a saved → spy_30min_{{train,val,test}}_2a.parquet")
print(f"  2b saved → spy_30min_{{train,val,test}}_2b.parquet")
