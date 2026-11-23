"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Train OLS, MLP, LSTM and Ridge stacking on 2017-2021 (validation 2022)
          for every (obs_bars, fcast_bars) combination with obs + fcast = 11
          30-minute bars, then evaluate on 2023 against a random walk.

          All models are scored on the same test rows: the LSTM's valid,
          day-grouped sequences with an exact 30-minute gap between bars.

Tables  : 1 primary metrics, 2 R2_OOS by DTE bucket, 3 R2_OOS by VIX tercile,
          4 top configurations, 5 model ranking, 6-7 vega-Sharpe sensitivity.
Input   : spy_30min_{train,val,test}_2b.parquet (from 04_train_val_test_split.py)
Output  : results/evaluation_report.txt (+ row-level predictions, not published)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data       import DataLoader, TensorDataset
from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
from datetime import time as dtime

from metrics import hit_rate, compute_metrics, vega_sharpe

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("OPTIONSDX_DIR", os.path.join(ROOT, "data"))
DIR  = f"{BASE}/30 minutes/SPY Option Chain/parquet"

TRAIN_PATH = f"{DIR}/spy_30min_train_2b.parquet"
VAL_PATH   = f"{DIR}/spy_30min_val_2b.parquet"
TEST_PATH  = f"{DIR}/spy_30min_test_2b.parquet"
REPORT_DIR = f"{ROOT}/results"

# ---------------------------------------------------------------------------
# Feature and target definitions
# ---------------------------------------------------------------------------

BASE_FEATURE_COLS = [
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO",
    "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO",
    "C_IV", "P_IV", "STRIKE_DISTANCE_PCT", "DTE", "C_SPREAD_REL", "P_SPREAD_REL",
    "SPY_RET", "SPY_REALIZED_VOL", "SPY_MOMENTUM",
    "VIX", "VIX_CHG",
]

TOTAL_BARS   = 11
COMBINATIONS = [(obs, TOTAL_BARS - obs) for obs in range(1, TOTAL_BARS)]  # (obs_bars, fcast_bars)
OPT_TYPES    = {"C": "Call", "P": "Put"}
MODELS       = ["OLS", "MLP", "LSTM", "Stack"]

# Trimmed trading window: drops the 9:30 opening-auction bar and the 16:00
# closing bar.
WINDOW_START = dtime(10, 0)
WINDOW_END   = dtime(15, 30)

# ---------------------------------------------------------------------------
# Hyperparameters (selected on the 2022 validation set)
# ---------------------------------------------------------------------------

MLP_HIDDEN_DIMS   = [64, 32]
MLP_DROPOUT       = 0.04
MLP_LEARNING_RATE = 2.5e-3

HIDDEN_SIZE        = 448
NUM_LAYERS         = 2
LSTM_DROPOUT       = 0.18
LSTM_LEARNING_RATE = 7e-4

BATCH_SIZE    = 512
MAX_EPOCHS    = 100
MLP_PATIENCE  = 30
LSTM_PATIENCE = 20
RIDGE_ALPHA   = 0.001
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list, output_dim: int, dropout: float):
        super().__init__()
        layers, in_dim = [], n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze()


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sequences(X: np.ndarray, y: np.ndarray, day_ids: np.ndarray,
                   times: np.ndarray, seq_len: int):
    """Day-grouped sequence builder (a day-
    group can still contain a mid-day hole from 02's screens, so consecutive
    bars are additionally required to be exactly 30 minutes apart)."""
    X_seq, y_seq, flat_idx = [], [], []
    thirty_min = np.timedelta64(30, "m")

    for gid in np.unique(day_ids):
        mask = day_ids == gid
        idx  = np.where(mask)[0]
        Xg, yg, tg = X[mask], y[mask], times[mask]
        if len(Xg) < seq_len:
            continue
        gaps_ok = np.diff(tg) == thirty_min if len(tg) > 1 else np.array([], dtype=bool)
        for i in range(len(Xg) - seq_len + 1):
            yi = yg[i + seq_len - 1]
            if np.isnan(yi):
                continue
            if seq_len > 1 and not gaps_ok[i : i + seq_len - 1].all():
                continue
            X_seq.append(Xg[i : i + seq_len])
            y_seq.append(yi)
            flat_idx.append(idx[i + seq_len - 1])

    if not X_seq:
        return (np.empty((0, seq_len, X.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=int))
    return np.array(X_seq), np.array(y_seq), np.array(flat_idx)


def train_model(model: nn.Module, loader: DataLoader,
                X_val_t: torch.Tensor, y_val_t: torch.Tensor,
                now_val: np.ndarray, lr: float, patience: int) -> nn.Module:
    """Selection criterion: validation MSE is primary, hit rate is a
    tiebreaker only (hit-rate-primary selection is magnitude-blind and
    collapsed test R² when tried)."""
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_hit_rate    = -1.0
    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    y_val_np         = y_val_t.cpu().numpy()

    for _ in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(Xb), yb).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_pred_t = model(X_val_t)
            val_loss   = criterion(val_pred_t, y_val_t).item()
            val_pred   = val_pred_t.cpu().numpy()

        val_hit_rate = hit_rate(now_val, val_pred, y_val_np)
        improved = (val_loss < best_val_loss) or (
            val_loss == best_val_loss and val_hit_rate > best_hit_rate
        )

        if improved:
            best_val_loss, best_hit_rate = val_loss, val_hit_rate
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


def train_lstm(X_train_base, y_train_full, X_val_base, y_val_full, train_days, val_days,
              train_times, val_times, seq_len, now_val_full):
    X_train_seq, y_train_seq, _            = make_sequences(X_train_base, y_train_full, train_days, train_times, seq_len)
    X_val_seq,   y_val_seq,   val_flat_idx  = make_sequences(X_val_base,   y_val_full,   val_days,   val_times,   seq_len)

    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
        return None

    now_val_seq = now_val_full[val_flat_idx]

    n_features = X_train_base.shape[1]
    lstm = LSTMModel(n_features, HIDDEN_SIZE, NUM_LAYERS, LSTM_DROPOUT).to(DEVICE)

    ds_lstm  = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                             torch.tensor(y_train_seq, dtype=torch.float32))
    ldr_lstm = DataLoader(ds_lstm, batch_size=BATCH_SIZE, shuffle=True)

    X_val_seq_t = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
    y_val_seq_t = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)

    return train_model(lstm, ldr_lstm, X_val_seq_t, y_val_seq_t, now_val_seq, LSTM_LEARNING_RATE, LSTM_PATIENCE)


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

print("Loading splits...")
train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")
val   = pd.read_parquet(VAL_PATH,   engine="pyarrow")
test  = pd.read_parquet(TEST_PATH,  engine="pyarrow")

train = train[train["QUOTE_READTIME"].dt.time.between(WINDOW_START, WINDOW_END)]
val   = val[val["QUOTE_READTIME"].dt.time.between(WINDOW_START, WINDOW_END)]
test  = test[test["QUOTE_READTIME"].dt.time.between(WINDOW_START, WINDOW_END)]
print(f"  Train : {len(train):,}   Val : {len(val):,}   Test : {len(test):,}  (anchors restricted to 10:00–15:30)")
print(f"  Device: {DEVICE}")

# Contract-day grouping key.
train_days = (train["EXPIRE_DATE"].astype(str) + "_" + train["STRIKE"].astype(str) + "_" + train["QUOTE_DATE"].astype(str)).values
val_days   = (val["EXPIRE_DATE"].astype(str)   + "_" + val["STRIKE"].astype(str)   + "_" + val["QUOTE_DATE"].astype(str)).values
test_days  = (test["EXPIRE_DATE"].astype(str)  + "_" + test["STRIKE"].astype(str)  + "_" + test["QUOTE_DATE"].astype(str)).values

train_times = train["QUOTE_READTIME"].values.astype("datetime64[m]")
val_times   = val["QUOTE_READTIME"].values.astype("datetime64[m]")
test_times  = test["QUOTE_READTIME"].values.astype("datetime64[m]")

# Base features for the LSTM: scaled once, globally (no NaNs by construction).
base_scaler  = StandardScaler()
X_train_base = base_scaler.fit_transform(train[BASE_FEATURE_COLS]).astype(np.float32)
X_val_base   = base_scaler.transform(val[BASE_FEATURE_COLS]).astype(np.float32)
X_test_base  = base_scaler.transform(test[BASE_FEATURE_COLS]).astype(np.float32)

# ---------------------------------------------------------------------------
# Train and evaluate — OLS + MLP + LSTM + Stacking, per combination and type
#
# Trained in three separate passes (rather than one interleaved per-combo
# loop) so that each model's sequence of random draws matches the conditions
# its hyperparameters were tuned under (one seed, then all 20 combos with the
# RNG state drifting from that model's own draws only). An interleaved loop would have MLP's training consume RNG
# state before LSTM initializes on every combo, silently perturbing LSTM's
# weight init/shuffling away from the conditions it was calibrated under.
# ---------------------------------------------------------------------------

print(f"\n{'Obs':>5} {'Fcast':>6} {'Type':<6} {'Model':<6} {'R²':>9} {'MSE':>10} {'MAE':>10} "
     f"{'HitRate':>8} {'R²_OOS_RW':>10} {'VegaSharpe':>11}")
print("-" * 96)

all_metrics = {}   # (obs_bars, fcast_bars, label, model) -> dict
regime_rows = []    # per-row records for DTE / VIX regime tables
pred_rows   = []    # for the saved predictions CSV
sharpe_rows = []    # vega-Sharpe sensitivity per cell (Table 6)

mlp_store  = {}   # (obs_bars, fcast_bars, label) -> ols/mlp fit artifacts
lstm_store = {}   # (obs_bars, fcast_bars, label) -> trained LSTM

# --- Pass 1/3: OLS + MLP -----------

for obs_bars, fcast_bars in COMBINATIONS:
    lag_cols = (
        [f"C_IV_LAG_{k}" for k in range(1, obs_bars)] +
        [f"P_IV_LAG_{k}" for k in range(1, obs_bars)]
    )
    mlp_feature_cols = BASE_FEATURE_COLS + lag_cols

    for prefix, label in OPT_TYPES.items():
        target = f"{prefix}_IV_FWD_{fcast_bars}"

        needed  = mlp_feature_cols + [target]
        train_c = train.dropna(subset=needed)
        val_c   = val.dropna(subset=needed)

        if train_c.empty or val_c.empty:
            continue

        y_train_c   = train_c[target].values.astype(np.float32)
        y_val_c     = val_c[target].values.astype(np.float32)
        now_val_c   = val_c[f"{prefix}_IV"].values.astype(np.float32)

        ols_scaler  = StandardScaler().fit(train_c[mlp_feature_cols])
        ols = LinearRegression().fit(ols_scaler.transform(train_c[mlp_feature_cols]), y_train_c)

        mlp_scaler  = StandardScaler().fit(train_c[mlp_feature_cols])
        X_train_mlp = mlp_scaler.transform(train_c[mlp_feature_cols])
        X_val_mlp   = mlp_scaler.transform(val_c[mlp_feature_cols])

        n_features = len(mlp_feature_cols)
        mlp = MLP(n_features, MLP_HIDDEN_DIMS, 1, MLP_DROPOUT).to(DEVICE)
        ds_mlp  = TensorDataset(torch.tensor(X_train_mlp, dtype=torch.float32),
                                torch.tensor(y_train_c, dtype=torch.float32))
        ldr_mlp = DataLoader(ds_mlp, batch_size=BATCH_SIZE, shuffle=True)
        X_val_mlp_t = torch.tensor(X_val_mlp, dtype=torch.float32).to(DEVICE)
        y_val_mlp_t = torch.tensor(y_val_c, dtype=torch.float32).to(DEVICE)
        mlp = train_model(mlp, ldr_mlp, X_val_mlp_t, y_val_mlp_t, now_val_c, MLP_LEARNING_RATE, MLP_PATIENCE)

        mlp_store[(obs_bars, fcast_bars, label)] = {
            "ols": ols, "ols_scaler": ols_scaler,
            "mlp": mlp, "mlp_scaler": mlp_scaler,
            "mlp_feature_cols": mlp_feature_cols,
        }

# --- Pass 2/3: LSTM ----------------

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

for obs_bars, fcast_bars in COMBINATIONS:
    for prefix, label in OPT_TYPES.items():
        target = f"{prefix}_IV_FWD_{fcast_bars}"

        y_train_full = train[target].values.astype(np.float32)
        y_val_full   = val[target].values.astype(np.float32)
        now_val_full = val[f"{prefix}_IV"].values.astype(np.float32)

        lstm = train_lstm(X_train_base, y_train_full, X_val_base, y_val_full,
                          train_days, val_days, train_times, val_times, seq_len=obs_bars,
                          now_val_full=now_val_full)
        if lstm is None:
            continue

        lstm_store[(obs_bars, fcast_bars, label)] = lstm

# --- Pass 3/3: Stacking + evaluation (deterministic — no RNG draws) --------

for obs_bars, fcast_bars in COMBINATIONS:
    for prefix, label in OPT_TYPES.items():
        key = (obs_bars, fcast_bars, label)

        if key not in mlp_store:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid rows'}")
            continue
        if key not in lstm_store:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid LSTM sequences'}")
            continue

        artifacts        = mlp_store[key]
        ols              = artifacts["ols"]
        ols_scaler       = artifacts["ols_scaler"]
        mlp              = artifacts["mlp"]
        mlp_scaler       = artifacts["mlp_scaler"]
        mlp_feature_cols = artifacts["mlp_feature_cols"]
        lstm             = lstm_store[key]

        target = f"{prefix}_IV_FWD_{fcast_bars}"

        y_val_full   = val[target].values.astype(np.float32)
        now_val_full = val[f"{prefix}_IV"].values.astype(np.float32)

        X_val_seq, y_val_seq, val_flat_idx = make_sequences(X_val_base, y_val_full, val_days, val_times, obs_bars)

        if len(val_flat_idx) == 0:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid LSTM sequences'}")
            continue

        # --- Stacking meta-model on validation meta-features ---

        mlp_val_pred  = predict(mlp, mlp_scaler.transform(val.iloc[val_flat_idx][mlp_feature_cols]))
        lstm_val_pred = predict(lstm, X_val_seq)
        y_val_aligned = y_val_seq
        meta_X_val    = np.column_stack([mlp_val_pred, lstm_val_pred])

        meta_valid_mask            = ~np.isnan(meta_X_val).any(axis=1)
        meta_X_val, y_val_aligned  = meta_X_val[meta_valid_mask], y_val_aligned[meta_valid_mask]

        if len(meta_X_val) == 0:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid meta-feature rows'}")
            continue

        meta_model = Ridge(alpha=RIDGE_ALPHA).fit(meta_X_val, y_val_aligned)

        # --- Test set: LSTM's sequence positions are the common alignment ---

        y_test_full = test[target].values.astype(np.float32)
        X_test_seq, y_test_seq, test_flat_idx = make_sequences(X_test_base, y_test_full, test_days, test_times, obs_bars)

        if len(test_flat_idx) == 0:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid test sequences'}")
            continue

        test_feat_rows = test.iloc[test_flat_idx]
        ols_test_pred  = ols.predict(ols_scaler.transform(test_feat_rows[mlp_feature_cols]))
        mlp_test_pred  = predict(mlp, mlp_scaler.transform(test_feat_rows[mlp_feature_cols]))
        lstm_test_pred = predict(lstm, X_test_seq)
        meta_X_test    = np.column_stack([mlp_test_pred, lstm_test_pred])

        now_test_full    = test_feat_rows[f"{prefix}_IV"].values
        vega_test_full   = test_feat_rows[f"{prefix}_VEGA"].values
        mid_test_full    = test_feat_rows[f"{prefix}_MID"].values
        spread_test_full = test_feat_rows[f"{prefix}_SPREAD_REL"].values
        dte_test_full    = test_feat_rows["DTE"].values
        vix_test_full    = test_feat_rows["VIX"].values

        test_valid_mask = (
            ~np.isnan(ols_test_pred) & ~np.isnan(meta_X_test).any(axis=1)
        )
        if test_valid_mask.sum() == 0:
            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {'—':<6} {'skipped: no valid test meta-feature rows'}")
            continue

        ols_test_pred  = ols_test_pred[test_valid_mask]
        mlp_test_pred  = mlp_test_pred[test_valid_mask]
        lstm_test_pred = lstm_test_pred[test_valid_mask]
        meta_X_test    = meta_X_test[test_valid_mask]
        y_test_masked  = y_test_seq[test_valid_mask]
        now_test       = now_test_full[test_valid_mask]
        vega_test      = vega_test_full[test_valid_mask]
        mid_test       = mid_test_full[test_valid_mask]
        spread_test    = spread_test_full[test_valid_mask]
        dte_test       = dte_test_full[test_valid_mask]
        vix_test       = vix_test_full[test_valid_mask]

        stack_test_pred = meta_model.predict(meta_X_test)

        preds_by_model = {
            "OLS": ols_test_pred, "MLP": mlp_test_pred,
            "LSTM": lstm_test_pred, "Stack": stack_test_pred,
        }

        # --- Metrics + regime rows for each model ----------------------------

        vix_q33, vix_q67 = np.percentile(vix_test, 33.3), np.percentile(vix_test, 66.7)
        dte_buckets = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 7, "6-7")]
        vix_regimes = [
            (vix_test <= vix_q33, "Low"),
            ((vix_test > vix_q33) & (vix_test <= vix_q67), "Mid"),
            (vix_test > vix_q67, "High"),
        ]

        for model_name, y_pred in preds_by_model.items():
            r2  = r2_score(y_test_masked, y_pred)
            mse = mean_squared_error(y_test_masked, y_pred)
            mae = mean_absolute_error(y_test_masked, y_pred)
            econ = compute_metrics(now_test, y_pred, y_test_masked, vega_test, mid_test, spread_test)

            key = (obs_bars, fcast_bars, label, model_name)
            all_metrics[key] = {"r2": r2, "mse": mse, "mae": mae, **econ}

            sharpe_rows.append({
                "obs_bars": obs_bars, "fcast_bars": fcast_bars, "type": label, "model": model_name,
                "vega_sharpe": econ["vega_sharpe"],
                "vega_sharpe_full_spread": vega_sharpe(now_test, y_pred, y_test_masked, vega_test,
                                                       mid_test, 2.0 * spread_test),
                "vega_sharpe_oracle": vega_sharpe(now_test, y_test_masked, y_test_masked, vega_test,
                                                  mid_test, spread_test),
            })

            print(f"  {obs_bars:>3}    {fcast_bars:>4}  {label:<6} {model_name:<6} {r2:>9.4f} {mse:>10.6f} {mae:>10.6f} "
                 f"{econ['hit_rate']:>8.4f} {econ['r2_oos_rw']:>10.4f} {econ['vega_sharpe']:>11.4f}")

            pred_rows.append(pd.DataFrame({
                "obs_bars": obs_bars, "fcast_bars": fcast_bars, "type": label, "model": model_name,
                "y_actual": y_test_masked, "y_pred": y_pred,
            }))

            for lo, hi, bucket_label in dte_buckets:
                mask = (np.floor(dte_test) >= lo) & (np.floor(dte_test) <= hi)
                if mask.sum() < 10:
                    continue
                regime_rows.append({
                    "obs_bars": obs_bars, "fcast_bars": fcast_bars, "type": label, "model": model_name,
                    "regime_kind": "DTE", "regime": bucket_label,
                    "r2_oos_rw": compute_metrics(now_test[mask], y_pred[mask], y_test_masked[mask],
                                                 vega_test[mask], mid_test[mask], spread_test[mask])["r2_oos_rw"],
                })

            for mask, regime_label in vix_regimes:
                if mask.sum() < 10:
                    continue
                regime_rows.append({
                    "obs_bars": obs_bars, "fcast_bars": fcast_bars, "type": label, "model": model_name,
                    "regime_kind": "VIX", "regime": regime_label,
                    "r2_oos_rw": compute_metrics(now_test[mask], y_pred[mask], y_test_masked[mask],
                                                 vega_test[mask], mid_test[mask], spread_test[mask])["r2_oos_rw"],
                })

# ---------------------------------------------------------------------------
# Table 1 — Primary metrics
# ---------------------------------------------------------------------------

W = 96
_metric_rows = []
for (obs_bars, fcast_bars, label, model_name), m in all_metrics.items():
    _metric_rows.append({
        "obs_bars": obs_bars, "fcast_bars": fcast_bars, "type": label, "model": model_name,
        "r2": round(m["r2"], 4), "mse": round(m["mse"], 6), "mae": round(m["mae"], 6),
        "hit_rate": round(m["hit_rate"], 4), "r2_oos_rw": round(m["r2_oos_rw"], 4),
        "vega_sharpe": round(m["vega_sharpe"], 4),
    })
table1_df = pd.DataFrame(_metric_rows)

print(f"\n{'=' * W}")
print("TABLE 1 — Primary Metrics (test set 2023)")
print(f"{'=' * W}")
print(table1_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Table 4 — Best configurations by R²_OOS_RW (quick lookup, Table 1 is in
# grid order and doesn't surface this on its own)
# ---------------------------------------------------------------------------

table4_df = table1_df.sort_values("r2_oos_rw", ascending=False).head(15).reset_index(drop=True)

print(f"\n{'=' * W}")
print("TABLE 4 — Top 15 Configurations by R²_OOS_RW")
print(f"{'=' * W}")
print(table4_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Table 5 — Model ranking: mean R²_OOS_RW / VegaSharpe / HitRate across all
# obs_bars/fcast_bars/type combinations — answers "which model wins overall".
# ---------------------------------------------------------------------------

table5_df = (
    table1_df.groupby("model")[["r2_oos_rw", "vega_sharpe", "hit_rate"]]
    .mean()
    .round(4)
    .sort_values("r2_oos_rw", ascending=False)
    .reset_index()
)

print(f"\n{'=' * W}")
print("TABLE 5 — Model Ranking (mean across all combinations)")
print(f"{'=' * W}")
print(table5_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Table 2 / 3 — Regime R²_OOS_RW by DTE bucket / VIX tercile
# ---------------------------------------------------------------------------

regime_df = pd.DataFrame(regime_rows)
table2_df = regime_df[regime_df["regime_kind"] == "DTE"].drop(columns="regime_kind").reset_index(drop=True)
table3_df = regime_df[regime_df["regime_kind"] == "VIX"].drop(columns="regime_kind").reset_index(drop=True)

print(f"\n{'=' * W}")
print("TABLE 2 — R²_OOS_RW by DTE Bucket")
print(f"{'=' * W}")
print(table2_df.to_string(index=False))

print(f"\n{'=' * W}")
print("TABLE 3 — R²_OOS_RW by VIX Regime (terciles computed per combination)")
print(f"{'=' * W}")
print(table3_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

fname = "evaluation_report.txt"

# Table 6 / 7 — vega-Sharpe sensitivity per cell and per model.
table6_df = pd.DataFrame(sharpe_rows).round(4)
table7_df = (
    pd.DataFrame(sharpe_rows)
    .assign(positive_half=lambda d: d["vega_sharpe"] > 0,
            positive_full=lambda d: d["vega_sharpe_full_spread"] > 0)
    .groupby("model", sort=False)
    .agg(mean_half_spread=("vega_sharpe", "mean"),
         mean_full_spread=("vega_sharpe_full_spread", "mean"),
         mean_oracle=("vega_sharpe_oracle", "mean"),
         positive_half=("positive_half", "sum"),
         positive_full=("positive_full", "sum"),
         cells=("vega_sharpe", "size"))
    .round(4)
    .reset_index()
)

content = "\n".join([
    f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "Models  : OLS | MLP | LSTM | Stacking  [Rolling Window / IV]",
    "",
    "Hyperparameters",
    "---------------",
    f"  MLP_HIDDEN_DIMS   : {MLP_HIDDEN_DIMS}",
    f"  MLP_DROPOUT       : {MLP_DROPOUT}",
    f"  MLP_LEARNING_RATE : {MLP_LEARNING_RATE}",
    "  SEQ_LEN           : = obs_bars (per combination)",
    f"  HIDDEN_SIZE       : {HIDDEN_SIZE}",
    f"  NUM_LAYERS        : {NUM_LAYERS}",
    f"  LSTM_DROPOUT      : {LSTM_DROPOUT}",
    f"  LSTM_LEARNING_RATE: {LSTM_LEARNING_RATE}",
    f"  BATCH_SIZE        : {BATCH_SIZE}",
    f"  MLP_PATIENCE      : {MLP_PATIENCE}",
    f"  LSTM_PATIENCE     : {LSTM_PATIENCE}",
    f"  RIDGE_ALPHA       : {RIDGE_ALPHA}",
    "",
    "TABLE 1 — Primary Metrics (test set 2023)",
    "-" * 70,
    table1_df.to_string(index=False),
    "",
    "TABLE 4 — Top 15 Configurations by R²_OOS_RW",
    "-" * 70,
    table4_df.to_string(index=False),
    "",
    "TABLE 5 — Model Ranking (mean across all combinations)",
    "-" * 70,
    table5_df.to_string(index=False),
    "",
    "TABLE 2 — R²_OOS_RW by DTE Bucket",
    "-" * 70,
    table2_df.to_string(index=False),
    "",
    "TABLE 3 — R²_OOS_RW by VIX Regime (terciles per combination)",
    "-" * 70,
    table3_df.to_string(index=False),
    "",
    "TABLE 6 — Vega-Sharpe sensitivity per cell",
    "  vega_sharpe              : half spread on entry (headline, as Table 1)",
    "  vega_sharpe_full_spread  : full spread (round trip)",
    "  vega_sharpe_oracle       : perfect-foresight forecast (pred = actual)",
    "-" * 70,
    table6_df.to_string(index=False),
    "",
    "TABLE 7 — Vega-Sharpe summary per model (positive_* = cells > 0 out of 20)",
    "-" * 70,
    table7_df.to_string(index=False),
])

os.makedirs(REPORT_DIR, exist_ok=True)
with open(os.path.join(REPORT_DIR, fname), "w", encoding="utf-8") as fh:
    fh.write(content)
print(f"\nSaved → {os.path.join(REPORT_DIR, fname)}")

preds_df  = pd.concat(pred_rows, ignore_index=True)
preds_csv = os.path.join(REPORT_DIR, fname.replace(".txt", "_preds.csv"))
preds_df.to_csv(preds_csv, index=False)
print(f"Preds  → {preds_csv}")
