"""
Master Thesis — Predicting Short-Term Option Returns Using Machine Learning
Author  : Stanislaw Jarzyna
Step 1  : Build Parquet Datasets
Purpose : Concatenate OptionsDX TXT files into Parquet format for all four
          datasets used in the thesis:
            1. SPY 30-minute option chain  (primary modelling dataset)
            2. SPY end-of-day option chain
            3. SPX end-of-day option chain
            4. VIX end-of-day option chain

          Files are written incrementally (one period at a time) to avoid
          loading the full dataset into memory simultaneously.
          Compression: Snappy.
"""

import os
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Dataset registry
# Each entry defines one source→output mapping.
# ---------------------------------------------------------------------------

# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("OPTIONSDX_DIR", os.path.join(ROOT, "data"))

DATASETS = [
    {
        "name"     : "SPY 30-minute",
        "txt_dir"  : f"{BASE}/30 minutes/SPY Option Chain/txt",
        "out_path" : f"{BASE}/30 minutes/SPY Option Chain/parquet/spy_30min_option_chain.parquet",
    },
    {
        "name"     : "SPY End-of-Day",
        "txt_dir"  : f"{BASE}/End of Day/SPY Option Chains/txt",
        "out_path" : f"{BASE}/End of Day/SPY Option Chains/parquet/spy_eod.parquet",
    },
    {
        "name"     : "SPX End-of-Day",
        "txt_dir"  : f"{BASE}/End of Day/SPX Option Chain/txt",
        "out_path" : f"{BASE}/End of Day/SPX Option Chain/parquet/spx_eod.parquet",
    },
    {
        "name"     : "VIX End-of-Day",
        "txt_dir"  : f"{BASE}/End of Day/VIX Option Chains/txt",
        "out_path" : f"{BASE}/End of Day/VIX Option Chains/parquet/vix_eod.parquet",
    },
]

# ---------------------------------------------------------------------------
# Columns that should be numeric (raw files contain mixed-type entries)
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "QUOTE_UNIXTIME", "QUOTE_TIME_HOURS", "UNDERLYING_LAST", "EXPIRE_UNIX", "DTE",
    "C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV", "C_VOLUME",
    "C_LAST", "C_BID", "C_ASK", "STRIKE",
    "P_BID", "P_ASK", "P_LAST", "P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA",
    "P_RHO", "P_IV", "P_VOLUME", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT",
]


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_parquet(name: str, txt_dir: str, out_path: str) -> None:
    files = sorted(glob.glob(os.path.join(txt_dir, "**", "*.txt"), recursive=True))

    print(f"\n{'=' * 60}")
    print(f"  Dataset : {name}")
    print(f"  Files   : {len(files)}")
    print(f"  Output  : {out_path}")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer     = None
    total_rows = 0

    for i, filepath in enumerate(files):
        df = pd.read_csv(filepath, sep=",", header=0, low_memory=False)

        # Strip brackets and whitespace from column names: " [C_BID] " -> "C_BID"
        df.columns = [col.strip().strip("[]") for col in df.columns]

        # Coerce numeric columns (some cells contain non-numeric placeholders)
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")

        writer.write_table(table)
        total_rows += len(df)

        print(f"  [{i + 1:>2}/{len(files)}]  {os.path.basename(filepath):<35}"
              f"  rows: {len(df):>10,}   cumulative: {total_rows:>12,}")

    if writer:
        writer.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Done — {total_rows:,} rows | {size_mb:,.1f} MB\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for dataset in DATASETS:
        build_parquet(**dataset)

    print("All datasets built successfully.")