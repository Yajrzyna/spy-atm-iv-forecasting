"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Figure: vega-scaled Sharpe ratio net of half the bid-ask spread, by
          model: (A) at each model's best combination, call and put;
          (B) all 20 cells per model.
Input   : results/evaluation_report.txt
Output  : results/figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import MODEL_COLORS, apply_thesis_style, save_figure

apply_thesis_style()

import os
# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_REPORT = f"{ROOT}/results/evaluation_report.txt"

text = open(EVAL_REPORT, encoding="utf-8").read()
table1_block = text.split("TABLE 1", 1)[1]
lines = [l for l in table1_block.splitlines() if l.strip() and not l.strip().startswith("-") and not l.strip().startswith("=")]
header_idx = next(i for i, l in enumerate(lines) if l.split()[:2] == ["obs_bars", "fcast_bars"])
header = lines[header_idx].split()
data_lines = []
for l in lines[header_idx + 1:]:
    if l.strip().startswith("TABLE"):
        break
    data_lines.append(l)
rows = [l.split() for l in data_lines]
df = pd.DataFrame(rows, columns=header)
for c in df.columns:
    if c not in ("type", "model"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

MODEL_ORDER = ["OLS", "MLP", "LSTM", "Stack"]
MODEL_LABELS = {"OLS": "OLS", "MLP": "MLP", "LSTM": "LSTM", "Stack": "Stacking"}
MODEL_COLOR_MAP = {"OLS": MODEL_COLORS["OLS"], "MLP": MODEL_COLORS["MLP"],
                   "LSTM": MODEL_COLORS["LSTM"], "Stack": MODEL_COLORS["Stacking"]}

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7, 5.6), gridspec_kw={"height_ratios": [1, 1.1], "hspace": 0.55}
)

# ---------------------------------------------------------------------------
# Panel A -- vega-scaled Sharpe ratio at each model's own R2_OOS,RW-best combo
# ---------------------------------------------------------------------------

x = np.arange(len(MODEL_ORDER))
width = 0.35

call_vals, put_vals = [], []
for m in MODEL_ORDER:
    sub = df[df["model"] == m]
    call_best = sub[sub["type"] == "Call"].loc[sub[sub["type"] == "Call"]["r2_oos_rw"].idxmax()]
    put_best = sub[sub["type"] == "Put"].loc[sub[sub["type"] == "Put"]["r2_oos_rw"].idxmax()]
    call_vals.append(call_best["vega_sharpe"])
    put_vals.append(put_best["vega_sharpe"])


# Call/Put must not depend on subtle alpha alone -- it needs to be legible on
# its own even if the color mapping (model) is stripped away (grayscale
# print, colorblind readers). Put bars get a diagonal hatch on top of the
# lighter fill; Call bars stay solid. The legend swatches are plain grey
# (not tied to any single model's color) so they read as "this is what
# Call/Put look like", independent of which model's bar you're looking at.
bars_call = ax_top.bar(x - width / 2, call_vals, width, label="Call",
                       color=[MODEL_COLOR_MAP[m] for m in MODEL_ORDER], alpha=1.0,
                       edgecolor="#262626", linewidth=0.6)
bars_put = ax_top.bar(x + width / 2, put_vals, width, label="Put",
                      color=[MODEL_COLOR_MAP[m] for m in MODEL_ORDER], alpha=0.55,
                      edgecolor="#262626", linewidth=0.6, hatch="///")

ax_top.axhline(0, color="#262626", linewidth=1)
ax_top.set_xticks(x)
ax_top.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax_top.set_ylabel("Vega-scaled Sharpe ratio")
ax_top.set_ylim(0, max(call_vals + put_vals) * 1.2)
ax_top.set_title("(A) At each model's own R2_OOS,RW-best combination", loc="left")

from matplotlib.patches import Patch
type_legend = [
    Patch(facecolor="#B0B0B0", edgecolor="#262626", linewidth=0.6, label="Call"),
    Patch(facecolor="#B0B0B0", edgecolor="#262626", linewidth=0.6, hatch="///", alpha=0.55, label="Put"),
]
ax_top.legend(handles=type_legend, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Option type")

for bar, val in zip(list(bars_call) + list(bars_put), call_vals + put_vals):
    # All values are positive -- label each bar just above its top.
    ax_top.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color="#262626")

# ---------------------------------------------------------------------------
# Panel B -- full-grid distribution (all 20 combos x call/put per model)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
TYPE_MARKER = {"Call": "o", "Put": "^"}
for i, m in enumerate(MODEL_ORDER):
    for opt_type, marker in TYPE_MARKER.items():
        vals = df[(df["model"] == m) & (df["type"] == opt_type)]["vega_sharpe"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_bot.scatter(np.full(len(vals), i) + jitter, vals, color=MODEL_COLOR_MAP[m],
                       marker=marker, s=24, alpha=0.75, edgecolor="white", linewidth=0.5)
    vals_all = df[df["model"] == m]["vega_sharpe"].values
    ax_bot.scatter([i], [vals_all.mean()], color=MODEL_COLOR_MAP[m], marker="D", s=55,
                   edgecolor="#262626", linewidth=1, zorder=5)

ax_bot.axhline(0, color="#262626", linewidth=1)
ax_bot.set_xticks(range(len(MODEL_ORDER)))
ax_bot.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax_bot.set_ylabel("Vega-scaled Sharpe ratio")
ax_bot.set_title("(B) Full grid (20 combinations x call/put per model, diamond = mean)", loc="left")

marker_legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#B0B0B0",
           markeredgecolor="white", markersize=6, label="Call"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#B0B0B0",
           markeredgecolor="white", markersize=6, label="Put"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="#B0B0B0",
           markeredgecolor="#262626", markersize=6, label="Model mean"),
]
ax_bot.legend(handles=marker_legend, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Option type")

png_path, pdf_path = save_figure(fig, "vega_sharpe")
print(f"Saved -> {png_path}")
print(f"Saved -> {pdf_path}")

print("\nPositive cells:", int((df["vega_sharpe"] > 0).sum()), "of", len(df))
print("Range across entire grid:", df["vega_sharpe"].min(), "to", df["vega_sharpe"].max())
