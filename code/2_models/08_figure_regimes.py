"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Figure: mean R2_OOS vs random walk by DTE bucket (A) and VIX
          tercile (B), per model, across all 20 cells.
Input   : results/evaluation_report.txt
Output  : results/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from plot_style import MODEL_COLORS, apply_thesis_style, save_figure

apply_thesis_style()

import os
# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_REPORT = f"{ROOT}/results/evaluation_report.txt"
text = open(EVAL_REPORT, encoding="utf-8").read()


def parse_table(text, start_marker, end_marker):
    block = text.split(start_marker, 1)[1]
    if end_marker:
        block = block.split(end_marker, 1)[0]
    lines = [l for l in block.splitlines() if l.strip() and not l.strip().startswith("-") and not l.strip().startswith("=")]
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
        if c not in ("type", "model", "regime"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


dte_df = parse_table(text, "TABLE 2", "TABLE 3")
vix_df = parse_table(text, "TABLE 3", None)

MODEL_ORDER = ["OLS", "MLP", "LSTM", "Stack"]
MODEL_LABELS = {"OLS": "OLS", "MLP": "MLP", "LSTM": "LSTM", "Stack": "Stacking"}
MODEL_COLOR_MAP = {"OLS": MODEL_COLORS["OLS"], "MLP": MODEL_COLORS["MLP"],
                   "LSTM": MODEL_COLORS["LSTM"], "Stack": MODEL_COLORS["Stacking"]}
DTE_ORDER = ["1-2", "3-5", "6-7"]
VIX_ORDER = ["Low", "Mid", "High"]

dte_means = dte_df.groupby(["model", "regime"])["r2_oos_rw"].mean().unstack()[DTE_ORDER]
vix_means = vix_df.groupby(["model", "regime"])["r2_oos_rw"].mean().unstack()[VIX_ORDER]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7, 5.6), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.6}
)


from matplotlib.patches import Patch

# Regimes must be distinguishable on their own, not just via alpha shading
# (which reads poorly in print/grayscale and is easy to misjudge by eye).
# Each regime gets its own hatch pattern on top of the alpha shading, and a
# dark edge so adjacent bars don't visually merge. Value labels are added
# above/below each bar so exact readings don't depend on the y-axis at all.
REGIME_HATCHES = ["", "///", "xxx"]


def grouped_bars(ax, means_df, regime_order, title):
    n_regimes = len(regime_order)
    width = 0.8 / n_regimes
    x = np.arange(len(MODEL_ORDER))
    for j, regime in enumerate(regime_order):
        vals = [means_df.loc[m, regime] for m in MODEL_ORDER]
        offset = (j - (n_regimes - 1) / 2) * width
        alpha = 1.0 - 0.22 * j
        bars = ax.bar(x + offset, vals, width, label=regime,
                      color=[MODEL_COLOR_MAP[m] for m in MODEL_ORDER], alpha=alpha,
                      edgecolor="#262626", linewidth=0.7, hatch=REGIME_HATCHES[j])
        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            offset_pts = 2 if val >= 0 else -2
            ax.annotate(f"{val:.2f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, offset_pts), textcoords="offset points",
                        ha="center", va=va, fontsize=6.5, color="#262626")
    ax.axhline(0, color="#262626", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Mean R2_OOS,RW")
    ax.set_title(title, loc="left")


grouped_bars(ax_top, dte_means, DTE_ORDER, "(A) By DTE bucket (days to expiration)")
ax_top.set_ylim(-0.25, 0.32)
grouped_bars(ax_bot, vix_means, VIX_ORDER, "(B) By VIX tercile")
ax_bot.set_ylim(-0.16, 0.30)

# Legend swatches are plain grey with the matching hatch, independent of
# model color, so "what does each regime look like" reads on its own.
# Placed outside the plotting area (right of the axes) so it never overlaps
# a bar or its value label, regardless of which bars happen to be tallest.
legend_elems_top = [Patch(facecolor="#B0B0B0", edgecolor="#262626", linewidth=0.7,
                          alpha=1.0 - 0.22 * j, hatch=REGIME_HATCHES[j], label=r)
                    for j, r in enumerate(DTE_ORDER)]
ax_top.legend(handles=legend_elems_top, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="DTE bucket", fontsize=8)
legend_elems_bot = [Patch(facecolor="#B0B0B0", edgecolor="#262626", linewidth=0.7,
                          alpha=1.0 - 0.22 * j, hatch=REGIME_HATCHES[j], label=r)
                    for j, r in enumerate(VIX_ORDER)]
ax_bot.legend(handles=legend_elems_bot, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="VIX tercile", fontsize=8)

png_path, pdf_path = save_figure(fig, "regimes")
print(f"Saved -> {png_path}")
print(f"Saved -> {pdf_path}")

print("\nDTE bucket means:\n", dte_means.round(4))
print("\nVIX tercile means:\n", vix_means.round(4))
