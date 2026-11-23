"""
Master Thesis: Short-Horizon Forecasting of ATM Implied Volatility on SPY Options
Author  : Stanislaw Jarzyna
Purpose : Figure: chronological train/validation/test split (A) and how the
          observation window, anchor bar and forecast horizon align within
          the 10:00-15:30 trading day (B).
Output  : results/figures/
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from plot_style import COLORS, apply_thesis_style, save_figure

apply_thesis_style()
# Figure 1 was embedded in the thesis as a low-quality metafile paste;
# regenerate at a higher raster resolution than the shared 300 dpi default
# so the re-embedded PNG holds up when zoomed/printed.
plt.rcParams["savefig.dpi"] = 450

# Panel A -- three shades of blue, dark to light with time.
TRAIN_C, VAL_C, TEST_C = "#08306B", "#4C72B0", "#9ECAE1"

# Panel B -- its own small palette, deliberately not reusing Panel A's colors.
OBS_FCAST_C = "#3B6FA0"   # single blue for both observation and forecast bars
ANCHOR_C    = COLORS["accent"]
UNUSED_C    = COLORS["neutral"]

bar_labels = ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
              "13:00", "13:30", "14:00", "14:30", "15:00", "15:30"]
n_bars = len(bar_labels)
positions = list(range(1, n_bars + 1))

OBS_BARS, FCAST_BARS = 5, 6
ANCHOR = 6  # position 6 = 12:30; observation window = positions 2-6, forecast target = position 12

obs_positions = list(range(ANCHOR - OBS_BARS + 1, ANCHOR + 1))
fcast_positions = list(range(ANCHOR + 1, ANCHOR + FCAST_BARS + 1))

BAR_WIDTH = 0.45  # narrow, non-touching -- reads as a discrete instant, not a span

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7, 3.6), gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.8}
)

# ---------------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------------

periods = [
    ("Train",      datetime(2017, 1, 1), datetime(2021, 12, 31), TRAIN_C, "white"),
    ("Validation", datetime(2022, 1, 1), datetime(2022, 12, 31), VAL_C,   "white"),
    ("Test",       datetime(2023, 1, 1), datetime(2023, 12, 31), TEST_C,  "#08306B"),
]
for label, start, end, color, text_color in periods:
    ax_top.barh(0, end - start, left=start, height=0.6, color=color)
    mid = start + (end - start) / 2
    ax_top.text(mid, 0, label, ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")

ax_top.set_xlim(datetime(2016, 10, 1), datetime(2024, 3, 1))
ax_top.set_ylim(-0.42, 0.5)
ax_top.set_yticks([])
ax_top.xaxis.set_major_locator(mdates.YearLocator())
ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_top.set_title("(A) Chronological train / validation / test split", loc="left")
for spine in ("left", "bottom"):
    ax_top.spines[spine].set_visible(False)
ax_top.tick_params(left=False, length=3, pad=2)

# ---------------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------------

for pos, label in zip(positions, bar_labels):
    if pos in obs_positions and pos != ANCHOR:
        color = OBS_FCAST_C
    elif pos == ANCHOR:
        color = ANCHOR_C
    elif pos in fcast_positions:
        color = OBS_FCAST_C
    else:
        color = UNUSED_C
    alpha = 0.25 if pos not in obs_positions and pos != ANCHOR and pos not in fcast_positions else 1.0
    ax_bot.bar(pos, 1, width=BAR_WIDTH, color=color, alpha=alpha)

ax_bot.set_xticks(positions)
ax_bot.set_xticklabels(bar_labels, fontsize=7.5)
ax_bot.tick_params(axis="x", length=0, pad=6)
ax_bot.set_yticks([])
ax_bot.set_xlim(0.3, n_bars + 0.7)
ax_bot.set_ylim(0, 1.4)
for spine in ("left", "top", "right", "bottom"):
    ax_bot.spines[spine].set_visible(False)

# Category header row, directly above each group -- the only text above the
# bars, so nothing competes with the time labels sitting below them.
ax_bot.text((obs_positions[0] + obs_positions[-2]) / 2, 1.08, "observation",
            ha="center", va="bottom", fontsize=8, color=OBS_FCAST_C, fontweight="bold")
ax_bot.text(ANCHOR, 1.08, "anchor",
            ha="center", va="bottom", fontsize=8, color=ANCHOR_C, fontweight="bold")
ax_bot.text((fcast_positions[0] + fcast_positions[-1]) / 2, 1.08, "forecast horizon",
            ha="center", va="bottom", fontsize=8, color=OBS_FCAST_C, fontweight="bold")

ax_bot.set_title(
    f"(B) Observation/forecast alignment (example: obs_bars={OBS_BARS}, fcast_bars={FCAST_BARS})",
    loc="left",
)

png_path, pdf_path = save_figure(fig, "train_val_test_split")
print(f"Saved -> {png_path}")
print(f"Saved -> {pdf_path}")
