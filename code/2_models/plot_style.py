"""
Master Thesis — Predicting Short-Term Option Returns Using Machine Learning
Author  : Stanislaw Jarzyna
Purpose : Shared matplotlib style for every thesis figure (Figure 1, Figure 2,
          and any later ones), so they are visually consistent without each
          figure script repeating its own formatting choices.

Usage   :
    from plot_style import apply_thesis_style, COLORS, save_figure
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 3))
    ...
    save_figure(fig, "figure_1_train_val_test_split")
"""

import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Palette -- colorblind-friendly qualitative set, reused by role across all
# figures (e.g. TRAIN/VAL/TEST always take these three colors; ACCENT marks
# the anchor bar / highlighted element in schematic figures; model-comparison
# figures such as Figure 2 index into MODEL_COLORS in a fixed order).
# ---------------------------------------------------------------------------

COLORS = {
    "train":   "#4C72B0",   # blue
    "val":     "#DD8452",   # orange
    "test":    "#55A868",   # green
    "accent":  "#C44E52",   # red -- anchor bar / highlighted element
    "neutral": "#808080",   # grey -- gridlines, unused/inactive elements
    "text":    "#262626",   # near-black -- all figure text
}

# Fixed color assignment for the four model classes, used consistently in
# any figure comparing OLS / MLP / LSTM / Stacking (e.g. Figure 2).
MODEL_COLORS = {
    "OLS":      "#4C72B0",
    "MLP":      "#DD8452",
    "LSTM":     "#55A868",
    "Stacking": "#C44E52",
}

# Paths are relative to the repository root; set OPTIONSDX_DIR to read the data elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURE_DIR = f"{ROOT}/results/figures"


def apply_thesis_style():
    """Call once at the top of every figure script, before creating axes."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "axes.edgecolor":    COLORS["neutral"],
        "axes.labelcolor":   COLORS["text"],
        "text.color":        COLORS["text"],
        "xtick.color":       COLORS["text"],
        "ytick.color":       COLORS["text"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
    })


def save_figure(fig, name, subfolder=""):
    """Save fig as both PNG (300 DPI, for inserting into Word) and PDF
    (vector, kept alongside for archival/re-editing) into
    FIGURE_DIR/<subfolder>/, creating it if needed.

    `subfolder` follows the numbered-per-figure convention, e.g.
    "1. Train-Val-Test", "2. Vega-Sharpe" -- one folder per thesis figure,
    named after what it shows."""
    out_dir = os.path.join(FIGURE_DIR, subfolder)
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return png_path, pdf_path
