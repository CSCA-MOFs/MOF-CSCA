
"""
2×2 scatter panel.

Reads:
  <plots_dir>/{prop}_plot_points_dedup.csv

Upper-left of each panel shows:
  - R^2  
  - MAE  
"""

from __future__ import annotations
import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 2×2 scatter panel from saved artifacts.")
    p.add_argument("--plots_dir", required=True, type=str,
                   help="Directory containing {prop}_plot_points_dedup.csv and {prop}_plot_meta.json.")
    p.add_argument("--props", type=str,
                   default="HenrysconstantN2,HenrysconstantO2,N2uptakemolkg,O2uptakemolkg",
                   help="Comma-separated list of exactly 4 properties to plot in panel order.")
    p.add_argument("--use_default_overrides", action="store_true",
                   help="If set, apply preset axis limits per property; else autoscale with 1–99% trimming.")
    return p.parse_args()

AXIS_OVERRIDES_PRESET = {
    "HenrysconstantN2": {"xleft": -1e-6, "xmax": 9e-6,  "yleft": -1e-6, "ymax": 0.8e-5},
    "HenrysconstantO2": {"xleft": -1e-6, "xmax": 10e-6, "yleft": -1e-6, "ymax": 1.0e-5},
    "N2uptakemolkg":    {"xleft": -1e-1, "xmax": 9e-1,  "yleft": -1e-1, "ymax": 9.0e-1},
    "O2uptakemolkg":    {"xleft": -1e-1, "xmax": 10e-1, "yleft": -1e-1, "ymax": 1.0e-0},
}

BASE_FONT = 22
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": BASE_FONT,
    "axes.titlesize": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "legend.fontsize": BASE_FONT,
    "xtick.labelsize": BASE_FONT,
    "ytick.labelsize": BASE_FONT,
})
EDGE        = "k"
EDGE_W      = 0.55
REG_LINE    = "#d62728"
MARKER_SIZE = 80
ALPHA       = 0.95

ORDER   = ["c_a","c_b","c_c","c_d","c_e","c_f","c_g","c_h","c_all","unknown"]
PALETTE = {
    "c_a":"#FFB000",
    "c_b":"#6A4C93",
    "c_c":"#0C7BDC",
    "c_d":"#9efbbaff",
    "c_e":"#8C613C",
    "c_f":"#E76F51",
    "c_g":"#3D405B",
    "c_h":"#1D3557",
    "c_all":"#E4572E",
    "unknown":"#7f7f7f",
}
MATH = {
    "c_a":r"$C_a$","c_b":r"$C_b$","c_c":r"$C_c$","c_d":r"$C_d$",
    "c_e":r"$C_e$","c_f":r"$C_f$","c_g":r"$C_g$","c_h":r"$C_h$",
    "c_all":r"$C_{\mathrm{all}}$","unknown":r"$\mathrm{unknown}$",
}

def _box(ax, lw=0.9):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(lw)

def _sci(ax):
    ax.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True, axis='both')

def _load_prop(plots_dir: str, prop: str):
    pts = pd.read_csv(os.path.join(plots_dir, f"{prop}_plot_points_dedup.csv"))
    meta_path = os.path.join(plots_dir, f"{prop}_plot_meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    return pts, meta

def _compute_limits(y, yhat, overrides: dict | None, pad_frac=0.03):
    y  = np.asarray(y, float); yhat = np.asarray(yhat, float)
    x_lo, x_hi = np.nanpercentile(y,    [1, 99]); x_pad = (x_hi - x_lo) * pad_frac if x_hi > x_lo else 1.0
    y_lo, y_hi = np.nanpercentile(yhat, [1, 99]); y_pad = (y_hi - y_lo) * pad_frac if y_hi > y_lo else 1.0
    xlim = (x_lo - x_pad, x_hi + x_pad); ylim = (y_lo - y_pad, y_hi + y_pad)
    if overrides:
        if "xleft" in overrides: xlim = (float(overrides["xleft"]), xlim[1])
        if "xmax"  in overrides: xlim = (xlim[0], float(overrides["xmax"]))
        if "yleft" in overrides: ylim = (float(overrides["yleft"]), ylim[1])
        if "ymax"  in overrides: ylim = (ylim[0], float(overrides["ymax"]))
    return xlim, ylim

def _fit_line(ax, x, y, num=400):
    a, b = np.polyfit(np.asarray(x,float), np.asarray(y,float), 1)
    x0 = np.linspace(*ax.get_xlim(), num=num)
    return x0, a*x0 + b

def _legend_handles():
    hs = []
    for lab in ORDER:
        if lab == "unknown": continue
        hs.append(mpl.lines.Line2D(
            [0],[0], marker="o", linestyle='',
            markerfacecolor=PALETTE[lab], markeredgecolor=EDGE,
            markersize=9.5, label=MATH[lab]
        ))
    return hs

def _scatter_by_category(ax, df, s=MARKER_SIZE):
    counts = df["cls"].value_counts()
    order = sorted(counts.index.tolist(), key=lambda k: counts.get(k, 0), reverse=True)
    for lab in order:
        sub = df[df["cls"] == lab]
        ax.scatter(sub["y"], sub["yhat"], s=s, alpha=ALPHA,
                   c=PALETTE.get(lab, "#7A7A7A"),
                   edgecolors=EDGE, linewidths=EDGE_W, marker="o", zorder=2)
    ax.set_aspect('auto')

def _R2_from_points(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0:
        return float("nan")
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def _MAE_from_points(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y)))

def plot_panel(plots_dir: str, props: list[str], use_default_overrides: bool):
    if len(props) != 4:
        raise SystemExit("--props must specify exactly 4 properties (comma-separated).")

    fig, axs = plt.subplots(2, 2, figsize=(18.0, 18.0), constrained_layout=False)
    letters = ['a','b','c','d']

    for i, prop in enumerate(props):
        pts_path = os.path.join(plots_dir, f"{prop}_plot_points_dedup.csv")
        if not os.path.isfile(pts_path):
            raise SystemExit(f"Missing points file for {prop}: {pts_path}")

        df, _meta = _load_prop(plots_dir, prop)
        ov = AXIS_OVERRIDES_PRESET.get(prop, {}) if use_default_overrides else {}

        ax = axs.flat[i]
        xlim, ylim = _compute_limits(df["y"], df["yhat"], ov)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.autoscale(False)

        _scatter_by_category(ax, df, s=MARKER_SIZE)
        x0, y0 = _fit_line(ax, df["y"], df["yhat"])
        ax.plot(x0, y0, color=REG_LINE, linewidth=2.0, zorder=3)

        R2  = _R2_from_points(df["y"].to_numpy(), df["yhat"].to_numpy())
        MAE = _MAE_from_points(df["y"].to_numpy(), df["yhat"].to_numpy())
        ax.text(0.02, 0.975, rf"$R^2$ = {R2:.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FONT)
        ax.text(0.02, 0.89,  f"MAE = {MAE:.2e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FONT)

        _sci(ax)
        ax.text(0.50, -0.10, f"({letters[i]})", transform=ax.transAxes,
                ha='center', va='top', fontsize=BASE_FONT)
        ax.set_xlabel(""); ax.set_ylabel(""); _box(ax)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.985,
                        bottom=0.3, wspace=0.15, hspace=0.2)

    xl = fig.supxlabel("True Values", y=0.14); xl.set_zorder(10)
    fig.supylabel("Predicted Values", x=0.035, y=0.66, va="center")

    leg = fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=9, frameon=True,
        bbox_to_anchor=(0.5, 0.18),
        borderaxespad=0.0,
        handletextpad=0.8, columnspacing=1.1, labelspacing=0.8
    )
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_facecolor("white")

    out_png = os.path.join(plots_dir, "four_props_scatter_panel_replot.png")
    fig.savefig(out_png, bbox_inches="tight")
    plt.show(); plt.close(fig)

def main():
    a = parse_args()
    props = [p.strip() for p in a.props.split(",") if p.strip()]
    plot_panel(a.plots_dir, props, a.use_default_overrides)

if __name__ == "__main__":
    main()

