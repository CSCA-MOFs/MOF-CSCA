"""
CACA MOF training + scatter plots.

- Trains GradientBoostingRegressor for each property over N splits.
- For each split, averages predictions from multiple model seeds.
- Saves per-property plotting artifacts:
    {prop}_plot_points_dedup.csv  (mof_id, y, yhat, cls)
    {prop}_plot_meta.json         (rp2_concat, mae_concat, rmse_concat, counts
"""

from __future__ import annotations
import os, re, json, argparse, math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib as mpl
import matplotlib.pyplot as plt

BASE_FONT   = 22
EDGE        = "k"
EDGE_W      = 0.55
REG_LINE    = "#d62728"
MARKER_SIZE = 70
ALPHA       = 0.95

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

ORDER   = ["c_a","c_b","c_c","c_d","c_e","c_f","c_g","c_h","c_all","unknown"]
PALETTE = {
    "c_a":"#FFB000",
    "c_b":"#6A4C93",
    "c_c":"#0C7BDC",
    "c_d":"#2A9D8F",
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

AXIS_OVERRIDES = {
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CA-only models and save publication-style scatter artifacts."
    )
    p.add_argument("--features_csv", required=True, type=str,
                   help="Path to features CSV (index = MOF id; columns = features).")
    p.add_argument("--label_dir", required=True, type=str,
                   help="Directory containing label files (xlsx/csv).")
    p.add_argument("--prop_map_json", default=None, type=str,
                   help="Optional JSON file mapping property -> filename inside label_dir.")
    p.add_argument("--properties", required=True, type=str,
                   help="Comma-separated list of property names (must match columns in label files).")
    p.add_argument("--panel_props", default="", type=str,
                   help="Comma-separated list of 4 properties to assemble into a 2x2 panel.")
    p.add_argument("--id_col_features", default="mof_id", type=str,
                   help="(Optional) name to use for feature index column in saved artifacts.")
    p.add_argument("--id_col_labels", default="MOFRefcodes", type=str,
                   help="ID column name in label files.")
    p.add_argument("--strip_suffix", default=".xyz", type=str,
                   help="Suffix to strip from feature index (e.g., .xyz). Empty to disable.")
    p.add_argument("--n_splits", default=10, type=int,
                   help="Number of data splits (train/val/test).")
    p.add_argument("--data_seed_start", default=23, type=int,
                   help="First seed used to make splits; uses range [start, start+n_splits).")
    p.add_argument("--seed_start", default=13, type=int,
                   help="First model seed for ensembling within a split.")
    p.add_argument("--seed_count", default=10, type=int,
                   help="Number of model seeds per split.")
    p.add_argument("--n_jobs", default=0, type=int,
                   help="Parallel workers for model seeds (0 -> use CPU-1).")
    p.add_argument("--out_dir", default=None, type=str,
                   help="Output directory (default: <features_dir>/ca_training).")
    p.add_argument("--n_estimators", default=10000, type=int)
    p.add_argument("--learning_rate", default=0.005, type=float)
    p.add_argument("--subsample", default=0.5, type=float)
    p.add_argument("--max_depth", default=7, type=int)
    p.add_argument("--max_features", default="sqrt", type=str)
    p.add_argument("--min_samples_split", default=2, type=int)
    p.add_argument("--min_samples_leaf", default=1, type=int)
    return p.parse_args()

def _canon(s: str) -> str:
    """Lowercase alnum key for alias-style lookups."""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _strip_suffix(s: str, suffix: str) -> str:
    s = str(s).strip()
    if suffix and s.lower().endswith(suffix.lower()):
        return s[: -len(suffix)]
    return s

def _pearson_r2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0 or np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    r = np.corrcoef(a, b)[0, 1]
    return float(r * r)

def _load_prop_map(label_dir: str, prop_map_json: Optional[str]) -> Dict[str, str]:

    if prop_map_json:
        with open(prop_map_json, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        return {str(k): str(v) for k, v in m.items()}

    files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    norm = {f: _canon(os.path.splitext(f)[0]) for f in files}
    rev = {}
    for fn, cn in norm.items():
        rev.setdefault(cn, fn)
    return rev  

def _resolve_label_path(prop: str, label_dir: str, prop_map: Dict[str, str]) -> str:
    """
    Resolve label file path for a property.
    - If prop_map has exact key prop -> filename, use it.
    - Else try canonicalized key.
    - Else raise.
    """
    if prop in prop_map:
        fn = prop_map[prop]
    else:
        key = _canon(prop)
        cand = None
        for k, v in prop_map.items():
            if _canon(k) == key:
                cand = v
                break
        if cand is None:
            for k, v in prop_map.items():
                if _canon(os.path.splitext(k)[0]) == key:
                    cand = k
                    break
        if cand is None:
            raise SystemExit(f"Could not find label file for property '{prop}' in {label_dir}. "
                             f"Provide --prop_map_json or place a file named similarly.")
        fn = cand
    path = os.path.join(label_dir, fn)
    if not os.path.isfile(path):
        raise SystemExit(f"Label file not found: {path}")
    return path

def _read_labels(path: str, id_col: str, value_col: str) -> pd.DataFrame:
    """Reads labels from .xlsx or .csv and returns df indexed by id_col with a single column value_col."""
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_table(path, sep=None, engine="python")
    if id_col not in df.columns or value_col not in df.columns:
        raise SystemExit(f"Expected columns '{id_col}' and '{value_col}' in {path}. "
                         f"Found: {list(df.columns)}")
    out = df[[id_col, value_col]].dropna()
    out[id_col] = out[id_col].astype(str).str.strip()
    return out.set_index(id_col)

def _auto_category_from_feature_counts(dfX: pd.DataFrame) -> pd.Series:
    """
    Estimate a dominant category label per row using columns C0_n_points..C7_n_points (if present),
    mapping to c_a..c_h. Fallback: 'unknown'.
    """
    cols = [f"C{k}_n_points" for k in range(8)]
    present = [c for c in cols if c in dfX.columns]
    if not present:
        return pd.Series(["unknown"] * len(dfX), index=dfX.index)
    arr = dfX[present].to_numpy(float)
    idx = np.argmax(arr, axis=1)
    sums = arr.sum(axis=1)
    c_map = {f"C{j}": lab for j, lab in enumerate(["c_a","c_b","c_c","c_d","c_e","c_f","c_g","c_h"])}
    labs = []
    for r, j in enumerate(idx):
        if sums[r] <= 0:
            labs.append("unknown")
        else:
            key = present[j].split("_")[0]  # e.g. "C3"
            labs.append(c_map.get(key, "unknown"))
    return pd.Series(labs, index=dfX.index)



_GLOBAL = {}
def _init_worker(Xtr, ytr, Xte, params):
    _GLOBAL["Xtr"] = Xtr; _GLOBAL["ytr"] = ytr; _GLOBAL["Xte"] = Xte; _GLOBAL["params"] = params

def _fit_predict_one_seed(seed: int) -> np.ndarray:
    p = _GLOBAL["params"]
    model = GradientBoostingRegressor(
        random_state=seed,
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        max_depth=p["max_depth"],
        max_features=p["max_features"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
    )
    model.fit(_GLOBAL["Xtr"], _GLOBAL["ytr"])
    return model.predict(_GLOBAL["Xte"])


def train_property(
    dfX: pd.DataFrame,
    labels_path: str,
    prop: str,
    id_col_labels: str,
    strip_suffix: str,
    n_splits: int,
    data_seed_start: int,
    seed_start: int,
    seed_count: int,
    n_jobs: int,
    gbdt_params: dict,
    out_dir: str
) -> Dict[str, str]:

    dfY = _read_labels(labels_path, id_col_labels, prop)
    dfX2 = dfX.copy()
    if strip_suffix:
        dfX2.index = dfX2.index.map(lambda s: _strip_suffix(s, strip_suffix))

    df = dfX2.join(dfY, how="inner")
    if df.empty:
        raise SystemExit(f"No overlap between features and labels for {prop} "
                         f"(features={dfX2.shape[0]}, labels={dfY.shape[0]}).")

    auto_class = _auto_category_from_feature_counts(dfX2).reindex(df.index)

    X_all = df.drop(columns=[prop]).to_numpy(float)
    y_all = df[prop].to_numpy(float)
    idx_all = df.index.to_numpy()

    X_all = StandardScaler().fit_transform(X_all)

    if n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    model_seeds = list(range(seed_start, seed_start + seed_count))
    y_true_all, y_pred_all, cls_all, ids_all = [], [], [], []

    pbar = tqdm(total=n_splits * seed_count, desc=f"Training {prop}")
    for i in range(n_splits):
        dseed = data_seed_start + i

        X_tr, X_tmp, y_tr, y_tmp, idx_tr, idx_tmp = train_test_split(
            X_all, y_all, idx_all, test_size=0.20, random_state=dseed
        )
        _, X_te, _, y_te, _, idx_te = train_test_split(
            X_tmp, y_tmp, idx_tmp, test_size=0.50, random_state=dseed
        )

        pred_sum = np.zeros_like(y_te, dtype=float)
        with ProcessPoolExecutor(max_workers=n_jobs,
                                 initializer=_init_worker,
                                 initargs=(X_tr, y_tr, X_te, gbdt_params)) as ex:
            futs = {ex.submit(_fit_predict_one_seed, s): s for s in model_seeds}
            for fut in as_completed(futs):
                pred_sum += fut.result()
                pbar.update(1)

        y_pred_avg = pred_sum / float(seed_count)
        y_true_all.append(y_te)
        y_pred_all.append(y_pred_avg)
        cls_all.append(auto_class.loc[idx_te])
        ids_all.append(pd.Index(idx_te, name="mof_id"))

    pbar.close()

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    cls_all    = pd.concat(cls_all)
    ids_all    = pd.Index(np.concatenate([idx.values for idx in ids_all]), name="mof_id")

    rp2  = _pearson_r2(y_true_all, y_pred_all)
    mae  = float(mean_absolute_error(y_true_all, y_pred_all))
    rmse = float(math.sqrt(mean_squared_error(y_true_all, y_pred_all)))

    # Save artifacts
    df_points = pd.DataFrame({
        "mof_id": ids_all.astype(str),
        "y": y_true_all.astype(float),
        "yhat": y_pred_all.astype(float),
        "cls": cls_all.astype(str).values
    })

    def _mode_or_first(s: pd.Series) -> str:
        m = s.mode()
        return str(m.iat[0]) if len(m) else str(s.iloc[0])

    df_dedup = (
        df_points
        .groupby("mof_id", as_index=False)
        .agg(y=("y","first"), yhat=("yhat","mean"), cls=("cls", _mode_or_first))
    )

    points_csv = os.path.join(plots_dir, f"{prop}_plot_points_dedup.csv")
    df_dedup.to_csv(points_csv, index=False)

    meta_json = os.path.join(plots_dir, f"{prop}_plot_meta.json")
    with open(meta_json, "w", encoding="utf-8") as fh:
        json.dump({
            "rp2_concat": rp2,
            "mae_concat": mae,
            "rmse_concat": rmse,
            "n_points_concat": int(df_points.shape[0]),
            "n_points_dedup": int(df_dedup.shape[0])
        }, fh, indent=2)

    return {"points_csv": points_csv, "meta_json": meta_json}


def _box(ax, lw=0.9):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True); ax.spines[s].set_linewidth(lw)

def _sci(ax):
    ax.ticklabel_format(style="sci", scilimits=(0,0), useMathText=True, axis="both")



def _compute_limits(y, yhat, ov: dict | None, pad_frac=0.03):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    x_lo, x_hi = np.nanpercentile(y,    [1, 99]); x_pad = (x_hi - x_lo) * pad_frac if x_hi > x_lo else 1.0
    y_lo, y_hi = np.nanpercentile(yhat, [1, 99]); y_pad = (y_hi - y_lo) * pad_frac if y_hi > y_lo else 1.0
    xlim = (x_lo - x_pad, x_hi + x_pad); ylim = (y_lo - y_pad, y_hi + y_pad)
    if ov:
        if "xleft" in ov: xlim = (float(ov["xleft"]), xlim[1])
        if "xmax"  in ov: xlim = (xlim[0], float(ov["xmax"]))
        if "yleft" in ov: ylim = (float(ov["yleft"]), ylim[1])
        if "ymax"  in ov: ylim = (ylim[0], float(ov["ymax"]))
    return xlim, ylim



def _fit_line(ax, x, y, num=400):
    a, b = np.polyfit(np.asarray(x,float), np.asarray(y,float), 1)
    x0 = np.linspace(*ax.get_xlim(), num=num)
    return x0, a*x0 + b



def _legend_handles():
    hs = []
    for lab in ORDER:
        if lab == "unknown": continue
        hs.append(mpl.lines.Line2D([0],[0], marker="o", linestyle="",
            markerfacecolor=PALETTE.get(lab, "#7A7A7A"), markeredgecolor=EDGE,
            markersize=9.5, label=MATH.get(lab, lab)))
    return hs

def _scatter_by_category(ax, df, s=MARKER_SIZE):
    counts = df["cls"].value_counts()
    order = sorted(counts.index.tolist(), key=lambda k: counts.get(k,0), reverse=True)
    for lab in order:
        sub = df[df["cls"] == lab]
        ax.scatter(sub["y"], sub["yhat"], s=s, alpha=ALPHA,
                   c=PALETTE.get(lab, "#7A7A7A"),
                   edgecolors=EDGE, linewidths=EDGE_W, marker="o", zorder=2)




def plot_four_panel(plots_dir: str, panel_props: List[str], out_png: str):
    if len(panel_props) != 4:
        raise SystemExit("--panel_props must contain exactly 4 property names.")

    fig, axs = plt.subplots(2, 2, figsize=(18.0, 18.0), constrained_layout=False)
    letters = ["a","b","c","d"]

    for i, prop in enumerate(panel_props):
        pts_path = os.path.join(plots_dir, f"{prop}_plot_points_dedup.csv")
        meta_path = os.path.join(plots_dir, f"{prop}_plot_meta.json")
        if not (os.path.isfile(pts_path) and os.path.isfile(meta_path)):
            raise SystemExit(f"Missing artifacts for {prop}: {pts_path} / {meta_path}")

        df = pd.read_csv(pts_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        ax = axs.flat[i]
        xlim, ylim = _compute_limits(df["y"], df["yhat"], AXIS_OVERRIDES.get(prop, {}))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.autoscale(False)

        _scatter_by_category(ax, df, s=MARKER_SIZE)
        x0, y0 = _fit_line(ax, df["y"], df["yhat"])
        ax.plot(x0, y0, color=REG_LINE, linewidth=2.0, zorder=3)

        rp2 = meta.get("rp2_concat", float("nan"))
        mae = meta.get("mae_concat", float("nan"))
        ax.text(0.02, 0.975, rf"$r_p^2$ = {rp2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FONT)
        ax.text(0.02, 0.89,  f"MAE = {mae:.2e}",     transform=ax.transAxes, ha="left", va="top", fontsize=BASE_FONT)

        _sci(ax)
        ax.text(0.50, -0.10, f"({letters[i]})", transform=ax.transAxes,
                ha="center", va="top", fontsize=BASE_FONT)
        ax.set_xlabel(""); ax.set_ylabel(""); _box(ax)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.985,
                        bottom=0.3, wspace=0.15, hspace=0.2)
    fig.supxlabel("True Values", y=0.14)
    fig.supylabel("Predicted Values", x=0.035, y=0.66, va="center")

    leg = fig.legend(
        handles=_legend_handles(), loc="lower center", ncol=9, frameon=True,
        bbox_to_anchor=(0.5, 0.18), borderaxespad=0.0,
        handletextpad=0.8, columnspacing=1.1, labelspacing=0.8
    )
    leg.get_frame().set_alpha(1.0); leg.get_frame().set_facecolor("white")

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)



def main():
    a = parse_args()
    props = [p.strip() for p in a.properties.split(",") if p.strip()]
    if not props:
        raise SystemExit("No properties provided via --properties.")

    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.features_csv)), "ca_training")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isfile(a.features_csv):
        raise SystemExit(f"Features CSV not found: {a.features_csv}")
    dfX = pd.read_csv(a.features_csv, index_col=0)
    dfX.index = dfX.index.astype(str).str.strip()

    gbdt_params = dict(
        n_estimators=a.n_estimators, learning_rate=a.learning_rate, subsample=a.subsample,
        max_depth=a.max_depth, max_features=a.max_features,
        min_samples_split=a.min_samples_split, min_samples_leaf=a.min_samples_leaf,
    )

    prop_map = _load_prop_map(a.label_dir, a.prop_map_json)

    for prop in props:
        labels_path = _resolve_label_path(prop, a.label_dir, prop_map)
        _ = train_property(
            dfX=dfX,
            labels_path=labels_path,
            prop=prop,
            id_col_labels=a.id_col_labels,
            strip_suffix=a.strip_suffix,
            n_splits=a.n_splits,
            data_seed_start=a.data_seed_start,
            seed_start=a.seed_start,
            seed_count=a.seed_count,
            n_jobs=a.n_jobs,
            gbdt_params=gbdt_params,
            out_dir=out_dir
        )

    panel_props = [p.strip() for p in a.panel_props.split(",") if p.strip()]
    if panel_props:
        missing = [p for p in panel_props
                   if not os.path.isfile(os.path.join(plots_dir, f"{p}_plot_points_dedup.csv"))]
        if missing:
            print(f"Skipping panel: missing artifacts for {missing}")
        else:
            out_png = os.path.join(plots_dir, "four_props_scatter_panel.png")
            plot_four_panel(plots_dir, panel_props, out_png)
            print(f"Saved panel: {out_png}")

    print(f"Artifacts saved under: {plots_dir}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()
