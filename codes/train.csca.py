"""
Train CSCAmodels (GBRT) saves
- {prop}_plot_points_dedup.csv  (mof_id, y, yhat, cls)
- {prop}_plot_meta.json         (R2_concat, MAE_concat, RMSE_concat, rp2_concat, counts)

- n_splits data splits (default 10; seeds start at --data_seed_start).
- For each split, average predictions from --seed_count model seeds
  (starting at --seed_start, default 13..22).
"""

from __future__ import annotations
import os, re, json, math, argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _strip_suffix(s: str, suffix: str) -> str:
    s = str(s).strip()
    return s[:-len(suffix)] if suffix and s.lower().endswith(suffix.lower()) else s

def _pearson_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0 or yhat.size == 0 or np.isclose(y.std(), 0.0) or np.isclose(yhat.std(), 0.0):
        return float("nan")
    r = np.corrcoef(y, yhat)[0, 1]
    return float(r * r)

def _coefficient_of_determination(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    if y.size == 0:
        return float("nan")
    y_mean = float(np.mean(y))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def _load_prop_map(label_dir: str, prop_map_json: Optional[str]) -> Dict[str, str]:
    if prop_map_json:
        with open(prop_map_json, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        return {str(k): str(v) for k, v in m.items()}
    files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]
    rev: Dict[str, str] = {}
    for f in files:
        key = _canon(os.path.splitext(f)[0])
        rev.setdefault(key, f)
    return rev

def _resolve_label_path(prop: str, label_dir: str, prop_map: Dict[str, str]) -> str:
    if prop in prop_map:
        fn = prop_map[prop]
    else:
        key = _canon(prop)
        fn = None
        for k, v in prop_map.items():
            if _canon(k) == key:
                fn = v; break
        if fn is None:
            for k, v in prop_map.items():
                if _canon(os.path.splitext(k)[0]) == key:
                    fn = v; break
        if fn is None:
            raise SystemExit(f"Could not find label file for property '{prop}' in {label_dir}.")
    path = os.path.join(label_dir, fn)
    if not os.path.isfile(path):
        raise SystemExit(f"Label file not found: {path}")
    return path

def _read_labels(path: str, id_col: str, value_col: str) -> pd.DataFrame:
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_table(path, sep=None, engine="python")
    if id_col not in df.columns or value_col not in df.columns:
        raise SystemExit(f"Expected columns '{id_col}' and '{value_col}' in {path}. Found: {list(df.columns)}")
    out = df[[id_col, value_col]].dropna()
    out[id_col] = out[id_col].astype(str).str.strip()
    return out.set_index(id_col)

def _auto_category_from_feature_counts(dfX: pd.DataFrame) -> pd.Series:
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
            labs.append(c_map.get(present[j].split("_")[0], "unknown"))
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
) -> dict:

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

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

    # Metrics 
    R2   = _coefficient_of_determination(y_true_all, y_pred_all)
    rp2  = _pearson_r2(y_true_all, y_pred_all)
    mae  = float(mean_absolute_error(y_true_all, y_pred_all))
    rmse = float(math.sqrt(mean_squared_error(y_true_all, y_pred_all)))

    # Save artifacts (dedup by mof_id; mean yhat across repeats; mode class)
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
            "R2_concat": R2,
            "rp2_concat": rp2,
            "MAE_concat": mae,
            "RMSE_concat": rmse,
            "n_points_concat": int(df_points.shape[0]),
            "n_points_dedup": int(df_dedup.shape[0]),
        }, fh, indent=2)

    return {"points_csv": points_csv, "meta_json": meta_json}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CA-only models and save artifacts for re-plotting.")
    p.add_argument("--features_csv", required=True, type=str,
                   help="Path to features CSV (index = MOF id; columns = features).")
    p.add_argument("--label_dir", required=True, type=str,
                   help="Directory containing label files (xlsx/csv).")
    p.add_argument("--out_dir", required=True, type=str,
                   help="Output directory where artifacts will be saved (under out_dir/plots).")
    p.add_argument("--prop_map_json", type=str, default=None,
                   help="Optional JSON mapping: property name -> filename inside label_dir.")
    p.add_argument("--properties", type=str, required=True,
                   help="Comma-separated property names to train.")
    p.add_argument("--id_col_labels", type=str, default="MOFRefcodes")
    p.add_argument("--strip_suffix", type=str, default=".xyz")
    p.add_argument("--n_splits", type=int, default=10)
    p.add_argument("--data_seed_start", type=int, default=23)
    p.add_argument("--seed_start", type=int, default=13)
    p.add_argument("--seed_count", type=int, default=10)
    p.add_argument("--n_jobs", type=int, default=0)
    # GBRT params
    p.add_argument("--n_estimators", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=0.005)
    p.add_argument("--subsample", type=float, default=0.5)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--max_features", type=str, default="sqrt")
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    return p.parse_args()

def main():
    a = parse_args()

    plots_dir = os.path.join(a.out_dir, "plots")
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
    props = [p.strip() for p in a.properties.split(",") if p.strip()]

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
            out_dir=a.out_dir
        )

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()
