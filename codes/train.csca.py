from __future__ import annotations
import argparse, os, re, json
import numpy as np, pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from scipy.stats import pearsonr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

PROPERTY_FILES = {
    "HenrysconstantN2": "HenrysconstantN2.xlsx",
    "HenrysconstantO2": "HenrysconstantO2.xlsx",
    "N2uptakemolkg": "N2uptakemolkg.xlsx",
    "O2uptakemolkg": "O2uptakemolkg.xlsx",
}

def parse_args():
    p = argparse.ArgumentParser(description="Train MOF properties with Category-Algebra (CA) features.")
    p.add_argument("--features_csv", type=str, default="data/features.csv")
    p.add_argument("--label_dir", type=str, default="data")
    p.add_argument("--id_col_features", type=str, default="MOFRefcodes")
    p.add_argument("--id_col_labels", type=str, default="MOFRefcodes")
    p.add_argument("--properties", type=str, default="HenrysconstantN2,HenrysconstantO2,N2uptakemolkg,O2uptakemolkg")
    p.add_argument("--n_jobs", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="ca_training")
    p.add_argument("--n_estimators", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=0.005)
    p.add_argument("--subsample", type=float, default=0.5)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--max_features", type=str, default="sqrt")
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    return p.parse_args()

def _pearson_r2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if _HAS_SCIPY:
        return float(pearsonr(a, b)[0] ** 2)
    r = np.corrcoef(a, b)[0, 1]
    return float(r * r)

def _fit_predict_one_seed(model_seed: int, X_tr, y_tr, X_te, gbdt_params) -> np.ndarray:
    model = GradientBoostingRegressor(
        random_state=model_seed,
        n_estimators=gbdt_params["n_estimators"],
        learning_rate=gbdt_params["learning_rate"],
        subsample=gbdt_params["subsample"],
        max_depth=gbdt_params["max_depth"],
        max_features=gbdt_params["max_features"],
        min_samples_split=gbdt_params["min_samples_split"],
        min_samples_leaf=gbdt_params["min_samples_leaf"],
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)

def _norm_cls_series(s: pd.Series) -> pd.Series:
    m = {
        "C_a":"c_a","C_b":"c_b","C_c":"c_c","C_d":"c_d",
        "C_e":"c_e","C_f":"c_f","C_g":"c_g","C_h":"c_h",
        "C_all":"c_all",
        "c_a":"c_a","c_b":"c_b","c_c":"c_c","c_d":"c_d",
        "c_e":"c_e","c_f":"c_f","c_g":"c_g","c_h":"c_h","c_all":"c_all",
    }
    return s.astype(str).map(lambda x: m.get(x.strip(), "unknown"))

def train_property(dfX: pd.DataFrame, labels_excel: str, prop: str,
                   id_col_features: str, id_col_labels: str, n_jobs: int,
                   gbdt_params: dict, cls_series: pd.Series | None,
                   collect_points: bool, points_out_dir: str) -> dict:
    dfY = pd.read_excel(labels_excel, engine="openpyxl" if labels_excel.lower().endswith(".xlsx") else None)
    if prop not in dfY.columns:
        raise SystemExit(f"Column '{prop}' not found in {labels_excel}")
    dfY = dfY[[id_col_labels, prop]].dropna()
    dfY[id_col_labels] = dfY[id_col_labels].astype(str).str.strip()

    if id_col_features in dfX.columns:
        idx = dfX[id_col_features].astype(str).str.strip()
        dfX_use = dfX.drop(columns=[id_col_features]).copy()
        dfX_use.index = idx
    else:
        dfX_use = dfX.copy()
        dfX_use.index = dfX_use.index.astype(str).str.strip()

    df = dfX_use.join(dfY.set_index(id_col_labels), how="inner")
    if df.empty:
        raise SystemExit(f"No overlap between features and labels for {prop}.")

    X_all = df.drop(columns=[prop]).values
    y_all = df[prop].values

    if cls_series is not None:
        cls_all = cls_series.reindex(df.index).fillna("unknown").astype(str).to_numpy()
    else:
        cls_all = np.array(["unknown"] * len(df), dtype=str)

    X_all = StandardScaler().fit_transform(X_all)

    if n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    split_seeds = list(range(23, 33))
    model_seeds = list(range(13, 23))

    rp2_splits, r2_splits, mae_splits, rmse_splits = [], [], [], []
    all_y_true, all_y_pred, all_cls = [], [], []

    for split_seed in tqdm(split_seeds, desc=f"Splits for {prop}"):
        idx = np.arange(len(X_all))
        tr_idx, tmp_idx = train_test_split(idx, test_size=0.20, random_state=split_seed)
        va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=split_seed)
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        pred_sum = np.zeros_like(y_te, dtype=float)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_fit_predict_one_seed, s, X_tr, y_tr, X_te, gbdt_params) for s in model_seeds]
            for fut in as_completed(futs):
                pred_sum += fut.result()
        y_pred = pred_sum / float(len(model_seeds))

        rp2 = _pearson_r2(y_te, y_pred)
        r2  = float(r2_score(y_te, y_pred))
        mae = float(mean_absolute_error(y_te, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

        rp2_splits.append(rp2); r2_splits.append(r2); mae_splits.append(mae); rmse_splits.append(rmse)

        if collect_points:
            all_y_true.append(y_te)
            all_y_pred.append(y_pred)
            all_cls.append(cls_all[te_idx])

    out = {
        "rp2_mean": float(np.mean(rp2_splits)),
        "r2_mean":  float(np.mean(r2_splits)),
        "mae_mean": float(np.mean(mae_splits)),
        "rmse_mean": float(np.mean(rmse_splits)),
    }

    if collect_points:
        y_concat    = np.concatenate(all_y_true, axis=0) if all_y_true else np.array([], float)
        yhat_concat = np.concatenate(all_y_pred, axis=0) if all_y_pred else np.array([], float)
        cls_concat  = np.concatenate(all_cls,     axis=0) if all_cls     else np.array([], str)
        cls_norm = _norm_cls_series(pd.Series(cls_concat))
        pts_df = pd.DataFrame({"y": y_concat, "yhat": yhat_concat, "cls": cls_norm}).drop_duplicates()

        os.makedirs(points_out_dir, exist_ok=True)
        pts_path  = os.path.join(points_out_dir, f"{prop}_plot_points_dedup.csv")
        meta_path = os.path.join(points_out_dir, f"{prop}_plot_meta.json")

        if len(pts_df):
            rp2_concat = _pearson_r2(pts_df["y"].values, pts_df["yhat"].values)
            r2_concat  = float(r2_score(pts_df["y"].values, pts_df["yhat"].values))
            mae_concat = float(mean_absolute_error(pts_df["y"].values, pts_df["yhat"].values))
        else:
            rp2_concat = r2_concat = mae_concat = float("nan")

        pts_df.to_csv(pts_path, index=False)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"rp2_concat": rp2_concat, "r2_concat": r2_concat, "mae_concat": mae_concat}, fh, indent=2)

    return out

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.isfile(args.features_csv):
        raise SystemExit(f"Features CSV not found: {args.features_csv}")
    dfX_raw = pd.read_csv(args.features_csv)

    if args.id_col_features in dfX_raw.columns:
        idx = dfX_raw[args.id_col_features].astype(str).str.strip()
        dfX_raw = dfX_raw.drop(columns=[args.id_col_features])
        dfX_raw.index = idx
    else:
        dfX_raw.index = dfX_raw.index.astype(str).str.strip()

    possible_cls_cols = ["category","cls","class","dominant_category","C_label","category_label","dominant"]
    found_col = next((c for c in possible_cls_cols if c in dfX_raw.columns), None)
    if found_col:
        cls_series = dfX_raw[found_col].astype(str)
        dfX = dfX_raw.drop(columns=[found_col])
    else:
        cls_series = None
        dfX = dfX_raw.copy()

    gbdt_params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    props = [p.strip() for p in re.split(r"[,\s]+", args.properties) if p.strip()]
    summary_rows = []

    for prop in props:
        if prop not in PROPERTY_FILES:
            raise SystemExit(f"Unknown property '{prop}'. Choose from: {list(PROPERTY_FILES.keys())}")
        labels_excel = os.path.join(args.label_dir, PROPERTY_FILES[prop])
        print(f"\nTraining {prop} with 10 splits (23–32) × 10 models (13–22)")
        res = train_property(
            dfX, labels_excel, prop,
            args.id_col_features, args.id_col_labels,
            args.n_jobs, gbdt_params,
            cls_series=cls_series,
            collect_points=True,
            points_out_dir=plots_dir,
        )
        summary_rows.append({"property": prop, **res})
        print(f"→ {prop}: r_p^2={res['rp2_mean']:.4f}  R^2={res['r2_mean']:.4f}  MAE={res['mae_mean']:.3e}  RMSE={res['rmse_mean']:.3e}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary_CA_means.csv"), index=False)

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    main()

