from __future__ import annotations
import argparse, os, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from scipy.stats import pearsonr
    _HASC = True
except Exception:
    _HASC = False

PROP_FILES = {
    "HenrysconstantN2":"HenrysconstantN2.xlsx",
    "HenrysconstantO2":"HenrysconstantO2.xlsx",
    "N2uptakemolkg":"N2uptakemolkg.xlsx",
    "O2uptakemolkg":"O2uptakemolkg.xlsx",
}
ALIASES = {"henryconstantn2":"HenrysconstantN2"}
SPLIT_SEEDS = list(range(23,33))   
MODEL_SEEDS = list(range(13,23))   

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--properties", default="all")
    p.add_argument("--id_col_features", default="mof_id")
    p.add_argument("--id_col_labels", default="MOFRefcodes")
    p.add_argument("--out_dir", default="ca_training")
    p.add_argument("--n_estimators", type=int, default=10000)
    p.add_argument("--learning_rate", type=float, default=0.005)
    p.add_argument("--subsample", type=float, default=0.5)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--max_features", default="sqrt")
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    return p.parse_args()

def canon(prop:str)->str:
    return ALIASES.get(re.sub(r"[^a-z0-9]","",prop.lower()), prop)

def label_path(prop,label_dir):
    f = PROP_FILES.get(canon(prop))
    if not f: raise SystemExit(f"Unknown property: {prop}")
    return os.path.join(label_dir, f)

def rp2(a,b):
    a = np.asarray(a,float); b = np.asarray(b,float)
    if a.size==0: return float("nan")
    if _HASC: return float(pearsonr(a,b)[0]**2)
    r = np.corrcoef(a,b)[0,1]; return float(r*r)

def fit_one(seed, Xtr, ytr, Xte, hp):
    m = GradientBoostingRegressor(
        random_state=seed,
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        max_depth=hp["max_depth"],
        max_features=hp["max_features"],
        min_samples_split=hp["min_samples_split"],
        min_samples_leaf=hp["min_samples_leaf"],
    )
    m.fit(Xtr,ytr); return m.predict(Xte)

def train_prop(dfX, labels_xlsx, prop, id_col_labels, hp):
    dfY = pd.read_excel(labels_xlsx)[[id_col_labels, prop]].dropna()
    dfY[id_col_labels] = dfY[id_col_labels].astype(str).str.strip()
    df = dfX.join(dfY.set_index(id_col_labels), how="inner")
    if df.empty: raise SystemExit(f"No overlap for {prop}")
    X = StandardScaler().fit_transform(df.drop(columns=[prop]).values)
    y = df[prop].values

    rps, r2s, maes = [], [], []
    for s in SPLIT_SEEDS:
        idx = np.arange(len(X))
        tr, tmp = train_test_split(idx, test_size=0.20, random_state=s)
        va, te  = train_test_split(tmp, test_size=0.50, random_state=s)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        pred = np.zeros_like(yte, float)
        for ms in MODEL_SEEDS:
            pred += fit_one(ms, Xtr, ytr, Xte, hp)
        pred /= float(len(MODEL_SEEDS))
        rps.append(rp2(yte, pred))
        r2s.append(float(r2_score(yte, pred)))
        maes.append(float(mean_absolute_error(yte, pred)))
    return {"rp2_mean": float(np.mean(rps)),
            "r2_mean":  float(np.mean(r2s)),
            "mae_mean": float(np.mean(maes))}

def main():
    a = args()
    os.makedirs(a.out_dir, exist_ok=True)
    dfX = pd.read_csv(a.features_csv, index_col=0)
    dfX.index = dfX.index.astype(str).str.strip()
    if a.id_col_features in dfX.columns:
        dfX = dfX.set_index(a.id_col_features)
        dfX.index = dfX.index.astype(str).str.strip()
    props = list(PROP_FILES.keys()) if a.properties.strip().lower()=="all" \
            else [ALIASES.get(p.strip().lower(), p.strip()) for p in a.properties.split(",")]
    hp = dict(n_estimators=a.n_estimators, learning_rate=a.learning_rate, subsample=a.subsample,
              max_depth=a.max_depth, max_features=a.max_features,
              min_samples_split=a.min_samples_split, min_samples_leaf=a.min_samples_leaf)
    rows = []
    for prop in props:
        res = train_prop(dfX, label_path(prop, a.label_dir), prop, a.id_col_labels, hp)
        rows.append({"property":prop, **res})
        print(f"{prop}: r_p^2={res['rp2_mean']:.4f}  R^2={res['r2_mean']:.4f}  MAE={res['mae_mean']:.3e}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir,"summary_CA_only_means.csv"), index=False)
    print("Saved →", os.path.join(a.out_dir,"summary_CA_only_means.csv"))

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
    main()
