#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


# -----------------------
# Helpers
# -----------------------

def safe_log1p(x: pd.Series) -> pd.Series:
    # works for non-negative numeric; if negatives exist, fallback to log(|x|+1) with sign (rare)
    x = pd.to_numeric(x, errors="coerce")
    neg = x < 0
    out = x.copy()
    out[~neg] = np.log1p(out[~neg])
    out[neg] = np.sign(out[neg]) * np.log1p(np.abs(out[neg]))
    return out


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def spearman(y_true, y_pred) -> float:
    if spearmanr is None:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if r == r else float("nan")


def infer_instance_key(df_feat: pd.DataFrame, df_lab: pd.DataFrame) -> Tuple[str, str]:
    """
    We assume:
      - features file has 'instance_name'
      - labels file has 'instance_folder' that often includes suffix like '-p1'
    We'll create a normalized key: strip trailing '-p#' if present.
    """
    feat_key = "instance_name" if "instance_name" in df_feat.columns else None
    lab_key = "instance_folder" if "instance_folder" in df_lab.columns else None
    if feat_key is None or lab_key is None:
        raise ValueError("Expected columns: features.instance_name and labels.instance_folder")

    def norm_name(s: str) -> str:
        # remove trailing -p1 / -p2 etc if present
        # keep base instance name for matching splits.json (which uses base names)
        return pd.Series(s).str.replace(r"-p\d+$", "", regex=True).iloc[0]

    # create normalized columns
    df_feat["_ins_base"] = df_feat[feat_key].astype(str).str.replace(r"-p\d+$", "", regex=True)
    df_lab["_ins_base"] = df_lab[lab_key].astype(str).str.replace(r"-p\d+$", "", regex=True)

    return "_ins_base", "_ins_base"


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Keep numeric columns only, drop obvious meta/ids/targets.
    """
    drop_like = {
        "instance_name", "filename", "graph_kind",
        "instance_folder", "log_path",
        "_ins_base",
        # labels that might appear after merge:
        "his_median", "his_q25", "his_q75", "his_iqr",
        "att_median", "att_q25", "att_q75", "att_iqr",
        "rep_his_len", "rep_max_attempts", "rep_rank", "rep_cfg_id",
        "n_configs",
    }
    # numeric candidates
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in numeric_cols if c not in drop_like]
    return feats


@dataclass
class TargetSpec:
    name: str
    y_col: str
    log_target: bool


def build_model(model_name: str, random_state: int = 20251228) -> Pipeline:
    """
    Returns a sklearn Pipeline:
      impute -> scale -> model
    """
    if model_name == "ridge":
        model = Ridge(alpha=1.0, random_state=random_state)
    elif model_name == "rf":
        model = RandomForestRegressor(
            n_estimators=800,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", model),
    ])
    return pipe


def run_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: TargetSpec,
    train_names: List[str],
    val_names: List[str],
    model_name: str,
    log_feature_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Train on train_names, evaluate on val_names.
    """
    log_feature_cols = log_feature_cols or []

    train_mask = df["_ins_base"].isin(train_names)
    val_mask = df["_ins_base"].isin(val_names)

    X_train = df.loc[train_mask, feature_cols].copy()
    X_val = df.loc[val_mask, feature_cols].copy()

    # feature log1p
    for c in log_feature_cols:
        if c in X_train.columns:
            X_train[c] = safe_log1p(X_train[c])
            X_val[c] = safe_log1p(X_val[c])

    y_train = df.loc[train_mask, target.y_col].astype(float).copy()
    y_val = df.loc[val_mask, target.y_col].astype(float).copy()

    # target log1p (recommended)
    if target.log_target:
        y_train = safe_log1p(y_train)
        y_val = safe_log1p(y_val)

    # drop missing labels
    tr_ok = y_train.notna()
    va_ok = y_val.notna()
    X_train = X_train.loc[tr_ok]
    y_train = y_train.loc[tr_ok]
    X_val = X_val.loc[va_ok]
    y_val = y_val.loc[va_ok]

    model = build_model(model_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    return {
        "mae": float(mean_absolute_error(y_val, pred)),
        "rmse": float(rmse(y_val, pred)),
        "spearman": float(spearman(y_val, pred)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True, help="evrp_instance_features_gpt.csv")
    ap.add_argument("--labels", type=str, required=True, help="labels_median_rep.csv")
    ap.add_argument("--splits", type=str, required=True, help="splits.json")
    ap.add_argument("--target_mode", type=str, default="median", choices=["median", "rep"],
                    help="Use median labels or representative labels")
    ap.add_argument("--models", type=str, default="ridge,rf",
                    help="Comma-separated: ridge,rf")
    ap.add_argument("--out", type=str, default="cv_results.csv")
    args = ap.parse_args()

    # load
    df_feat = pd.read_csv(args.features)
    df_lab = pd.read_csv(args.labels)
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # normalize + merge keys
    feat_key, lab_key = infer_instance_key(df_feat, df_lab)
    df = df_feat.merge(df_lab, left_on=feat_key, right_on=lab_key, how="inner", suffixes=("", "_lab"))

    # check test list exists
    test_list = splits.get("test", [])
    if not test_list:
        raise ValueError("splits.json has no 'test' list")

    # features to use
    feature_cols = select_feature_columns(df)

    # suggested log-features (edit as you like)
    log_feature_cols = [
        "N_nodes", "N_total", "num_customers", "num_stations",
        "mst_weight_per_node", "mst_weight",
        "nn_dist_mean", "cust_to_station_nn_mean",
        "total_demand_to_capacity",
        "vehicle_capacity", "battery_capacity", "energy_consumption",
        "max_evals",
    ]
    log_feature_cols = [c for c in log_feature_cols if c in feature_cols]

    # targets
    if args.target_mode == "median":
        targets = [
            TargetSpec("his_len", "his_median", True),
            TargetSpec("max_attempts", "att_median", True),
        ]
    else:
        targets = [
            TargetSpec("his_len", "rep_his_len", True),
            TargetSpec("max_attempts", "rep_max_attempts", True),
        ]

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    # CV folds
    cv = splits["cv"]
    records = []

    for model_name in model_names:
        for t in targets:
            for fold_name, fold in cv.items():
                train_names = fold["train"]
                val_names = fold["val"]

                res = run_fold(
                    df=df,
                    feature_cols=feature_cols,
                    target=t,
                    train_names=train_names,
                    val_names=val_names,
                    model_name=model_name,
                    log_feature_cols=log_feature_cols,
                )
                records.append({
                    "model": model_name,
                    "target": t.name,
                    "fold": fold_name,
                    **res
                })

    out_df = pd.DataFrame(records)
    out_df.to_csv(args.out, index=False)

    # print summary
    print("=== CV Summary (mean ± std) ===")
    for model_name in model_names:
        for t in targets:
            sub = out_df[(out_df["model"] == model_name) & (out_df["target"] == t.name)]
            mae_mean, mae_std = sub["mae"].mean(), sub["mae"].std()
            rmse_mean, rmse_std = sub["rmse"].mean(), sub["rmse"].std()
            sp_mean, sp_std = sub["spearman"].mean(), sub["spearman"].std()
            print(f"{model_name:5s} | {t.name:12s} | MAE {mae_mean:.4f}±{mae_std:.4f} | "
                  f"RMSE {rmse_mean:.4f}±{rmse_std:.4f} | Spearman {sp_mean:.3f}±{sp_std:.3f}")

    print(f"\n[OK] Wrote per-fold results to: {args.out}")
    print(f"Features used: {len(feature_cols)} (log1p on {len(log_feature_cols)})")


if __name__ == "__main__":
    main()

