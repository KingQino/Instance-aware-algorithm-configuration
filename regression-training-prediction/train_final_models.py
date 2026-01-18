#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def safe_log1p_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    neg = s < 0
    out = s.copy()
    out[~neg] = np.log1p(out[~neg])
    out[neg] = np.sign(out[neg]) * np.log1p(np.abs(out[neg]))
    return out


def rmse(y_true, y_pred) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def add_base_key(df: pd.DataFrame, col: str, outcol: str = "_ins_base") -> pd.DataFrame:
    df[outcol] = df[col].astype(str).str.replace(r"-p\d+$", "", regex=True)
    return df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_like = {
        "instance_name", "filename", "graph_kind",
        "instance_folder", "log_path",
        "_ins_base",
        # labels:
        "his_median", "his_q25", "his_q75", "his_iqr",
        "att_median", "att_q25", "att_q75", "att_iqr",
        "rep_his_len", "rep_max_attempts", "rep_rank", "rep_cfg_id",
        "n_configs",
    }
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in numeric_cols if c not in drop_like]
    return feats


def build_ridge(alpha: float = 1.0, random_state: int = 20251228) -> Pipeline:
    # NOTE: Ridge supports random_state only for 'sag'/'saga' solvers; default solver ignores it.
    model = Ridge(alpha=alpha)
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", model),
    ])
    return pipe


def export_ridge_coeffs(pipe: Pipeline, feature_cols: List[str], out_csv: str) -> None:
    """
    Export coefficients in standardized feature space:
      y = intercept + sum_i coef_i * z_i
    where z_i are standardized features after imputer+scaler.
    """
    ridge = pipe.named_steps["ridge"]
    scaler = pipe.named_steps["scaler"]

    coefs = ridge.coef_.ravel()
    intercept = float(ridge.intercept_)

    dfc = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    # also export scaler stats for reproducibility / interpretation
    # (mean_ and scale_ correspond to post-imputation feature distribution)
    df_scaler = pd.DataFrame({
        "feature": feature_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    })

    # write two tables into one file by merging
    out = dfc.merge(df_scaler, on="feature", how="left")
    out["intercept"] = intercept  # repeated, but convenient
    out.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True)
    ap.add_argument("--out_prefix", type=str, default="final_ridge")
    ap.add_argument("--alpha_his", type=float, default=1.0)
    ap.add_argument("--alpha_att", type=float, default=1.0)
    ap.add_argument("--target_mode", type=str, default="median", choices=["median", "rep"])
    ap.add_argument("--log_y", action="store_true", help="Apply log1p to targets (recommended).")
    ap.add_argument("--log_x", action="store_true", help="Apply log1p to selected scale features.")
    args = ap.parse_args()

    df_feat = pd.read_csv(args.features)
    df_lab = pd.read_csv(args.labels)

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # keys
    if "instance_name" not in df_feat.columns:
        raise ValueError("features file must contain 'instance_name'")
    if "instance_folder" not in df_lab.columns:
        raise ValueError("labels file must contain 'instance_folder'")

    df_feat = add_base_key(df_feat, "instance_name", "_ins_base")
    df_lab = add_base_key(df_lab, "instance_folder", "_ins_base")

    # merge
    df = df_feat.merge(df_lab, on="_ins_base", how="inner", suffixes=("", "_lab"))

    # define targets
    if args.target_mode == "median":
        y_h_col = "his_median"
        y_a_col = "att_median"
    else:
        y_h_col = "rep_his_len"
        y_a_col = "rep_max_attempts"

    # feature columns
    feature_cols = select_feature_columns(df)

    # optional log features (same philosophy as CV)
    log_feature_cols = [
        "N_nodes", "N_total", "num_customers", "num_stations",
        "mst_weight_per_node", "mst_weight",
        "nn_dist_mean", "cust_to_station_nn_mean",
        "total_demand_to_capacity",
        "vehicle_capacity", "battery_capacity", "energy_consumption",
        "max_evals",
    ]
    log_feature_cols = [c for c in log_feature_cols if c in feature_cols]

    # split: train = everything not in test
    test_list = splits.get("test", [])
    if not test_list:
        raise ValueError("splits.json must contain a non-empty 'test' list")

    # normalize test names to base key format (strip -p# if present)
    test_base = pd.Series(test_list).astype(str).str.replace(r"-p\d+$", "", regex=True).tolist()

    train_df = df[~df["_ins_base"].isin(test_base)].copy()
    test_df = df[df["_ins_base"].isin(test_base)].copy()

    # save which instances used
    train_df[["_ins_base"]].drop_duplicates().sort_values("_ins_base").to_csv(
        f"{args.out_prefix}_training_instances_used.csv", index=False
    )
    test_df[["_ins_base"]].drop_duplicates().sort_values("_ins_base").to_csv(
        f"{args.out_prefix}_test_instances_heldout.csv", index=False
    )

    # X/Y
    X_train = train_df[feature_cols].copy()
    if args.log_x:
        for c in log_feature_cols:
            X_train[c] = safe_log1p_series(X_train[c])

    y_h = train_df[y_h_col].astype(float)
    y_a = train_df[y_a_col].astype(float)
    if args.log_y:
        y_h = safe_log1p_series(y_h)
        y_a = safe_log1p_series(y_a)

    # drop missing labels
    ok_h = y_h.notna()
    ok_a = y_a.notna()

    # Train his_len model
    pipe_h = build_ridge(alpha=args.alpha_his)
    pipe_h.fit(X_train.loc[ok_h], y_h.loc[ok_h])

    # Train max_attempts model
    pipe_a = build_ridge(alpha=args.alpha_att)
    pipe_a.fit(X_train.loc[ok_a], y_a.loc[ok_a])

    # export models
    joblib.dump(pipe_h, f"{args.out_prefix}_his_len.joblib")
    joblib.dump(pipe_a, f"{args.out_prefix}_max_attempts.joblib")

    # export coefficients
    export_ridge_coeffs(pipe_h, feature_cols, f"{args.out_prefix}_coeffs_his_len.csv")
    export_ridge_coeffs(pipe_a, feature_cols, f"{args.out_prefix}_coeffs_max_attempts.csv")

    # sanity: training fit metrics (NOT a generalization metric)
    pred_h = pipe_h.predict(X_train.loc[ok_h])
    pred_a = pipe_a.predict(X_train.loc[ok_a])

    fit_metrics = pd.DataFrame([
        {"target": "his_len", "n": int(ok_h.sum()),
         "train_mae": float(mean_absolute_error(y_h.loc[ok_h], pred_h)),
         "train_rmse": float(rmse(y_h.loc[ok_h], pred_h))},
        {"target": "max_attempts", "n": int(ok_a.sum()),
         "train_mae": float(mean_absolute_error(y_a.loc[ok_a], pred_a)),
         "train_rmse": float(rmse(y_a.loc[ok_a], pred_a))}
    ])
    fit_metrics.to_csv(f"{args.out_prefix}_train_fit_metrics.csv", index=False)

    print("[OK] Trained final models on TRAIN only (test held out).")
    print(f"[OK] Saved: {args.out_prefix}_his_len.joblib")
    print(f"[OK] Saved: {args.out_prefix}_max_attempts.joblib")
    print(f"[OK] Saved: {args.out_prefix}_coeffs_his_len.csv")
    print(f"[OK] Saved: {args.out_prefix}_coeffs_max_attempts.csv")
    print(f"[OK] Saved: {args.out_prefix}_train_fit_metrics.csv")
    print(f"Train instances: {train_df['_ins_base'].nunique()} | Test instances held out: {test_df['_ins_base'].nunique()}")
    print(f"Features used: {len(feature_cols)} | log_x={'ON' if args.log_x else 'OFF'} | log_y={'ON' if args.log_y else 'OFF'}")


if __name__ == "__main__":
    main()

