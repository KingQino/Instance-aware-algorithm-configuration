#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np
import pandas as pd
import joblib


def safe_expm1(x):
    return np.expm1(x)


def add_base_key(df, col, outcol="_ins_base"):
    df[outcol] = df[col].astype(str).str.replace(r"-p\d+$", "", regex=True)
    return df


def select_feature_columns(df):
    drop_like = {
        "instance_name", "filename", "graph_kind",
        "instance_folder", "log_path",
        "_ins_base",
        # labels if present
        "his_median", "his_q25", "his_q75", "his_iqr",
        "att_median", "att_q25", "att_q75", "att_iqr",
        "rep_his_len", "rep_max_attempts", "rep_rank",
        "n_configs",
    }
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in numeric_cols if c not in drop_like]
    return feats


def safe_log1p_series(s):
    s = pd.to_numeric(s, errors="coerce")
    neg = s < 0
    out = s.copy()
    out[~neg] = np.log1p(out[~neg])
    out[neg] = np.sign(out[neg]) * np.log1p(np.abs(out[neg]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--model_his", required=True)
    ap.add_argument("--model_att", required=True)
    ap.add_argument("--out", default="predicted_params_test.csv")
    ap.add_argument("--log_x", action="store_true")
    args = ap.parse_args()

    # load
    df_feat = pd.read_csv(args.features)
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if "instance_name" not in df_feat.columns:
        raise ValueError("features file must contain instance_name")

    df_feat = add_base_key(df_feat, "instance_name")

    test_list = splits.get("test", [])
    test_base = pd.Series(test_list).astype(str).str.replace(r"-p\d+$", "", regex=True).tolist()

    test_df = df_feat[df_feat["_ins_base"].isin(test_base)].copy()

    feature_cols = select_feature_columns(test_df)

    # log-transform same features as training
    log_feature_cols = [
        "N_nodes", "N_total", "num_customers", "num_stations",
        "mst_weight_per_node", "mst_weight",
        "nn_dist_mean", "cust_to_station_nn_mean",
        "total_demand_to_capacity",
        "vehicle_capacity", "battery_capacity", "energy_consumption",
        "max_evals",
    ]
    log_feature_cols = [c for c in log_feature_cols if c in feature_cols]

    X = test_df[feature_cols].copy()
    if args.log_x:
        for c in log_feature_cols:
            X[c] = safe_log1p_series(X[c])

    # load models
    model_h = joblib.load(args.model_his)
    model_a = joblib.load(args.model_att)

    # predict (log-space)
    y_h_log = model_h.predict(X)
    y_a_log = model_a.predict(X)

    # back to original scale
    his_pred = safe_expm1(y_h_log)
    att_pred = safe_expm1(y_a_log)

    out = pd.DataFrame({
        "instance_name": test_df["instance_name"].values,
        "his_len_pred": np.round(his_pred).astype(int),
        "max_attempts_pred": np.clip(np.round(att_pred), 1, None).astype(int),
    })

    out.to_csv(args.out, index=False)

    print("[OK] Predicted parameters for TEST instances")
    print(out)
    print(f"[OK] Saved to: {args.out}")


if __name__ == "__main__":
    main()

