#!/usr/bin/env python3
import sys
import json
import random
import pandas as pd

def size_bin(n: int) -> str:
    # Fixed bins for reproducibility
    if n <= 100:
        return "S"
    elif n <= 300:
        return "M"
    else:
        return "L"

def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df["instance name"].astype(str).str[0]
    df["n"] = df["#customers"].astype(int)
    df["size_bin"] = df["n"].apply(size_bin)
    df["maxEvals"] = df["maxEvals"].astype(int)

    # budget_bin: within each family, split by family-median maxEvals
    df["budget_bin"] = "Low"
    for fam, g in df.groupby("family"):
        med = g["maxEvals"].median()
        df.loc[g.index, "budget_bin"] = df.loc[g.index, "maxEvals"].apply(
            lambda x: "Low" if x <= med else "High"
        )

    df["stratum"] = df["family"] + "_" + df["size_bin"] + "_" + df["budget_bin"]
    return df

def stratified_greedy_kfold(names_df: pd.DataFrame, k: int, seed: int, strat_col: str = "stratum"):
    """
    Greedy, stratum-aware assignment to balance fold sizes.
    Process strata from large to small; within each stratum shuffle, then assign
    each item to the currently smallest fold.
    """
    rng = random.Random(seed)
    folds = {i: [] for i in range(k)}
    counts = {i: 0 for i in range(k)}

    strata = [(s, g.copy()) for s, g in names_df.groupby(strat_col)]
    strata.sort(key=lambda x: len(x[1]), reverse=True)

    for s, g in strata:
        items = g["instance name"].tolist()
        rng.shuffle(items)
        for item in items:
            f = min(counts.keys(), key=lambda i: (counts[i], i))
            folds[f].append(item)
            counts[f] += 1

    return {i + 1: sorted(folds[i]) for i in range(k)}

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_splits.py Instance.csv", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    df = add_strata(df)

    # ---- Fixed test set (as produced earlier) ----
    test_set = [
        "E-n22-k4",
        "E-n112-k8-s11",
        "M-n212-k16-s12",
        "F-n140-k5-s5",
        "X-n221-k11-s7",
        "X-n469-k26-s10",
        "X-n698-k75-s13",
        "X-n1006-k43-s5",
    ]

    # Sanity check
    all_names = set(df["instance name"])
    missing = [x for x in test_set if x not in all_names]
    if missing:
        raise RuntimeError(f"These test instances are not found in CSV: {missing}")

    remain = df[~df["instance name"].isin(test_set)].copy()
    assert len(remain) == 33, f"Expected 33 remaining instances, got {len(remain)}"

    # ---- 5-fold CV on remaining ----
    seed = 20251228
    folds_val = stratified_greedy_kfold(remain, k=5, seed=seed, strat_col="stratum")

    # Build train/val per fold (within remaining set)
    remain_names = sorted(remain["instance name"].tolist())
    folds = {}
    for f, val_list in folds_val.items():
        val_set = set(val_list)
        train_list = [n for n in remain_names if n not in val_set]
        folds[f"fold_{f}"] = {"train": train_list, "val": val_list}

    out = {
        "seed": seed,
        "stratification": {
            "family": "first character of instance name",
            "size_bin": "S: n<=100, M: 100<n<=300, L: n>300",
            "budget_bin": "within-family median split on maxEvals",
            "stratum": "family_size_bin_budget_bin",
        },
        "test": sorted(test_set),
        "cv": folds,
    }

    with open("splits.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print("Test set (n={}):".format(len(test_set)))
    print("  " + ", ".join(sorted(test_set)))
    print("\n5-fold CV validation sizes:")
    for f in range(1, 6):
        print(f"  fold_{f}: |val|={len(folds[f'fold_{f}']['val'])}, |train|={len(folds[f'fold_{f}']['train'])}")

    print("\nWrote splits.json")

if __name__ == "__main__":
    main()

