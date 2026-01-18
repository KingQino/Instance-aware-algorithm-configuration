#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect irace topK configs per instance folder and compute:
- median / Q1 / Q3 (IQR) for each parameter
- representative configuration: closest to median (IQR-normalized distance)
Outputs:
- labels_median_rep.csv
- labels_median_rep.json

Expected layout:
ROOT/
  ins_folders.txt
  <instance_folder_1>/irace.log
  <instance_folder_2>/irace.log
  ...

irace.log should contain a block like:
# Best configurations as commandlines (first number is the configuration ID; listed from best to worst ...):
100 -his_len 3258 -max_attempts 19
...

If there are multiple such blocks, we take the LAST one (final results).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


@dataclass
class Config:
    rank: int          # 1..K in the order printed by irace
    cfg_id: int
    his_len: int
    max_attempts: int


BEST_BLOCK_HEADER_RE = re.compile(r"^#\s*Best configurations as commandlines", re.IGNORECASE)
CONFIG_LINE_RE = re.compile(
    r"^\s*(\d+)\s+.*?-his_len\s+(\d+)\s+.*?-max_attempts\s+(\d+)\s*$"
)

# Also accept "Elite configurations as commandlines" as fallback
ELITE_BLOCK_HEADER_RE = re.compile(r"^#\s*Elite configurations", re.IGNORECASE)


def read_ins_folders(path: str) -> List[str]:
    folders = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            folders.append(s)
    return folders


def parse_irace_log_for_topk(log_path: str, topk: int = 10) -> List[Config]:
    """
    Parse the LAST 'Best configurations as commandlines' block in irace.log.
    If not found, try the LAST 'Elite configurations' block.
    """
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find all block start indices
    best_starts = [i for i, line in enumerate(lines) if BEST_BLOCK_HEADER_RE.search(line)]
    elite_starts = [i for i, line in enumerate(lines) if ELITE_BLOCK_HEADER_RE.search(line)]

    start_idx = None
    header_kind = None
    if best_starts:
        start_idx = best_starts[-1]
        header_kind = "best"
    elif elite_starts:
        start_idx = elite_starts[-1]
        header_kind = "elite"
    else:
        return []

    # Parse subsequent lines until blank line or next header or separator
    configs: List[Config] = []
    rank = 0
    for j in range(start_idx + 1, len(lines)):
        line = lines[j].rstrip("\n")

        # stop conditions
        if line.strip() == "":
            break
        if line.startswith("===") or line.startswith("---") or line.startswith("+-+"):
            break
        if BEST_BLOCK_HEADER_RE.search(line) or ELITE_BLOCK_HEADER_RE.search(line):
            break
        if line.startswith("#"):
            # irace comments: typically ends the block soon
            # but sometimes configs are followed by more comments, so we allow configs parsing to stop here
            break

        m = CONFIG_LINE_RE.match(line)
        if m:
            cfg_id = int(m.group(1))
            his_len = int(m.group(2))
            max_attempts = int(m.group(3))
            rank += 1
            configs.append(Config(rank=rank, cfg_id=cfg_id, his_len=his_len, max_attempts=max_attempts))
            if rank >= topk:
                break

    return configs


def robust_stats(values: List[int]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))
    return {"q25": q25, "median": q50, "q75": q75, "iqr": (q75 - q25)}


def choose_representative(configs: List[Config],
                          his_stats: Dict[str, float],
                          att_stats: Dict[str, float]) -> Optional[Config]:
    """
    Representative = closest to median (IQR-normalized distance).
    - his_len compared in log-scale (more meaningful for wide ranges)
    - max_attempts in linear scale
    Distance:
      ((log(h)-log(med_h))/scale_h)^2 + ((a-med_a)/scale_a)^2
    where scale_h = max(log(q75_h)-log(q25_h), eps), scale_a = max(iqr_a, eps)

    Tie-break: smaller rank (better in irace list).
    """
    if not configs:
        return None

    med_h = his_stats["median"]
    q25_h = his_stats["q25"]
    q75_h = his_stats["q75"]

    med_a = att_stats["median"]
    iqr_a = att_stats["iqr"]

    # scales
    eps = 1e-9
    # protect against non-positive due to parsing issues
    med_h = max(med_h, 1.0)
    q25_h = max(q25_h, 1.0)
    q75_h = max(q75_h, 1.0)

    scale_h = max(math.log(q75_h) - math.log(q25_h), eps)
    scale_a = max(iqr_a, eps)

    best = None
    best_dist = None
    for c in configs:
        # log-scale on his_len
        dh = (math.log(max(c.his_len, 1)) - math.log(med_h)) / scale_h
        da = (c.max_attempts - med_a) / scale_a
        dist = dh * dh + da * da

        if best is None or dist < best_dist - 1e-12 or (abs(dist - best_dist) <= 1e-12 and c.rank < best.rank):
            best = c
            best_dist = dist

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Root directory containing instance folders")
    ap.add_argument("--ins_folders", type=str, required=True, help="Path to ins_folders file (one folder per line)")
    ap.add_argument("--log_name", type=str, default="irace.log", help="irace log filename inside each folder")
    ap.add_argument("--topk", type=int, default=10, help="How many top configs to parse per instance")
    ap.add_argument("--out_csv", type=str, default="labels_median_rep.csv")
    ap.add_argument("--out_json", type=str, default="labels_median_rep.json")
    args = ap.parse_args()

    folders = read_ins_folders(args.ins_folders)

    rows = []
    out_json: Dict[str, Any] = {"instances": {}}

    for folder in folders:
        log_path = os.path.join(args.root, folder, args.log_name)
        configs = parse_irace_log_for_topk(log_path, topk=args.topk)

        if not configs:
            # still record, but mark missing
            rows.append({
                "instance_folder": folder,
                "n_configs": 0,
                "his_median": np.nan, "his_q25": np.nan, "his_q75": np.nan, "his_iqr": np.nan,
                "att_median": np.nan, "att_q25": np.nan, "att_q75": np.nan, "att_iqr": np.nan,
                "rep_his_len": np.nan, "rep_max_attempts": np.nan, "rep_rank": np.nan, "rep_cfg_id": np.nan,
                "log_path": log_path,
            })
            out_json["instances"][folder] = {"n_configs": 0, "log_path": log_path, "configs": []}
            continue

        his_vals = [c.his_len for c in configs]
        att_vals = [c.max_attempts for c in configs]

        his_stats = robust_stats(his_vals)
        att_stats = robust_stats(att_vals)
        rep = choose_representative(configs, his_stats, att_stats)

        rows.append({
            "instance_folder": folder,
            "n_configs": len(configs),
            "his_median": his_stats["median"],
            "his_q25": his_stats["q25"],
            "his_q75": his_stats["q75"],
            "his_iqr": his_stats["iqr"],
            "att_median": att_stats["median"],
            "att_q25": att_stats["q25"],
            "att_q75": att_stats["q75"],
            "att_iqr": att_stats["iqr"],
            "rep_his_len": rep.his_len if rep else np.nan,
            "rep_max_attempts": rep.max_attempts if rep else np.nan,
            "rep_rank": rep.rank if rep else np.nan,
            "rep_cfg_id": rep.cfg_id if rep else np.nan,
            "log_path": log_path,
        })

        out_json["instances"][folder] = {
            "n_configs": len(configs),
            "log_path": log_path,
            "his_stats": his_stats,
            "att_stats": att_stats,
            "representative": {
                "cfg_id": rep.cfg_id,
                "rank": rep.rank,
                "his_len": rep.his_len,
                "max_attempts": rep.max_attempts,
            } if rep else None,
            "configs": [
                {"rank": c.rank, "cfg_id": c.cfg_id, "his_len": c.his_len, "max_attempts": c.max_attempts}
                for c in configs
            ],
        }

    # Write CSV
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    print(f"[OK] Wrote {args.out_csv}")
    print(f"[OK] Wrote {args.out_json}")


if __name__ == "__main__":
    main()

