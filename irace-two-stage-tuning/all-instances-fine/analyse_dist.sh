#!/usr/bin/env bash
set -euo pipefail

ROOT="."
INS_FOLDERS="ins_folders.txt"

module load GCCcore/13.2.0 Python/3.11.5

python collect_median_representative.py --root "$ROOT" --ins_folders "$INS_FOLDERS" --topk 10

