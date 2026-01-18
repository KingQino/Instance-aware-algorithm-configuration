#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./submit_all_irace.sh ins_folders
# default file name: ins_folders

LIST_FILE="${1:-ins_folders}"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "Error: file not found: $LIST_FILE" >&2
  exit 1
fi

while IFS= read -r dir || [[ -n "$dir" ]]; do
  # skip empty lines / comments
  [[ -z "${dir// }" ]] && continue
  [[ "${dir:0:1}" == "#" ]] && continue

  if [[ ! -d "$dir" ]]; then
    echo "Skip (folder not found): $dir"
    continue
  fi

  if [[ ! -f "$dir/submit_irace.slurm" ]]; then
    echo "Skip (submit_irace.slurm not found): $dir"
    continue
  fi

  echo "Submitting: $dir"
  (
    cd "$dir"
    sbatch submit_irace.slurm
  )

  # 可选：防止提交太快被 Slurm 拒绝
  # sleep 1

done < "$LIST_FILE"

