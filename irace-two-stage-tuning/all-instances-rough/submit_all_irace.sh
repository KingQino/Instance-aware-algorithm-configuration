#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./submit_all_irace.sh ins

INS_FILE="${1:-ins}"

if [[ ! -f "$INS_FILE" ]]; then
  echo "Error: file not found: $INS_FILE" >&2
  exit 1
fi

while IFS= read -r inst || [[ -n "$inst" ]]; do
  # skip empty lines / comments
  [[ -z "${inst// }" ]] && continue
  [[ "${inst:0:1}" == "#" ]] && continue

  # only handle .evrp
  if [[ "$inst" != *.evrp ]]; then
    echo "Skip (not .evrp): $inst"
    continue
  fi

  dir="${inst%.evrp}"

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

done < "$INS_FILE"

