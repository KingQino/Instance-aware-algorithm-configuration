#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./collect_top10_from_irace_logs.sh ins_folders > top10_all.txt
# or:
#   ./collect_top10_from_irace_logs.sh ins_folders top10_all.txt

FOLDERS_FILE="${1:-ins_folders}"
OUT_FILE="${2:-}"

if [[ ! -f "$FOLDERS_FILE" ]]; then
  echo "Error: file not found: $FOLDERS_FILE" >&2
  exit 1
fi

# If output file is provided, redirect stdout
if [[ -n "$OUT_FILE" ]]; then
  exec > "$OUT_FILE"
fi

while IFS= read -r dir || [[ -n "$dir" ]]; do
  # skip empty lines / comments
  [[ -z "${dir// }" ]] && continue
  [[ "${dir:0:1}" == "#" ]] && continue

  log="${dir}/irace.log"

  echo "================================================================"
  echo "Instance folder: ${dir}"

  if [[ ! -f "$log" ]]; then
    echo "Status: MISSING irace.log"
    continue
  fi

  # Extract Top-10 from:
  # "# Best configurations as commandlines"
  awk '
    BEGIN {flag=0; n=0}
    /^# Best configurations as commandlines/ {
        flag=1
        next
    }
    flag==1 {
        # stop when next section or empty line
        if ($0 ~ /^#/ || $0 ~ /^[[:space:]]*$/) exit
        print
        n++
        if (n >= 10) exit
    }
  ' "$log"

done < "$FOLDERS_FILE"

