#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./collect_top10_from_irace_logs.sh ins > top10_all.txt
# or:
#   ./collect_top10_from_irace_logs.sh ins top10_all.txt

INS_FILE="${1:-ins}"
OUT_FILE="${2:-}"

if [[ ! -f "$INS_FILE" ]]; then
  echo "Error: file not found: $INS_FILE" >&2
  exit 1
fi

# write to stdout by default; or to file if provided
if [[ -n "$OUT_FILE" ]]; then
  exec > "$OUT_FILE"
fi

while IFS= read -r inst || [[ -n "$inst" ]]; do
  [[ -z "${inst// }" ]] && continue
  [[ "${inst:0:1}" == "#" ]] && continue
  [[ "$inst" != *.evrp ]] && continue

  dir="${inst%.evrp}"
  log="${dir}/irace.log"

  echo "================================================================"
  echo "Folder:   ${dir}"
  if [[ ! -f "$log" ]]; then
    echo "Status:   MISSING irace.log"
    continue
  fi

  # Extract the first 10 lines after:
  # "# Best configurations as commandlines"
  # Stop when hitting an empty line or the next section starting with "#"
  top10=$(
    awk '
      BEGIN {flag=0; n=0}
      /^# Best configurations as commandlines/ {flag=1; next}
      flag==1 {
        # stop at blank line or next section header
        if ($0 ~ /^#/ || $0 ~ /^[[:space:]]*$/) { exit }
	print "(" $3 ", " $5 "),"
        #print $3,$5
        n++
        if (n>=10) { exit }
      }
    ' "$log"
  )

  if [[ -z "$top10" ]]; then
    echo "Status:   Could not find \"Best configurations as commandlines\" section"
    continue
  fi

  echo "Top-10 (commandlines):"
  echo "$top10"

done < "$INS_FILE"

