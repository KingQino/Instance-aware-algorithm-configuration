#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./gen_irace_folders.sh ins

INS_FILE="${1:-ins}"

FOLDERS_LIST="ins_folders"
: > "$FOLDERS_LIST"   # 清空/创建

if [[ ! -f "$INS_FILE" ]]; then
  echo "Error: file not found: $INS_FILE" >&2
  exit 1
fi

BASE_OUTPUT_PARENT="/home/e/exx866/IRACE-b-LAHC/all-instances-fine"

trim() { echo "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'; }

write_common_files() {
  local folder="$1"          # e.g. E-n33-k4-p1
  local inst="$2"            # e.g. E-n33-k4.evrp
  local hisL="$3" hisU="$4"  # range
  local attL="$5" attU="$6"  # range

  mkdir -p "$folder"

  # instances.txt (10 lines, identical instance)
  {
    for _ in {1..10}; do
      echo "$inst"
    done
  } > "${folder}/instances.txt"

  # ===== compute default values (midpoint) =====
  local his_def att_def
  his_def=$(( (hisL + hisU) / 2 ))
  att_def=$(( (attL + attU) / 2 ))

  # safety (in case of weird bounds)
  (( his_def < hisL )) && his_def="$hisL"
  (( his_def > hisU )) && his_def="$hisU"
  (( att_def < attL )) && att_def="$attL"
  (( att_def > attU )) && att_def="$attU"

  # ===== configurations.txt (DEFAULT CONFIG INSIDE THIS PEAK) =====
  cat > "${folder}/configurations.txt" << EOF
his_len max_attempts
${his_def}    ${att_def}
EOF

  # parameters.txt (single-peak only)
  cat > "${folder}/parameters.txt" << EOF
# 1:            2:                   3:     4:
his_len         "-his_len "          i      (${hisL}, ${hisU})
max_attempts    "-max_attempts "     i      (${attL}, ${attU})


[global]
digits = 2
EOF

  # run_irace.R
  cat > "${folder}/run_irace.R" << 'EOF'
library(irace)

message("Starting irace...")
scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())
irace_main(scenario = scenario)

if (file.exists("irace.Rdata")) {
  load("irace.Rdata")
  best <- iraceResults$allConfigurations[tail(iraceResults$iterationElites, 1)[[1]][1], ]
  write.table(best, file = "best-config.txt", sep = "\t", quote = FALSE, row.names = FALSE)
}
EOF

  # scenario.txt
  cat > "${folder}/scenario.txt" << 'EOF'
parameterFile = "parameters.txt"
trainInstancesFile = "instances.txt"
targetRunner = "./target-runner"
logFile = "irace.Rdata"
execDir = "./"

parallel = 128
loadBalancing = 1

maxExperiments = 6000
minExperiments = NA

blockSize = 1

firstTest = 10
eachTest  = 10

configurationsFile = "configurations.txt"

postselection = 1
minNbSurvival = 10
EOF

  # submit_irace.slurm (两处用 folder 名)
  cat > "${folder}/submit_irace.slurm" << EOF
#!/bin/bash

#SBATCH --job-name=${folder}
#SBATCH --output=${BASE_OUTPUT_PARENT}/${folder}/irace.log
#SBATCH --time=48:0:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1G
#SBATCH --account=su008-exx866

module load GCCcore/13.2.0 CMake GCC/13.2.0 OpenMPI/4.1.6 R/4.3.3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export IRACE_HOME="/home/e/exx866/R/x86_64-pc-linux-gnu-library/4.3/irace"
export PATH="\$IRACE_HOME/bin:\$PATH"

Rscript run_irace.R
EOF
  chmod +x "${folder}/submit_irace.slurm"

  # target-runner（原样：只接受 -his_len / -max_attempts）
  cat > "${folder}/target-runner" << 'EOF'
#!/bin/bash
###############################################################################
# This script is the command that is executed every run.
###############################################################################
error() {
    echo "`TZ=UTC date`: $0: error: $@"
    exit 1
}

CONFIG_ID=$1
INSTANCE_ID=$2
SEED=$3
INSTANCE=$4
shift 4 || error "Not enough parameters"
CONFIG_PARAMS=$*

EXE=/home/e/exx866/IRACE-b-LAHC/E-CVRP-Learning/build/Run
EXE_PARAMS="-alg lahc -ins $INSTANCE -log 0 -stp 0 -mth 0 -seed ${SEED} -low_margin 1.01 -noise_lb 0.95 -noise_ub 1.05 -exp 0 ${CONFIG_PARAMS}"

if [ ! -x "$(command -v ${EXE})" ]; then
    error "${EXE}: not found or not executable (pwd: $(pwd))"
fi

STDOUT=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stdout
STDERR=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stderr

START=$(date +%s.%N)
$EXE ${EXE_PARAMS} 1> ${STDOUT} 2> ${STDERR}
END=$(date +%s.%N)
TIME=$(echo "$END - $START" | bc)

if [ ! -s "${STDOUT}" ]; then
    error "${STDOUT}: No such file or directory"
fi

COST=$(tail -n 1 ${STDOUT} | grep -e '^[[:space:]]*[+-]\?[0-9]' | cut -f1)
echo "$COST $TIME"
rm -f "${STDOUT}" "${STDERR}"
exit 0
EOF
  chmod +x "${folder}/target-runner"
}

while IFS= read -r line || [[ -n "$line" ]]; do
  line="$(trim "$line")"
  [[ -z "$line" ]] && continue
  [[ "${line:0:1}" == "#" ]] && continue

  inst="${line%%,*}"
  inst="$(trim "$inst")"
  [[ "$inst" == *.evrp ]] || { echo "Skip bad line: $line" >&2; continue; }

  base="${inst%.evrp}"
  rest=""
  if [[ "$line" == *","* ]]; then
    rest="${line#*,}"
  fi

  # 只从 rest 提取数字，避免把实例名里的 n22/k4 抓进去
  nums=()
  if [[ -n "$rest" ]]; then
    while IFS= read -r n; do nums+=("$n"); done < <(printf "%s" "$rest" | grep -oE '[0-9]+' || true)
  fi

  if [[ ${#nums[@]} -eq 4 ]]; then
    # 单峰：base/
    write_common_files "$base" "$inst" "${nums[0]}" "${nums[1]}" "${nums[2]}" "${nums[3]}"
    echo "Generated: $base (single peak)"
    echo "$base" >> "$FOLDERS_LIST"
  elif [[ ${#nums[@]} -eq 8 ]]; then
    # 双峰：base-p1/ 和 base-p2/
    write_common_files "${base}-p1" "$inst" "${nums[0]}" "${nums[1]}" "${nums[2]}" "${nums[3]}"
    write_common_files "${base}-p2" "$inst" "${nums[4]}" "${nums[5]}" "${nums[6]}" "${nums[7]}"
    echo "Generated: ${base}-p1 and ${base}-p2 (two peaks)"
    echo "${base}-p1" >> "$FOLDERS_LIST"
    echo "${base}-p2" >> "$FOLDERS_LIST"
  else
    echo "Skip (need 4 or 8 numbers after instance, got ${#nums[@]}): $line" >&2
  fi

done < "$INS_FILE"

