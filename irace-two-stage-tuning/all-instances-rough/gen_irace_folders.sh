#!/usr/bin/env bash
set -euo pipefail

# usage:
#   ./gen_irace_folders.sh ins
#
# ins: one column, each line like "E-n22-k4.evrp"

INS_FILE="${1:-ins}"

if [[ ! -f "$INS_FILE" ]]; then
  echo "Error: file not found: $INS_FILE" >&2
  exit 1
fi

# You only need to change this if you want the slurm --output path elsewhere.
# It should be the parent directory that contains all instance folders.
# Example in your template:
#   /home/e/exx866/IRACE-b-LAHC/all-instances-rough/<INSTANCE_DIR>/irace.log
BASE_OUTPUT_PARENT="/home/e/exx866/IRACE-b-LAHC/all-instances-rough"

# Create each instance folder + files
while IFS= read -r inst || [[ -n "$inst" ]]; do
  # skip empty lines and comments
  [[ -z "${inst// }" ]] && continue
  [[ "${inst:0:1}" == "#" ]] && continue

  # must end with .evrp
  if [[ "$inst" != *.evrp ]]; then
    echo "Skip (not .evrp): $inst" >&2
    continue
  fi

  dir="${inst%.evrp}"   # E-n22-k4.evrp -> E-n22-k4

  mkdir -p "$dir"

  # 1) instances.txt (10 lines, all identical instance filename)
  {
    for _ in {1..10}; do
      echo "$inst"
    done
  } > "${dir}/instances.txt"

  # 2) configurations.txt
  cat > "${dir}/configurations.txt" << 'EOF'
his_len max_attempts
5723    60
EOF

  # 3) parameters.txt
  cat > "${dir}/parameters.txt" << 'EOF'
# 1:            2:                   3:     4:      5:
his_len         "-his_len "          i      (300, 20000)
max_attempts    "-max_attempts "     i      (1, 500)


[forbidden]
#mode == "x1" & mutation == "low"


[global]
digits = 2 # Maximum number of decimal places that are significant for numerical (real) parameters.
EOF

  # 4) run_irace.R
  cat > "${dir}/run_irace.R" << 'EOF'
library(irace)

message("Starting irace...")
scenario <- readScenario(filename = "scenario.txt", scenario = defaultScenario())
# checkIraceScenario(scenario = scenario)
irace_main(scenario = scenario)

# load results
if (file.exists("irace.Rdata")) {
  load("irace.Rdata")

  best <- iraceResults$allConfigurations[tail(iraceResults$iterationElites, 1)[[1]][1], ]
  write.table(best, file = "best-config.txt", sep = "\t", quote = FALSE, row.names = FALSE)

}
EOF

  # 5) scenario.txt
  cat > "${dir}/scenario.txt" << 'EOF'
parameterFile = "parameters.txt"
trainInstancesFile = "instances.txt"
targetRunner = "./target-runner"
logFile = "irace.Rdata"
execDir = "./"

# parallel
parallel = 64
loadBalancing = 1

# budget
maxExperiments = 3500
minExperiments = NA

# instance-related
blockSize = 1
# sampleInstances = 1 # Random sampling - don't iUse instances in the order specified in instances.txt (fast to slow)

# racing
firstTest = 10 
eachTest  = 10 

# default config
configurationsFile = "configurations.txt"

# post selection and select top 10 configs
postselection = 0
minNbSurvival = 10
EOF

  # 6) submit_irace.slurm (2 places replaced with instance dir name)
  #   - #SBATCH --job-name=<dir>
  #   - #SBATCH --output=<BASE_OUTPUT_PARENT>/<dir>/irace.log
  cat > "${dir}/submit_irace.slurm" << EOF
#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=${dir}                       # Job name set to the parent directory name
#SBATCH --output=${BASE_OUTPUT_PARENT}/${dir}/irace.log      # Output log file path in the log folder
#SBATCH --time=48:0:0                                  # Request 48 hours of compute time
#SBATCH --nodes=1                                      # Request 1 node
#SBATCH --tasks-per-node=1                             # One task per node
#SBATCH --cpus-per-task=64                             # Each task uses 10 CPUs (threads)
#SBATCH --mem-per-cpu=1G                               # Memory per CPU
#SBATCH --account=su008-exx866

# Load necessary modules
module load GCCcore/13.2.0 CMake GCC/13.2.0 OpenMPI/4.1.6 R/4.3.3

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export IRACE_HOME="/home/e/exx866/R/x86_64-pc-linux-gnu-library/4.3/irace"
export PATH="\$IRACE_HOME/bin:\$PATH"

Rscript run_irace.R
EOF

  chmod +x "${dir}/submit_irace.slurm"

  # 7) target-runner
  cat > "${dir}/target-runner" << 'EOF'
#!/bin/bash
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# $1 is the candidate configuration number
# $2 is the instance ID
# $3 is the seed
# $4 is the instance name
# The rest ($* after `shift 4') are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################
error() {
    echo "`TZ=UTC date`: $0: error: $@"
    exit 1
}

# This parses the arguments given by irace. Do not touch it!
CONFIG_ID=$1
INSTANCE_ID=$2
SEED=$3
INSTANCE=$4
shift 4 || error "Not enough parameters"
CONFIG_PARAMS=$*
# End of parsing

EXE=/home/e/exx866/IRACE-b-LAHC/E-CVRP-Learning/build/Run
EXE_PARAMS="-alg lahc -ins $INSTANCE -log 0 -stp 0 -mth 0 -seed ${SEED} -low_margin 1.01 -noise_lb 0.95 -noise_ub 1.05 -exp 0 ${CONFIG_PARAMS}"

if [ ! -x "$(command -v ${EXE})" ]; then
    error "${EXE}: not found or not executable (pwd: $(pwd))"
fi

STDOUT=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stdout
STDERR=c${CONFIG_ID}-${INSTANCE_ID}-${SEED}.stderr

# echo "$EXE ${EXE_PARAMS} 1> ${STDOUT} 2> ${STDERR}"
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

  chmod +x "${dir}/target-runner"

  echo "Generated: ${dir}/ (from ${inst})"

done < "$INS_FILE"

