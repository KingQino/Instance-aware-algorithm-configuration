#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=M-n212-k16-s12                       # Job name set to the parent directory name
#SBATCH --output=/home/e/exx866/IRACE-b-LAHC/all-instances-rough/M-n212-k16-s12/irace.log      # Output log file path in the log folder
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
export PATH="$IRACE_HOME/bin:$PATH"

Rscript run_irace.R
