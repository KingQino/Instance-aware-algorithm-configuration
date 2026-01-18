#!/bin/bash

#SBATCH --job-name=E-n51-k5-p1
#SBATCH --output=/home/e/exx866/IRACE-b-LAHC/all-instances-fine/E-n51-k5-p1/irace.log
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
export PATH="$IRACE_HOME/bin:$PATH"

Rscript run_irace.R
