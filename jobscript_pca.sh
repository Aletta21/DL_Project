#!/bin/sh
### LSF options for CPU queue
#BSUB -q gpuv100
#BSUB -J pca_iso
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -oo logs/pca_%J.out
#BSUB -eo logs/pca_%J.err
#BSUB -u s252976@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

# Keep MKL/BLAS thread counts aligned with allocated cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

module purge
source ~/miniforge3/bin/activate IDLCV2

cd /zhome/6b/3/223370/projects/DL_Project
mkdir -p logs

echo "Running on host: $(hostname)"
python --version

# Run PCA preprocessing (uses /work3 data paths inside the script)
python -u data/pca.py
