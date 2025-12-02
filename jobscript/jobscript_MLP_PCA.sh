#!/bin/sh
### LSF options for DTU GPU queues
#BSUB -q gpuv100
#BSUB -J train_MLP_PCA
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -oo logs/train_%J.out
#BSUB -eo logs/train_%J.err
#BSUB -u s252976@dtu.dk
#BSUB -B
#BSUB -N

set -euo pipefail

module purge
module load cuda/11.6 || true
source ~/miniforge3/bin/activate IDLCV2

cd /zhome/6b/3/223370/projects/DL_Project
mkdir -p logs

echo "Running on host: $(hostname)"
nvidia-smi || true

# Quick check: does PyTorch see the GPU?
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device(), "name:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

# Run training on full dataset (50/35/15 split)
python -u train_MLP.py --whole-dataset --epochs 300 --batch-size 64