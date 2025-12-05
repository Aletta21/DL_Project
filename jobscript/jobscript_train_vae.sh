#!/bin/sh
### LSF options for DTU GPU queues
#BSUB -q gpuv100
#BSUB -J vae_only_run
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=24GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -oo logs/train_vae_%J.out
#BSUB -eo logs/train_vae_%J.err
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

# Train VAE only on gene inputs (no predictor)
python -u train_VAE.py \
  --whole-dataset \
  --epochs 200 \
  --batch-size 128 \
  --latent 128 \
  --hidden 1024 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --beta 0.5 \
  --beta-warmup 20 \
  --grad-clip 5.0 \
  --patience 30
