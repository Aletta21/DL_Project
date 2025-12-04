#!/bin/sh
### LSF options for DTU GPU queues
#BSUB -q gpuv100
#BSUB -J new_vae_run
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -oo logs/new_vae_%J.out
#BSUB -eo logs/new_vae_%J.err
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

# Train VAE + predictor (new_vae_train)
python -u new_vae_train.py \
  --whole-dataset \
  --epochs 200 \
  --batch-size 128 \
  --latent 128 \
  --hidden 1024 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --beta 0.5 \
  --beta-warmup 20 \
  --patience 30 \
  --grad-clip 5.0 \
  --pred-epochs 200 \
  --pred-batch 128 \
  --pred-lr 5e-4 \
  --pred-weight-decay 1e-4 \
  --pred-dropout 0.25 \
  --hidden1 1536 --hidden2 1024 --hidden3 512
