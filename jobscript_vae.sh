#!/bin/sh
### LSF options for DTU GPU queues
#BSUB -q gpuv100
#BSUB -J vae_latent_iso
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -oo logs/vae_latent_%J.out
#BSUB -eo logs/vae_latent_%J.err
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

# Train VAE on full dataset with latent->isoform predictor
python -u train_vae.py \
  --train-n 2000 \
  --val-n 1400 \
  --test-n 600 \
  --vae-latent 256 \
  --vae-hidden 768 \
  --vae-epochs 100 \
  --vae-beta 0.5 \
  --vae-beta-warmup 20 \
  --vae-grad-clip 5.0 \
  --vae-batch 128 \
  --vae-lr 1e-3 \
  --vae-weight-decay 1e-5 \
  --epochs 250 \
  --batch-size 128 \
  --lr 2e-4 \
  --weight-decay 3e-4 \
  --dropout 0.25 \
  --hidden1 1536 --hidden2 1024 --hidden3 512
