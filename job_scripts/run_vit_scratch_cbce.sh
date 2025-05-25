#!/bin/bash
#SBATCH --job-name=vit-scratch-reefnet-cbce
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:3
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=196G
#SBATCH --time=23:30:00
#SBATCH --output=output/ViT-Scratch_CBCE/logs/%j-%x.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=[YOUR_EMAIL]

# Environment setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9902  # fallback if dynamic one fails
export GPUS_PER_NODE=3

export OMP_NUM_THREADS=10

source ~/.bashrc
conda activate pytoenv

# Dynamically pick an open port
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
    PORT=$(shuf -i $LOWERPORT-$UPPERPORT -n 1)
    ss -lpn | grep -q ":$PORT " || break
done


echo "Running on $(hostname) with $GPUS_PER_NODE GPUs, port $PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Master address: $MASTER_ADDR"

torchrun --nproc-per-node=$GPUS_PER_NODE --master_port=$PORT train.py \
  --model vit_large_patch16_384 \
  --pretrained False\
  --dataset reefnet_csv \
  --csv_file [csv_file_path] \
  --num-classes 63 \
  --batch-size 128 \
  --epochs 100 \
  --loss cbce --beta 0.999\
  --smoothing 0.2 \
  --amp \
  --channels-last \
  --input-size 3 384 384 \
  --interpolation bicubic \
  --train-interpolation bicubic \
  --color-jitter 0.2 \
  --hflip 0.5 \
  --aa rand-m7-mstd0.5-inc1 \
  --reprob 0.1 \
  --mixup 0.4 \
  --cutmix 0.5 \
  --mean 0.38453767 0.41910657 0.34101963 \
  --std 0.16151004 0.17718494 0.15782207 \
  --log-wandb \
  --wandb-project coral-benchmark \
  --wandb-tags reefnet vit scratch \
  --wandb-resume-id "" \
  --opt adamw \
  --opt-betas 0.9 0.999 \
  --opt-eps 1e-8 \
  --lr 5e-4 \
  --warmup-epochs 20 \
  --layer-decay 1.0 \
  --drop-path 0.3 \
  --model-ema \
  --weight-decay 0.05 \
  --clip-grad 1.0 \
  --workers 10 \
  --output output/ViT-Scratch_CBCE/output/