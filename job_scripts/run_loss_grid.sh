#!/usr/bin/env bash


# ce	–
# lsce	--smoothing 0.1
# asl	– (now uses SafeASLSingleLabel)
# focal	--gamma 2 --alpha 0.25
# cbce	--beta 0.999
# cbfocal	--beta 0.999 --gamma 2


set -e
DATA=/coral/images
CSV=/coral/splits             # train.csv / val.csv

for MODEL in convnext_base.fb_in22k_ft_in1k vit_base_patch16_224; do
  for LOSS in ce lsce asl focal cbce cbfocal; do
    for SEED in 42 91 2025; do
      torchrun --nproc_per_node=4 train_ablate.py \
        $DATA --dataset csv \
        --csv-train $CSV/train.csv \
        --csv-val   $CSV/val.csv \
        --model $MODEL \
        --epochs 100 --batch-size 64 \
        --sched cosine --warmup-epochs 5 \
        --opt adamw --lr 5e-4 --weight-decay 0.05 \
        --loss $LOSS \
        --gamma 2 --alpha 0.25 --beta 0.999 \
        --seed $SEED \
        --log-wandb --experiment "abl_${MODEL}_${LOSS}_s${SEED}" \
        "$@"
    done
  done
done
