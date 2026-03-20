#!/bin/bash

random_seed=12 
export PYTHONHASHSEED=$random_seed
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=disabled

MODEL_PTH='./models'
TRAIN_JS='/data1/yaxiong/dataset/DGM4/metadata_split/guardian/train.json'
VAL_JS='/data1/yaxiong/dataset/DGM4/metadata_split/guardian/val.json'
IMAGE_ROOT='/data1/yaxiong/dataset/'

python scripts/train.py \
    --AMD-init-pth "$MODEL_PTH" \
    --dataset-type DGM4 \
    --batch-size 5 \
    --epochs 13 \
    --lr 1e-6 \
    --eval-steps 2000 \
    --run-name "amd_train_run_$(date +%Y%m%d_%H%M)" \
    --max-val-item-count 2000 \
    --regular-weight 2000 \
    --train-js "$TRAIN_JS" \
    --val-js "$VAL_JS" \
    --train-domain "guardian" \
    --seed $random_seed \
    --image-root "$IMAGE_ROOT"

echo "Training completed at: $(date)"
