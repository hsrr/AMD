#!/usr/bin/env bash
set -euo pipefail

random_seed=12
export PYTHONHASHSEED="${random_seed}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# -------- required paths --------
MODEL_PTH="/data1/hsiri/florence_base"
TRAIN_JS="/data1/yaxiong/dataset/DGM4/metadata_split/guardian/train.json"
VAL_JS="/data1/yaxiong/dataset/DGM4/metadata_split/guardian/val_mini.json"
IMAGE_ROOT="/data1/yaxiong/dataset"

# -------- train config --------
BATCH_SIZE=5
EPOCHS=13
LR=1e-6
EVAL_STEPS=2000
MAX_VAL_ITEM_COUNT=2000
REGULAR_WEIGHT=2000
TRAIN_DOMAIN="NYT"
RUN_NAME="amd_train_run_$(date +%Y%m%d_%H%M)"

# 1 = only classification supervision; 0 = keep full supervision
CLASSIFICATION_ONLY_SUPERVISION=1

cmd=(
  python scripts/train.py
  --AMD-init-pth "${MODEL_PTH}"
  --dataset-type DGM4
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --eval-steps "${EVAL_STEPS}"
  --run-name "${RUN_NAME}"
  --max-val-item-count "${MAX_VAL_ITEM_COUNT}"
  --regular-weight "${REGULAR_WEIGHT}"
  --train-js "${TRAIN_JS}"
  --val-js "${VAL_JS}"
  --train-domain "${TRAIN_DOMAIN}"
  --seed "${random_seed}"
  --image-root "${IMAGE_ROOT}"
)

if [[ "${CLASSIFICATION_ONLY_SUPERVISION}" == "1" ]]; then
  cmd+=(--classification-only-supervision)
fi

echo "Running command:"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"

echo "Training completed at: $(date)"
