#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
EPOCHS="$1"
BATCH="$2"
shift 2
RESUME_ARG=""
if [ -f data/checkpoints/TextReIDNet_latest.pth.tar ]; then
  RESUME_ARG="--resume"
fi
python train.py --dataset_source original --epochs "$EPOCHS" --batch_size "$BATCH" $RESUME_ARG "$@"