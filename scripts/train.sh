#!/bin/bash
cd "$(dirname "$0")/.."
source .venv/bin/activate
RESUME_ARG=""
if [ -f data/checkpoints/TextReIDNet_latest.pth.tar ]; then
  RESUME_ARG="--resume"
fi
python train.py --dataset_source original --epochs "$1" --batch_size "$2" $RESUME_ARG