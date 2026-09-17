#!/bin/bash
# M2 — Entrenamiento bilingüe EN+ES con capacidad baseline (emb 512, feat 1024).
# FOREGROUND (visible), bf16, num_workers=16, pin_memory.
#
# Uso:
#   ./scripts/train_multilingue.sh [EPOCHS] [BATCH_SIZE]
#     (defaults: 100 128)
#
# Auto-retoma: si existe OUT_DIR/TextReIDNet_latest.pth.tar pasa --resume
# y continua en la ultima epoca guardada. La barra de progreso se ve en la terminal.

cd "$(dirname "$0")/.."
source .venv/bin/activate

EPOCHS="${1:-100}"
BATCH="${2:-128}"
OUT_DIR="data/checkpoints/bpe24_512x1024"

RESUME_ARG=""
if [ -f "$OUT_DIR/TextReIDNet_latest.pth.tar" ]; then
  RESUME_ARG="--resume"
  echo "Checkpoint previo encontrado -> continuando desde el último"
fi

.venv/bin/python train.py \
  --dataset_source original \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --tokenizer_type bpe_en_es \
  --vocab_size 24000 \
  --embedding_dim 512 \
  --feature_length 1024 \
  --bilingual \
  --output_dir "$OUT_DIR" \
  $RESUME_ARG