#!/bin/bash
# M3 — Evaluación multilingue (EN y ES, cross-lingual) del checkpoint bilingue.
#
# Uso:
#   ./scripts/evaluate_multilingue.sh [CHECKPOINT]
#     (default: data/checkpoints/bpe24_256x512/TextReIDNet_latest.pth.tar)

cd "$(dirname "$0")/.."
source .venv/bin/activate

CKPT="${1:-data/checkpoints/bpe24_256x512/TextReIDNet_latest.pth.tar}"

for LANG in en es; do
  echo "============================================================"
  echo "Evaluando LANG=$LANG  checkpoint=$CKPT"
  echo "============================================================"
  .venv/bin/python evaluate.py \
    --checkpoint "$CKPT" \
    --dataset_source original \
    --tokenizer_type bpe_en_es \
    --vocab_size 24000 \
    --embedding_dim 256 \
    --feature_length 512 \
    --evaluate_language "$LANG"
  echo ""
done

echo "Evaluación completa."