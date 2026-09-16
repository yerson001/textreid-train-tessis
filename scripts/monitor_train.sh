#!/bin/bash
# Monitor visual del entrenamiento M2 (bilingue).
# Muestra: periodo del proceso, época actual, batches, %, pérdidas, ETA,
# checkpoint más reciente y si ya terminó.
# Salir: Ctrl+C
#
# Uso:
#   ./scripts/monitor_train.sh [segundos_entre_refrescos]

cd "$(dirname "$0")/.."

LOG="logs/train_bpe_multilingue.log"
OUT="data/checkpoints/bpe24_256x512"
INTERVAL="${1:-3}"
TOTAL_EPOCHS=""
SPIN=("◐" "◓" "◑" "◒")
i=0

while true; do
  i=$(( (i + 1) % 4 ))
  clear
  printf "MONITOR TEXTREIDNET M2  (%s)   %s\n" "$(date '+%H:%M:%S')" "${SPIN[$i]}"

  # ---- proceso en GPU (evita self-match, usa nvidia-smi) ----
  GPUPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | head -1)
  if [ -n "$GPUPID" ]; then
    printf "● PROCESO GPU: PID %s  (corriendo)\n" "$GPUPID"
  else
    printf "○ SIN proceso python en GPU\n"
  fi

  # ---- última línea de progreso (tqdm usa \r) ----
  LINE=$(tail -c 4000 "$LOG" 2>/dev/null | tr '\r' '\n' | grep "Epoch" | tail -1)
  if [ -n "$LINE" ]; then
    EPOCH=$(printf '%s' "$LINE" | sed -E 's/.*Epoch ([0-9]+)\/([0-9]+).*/\1/')
    TOTAL_EPOCHS=$(printf '%s' "$LINE" | sed -E 's/.*Epoch ([0-9]+)\/([0-9]+).*/\2/')
    PIPE=$(printf '%s' "$LINE" | grep -oE '[0-9]+/[0-9]+' | head -1)
    PCT=$(printf '%s' "$LINE" | grep -oE '[0-9]+%' | head -1)
    SPD=$(printf '%s' "$LINE" | grep -oE '[0-9.]+batch/s' | head -1)
    LOSS=$(printf '%s' "$LINE" | grep -oE '(rank|id|tot)=[0-9.]+' | tr '\n' ' ')

    DONE=${PIPE%%/*}
    TOTP=${PIPE##*/}
    printf "ÉPOCA   %s/%s   (%s)   batches %s\n" "$EPOCH" "$TOTAL_EPOCHS" "$PCT" "$PIPE"
    if [ -n "$SPD" ]; then
      BPS=$(printf '%s' "$SPD" | grep -oE '[0-9.]+')
      ETAS=$(awk -v r=$((TOTP - DONE)) -v b="$BPS" 'BEGIN{printf "%d min %d s", int(r/b/60), int(r/b)%60}')
      printf "VELOCID %.2f batch/s | ETA época: %s\n" "$BPS" "$ETAS"
    fi
    printf "PÉRDIDAS %s\n" "$LOSS"
  else
    printf "Sin progreso aún (arrancando)...\n"
  fi

  # ---- checkpoint más reciente ----
  LATEST=$(ls -t "$OUT"/TextReIDNet_epoch*.pth.tar 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    printf "CHECKP   %s  (último guardado)\n" "$(basename "$LATEST")"
  fi

  # ---- fin del entrenamiento ----
  if [ -z "$GPUPID" ] && [ -n "$TOTAL_EPOCHS" ] && [ "$EPOCH" = "$TOTAL_EPOCHS" ]; then
    printf ">>> ENTRENAMIENTO TERMINADO (época %s/%s) <<<\n" "$EPOCH" "$TOTAL_EPOCHS"
    printf ">>> Luego corre: ./scripts/evaluate_multilingue.sh\n"
  fi

  sleep "$INTERVAL"
done