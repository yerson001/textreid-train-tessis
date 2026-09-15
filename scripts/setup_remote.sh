#!/usr/bin/env bash
# ==================================================================
# setup_remote.sh — Setup automatizado para dc-2019 (rama `remote`)
# GPU: RTX 4070 Super (12 GB) | Python 3.14 (sistema) | CUDA 12.6 wheel
#
# CERO instalaciones al sistema:
#   - NO instala CUDA (el wheel de torch trae su propia CUDA)
#   - NO usa pyenv ni compila Python
#   - NO usa sudo/apt (solo crea el venv e instala paquetes pip adentro)
#
# Uso:
#   bash scripts/setup_remote.sh                         # setup basico
#   bash scripts/setup_remote.sh --with-dataset         # + pasar dataset a data/
# ==================================================================
set -euo pipefail

WITH_DATASET=false
for arg in "$@"; do
    case "$arg" in
        --with-dataset) WITH_DATASET=true ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"

echo "============================================================"
echo "  SETUP REMOTE — dc-2019"
echo "  Repo: $REPO_DIR"
echo "============================================================"

# ----------------------------------------------------------
# 1. Verificar prerequisites
# ----------------------------------------------------------
echo ""
echo "[1/5] Verificando prerequisites..."

if ! command -v python3 > /dev/null 2>&1; then
    echo "ERROR: python3 no encontrado." >&2
    exit 1
fi
if ! command -v curl > /dev/null 2>&1; then
    echo "ERROR: curl no encontrado." >&2
    exit 1
fi

PYTHON_VERSION="$(python3 --version)"
echo "  OK: $PYTHON_VERSION"
if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "  AVISO: tarjeta NVIDIA no detectada (GPU no disponible)."
    echo "  El entrenamiento con cuda no funcionara."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 \
        | sed 's/^/  GPU detectada: /'
fi

# ----------------------------------------------------------
# 2. Crear entorno virtual (sin pip)
# ----------------------------------------------------------
echo ""
echo "[2/5] Creando entorno virtual en $VENV_DIR ..."

if [ -d "$VENV_DIR" ]; then
    echo "  .venv ya existe. Eliminando..."
    rm -rf "$VENV_DIR"
fi

python3 -m venv --without-pip "$VENV_DIR"
echo "  OK: venv creado"

# ----------------------------------------------------------
# 3. Bootstrap pip dentro del venv (sin sudo)
# ----------------------------------------------------------
echo ""
echo "[3/5] Instalando pip dentro del venv..."
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV_DIR/bin/python" /tmp/get-pip.py > /dev/null 2>&1
rm -f /tmp/get-pip.py
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "  OK: $("$VENV_DIR/bin/python" -m pip --version)"

# ----------------------------------------------------------
# 4. Instalar PyTorch (CUDA 12.6) + dependencias
# ----------------------------------------------------------
echo ""
echo "[4/5] Instalando PyTorch 2.14.0 + CUDA 12.6 ..."
echo "  (descarga grande ~3GB, puede tardar varios minutos)"

"$VENV_DIR/bin/python" -m pip install \
    torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu126 \
    > /dev/null 2>&1

echo "  Instalando resto de dependencias..."
"$VENV_DIR/bin/python" -m pip install -r "$REPO_DIR/requirements-remote.txt" \
    > /dev/null 2>&1

echo "  OK: dependencias instaladas"

# ----------------------------------------------------------
# 5. Verificar instalacion
# ----------------------------------------------------------
echo ""
echo "[5/5] Verificando instalacion..."
"$VENV_DIR/bin/python" -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:    {torch.version.cuda}')
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
    p = torch.cuda.get_device_properties(0)
    print(f'  VRAM:            {p.total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA no disponible!')
"
echo ""

# ----------------------------------------------------------
# 6. Dataset original (opcional)
# ----------------------------------------------------------
if [ "$WITH_DATASET" = true ]; then
    echo "============================================================"
    echo "  Preparando dataset original:"
    echo "    ~/Downloads/CUHK-PEDES.zip -> data/CUHK-PEDES/"
    echo "============================================================"
    mkdir -p "$REPO_DIR/data/CUHK-PEDES"
    unzip -o "$HOME/Downloads/CUHK-PEDES.zip" "CUHK-PEDES/*" \
        -d /tmp/cuhkpedes_unpack > /dev/null 2>&1
    cp -r /tmp/cuhkpedes_unpack/CUHK-PEDES/* "$REPO_DIR/data/CUHK-PEDES/"
    rm -rf /tmp/cuhkpedes_unpack
    "$VENV_DIR/bin/python" "$REPO_DIR/scripts/convert_original_dataset.py"
fi

# ----------------------------------------------------------
# Resumen
# ----------------------------------------------------------
echo "============================================================"
echo "  SETUP REMOTE COMPLETADO"
echo ""
echo "  Activar entorno:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Convertir dataset original (si no se hizo):"
echo "    python scripts/convert_original_dataset.py"
echo ""
echo "  Smoke test:"
echo "    python scripts/test_one_epoch.py --dataset_source original --n_batches 50 --batch_size 16"
echo ""
echo "  Entrenamiento:"
echo "    python train.py --dataset_source original --epochs 60 --batch_size 16"
echo ""
echo "  Evaluacion:"
echo "    python evaluate.py --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar --dataset_source original"
echo "============================================================"