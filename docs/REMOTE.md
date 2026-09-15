# Entorno remoto — especificaciones

> **Plantilla a completar.**
> Cuando se configure la máquina remota para entrenamiento,
> llenar esta tabla con las especificaciones reales y cualquier
> diferencia respecto al entorno local.

## Identificación

| Campo | Valor |
|-------|-------|
| Hostname | _completar_ |
| Proveedor | _ej: AWS p3.2xlarge, GCP n1-standard-8, máquina del lab, etc._ |
| IP / hostname completo | _completar_ |
| Sistema operativo | _ej: Ubuntu 22.04 LTS_ |
| Kernel | _completar_ |
| Usuario | _ej: yerson_ |
| Fecha de setup | _YYYY-MM-DD_ |
| Conexión SSH | _ej: `ssh user@host`_ |

## Hardware

| Componente | Especificación | Notas |
|-----------|----------------|-------|
| CPU | _ej: Intel Xeon Gold 6248 @ 2.5 GHz_ | _# cores, # threads_ |
| RAM | _ej: 64 GB_ | |
| Swap | _ej: 8 GB_ | |
| GPU(s) | _ej: 1× NVIDIA A100 40 GB_ | _modelo exacto + VRAM_ |
| Compute capability | _ej: 8.0 (Ampere)_ | |
| Driver NVIDIA | _ej: 535.xx_ | `nvidia-smi` |
| CUDA toolkit (build) | _ej: 12.1_ | `nvcc --version` |
| Disco | _ej: 500 GB SSD_ | _cuánto libre_ |

## Software

| Componente | Versión | Notas |
|-----------|---------|-------|
| Python | _ej: 3.8.10_ | |
| PyTorch | _ej: 1.13.1+cu117_ | |
| CUDA (PyTorch) | _ej: 11.7_ | |
| cuDNN | _ej: 8500_ | |
| transformers | _ej: 4.46.3_ | |
| datasets | _ej: 3.1.0_ | |
| numpy | _ej: 1.24.4_ | |
| Pillow | _ej: 10.4.0_ | |

Instalación (referencia):
```bash
# Ajustar el comando a la versión de CUDA disponible
pip install torch==X.X.X+cuXXX torchvision \
    --extra-index-url https://download.pytorch.org/whl/cuXXX
pip install -r requirements.txt
```

## Entorno virtual

```bash
# Comandos exactos para activar el entorno
cd /path/to/textreid-train
source .venv/bin/activate  # o conda activate, etc.
```

## Configuración usada para entrenamiento

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `batch_size` | _ej: 32_ | _depende de VRAM disponible_ |
| `num_workers` | _ej: 8_ | _# CPUs / 2 típicamente_ |
| `epochs` | _ej: 60_ | |
| Learning rate | _ej: 0.001_ | |
| `epoch_decay` | _ej: [20, 40]_ | |

## Rendimiento esperado

| Métrica | Valor estimado | Medido real |
|---------|---------------|-------------|
| Tiempo por batch | _ej: 0.4 s_ | |
| VRAM peak (train) | _ej: 8 GB_ | |
| Batches por epoch | 22\,431 (HF) / 5\,008 (original) | |
| Tiempo por epoch | _estimar_ | |
| Tiempo total (60 epochs) | _estimar_ | |

## Pasos de deployment

```bash
# 1. Clonar repo
git clone https://github.com/yerson001/textreid-train-tessis.git
cd textreid-train-tessis

# 2. Crear venv e instalar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cuXXX

# 3. Bajar dataset (si se usa HF)
python scripts/download_hf_dataset.py

# 4. Smoke test
python scripts/test_one_epoch.py --n_batches 50 --batch_size 8

# 5. Entrenamiento completo
nohup python train.py --epochs 60 --batch_size 32 \
    --output_dir ./data/checkpoints > logs/train.log 2>&1 &

# 6. Monitorear
tail -f logs/train.log
nvidia-smi

# 7. Evaluación
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source huggingface \
    --split test
```

## Transferencia de checkpoints a local

```bash
# Desde la PC remota
scp user@remote:/path/textreid-train/data/checkpoints/TextReIDNet_latest.pth.tar ./

# En la PC local
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source huggingface \
    --split test
```

## Issues conocidos / Diferencias con local

> Anotar cualquier incompatibilidad observada durante el setup o
> entrenamiento (versiones diferentes, drivers faltantes, etc.)

- _ejemplo: driver NVIDIA 535 vs 580 de local_
- _ejemplo: TF32 habilitado por defecto en A100, distinto a GTX 1050_

## Métricas reportadas

| Split | Top-1 | Top-5 | Top-10 | mAP |
|-------|-------|-------|--------|-----|
| val   |       |       |        |     |
| test  |       |       |        |     |

---

## Plantilla rápida para llenar

Si solo querés copiar y pegar lo básico:

```markdown
## Identificación
- Hostname: 
- OS: 
- Fecha: 

## Hardware
- CPU: 
- RAM: 
- GPU: 
- VRAM: 
- CUDA: 

## Software
- Python: 
- PyTorch: 

## Config
- batch_size: 
- epochs: 

## Resultados
- test Top-1: 
- test mAP: 
```
