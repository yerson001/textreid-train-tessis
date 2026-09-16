# Entorno remoto — dc-2019

> Configuración de la máquina remota de entrenamiento.
> Rama `remote` del repositorio.
> **Cero instalaciones al sistema** — el wheel de PyTorch trae su propia CUDA.

## Identificación

| Campo | Valor |
|-------|-------|
| Hostname | `dc-2019` |
| Sistema operativo | Ubuntu 26.04.1 LTS (Resolute Raccoon) |
| Fecha de setup | 2025-09-15 |

## Hardware

| Componente | Especificación | Notas |
|-----------|----------------|-------|
| CPU | Intel Core i7-14700F @ 2.10 GHz | 20 cores / 28 threads |
| RAM | 30 GB | |
| GPU | NVIDIA GeForce RTX 4070 SUPER | 12.4 GB VRAM, Ada Lovelace (sm_89) |
| Driver NVIDIA | 595.84 | soporta CUDA hasta 13.2 |
| CUDA toolkit (sistema) | 12.4 | **NO se usa para entrenar** — solo compilar si hiciera falta |
| Disco | 645 GB total, ~580 GB libre | NVMe |

## Software

| Componente | Versión |
|-----------|---------|
| Python | 3.14.4 (sistema, no se instala nada) |
| PyTorch | **2.14.0+cu126** |
| CUDA (dentro del wheel) | 12.6 (se trae consigo, no usa el toolkit del sistema) |
| torchvision | 0.29.0+cu126 |
| transformers | 4.57.6 |
| datasets | 5.0.1 |
| numpy | 2.5.2 |
| Pillow | 12.3.0 |

## Setup (primera vez)

```bash
cd textreid_train

# 1. Crear entorno + instalar todo (sin sudo, sin pyenv)
bash scripts/setup_remote.sh

# 2. Activar entorno
source .venv/bin/activate

# 3. Convertir dataset original (si ya está en ~/Downloads)
bash scripts/setup_remote.sh --with-dataset

# 3b. O manualmente:
mkdir -p data/CUHK-PEDES
unzip ~/Downloads/CUHK-PEDES.zip "CUHK-PEDES/*" -d /tmp/u && cp -r /tmp/u/CUHK-PEDES/* data/CUHK-PEDES/ && rm -rf /tmp/u
python scripts/convert_original_dataset.py
```

## Dataset original (CUHK-PEDES)

El zip del dataset (`~/Downloads/CUHK-PEDES.zip`) contiene:
- `caption_all.json` — 40,206 anotaciones (sin campo `split`)
- `imgs/` — imágenes BMP (cam_a, cam_b) y JPG (Market, CUHK03, CUHK01, queries)

El script `scripts/convert_original_dataset.py` genera `reid_raw.json` con:
- Split por rango de id: train (1–11003) / val (11004–12003) / test (12004–13003)
- 40,206 imágenes, 13,003 personas, ~2 captions/imagen

## Configuración de entrenamiento

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `batch_size` | 16 | RTX 4070 Super (12 GB, seguro) |
| `num_workers` | 8 | i7-14700F (28 threads) |
| `epochs` | 60 | Default |
| Learning rate | 0.001 | Default |
| `epoch_decay` | [20, 40] | Default |

## Comandos de entrenamiento

```bash
source .venv/bin/activate

# Smoke test
python scripts/test_one_epoch.py --dataset_source original --n_batches 50 --batch_size 16

# Entrenamiento completo
python train.py --dataset_source original --epochs 60 --batch_size 16

# Evaluacion
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source original \
    --split test
```

## Issues conocidos

- La CUDA 12.4 del sistema (`nvcc`) no se usa: el wheel cu126 trae su propio CUDA runtime dentro de `.venv`.
- `pysqlite3-binary` removido de requirements (innecesario en Python 3.11+).
- `caption_all.json` no trae campo `split`: se genera con `convert_original_dataset.py`.
