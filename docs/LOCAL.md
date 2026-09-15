# Guía de ejecución — `textreid-train-tessis`

> Documento complementario al README.
> Contiene los pasos detallados para poner a funcionar el proyecto desde cero.

---

## Índice

1. [Requisitos previos](#1-requisitos-previos)
2. [Instalación](#2-instalación)
3. [Obtener el dataset](#3-obtener-el-dataset)
4. [Verificación rápida (smoke test)](#4-verificación-rápida-smoke-test)
5. [Entrenamiento](#5-entrenamiento)
6. [Evaluación](#6-evaluación)
7. [Solución de problemas](#7-solución-de-problemas)
8. [Estructura esperada del dataset original](#8-estructura-esperada-del-dataset-original)

---

## 1. Requisitos previos

### Hardware

| Componente | Mínimo | Recomendado |
|-----------|--------|-------------|
| GPU | NVIDIA con 4 GB VRAM | RTX 3090 / A100 (16+ GB) |
| RAM | 8 GB | 16 GB |
| Disco | 5 GB | 20 GB |

### Software

- **Python 3.8+** (probado con 3.8.10)
- **CUDA 11.7** (para PyTorch con soporte GPU)
- **pip** actualizado

Verificar versión de Python:
```bash
python --version
```

Verificar GPU (si tenés NVIDIA):
```bash
nvidia-smi
```

---

## 2. Instalación

### 2.1 Clonar el repositorio

```bash
git clone https://github.com/yerson001/textreid-train-tessis.git
cd textreid-train-tessis
```

### 2.2 Crear entorno virtual (recomendado)

```bash
python3.8 -m venv .venv
source .venv/bin/activate
```

En Windows:
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2.3 Instalar PyTorch con CUDA

Para CUDA 11.7 (recomendado):
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

Para CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Para CPU solamente (lento):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2.4 Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```
torch==1.13.1+cu117
torchvision==0.14.1+cu117
numpy>=1.21,<1.25
Pillow>=9.0,<11
tqdm
natsort
matplotlib
transformers>=4.20,<5
datasets
pyarrow
nltk
tiktoken
ftfy
pysqlite3-binary
```

### 2.5 Verificar instalación

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Salida esperada (con GPU):
```
Torch: 1.13.1+cu117
CUDA: True
```

---

## 3. Obtener el dataset

Este proyecto soporta **dos fuentes** para CUHK-PEDES. Elegí la que prefieras.

### Opción A — HuggingFace (`PeterPanTheGenius/CUHK-PEDES`)

**Ventaja:** un solo comando, sin permisos especiales.

**Desventaja:** imágenes a 128×128 (menor calidad que el original).

```bash
python scripts/download_hf_dataset.py
```

El script:
1. Descarga el Parquet (~550 MB) desde HuggingFace
2. Extrae las 34,042 imágenes a `data/CUHK-PEDES-HF/imgs/all/*.jpg`
3. Genera `data/CUHK-PEDES-HF/reid_raw.json` con los captions

**Salida esperada:**
```
Reading parquet from data/CUHK-PEDES-HF/raw.parquet...
Total rows: 238294
Unique images: 34042
Avg captions per image: 7.01
Wrote 34042 entries to data/CUHK-PEDES-HF/reid_raw.json
```

**Verificación:**
```bash
ls data/CUHK-PEDES-HF/
# raw.parquet  reid_raw.json  imgs/

ls data/CUHK-PEDES-HF/imgs/all/ | wc -l
# 34042
```

### Opción B — Original CUHK-PEDES (por email)

**Ventaja:** imágenes en resolución completa, ~54% Top-1 (paper).

**Pasos:**

1. **Email de solicitud:**
   - Destinatario: `tong.xiao.work@gmail.com`
   - From: **email universitario** (gmail/edu, no sirve personal)
   - Asunto: "CUHK-PEDES dataset for research purposes"
   - Cuerpo: una breve descripción de tu investigación

2. **Respuesta:** suelen responder en 1-3 días con un link (Google Drive o Baidu Pan).

3. **Descargar y descomprimir:**
   ```bash
   # Asumiendo que lo descargaste a ~/Downloads/CUHK-PEDES.zip
   mkdir -p data/CUHK-PEDES
   unzip ~/Downloads/CUHK-PEDES.zip -d data/CUHK-PEDES/
   ```

4. **Verificar estructura** (ver [sección 8](#8-estructura-esperada-del-dataset-original)):
   ```
   data/CUHK-PEDES/
   ├── reid_raw.json
   └── imgs/
       ├── cam_a/
       ├── cam_b/
       └── ...
   ```

---

## 4. Verificación rápida (smoke test)

Antes de comprometerte a un entrenamiento completo (60 epochs, ~2-4 días en RTX 2060), corré un test rápido de 50 batches:

```bash
python scripts/test_one_epoch.py \
    --n_batches 50 \
    --batch_size 8 \
    --output_dir ./data/test_checkpoints
```

**Salida esperada:**
```
============================================================
TEST: 50 batches @ batch_size=8
============================================================
Num classes: ~25531
Total parámetros: 32.27M
GPU: cuda
  Batch 10/50: rank=1.000 id=20.295 tot=21.295
  ...
  Batch 50/50: rank=1.939 id=20.296 tot=22.235
Checkpoint guardado en: data/test_checkpoints/TextReIDNet_test.pth.tar
TEST OK - pipeline end-to-end funcional
VRAM peak: ~2.82 GB
```

**Lo que valida este test:**
- ✓ Imports funcionan
- ✓ Dataset se carga correctamente
- ✓ Forward + backward sin crashes
- ✓ Checkpoint se guarda en formato compatible con `evaluate.py`
- ✓ VRAM cabe en GPUs modestas (~2.8 GB en GTX 1050)

Si todo salió bien, podés saltar a entrenamiento o evaluación.

### Probar evaluación con el checkpoint de test

```bash
python evaluate.py \
    --checkpoint data/test_checkpoints/TextReIDNet_test.pth.tar \
    --dataset_source huggingface \
    --split test \
    --batch_size 32
```

**Salida esperada** (con sólo 50 batches, el modelo no aprendió nada):
```
Results on test split:
  Top-1:  0.02%   ← casi 0% porque el modelo no entrenó
  Top-5:  0.08%
  Top-10: 0.17%
  mAP:    0.10%
```

Si ves estos números (≈0%), confirma que el pipeline funciona. Con un entrenamiento real, vas a ver Top-1 cercano al 50% (HF) o 54% (original).

---

## 5. Entrenamiento

### 5.1 Comando básico

**Con HuggingFace dataset:**
```bash
python train.py --dataset_source huggingface --epochs 60
```

**Con dataset original:**
```bash
python train.py --dataset_source original --epochs 60
```

### 5.2 Argumentos CLI

| Flag | Default | Descripción |
|------|---------|-------------|
| `--dataset_source` | `huggingface` | `huggingface` o `original` |
| `--epochs` | 60 (config) | Número de epochs |
| `--batch_size` | 8 (config, depende del hostname) | Tamaño de batch |
| `--lr` | 0.001 (config) | Learning rate |
| `--seed` | 3407 (config) | Semilla para reproducibilidad |
| `--output_dir` | `./data/checkpoints` | Dónde guardar checkpoints |

### 5.3 Selección de batch_size según GPU

| GPU | VRAM | Batch recomendado |
|-----|------|-------------------|
| GTX 1050 | 4 GB | 4 |
| RTX 2060 | 6 GB | 8 |
| RTX 3060 | 12 GB | 8-12 |
| RTX 3090 | 24 GB | 16-24 |
| A100 | 40-80 GB | 32-64 |
| Jetson Nano | 4 GB | 2-4 |

Si ves `CUDA out of memory`, bajá `--batch_size` a la mitad.

### 5.4 Salida esperada durante el entrenamiento

```
Epoch 1/1:   1%|          | 200/22431 [07:18<13:32:35,  2.27s/batch, rank=1.224, id=20.296, tot=21.519]
Epoch 1/1:   2%|          | 400/22431 [14:36<13:25:19,  2.27s/batch, tot=21.519]
...
Epoch 1 | Ranking Loss: 0.4521 | Identity Loss: 4.3210 | Total Loss: 4.7731
Checkpoint guardado en: data/checkpoints/TextReIDNet_epoch1.pth.tar
```

Por cada epoch se imprimen las pérdidas y se guardan dos checkpoints:
- `TextReIDNet_epochN.pth.tar` — el del epoch N
- `TextReIDNet_latest.pth.tar` — siempre el último (lo que usa `evaluate.py`)

### 5.5 Cómo reanudar un entrenamiento interrumpido

El código actual NO implementa resume automático. Si tu entrenamiento se cortó:

1. **Último checkpoint disponible**: `data/checkpoints/TextReIDNet_latest.pth.tar`
2. **Reanudar manualmente**: hay que modificar `train.py` para que cargue el checkpoint y arranque desde el epoch siguiente.

Truco rápido si querés agregar resume (5 líneas en `train.py`):
```python
import os
if os.path.exists('./data/checkpoints/TextReIDNet_latest.pth.tar'):
    ckpt = torch.load('./data/checkpoints/TextReIDNet_latest.pth.tar',
                      map_location=config.device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
else:
    start_epoch = 1

# Cambiar el for loop:
for current_epoch in range(start_epoch, config.epoch + 1):
    ...
```

### 5.6 Tiempo estimado

| GPU | Tiempo por epoch (HF) | Tiempo total (60 epochs) |
|-----|----------------------|--------------------------|
| Jetson Nano | ~5h | 12 días |
| GTX 1050 | ~13h | 32 días (no recomendado) |
| RTX 2060 | ~2-3h | 5-7 días |
| RTX 3090 | ~30min | 1 día |
| A100 | ~15-20min | 12-20 horas |

Con el dataset original (40k imágenes vs 179k del HF), el tiempo es ~4× menor.

---

## 6. Evaluación

### 6.1 Comando

```bash
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source huggingface \
    --split test
```

### 6.2 Argumentos

| Flag | Default | Descripción |
|------|---------|-------------|
| `--checkpoint` | (requerido) | Path al checkpoint `.pth.tar` |
| `--dataset_source` | `huggingface` | `huggingface` o `original` |
| `--split` | `test` | `test` o `val` |
| `--batch_size` | 32 | Tamaño de batch para extracción |
| `--device` | `cuda` | `cuda` o `cpu` |

### 6.3 Salida esperada

```
============================================================
Checkpoint:     data/checkpoints/TextReIDNet_latest.pth.tar
Dataset source: huggingface
Split:          test
Device:         cuda
============================================================
Images: 4256
Texts:  29930
Loading checkpoint from data/checkpoints/TextReIDNet_latest.pth.tar...
Checkpoint loaded.
Extracting image features: 100%|████████████| 133/133 [02:00<00:00,  1.10it/s]
Extracting text features:  100%|████████████| 936/936 [00:44<00:00, 22.20it/s]
Computing similarity matrix...

============================================================
Results on test split:
  Top-1:  XX.XX%
  Top-5:  XX.XX%
  Top-10: XX.XX%
  mAP:    XX.XX%
============================================================
```

Los resultados también se guardan en `logs/test.log` (modo append).

### 6.4 Interpretación de resultados

**Top-1 ~50%, mAP ~45%**: modelo entrenado decentemente.

**Top-1 bajo pero mAP razonable**: el modelo encuentra imágenes cercanas pero no acierta el top-1. Considerá entrenar más epochs.

**Top-1 alto, mAP bajo**: improbable, indicaría overfitting a un subconjunto.

Ver [Cap. 13 del PDF del libro] (`book/main.pdf`) para explicación detallada de cada métrica.

---

## 7. Solución de problemas

### Error: `CUDA out of memory`
Reducí `--batch_size`:
```bash
python train.py --batch_size 4 ...
```

### Error: `RuntimeError: Expected tensor ... to have one of the following scalar types: Long, Int`
Bug ya parcheado en `datasets/bases.py:215`. Si lo ves, asegurate de tener la versión actualizada:
```python
# datasets/bases.py línea 215
token_ids = token_ids.to(torch.long)
```

### Error: `ValueError: step must be greater than zero`
Bug ya parcheado en `evaluation/evaluations.py:63`. Si lo ves, verificá:
```python
# evaluation/evaluations.py línea 63
ap_i, cmc_i = calculate_ap(similarity[:, i], label_query[i], label_gallery)
# NOT similarity[i, :]
```

### Error: `Unicode character ... not set up for use in LaTeX`
Sólo afecta al compilar el PDF del libro, no al entrenamiento. Reemplazar el carácter Unicode problemático en el archivo `.tex`.

### El entrenamiento no mejora (Top-1 cercano a 0 después de muchos epochs)
Posibles causas:
1. **Learning rate muy alto**: bajalo con `--lr 0.0005`
2. **Datos no cargados correctamente**: verificá `data/CUHK-PEDES-HF/reid_raw.json`
3. **GPU no está usando**: verificá `nvidia-smi` durante el entrenamiento

### La evaluación da NaN o Inf
El checkpoint se corrompió o se entrenó con un config diferente. Volvé a entrenar desde cero.

---

## 8. Estructura esperada del dataset original

Si conseguiste el dataset original por email, debe tener esta estructura:

```
data/CUHK-PEDES/
├── reid_raw.json          # archivo crítico con anotaciones
└── imgs/
    ├── cam_a/             # o el nombre que tenga cada cámara
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    ├── cam_b/
    │   └── ...
    └── ...
```

### Formato de `reid_raw.json`

Cada elemento es un diccionario con:

```json
{
    "split": "train",
    "captions": ["caption 1", "caption 2"],
    "file_path": "cam_a/0001.jpg",
    "processed_tokens": [],
    "id": 11004
}
```

Campos requeridos:
- `file_path`: ruta relativa de la imagen (sin prefijo `imgs/`)
- `captions`: lista de strings (1 o más descripciones)
- `split`: `"train"`, `"val"` o `"test"`
- `id`: identificador entero de persona (1 a 13003)

Si el archivo que te pasaron tiene otro formato (ej. nombres de campos distintos), hay que adaptar `datasets/cuhkpedes.py:75-115` (`_process_anno`).

---

## 9. Checklist pre-entrenamiento

Antes de empezar un entrenamiento de varios días, verificá:

- [ ] GPU detectada (`nvidia-smi`)
- [ ] Dataset descargado y en `data/CUHK-PEDES-HF/` o `data/CUHK-PEDES/`
- [ ] Smoke test pasa: `python scripts/test_one_epoch.py --n_batches 50`
- [ ] Evaluación del checkpoint de test funciona: `python evaluate.py --checkpoint data/test_checkpoints/TextReIDNet_test.pth.tar --dataset_source huggingface`
- [ ] Espacio en disco suficiente (~10 GB para checkpoints de 60 epochs)
- [ ] Batch size adecuado para tu GPU (ver [sección 5.3](#53-selección-de-batch_size-según-gpu))

Si todo OK, estás listo para `python train.py`.

---

## 10. Referencias rápidas

- **Paper**: `docs/paper.md` (traducción al español)
- **README del proyecto**: `README.md` (resumen general)
- **Cuaderno de validación**: `notebooks/01_lab_validate_dataset.ipynb`
- **PDF libro completo** (no necesario para ejecutar): `book/main.pdf`
  - Si necesitás compilarlo: `cd book && make`

---

## 11. Entornos de ejecución

Este proyecto se ha ejecutado en dos tipos de máquina. Las
especificaciones exactas se documentan abajo para que al migrar a
otra PC puedas comparar.

### 11.1 Local — esta PC (host: `yrsn`)

Configuración usada durante el desarrollo y las pruebas.

#### Hardware

| Componente | Especificación |
|-----------|----------------|
| Hostname | `yrsn` |
| OS | Ubuntu 26.04 LTS (Resolute Raccoon) |
| Kernel | 7.0.0-31-generic |
| CPU | Intel Core i7-7700HQ @ 2.80 GHz (max 3.8 GHz) |
| Cores / Threads | 4 cores / 8 threads |
| RAM | 14 GB total (11 GB disponible) |
| Swap | 4 GB |
| GPU | NVIDIA GeForce GTX 1050 (4 GB VRAM) |
| Compute capability | 6.1 (Pascal) |
| Multiprocesadores | 5 SMs |
| Driver NVIDIA | 580.173.02 |
| CUDA toolkit (build) | 13.2 |
| Disco | 439 GB total, ~86 GB disponible |

#### Software

| Componente | Versión |
|-----------|---------|
| Python | 3.8.10 (vía pyenv) |
| PyTorch | 1.13.1+cu117 |
| CUDA (PyTorch) | 11.7 |
| cuDNN | 8500 |
| torchvision | 0.14.1+cu117 |
| transformers | 4.46.3 |
| datasets | 3.1.0 |
| tokenizers | 0.20.3 |
| pyarrow | 17.0.0 |
| numpy | 1.24.4 |
| Pillow | 10.4.0 |
| matplotlib | 3.7.5 |
| tqdm | 4.70.1 |
| tiktoken | 0.7.0 |
| ftfy | 6.2.3 |
| nltk | 3.9.1 |
| natsort | 8.4.0 |
| huggingface_hub | 0.36.2 |

#### Entorno virtual

Activación:
```bash
cd /home/yrsn/Dev/textreid-train
export PYENV_VERSION=3.8.10
source .venv/bin/activate
```

Python resuelto vía `.venv` (pyenv 3.8.10). Verificable con:
```bash
which python  # debe apuntar a /home/yrsn/Dev/textreid-train/.venv/bin/python
python --version  # Python 3.8.10
```

#### Rendimiento medido (en esta PC)

Con GTX 1050 + batch_size=8 + FP16:

| Métrica | Valor |
|---------|-------|
| Tiempo por batch | ~2.16 s |
| VRAM peak (train) | ~2.82 GB |
| VRAM peak (eval, batch 32) | ~2.0 GB |
| Batches por epoch (HF) | 22\,431 |
| Tiempo por epoch | \textasciitilde 13 h |
| Tiempo total (60 epochs) | \textasciitilde 32 días (no viable) |
| Tiempo evaluación completa | \textasciitilde 3 min (test, 4256 img + 29\,930 txt) |

**Conclusión**: esta PC sirve para desarrollo y validación, pero el
entrenamiento completo de 60 epochs es inviable. Usar el smoke
test (50 batches) o migrar a una máquina remota más potente.

#### Limitaciones observadas

- GPU clase 6.1 (Pascal, 2016): no soporta algunas instrucciones
  modernas de CUDA, pero funciona correctamente.
- 4 GB VRAM: limita `batch_size` a 4–8. Cualquier cosa mayor da OOM.
- CPU clase i7-7700HQ (2017): suficientemente rápido para data
  loading, pero no para cómputo numérico pesado.

### 11.2 Remoto — máquina potente

Ver [`docs/REMOTE.md`](REMOTE.md) para las especificaciones de la
máquina remota que se usará para el entrenamiento completo.

---

## 12. Reproducibilidad entre máquinas

Para reproducir los mismos resultados en otra máquina:

1. **Misma semilla**: `--seed 3407` (config default).
2. **Mismo dataset**: usar el mismo split (HF o original).
3. **Mismas versiones de PyTorch y CUDA**: idealmente `torch==1.13.1+cu117`.
4. **Misma configuración de hardware**: misma VRAM disponible y
   mismo `batch_size`.

Diferencias inevitables:
- Orden de operaciones de punto flotante (no determinístico en
  CUDA aunque se use `set_seed`).
- Velocidad: cambia con la GPU.
- Número de workers: ajustar `num_workers` al CPU disponible.
