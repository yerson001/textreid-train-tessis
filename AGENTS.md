# AGENTS.md

Repo de entrenamiento/evaluación de **TextReIDNet** (EfficientNet-B0 + BERT/BiGRU + DSC, ~32.27M params) sobre **CUHK-PEDES**, basado en el paper de Agyeman & Rinner (IEEE Access 2024). Guía autoritativa: `docs/REMOTE.md` (español).

## Ramas git

- `main` = base limpia, **no tocar**.
- `remote` = esta máquina (`dc-2019`), **todo el trabajo va aquí**. Es la rama actual.
- `local` = máquina `yrsn` (GTX 1050). No crear/mezclar a menos que el usuario lo pida: **las ramas son entornos separados, no se fusionan** (comparten solo `main`).
- Identidad git del repo: `yerson001 <yhon.sanchez@ucsp.edu.pe>`.

## Entorno (cero instalaciones al sistema)

- `.venv/` dentro del repo, Python 3.14 del sistema. Buscar `source .venv/bin/activate` o usar `.venv/bin/python` directamente.
- PyTorch **2.14.0+cu126** (el wheel trae su propia CUDA 12.6). **No instalar nada con sudo/apt/pyenv**; el CUDA toolkit 12.4 del sistema (`nvcc`) no se usa para entrenar.
- Install/limpieza desde cero: `bash scripts/setup_remote.sh [--with-dataset]`.

## Comandos exactos (dataset source = `original` en esta máquina)

```bash
# Convertir dataset original en data/CUHK-PEDES/ (solo si falta reid_raw.json)
python scripts/convert_original_dataset.py

# Smoke test del pipeline (~50 batches, ~1 min)
python scripts/test_one_epoch.py --dataset_source original --n_batches 50 --batch_size 16

# Entrenar (total de épocas SIEMPRE; auto-retoma si existe checkpoint)
./scripts/train.sh 60 16        # o: python train.py --dataset_source original --epochs 60 --batch_size 16

# Retomar manualmente desde un checkpoint
python train.py --dataset_source original --epochs 60 --batch_size 16 --resume
python train.py --dataset_source original --epochs 60 --batch_size 16 --resume data/checkpoints/TextReIDNet_epoch10.pth.tar

# Evaluar en test
python evaluate.py --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar --dataset_source original --split test
```

`./scripts/train.sh` resuelve su propio path (`cd $(dirname "$0")/..`) → funciona desde cualquier directorio. Si existe `data/checkpoints/TextReIDNet_latest.pth.tar`, pasa `--resume` automáticamente y continúa en la época guardada+1 hasta el total pedido.

## Dataset

- `data/CUHK-PEDES/` = original: `imgs/` + `caption_all.json`. **`caption_all.json` no trae campo `split`**; `convert_original_dataset.py` genera `reid_raw.json` con split por rango de id: train 1–11003 / val 11004–12003 / test 12004–13003 (40,206 imgs, 13,003 personas, ~2 captions/img).
- `~/Downloads/CUHK-SYSU.zip` es de otro proyecto (person search), **no se usa**.
- Fuente alternativa `huggingface` (`PeterPanTheGenius/CUHK-PEDES`, resized 128×128) existe en el código pero en esta máquina se entrena con `original`.

## Convenciones de versión (bugs ya corregidos — no reintroducir)

- Pillow ≥10 eliminó `Image.ANTIALIAS` → en `datasets/bases.py` usar `_RESAMPLE` (ya definido = `Image.Resampling.LANCZOS`).
- numpy ≥2 eliminó `np.in1d` → `np.isin` (ya aplicado en `evaluation/evaluations.py`).
- torch ≥2: `torch.amp.GradScaler('cuda')` y `torch.autocast(device_type=...)` (ya en `train.py`/`scripts/test_one_epoch.py`).
- Checkpoints (por época `TextReIDNet_epoch{N}.pth.tar` + `_latest.pth.tar`): claves `epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, loss`. `train.py`/`evaluate.py` los cargan con `weights_only=False` (archivos propios); el `loss` se guarda como `float` para que el `weights_only` por defecto no falle en otros lectores.
- `config.py` se selecciona por **hostname**: la rama se llama `remote`, pero el caso es `'dc-2019'` (`num_workers=8`, `batch_size=16`). No cambiar a menos que se migre de equipo.

## Operaciones (gotchas de entorno)

- Rendimiento real: ~10.5 batch/s → **~6.5 min/época** (4,251 batches a bs16); VRAM ~4.4 GB de 12.4 GB. bs16 es el default seguro de esta GPU.
- **No** lanzar entrenamientos largos con `nohup ... &` dentro de un tool call de bash: al expirar el timeout (120 s por defecto) el tool mata el grupo de procesos y se pierde la corrida. Usar `setsid ... < /dev/null >> log 2>&1 &` para desacoplar, o pasar `timeout` grande al tool, o darle el comando al usuario para su propia terminal.
- `pkill -f train.py` / `pgrep -f train.py` pueden matchear la propia shell que ejecuta el comando (self-kill). Para saber qué corre en GPU usar `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory`.

## Verificación

- No hay framework de lint/test. Verificación = smoke test (`test_one_epoch.py`) + `evaluate.py`. Chequear sintaxis con `.venv/bin/python -m py_compile <file>` y `bash -n` para scripts; el setup corre basicamente con `bash scripts/setup_remote.sh`.
- `data/`, `.venv/`, `logs/`, `*.pth` están en `.gitignore`: los artefactos de entrenamiento nunca se commitean.