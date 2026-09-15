# textreid-train-tessis

A clean, training-only version of the **DecentralizedTextReIDNet** project.

Trains the **TextReIDNet** model (EfficientNet-B0 + BERT + BiGRU) on **CUHK-PEDES** for text-based person re-identification.

> The original project is a full decentralized multi-camera surveillance system with inference scripts, a Flask web app, MySQL/MQTT, and human detection. This version strips all of that — it only has what you need to **train** and **evaluate**.

---

## What's in this project

```
textreid-train/
├── config.py                  # Simplified config (training + eval only)
├── train.py                   # Training script
├── evaluate.py                # Evaluation script (Top-1, Top-5, Top-10, mAP)
├── requirements.txt           # Python dependencies
├── model/
│   ├── textreidnet.py         # Main ReID model
│   ├── visual_network.py      # EfficientNet-B0 backbone
│   ├── language_network.py    # BERT + BiGRU
│   ├── efficientnet_backbone.py
│   └── model_utils.py
├── datasets/
│   ├── bases.py               # ImageTextDataset, ImageDataset, TextDataset
│   ├── cuhkpedes.py           # Loader for original CUHK-PEDES
│   ├── cuhkpedes_hf.py        # Loader for the PeterPanTheGenius/HF version
│   ├── cuhkpedes_dataloader.py # Unified dataloader builder
│   ├── bert_tokenizer.py
│   ├── simple_tokenizer.py
│   └── tiktoken_tokenizer.py
├── evaluation/
│   ├── ranking_loss.py        # Triplet-style ranking loss
│   ├── identity_loss.py       # Cross-entropy identity loss
│   ├── focal_loss.py          # Pure-Python sigmoid focal loss
│   └── evaluations.py         # compute Top-k, mAP
├── utils/
│   ├── iotools.py
│   └── miscellaneous_utils.py
├── scripts/
│   ├── download_hf_dataset.py # Downloads and converts HF dataset
│   └── test_one_epoch.py      # Quick smoke test (50 batches)
├── data/                      # Created at runtime
│   ├── CUHK-PEDES-HF/         # HF dataset (after download script)
│   ├── CUHK-PEDES/            # Original dataset (after email request)
│   └── checkpoints/           # Saved model weights
├── docs/
│   ├── paper.md               # Spanish translation of paper
│   └── LOCAL.md           # Detailed execution guide
├── logs/                      # Training/eval logs
└── notebooks/
    └── 01_lab_validate_dataset.ipynb
```

---

## Quick start

**Para instrucciones detalladas** (instalación paso a paso, smoke test, troubleshooting, batch size por GPU, estructura del dataset): ver **`docs/LOCAL.md`**.

Resumen rápido:

```bash
# 1. Instalar dependencias
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

# 2. Bajar el dataset (Opción A — HuggingFace, recomendado)
python scripts/download_hf_dataset.py

# 3. Test rápido (valida pipeline en ~1 minuto)
python scripts/test_one_epoch.py --n_batches 50 --batch_size 8

# 4. Entrenar
python train.py --dataset_source huggingface --epochs 60

# 5. Evaluar
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source huggingface \
    --split test
```

Para el dataset original por email, ver `docs/LOCAL.md` §3 (Opción B).

---

## Two dataset sources

### Option A: HuggingFace (`PeterPanTheGenius/CUHK-PEDES`) — recommended

- Resized to 128×128 thumbnails.
- One-command setup, no email needed.
- Expected Top-1: ~45-52%.

### Option B: Original CUHK-PEDES — best quality, needs email

- Email `tong.xiao.work@gmail.com` from a **university email**, ask for "CUHK-PEDES dataset for research purposes".
- Original image quality.
- Expected Top-1: ~54% (matches the paper).

For detailed instructions on how to install, run, and troubleshoot both options, see **`docs/LOCAL.md`**.

---

## Environments

The project documents two execution environments:

| Env | File | Description |
|-----|------|-------------|
| **Local** | [`docs/LOCAL.md`](docs/LOCAL.md) | This PC (`yrsn`, GTX 1050, Ubuntu 26.04). Used for development and smoke tests. |
| **Remote** | [`docs/REMOTE.md`](docs/REMOTE.md) | Template for the remote training machine (to be filled). |

---

## Expected results

| Dataset source | Expected Top-1 | Expected mAP |
|----------------|----------------|--------------|
| HuggingFace (128×128) | ~45-52% | ~40-48% |
| Original CUHK-PEDES | ~54% | ~50% |

---

## Training details

- **Optimizer:** AdamW (β1=0.90, β2=0.999)
- **Learning rate:** 1e-3, decays at epoch 20 and 40 (×0.1)
- **Loss:** α·RankingLoss + β·IdentityLoss (α=β=1.0)
- **Triplet margin:** 0.5
- **Embedding dim:** 1024
- **Mixed precision:** FP16 (GPU) / BF16 (CPU)
- **Total parameters:** ~32.27 M

---

## Architecture summary

- **Visual branch:** EfficientNet-B0 (ImageNet pretrained) → DSC → Pool → DSC
- **Text branch:** BERT tokenizer → Embedding → BiGRU → max-pool → DSC
- **Joint embedding:** Both branches share the last DSC → 1024-dim vector
- **Similarity:** Cosine

For the full architecture with line-by-line explanations, see `docs/LOCAL.md` and the PDF book at `book/main.pdf`.

---

## What's different from the original project

| Removed | Reason |
|---------|--------|
| `model/unity.py` | Wraps detection + re-ID; not needed for training |
| `model/human_detection_network.py` | Person detection model |
| `model/feature_pyramid_network.py`, `mask_head.py`, `parsing_head.py` | Used only by detection |
| `inference/` scripts | For running on images/videos |
| `surveillance_application/` | Flask app, MySQL, MQTT |
| `demo/` | Web demo |
| `setup.py` | Was for building focalloss CUDA ext |

What stayed: TextReIDNet, VisualNetwork (EfficientNet-B0), LanguageNetwork (BERT + BiGRU), dataset loaders, loss functions, evaluation metrics.

---

## Known issues / patches applied

Three bugs were found and fixed during testing:

1. **`datasets/bases.py:215`** — `TextDataset` converts `token_ids` to `torch.long`.
2. **`evaluation/evaluations.py:17`** — `calculate_similarity` returns numpy (not torch tensor).
3. **`evaluation/evaluations.py:63`** — `evaluate` indexes similarity columns (not rows).

---

## References

- **Agyeman, R. & Rinner, B.** (2024). Decentralized Text-Based Person Re-Identification in Multicamera Networks. *IEEE Access*. DOI: [10.1109/ACCESS.2024.3501382](https://doi.org/10.1109/ACCESS.2024.3501382). Spanish translation in `docs/paper.md`.
- **Tan, M. & Le, Q. V.** (2019). EfficientNet. *ICML*.
- **Devlin, J. et al.** (2018). BERT. *NAACL*.
- **Li, S. et al.** (2017). Person Search with Natural Language Description. *ICCV*.

---

## License

Based on the paper by Agyeman & Rinner (IEEE Access, 2024). Please cite their work if you use this code.
