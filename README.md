# textreid-train

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
│   ├── cuhkpedes.py           # Loader for the original CUHK-PEDES
│   ├── cuhkpedes_hf.py        # Loader for the PeterPanTheGenius/HF version
│   ├── cuhkpedes_dataloader.py # Unified dataloader builder
│   ├── bases.py               # ImageTextDataset, ImageDataset, TextDataset
│   ├── bert_tokenizer.py
│   └── ...
├── evaluation/
│   ├── ranking_loss.py        # Triplet-style ranking loss
│   ├── identity_loss.py       # Cross-entropy identity loss
│   ├── focal_loss.py          # Patched: uses pure-Python implementation
│   └── evaluations.py         # compute Top-k, mAP
├── utils/
│   ├── iotools.py
│   └── miscellaneous_utils.py
├── scripts/
│   └── download_hf_dataset.py # Downloads and converts the HF dataset
├── data/                      # Created at runtime
│   ├── CUHK-PEDES-HF/         # HF dataset (after running the download script)
│   └── checkpoints/           # Saved model weights
└── logs/                      # Training/eval logs
```

---

## Two dataset sources

This project supports two ways to get the CUHK-PEDES dataset:

### Option A: HuggingFace (PeterPanTheGenius/CUHK-PEDES) — works now

The HuggingFace version has the same images but:
- Resized to 128×128 thumbnails
- Stored in Parquet format (no folders, no `reid_raw.json`)
- No person IDs, no train/val/test splits

Pros:
- Can download and start training **immediately**
- No email needed

Cons:
- Lower image quality (128×128 vs original ~256×128+)
- Expected Top-1: ~45-52% (vs paper's 54.02%)

### Option B: Original CUHK-PEDES — best quality, needs email

The real dataset with original-resolution images and proper annotations. To get it:
1. Email `tong.xiao.work@gmail.com` from a **university email**
2. Ask for "CUHK-PEDES dataset for research purposes"
3. They'll send a download link (or instructions)

Pros:
- Original image quality
- Expected Top-1: ~54% (matching the paper)

---

## Quick start

### 1. Install dependencies

```bash
export PYENV_VERSION=3.8.10  # adjust if you use a different Python manager
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

### 2. Get the dataset

For the HuggingFace version (fastest):
```bash
python scripts/download_hf_dataset.py
```

For the original version:
1. Email `tong.xiao.work@gmail.com`
2. Unzip what they send into `data/CUHK-PEDES/`
3. Make sure it contains `reid_raw.json` and `imgs/` with subfolders

### 3. Train

```bash
# With HuggingFace dataset
python train.py --dataset_source huggingface --epochs 60

# With original dataset
python train.py --dataset_source original --epochs 60
```

Other useful flags:
- `--batch_size 8` — lower if OOM
- `--lr 0.001` — learning rate
- `--seed 3407` — for reproducibility
- `--output_dir ./data/checkpoints` — where to save weights

### 4. Evaluate

```bash
python evaluate.py \
    --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar \
    --dataset_source huggingface \
    --split test
```

Outputs:
- Top-1, Top-5, Top-10 accuracy
- mAP (mean Average Precision)

---

## Expected results

| Dataset source | Expected Top-1 | Notes |
|----------------|----------------|-------|
| HuggingFace (128×128) | ~45-52% | Limited by image resolution |
| Original CUHK-PEDES | ~54% | Matches paper |

---

## What's different from the original project

| Removed | Reason |
|---------|--------|
| `model/unity.py` | Wraps detection + re-ID; not needed for pure training |
| `model/human_detection_network.py` | Person detection model; not used in ReID-only training |
| `model/feature_pyramid_network.py` | Used only by detection |
| `model/mask_head.py`, `parsing_head.py` | Used only by detection |
| `model/process_parsing_result.py` | Used only by detection |
| `inference/` scripts | For running on images/videos, not training |
| `surveillance_application/` | Flask app, MySQL, MQTT — decentralized demo |
| `demo/` | Web demo |
| `dataset/mhpv2*`, `dataset/mals*` | Other datasets (not needed for CUHK-PEDES training) |
| `setup.py` | Was for building the focalloss CUDA ext (doesn't compile here) |

What stayed:
- TextReIDNet (the actual model being trained)
- VisualNetwork (EfficientNet-B0 backbone)
- LanguageNetwork (BERT + BiGRU)
- Dataset loaders (adapted for both sources)
- Loss functions
- Evaluation metrics

---

## Training details

- **Optimizer:** AdamW (β1=0.90, β2=0.999)
- **Learning rate:** 1e-3, decays at epoch 20 and 40
- **Batch size:** 8 (RTX 2060) / 16 (RTX 3090/A100)
- **Epochs:** 60
- **Mixed precision:** FP16 (RTX) / BF16 (CPU)
- **Loss:** α·RankingLoss + β·IdentityLoss (α=β=1.0)
- **Expected time:** 2-4 days on RTX 2060 for full training
# textreid-train-tessis
