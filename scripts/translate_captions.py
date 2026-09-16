#!/usr/bin/env python
"""
M1 — Traduccion EN->ES de captions CUHK-PEDES.

Genera `data/CUHK-PEDES/reid_raw_bilingue.json` (copia de reid_raw.json con una
clave nueva `captions_es` por entrada; lista vacia si no se traduce).

Estrategia barata por ahora:
  - traduce TODAS las captions de val y test (para poder evaluar ES y cross-lingual)
  - mas las captions de los primeros `--max_train_entries` de train (~5000 captions)

Motor: Helsinki-NLP/opus-mt-en-es (MarianMT, ~310MB). GPU si esta disponible.

Uso:
  .venv/bin/python scripts/translate_captions.py [--max_train_entries 2500] [--batch 64]
"""

import os
import json
import argparse
import time
import torch
from transformers import MarianMTModel, MarianTokenizer

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PARENT_DIR, '..', 'data', 'CUHK-PEDES')
ANNO_PATH = os.path.join(DATASET_DIR, 'reid_raw.json')
OUT_PATH = os.path.join(DATASET_DIR, 'reid_raw_bilingue.json')

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_train_entries', type=int, default=2500,
                    help='numero de entradas train cuyas captions se traducen (~2 captions/entrada)')
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()

    annos = json.load(open(ANNO_PATH))
    print(f"Total entradas: {len(annos)}")

    # seleccionar entradas a traducir
    to_translate = []
    train_count = 0
    for i, anno in enumerate(annos):
        if anno['split'] in ('val', 'test'):
            to_translate.append(i)
        elif anno['split'] == 'train' and train_count < args.max_train_entries:
            to_translate.append(i)
            train_count += 1

    n_captions = sum(len(annos[i]['captions']) for i in to_translate)
    print(f"Entradas a traducir: {len(to_translate)} | captions: {n_captions} "
          f"(train subset: {train_count})")

    print(f"Cargando modelo {MODEL_NAME} ...")
    tok = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Device: {device}")

    # lista plana "indice -> (tex_index, caption_index)"
    flat = []
    for idx in to_translate:
        for c_idx, cap in enumerate(annos[idx]['captions']):
            flat.append((idx, c_idx, cap))
    print(f"Total captions a traducir: {len(flat)}")

    t0 = time.time()
    for start in range(0, len(flat), args.batch):
        chunk = flat[start:start + args.batch]
        texts = [c for _, _, c in chunk]
        inputs = tok(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128)
        decoded = [tok.decode(t, skip_special_tokens=True) for t in out.cpu()]
        for j, (idx, c_idx, _) in enumerate(chunk):
            annos[idx].setdefault('captions_es', [])
            while len(annos[idx]['captions_es']) <= c_idx:
                annos[idx]['captions_es'].append('')
            annos[idx]['captions_es'][c_idx] = decoded[j]
        done = start + len(chunk)
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        print(f"  {done}/{len(flat)}  ({len(chunk)})  {rate:.1f} capt/s", flush=True)

    for anno in annos:
        anno.setdefault('captions_es', [])

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(annos, f, ensure_ascii=False, indent=1)

    es_total = sum(len(a['captions_es']) for a in annos)
    print(f"\nOK -> {OUT_PATH}")
    print(f"Captions ES generadas: {es_total}")

    # muestra
    sample_idx = next(i for i in to_translate if annos[i]['captions_es'])
    s = annos[sample_idx]
    for en, es in zip(s['captions'], s['captions_es']):
        print(f"  EN: {en}\n  ES: {es}\n")


if __name__ == '__main__':
    main()