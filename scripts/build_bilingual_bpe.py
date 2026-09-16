#!/usr/bin/env python
"""
M1 — Entrenar tokenizador BPE bilingue (EN+ES) con la libreria `tokenizers`.

Corpus: captions EN de CUHK-PEDES (reid_raw.json) + captions ES traducidas
(reid_raw_bilingue.json). Genera data/tokenizer_bpe_en_es/{vocab.json,merges.txt,config.json}.

Especial: <pad> (id 0) y <unk> (id 1) al inicio. El nn.Embedding usa padding_idx=0.

Uso: .venv/bin/python scripts/build_bilingual_bpe.py [--vocab_size 24000]
"""

import os
import json
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PARENT_DIR, '..', 'data', 'CUHK-PEDES')
OUT_DIR = os.path.join(PARENT_DIR, '..', 'data', 'tokenizer_bpe_en_es')

SPECIAL_TOKENS = ["<pad>", "<unk>"]


def iter_corpus():
    for path in (os.path.join(DATASET_DIR, 'reid_raw.json'),
                 os.path.join(DATASET_DIR, 'reid_raw_bilingue.json')):
        if not os.path.isfile(path):
            continue
        annos = json.load(open(path, encoding='utf-8'))
        for anno in annos:
            for cap in anno['captions']:
                yield cap
            for cap in anno.get('captions_es', []):
                if cap:
                    yield cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab_size', type=int, default=24000)
    args = ap.parse_args()

    count = 0
    for _ in iter_corpus():
        count += 1
    print(f"Captions del corpus (EN+ES): {count}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size,
                                  min_frequency=2,
                                  special_tokens=SPECIAL_TOKENS,
                                  show_progress=True,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    tokenizer.train_from_iterator(iter_corpus(), trainer=trainer, length=count)

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_path = os.path.join(OUT_DIR, 'config.json')
    tokenizer.save(str(cfg_path))

    saved = Tokenizer.from_file(str(cfg_path))
    vocab = saved.get_vocab()
    print(f"\nOK -> {OUT_DIR}")
    print(f"Vocab total: {len(vocab)} (target {args.vocab_size})")
    print(f"<pad> id = {vocab.get('<pad>')}, <unk> id = {vocab.get('<unk>')}")

    for s in ["She wears a purple long sleeved, ankle length dress.",
              "Ella lleva un vestido purpura de manga larga y unos zapatos azules."]:
        enc = saved.encode(s)
        print(f"  ENCODER[{s!r}] ids[:16]={enc.ids[:16]} n_tokens={len(enc.ids)}")


if __name__ == '__main__':
    main()