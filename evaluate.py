"""
C. September 2024
Doc: Evaluation script for TextReIDNet on CUHK-PEDES.
      Computes Top-1, Top-5, Top-10 accuracy and mAP.

Usage:
    python evaluate.py --checkpoint data/checkpoints/TextReIDNet_latest.pth.tar
"""

import os
import sys
import argparse
import logging

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('../textreid-train'))

from config import sys_configuration
from utils.miscellaneous_utils import setup_logger
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet
from evaluation.evaluations import calculate_similarity, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TextReIDNet on CUHK-PEDES')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--dataset_source', type=str, default='huggingface',
                        choices=['huggingface', 'original'])
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tokenizer_type', type=str, default=None,
                        choices=['bert', 'bpe_en_es'],
                        help='Override tokenizer (bpe_en_es = bilingue EN-ES)')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Override vocab size (debe coincidir con el tokenizador)')
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help='Override text embedding dim')
    parser.add_argument('--feature_length', type=int, default=None,
                        help='Override joint feature dim')
    parser.add_argument('--evaluate_language', type=str, default=None,
                        choices=['en', 'es'],
                        help='Idioma de las captions de val/test')
    return parser.parse_args()


@torch.no_grad()
def extract_image_features(model, img_loader, device):
    model.eval()
    all_features = []
    all_pids = []
    all_paths = []

    for batch in tqdm(img_loader, desc='Extracting image features'):
        if isinstance(batch, (list, tuple)):
            imgs, pids, paths = batch[0], batch[1], batch[2]
        else:
            imgs, pids, paths = batch

        imgs = imgs.to(device)
        feats = model.image_embedding(imgs)
        feats = feats.view(feats.size(0), -1).cpu()

        all_features.append(feats)
        all_pids.extend(pids.tolist() if torch.is_tensor(pids) else list(pids))
        all_paths.extend(paths if isinstance(paths, list) else [paths])

    return torch.cat(all_features, 0), all_pids, all_paths


@torch.no_grad()
def extract_text_features(model, txt_loader, device):
    model.eval()
    all_features = []
    all_pids = []

    for batch in tqdm(txt_loader, desc='Extracting text features'):
        if isinstance(batch, (list, tuple)):
            pids, token_ids, token_lengths = batch[0], batch[1], batch[2]
        else:
            pids, token_ids, token_lengths = batch

        token_ids = token_ids.to(device)
        token_lengths = token_lengths.to(device)
        feats = model.text_embedding(token_ids, token_lengths)
        feats = feats.view(feats.size(0), -1).cpu()

        all_features.append(feats)
        all_pids.extend(pids.tolist() if torch.is_tensor(pids) else list(pids))

    return torch.cat(all_features, 0), all_pids


def main():
    args = parse_args()

    config = sys_configuration(dataset_name="CUHK-PEDES", dataset_source=args.dataset_source)
    config['batch_size'] = args.batch_size
    config['device'] = args.device
    config['model_testing_data_split'] = args.split

    if args.tokenizer_type is not None:
        config['tokenizer_type'] = args.tokenizer_type
    if args.vocab_size is not None:
        config['vocab_size'] = args.vocab_size
    if args.embedding_dim is not None:
        config['embedding_dim'] = args.embedding_dim
    if args.feature_length is not None:
        config['feature_length'] = args.feature_length
    if args.evaluate_language is not None:
        config['evaluate_language'] = args.evaluate_language
        config['bilingual'] = True
    if args.tokenizer_type == 'bpe_en_es':
        config['bilingual'] = True

    print("=" * 60)
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Dataset source: {args.dataset_source}")
    print(f"Split:          {args.split}")
    print(f"Tokenizer:      {config.tokenizer_type} | vocab {config.vocab_size}")
    print(f"Lang (eval):    {config.evaluate_language}")
    print(f"Device:         {args.device}")
    print("=" * 60)

    # Build dataloaders
    _, img_loader, txt_loader, _ = build_cuhkpedes_dataloader(config)
    print(f"Images: {len(img_loader.dataset)}")
    print(f"Texts:  {len(txt_loader.dataset)}")

    # Build model
    model = TextReIDNet(config).to(config.device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # Compatibilidad con checkpoints antiguos (baseline): la rama textual usaba
    # la misma DSC que la visual (no existia text_final_convolution).
    if not any(k.startswith('text_final_convolution') for k in sd):
        model.text_final_convolution = model.depthwise_seperable_convolution
        model.load_state_dict(sd, strict=False)
        print("(checkpoint antiguo: rama textual usa el DSC compartido)")
    else:
        model.load_state_dict(sd)
    print("Checkpoint loaded.")

    # Extract features
    img_feats, img_pids, img_paths = extract_image_features(model, img_loader, config.device)
    txt_feats, txt_pids = extract_text_features(model, txt_loader, config.device)

    # Compute similarity
    print("Computing similarity matrix...")
    similarity = calculate_similarity(img_feats, txt_feats)

    # Evaluate
    img_pids_t = torch.tensor(img_pids)
    txt_pids_t = torch.tensor(txt_pids)
    cmc, mAP = evaluate(similarity, txt_pids_t, img_pids_t)

    top1 = cmc[0] * 100
    top5 = cmc[4] * 100 if len(cmc) > 4 else 0
    top10 = cmc[9] * 100 if len(cmc) > 9 else 0

    print("\n" + "=" * 60)
    print(f"Results on {args.split} split:")
    print(f"  Top-1:  {top1:.2f}%")
    print(f"  Top-5:  {top5:.2f}%")
    print(f"  Top-10: {top10:.2f}%")
    print(f"  mAP:    {mAP*100:.2f}%")
    print("=" * 60)

    # Save log
    os.makedirs(os.path.dirname(config.test_log_path), exist_ok=True)
    logger = setup_logger(name='eval_logger',
                           log_file_path=config.test_log_path,
                           write_mode='append')
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | Top-10: {top10:.2f}% | mAP: {mAP*100:.2f}%")


if __name__ == '__main__':
    main()
