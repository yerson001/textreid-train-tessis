"""
C. September 2024
Doc: Clean training script for TextReIDNet on CUHK-PEDES.
      Uses only TextReIDNet (no Unity wrapper).
      Supports both 'huggingface' and 'original' dataset sources.

Usage:
    export PYENV_VERSION=3.8.10
    python train.py [--dataset_source huggingface|original] [--epochs 60] [--batch_size 8]
"""

import os
import sys
import argparse
import logging
import datetime

import torch
import numpy as np
from tqdm import tqdm
from torch import optim

sys.path.insert(0, os.path.abspath('../textreid-train'))

from config import sys_configuration
from utils.miscellaneous_utils import set_seed, setup_logger, save_model_checkpoint, SavePlots
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet
from evaluation.ranking_loss import RankingLoss
from evaluation.identity_loss import IdentityLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train TextReIDNet on CUHK-PEDES')
    parser.add_argument('--dataset_source', type=str, default='huggingface',
                        choices=['huggingface', 'original'],
                        help='Which dataset source to use')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override seed from config')
    parser.add_argument('--output_dir', type=str, default='./data/checkpoints',
                        help='Where to save model checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()

    # Init config
    config = sys_configuration(dataset_name="CUHK-PEDES", dataset_source=args.dataset_source)

    # Apply overrides
    if args.epochs is not None:
        config['epoch'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.seed is not None:
        config['seed'] = args.seed
    if args.output_dir is not None:
        config['model_save_path'] = os.path.abspath(args.output_dir)
        os.makedirs(config['model_save_path'], exist_ok=True)

    set_seed(config.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    scaler = torch.amp.GradScaler('cuda')

    print("=" * 60)
    print(f"Dataset source: {args.dataset_source}")
    print(f"Dataset path:   {config.dataset_path}")
    print(f"Epochs:         {config.epoch}")
    print(f"Batch size:     {config.batch_size}")
    print(f"Learning rate:  {config.lr}")
    print(f"Device:         {config.device}")
    print("=" * 60)

    # Build dataloaders
    train_data_loader, _, _, train_num_classes = build_cuhkpedes_dataloader(config)
    print(f"Number of training classes: {train_num_classes}")
    print(f"Number of training batches per epoch: {len(train_data_loader)}")

    # Build model
    model = TextReIDNet(config).to(config.device)
    identity_loss_fnx = IdentityLoss(config=config, class_num=train_num_classes).to(config.device)
    ranking_loss_fnx = RankingLoss(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), betas=(config.adam_alpha, config.adam_beta), lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)

    # Logging
    os.makedirs(os.path.dirname(config.train_log_path), exist_ok=True)
    train_logger = setup_logger(name='train_logger',
                                 log_file_path=config.train_log_path,
                                 write_mode=config.write_mode)

    time_stamp = str(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    train_logger.info(f"\nStarted on {time_stamp}\n{'='*35}")
    train_logger.info(f"Dataset source: {args.dataset_source}")
    train_logger.info(f"Number of classes: {train_num_classes}")

    # Training loop
    for current_epoch in range(1, config.epoch + 1):
        train_ranking_loss_list = []
        train_identity_loss_list = []
        train_total_loss_list = []

        model.train()
        with tqdm(train_data_loader, unit='batch') as tepoch:
            tepoch.set_description(f"Epoch {current_epoch}/{config.epoch}")

            for batch in tepoch:
                preprocessed_images = batch['preprocessed_images'].to(config.device)
                labels = batch['pids'].to(config.device)
                token_ids = batch['token_ids'].to(config.device)
                orig_token_length = batch['orig_token_lengths'].to(config.device)

                optimizer.zero_grad()

                precision_dtype = torch.bfloat16 if config.device == 'cpu' else torch.float16
                with torch.autocast(device_type=config.device, dtype=precision_dtype):
                    visual_embeddings, textual_embeddings = model(
                        image=preprocessed_images,
                        text_ids=token_ids,
                        text_length=orig_token_length
                    )

                    # Flatten embeddings to (B, feature_length) for losses
                    visual_emb_flat = visual_embeddings.view(visual_embeddings.size(0), -1)
                    text_emb_flat = textual_embeddings.view(textual_embeddings.size(0), -1)
                    labels_flat = labels.view(-1)

                    ranking_loss = ranking_loss_fnx(visual_emb_flat, text_emb_flat, labels_flat)
                    identity_loss = identity_loss_fnx(visual_embeddings, textual_embeddings, labels_flat)
                    total_loss = (config.ranking_loss_alpha * ranking_loss +
                                  config.identity_loss_beta * identity_loss)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_ranking_loss_list.append(ranking_loss.item())
                train_identity_loss_list.append(identity_loss.item())
                train_total_loss_list.append(total_loss.item())

                tepoch.set_postfix({
                    'rank': f"{np.mean(train_ranking_loss_list):.3f}",
                    'id':   f"{np.mean(train_identity_loss_list):.3f}",
                    'tot':  f"{np.mean(train_total_loss_list):.3f}",
                })

        scheduler.step()

        log_msg = (f"Epoch {current_epoch}/{config.epoch} | "
                   f"Ranking Loss: {np.mean(train_ranking_loss_list):.4f} | "
                   f"Identity Loss: {np.mean(train_identity_loss_list):.4f} | "
                   f"Total Loss: {np.mean(train_total_loss_list):.4f}")
        print(log_msg)
        train_logger.info(log_msg)

        # Save checkpoint every epoch
        ckpt_path = os.path.join(config.model_save_path,
                                 f"TextReIDNet_epoch{current_epoch}.pth.tar")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(train_total_loss_list),
        }, ckpt_path)

        # Also save 'latest'
        latest_path = os.path.join(config.model_save_path, "TextReIDNet_latest.pth.tar")
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(train_total_loss_list),
        }, latest_path)

    print("Training complete.")
    train_logger.info("Training complete.")


if __name__ == '__main__':
    main()
