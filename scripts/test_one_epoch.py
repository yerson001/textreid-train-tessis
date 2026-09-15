"""
Test rápido: ejecuta N batches del training para validar que
todo el pipeline funciona end-to-end (forward, loss, backward,
checkpoint save, evaluate).

Uso: .venv/bin/python scripts/test_one_epoch.py [N_BATCHES]
Por defecto N_BATCHES = 100.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
from torch import optim
from torch.cuda.amp import GradScaler

from config import sys_configuration
from utils.miscellaneous_utils import set_seed
from datasets.cuhkpedes_dataloader import build_cuhkpedes_dataloader
from model.textreidnet import TextReIDNet
from evaluation.ranking_loss import RankingLoss
from evaluation.identity_loss import IdentityLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./data/test_checkpoints')
    args = parser.parse_args()

    config = sys_configuration(dataset_name='CUHK-PEDES',
                                dataset_source='huggingface')
    config['batch_size'] = args.batch_size
    config['num_workers'] = 0
    config['model_save_path'] = os.path.abspath(args.output_dir)
    os.makedirs(config['model_save_path'], exist_ok=True)

    set_seed(config.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')

    print('=' * 60)
    print(f'TEST: {args.n_batches} batches @ batch_size={args.batch_size}')
    print(f'Output: {config["model_save_path"]}')
    print('=' * 60)

    train_loader, _, _, num_classes = build_cuhkpedes_dataloader(config)
    print(f'Num classes: {num_classes}')
    print(f'Total batches disponibles: {len(train_loader)}')

    model = TextReIDNet(config).to(config.device)
    identity_loss_fnx = IdentityLoss(config=config,
                                     class_num=num_classes).to(config.device)
    ranking_loss_fnx = RankingLoss(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parámetros: {total_params / 1e6:.2f}M')

    optimizer = optim.AdamW(model.parameters(),
                            betas=(config.adam_alpha, config.adam_beta),
                            lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.epoch_decay)
    scaler = GradScaler()

    print('\n--- Entrenando {} batches ---'.format(args.n_batches))
    rank_losses, id_losses, tot_losses = [], [], []

    model.train()
    for i, batch in enumerate(train_loader):
        if i >= args.n_batches:
            break

        imgs = batch['preprocessed_images'].to(config.device)
        labels = batch['pids'].to(config.device)
        tokens = batch['token_ids'].to(config.device)
        lengths = batch['orig_token_lengths'].to(config.device)

        optimizer.zero_grad()

        with torch.autocast(device_type=config.device,
                            dtype=torch.float16):
            vis_emb, txt_emb = model(image=imgs, text_ids=tokens,
                                     text_length=lengths)
            vis_flat = vis_emb.view(vis_emb.size(0), -1)
            txt_flat = txt_emb.view(txt_emb.size(0), -1)
            labels_flat = labels.view(-1)

            r_loss = ranking_loss_fnx(vis_flat, txt_flat, labels_flat)
            i_loss = identity_loss_fnx(vis_emb, txt_emb, labels_flat)
            total = (config.ranking_loss_alpha * r_loss
                     + config.identity_loss_beta * i_loss)

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        rank_losses.append(r_loss.item())
        id_losses.append(i_loss.item())
        tot_losses.append(total.item())

        if (i + 1) % 10 == 0:
            print(f'  Batch {i+1}/{args.n_batches}: '
                  f'rank={np.mean(rank_losses):.3f} '
                  f'id={np.mean(id_losses):.3f} '
                  f'tot={np.mean(tot_losses):.3f}')

    scheduler.step()

    # Guardar checkpoint
    ckpt_path = os.path.join(config['model_save_path'],
                             'TextReIDNet_test.pth.tar')
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(np.mean(tot_losses)),
    }, ckpt_path)
    print(f'\nCheckpoint guardado en: {ckpt_path}')
    print(f'Tamaño: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB')

    print('\n--- RESUMEN ---')
    print(f'Ranking loss final:  {np.mean(rank_losses):.4f}')
    print(f'Identity loss final: {np.mean(id_losses):.4f}')
    print(f'Total loss final:    {np.mean(tot_losses):.4f}')
    if torch.cuda.is_available():
        print(f'VRAM peak:           {torch.cuda.max_memory_allocated()/1e9:.2f} GB')

    print('\nTEST OK - pipeline end-to-end funcional')


if __name__ == '__main__':
    main()
