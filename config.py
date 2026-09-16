"""
C. September 2024
Doc: Simplified configuration file for textreid-train.
      Only includes what's needed for training + evaluation.
      No demos, no inference, no decentralized app.
"""

import os
import platform

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def sys_configuration(platform_name: str = platform.node(),
                     dataset_name: str = "CUHK-PEDES",
                     dataset_source: str = "huggingface",
                     show_configs: bool = False) -> DotDict:
    """
    Args.:
        platform_name: hostname of the machine
        dataset_name:  "CUHK-PEDES" (only one supported for now)
        dataset_source: "huggingface" (PeterPanTheGenius/CUHK-PEDES)
                       or "original" (when you get the real dataset by email)
    """

    configs = dict()

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++[Language Model Configurations]+++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['dropout_rate'] = 0.30
    configs['num_layers'] = 1
    configs['tokens_length_max'] = 100
    configs['feature_length'] = 1024
    configs['tokenizer_type'] = 'bert'
    configs['vocab_size'] = 29609 + 1  # BERT vocab size + 1
    configs['embedding_dim'] = 512
    configs['bilingual'] = False  # entrenar con captions EN+ES (reid_raw_bilingue.json)
    configs['evaluate_language'] = 'en'  # 'es' usa captions_es en val/test

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++[Visual Model Configuration]+++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['CUHKPEDES_image_size'] = (384, 128)
    configs['MHPV2_image_size'] = (512, 512)
    configs['MHPV2_means'] = [0.3555248975753784, 0.3295726776123047, 0.3115333914756775]
    configs['MHPV2_stds'] = [0.3482806384563446, 0.3339986503124237, 0.3296058773994446]
    configs['CUHKPEDES_means'] = [0.485, 0.456, 0.406]
    configs['CUHKPEDES_stds'] = [0.229, 0.224, 0.225]
    configs['select_multi_person'] = False
    configs['reID_confidence_threshold'] = 0.50

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++[Human Parsing Configurations]+++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['nms_pre'] = 500
    configs['score_thr'] = 0.3
    configs['mask_thr'] = 0.3
    configs['update_thr'] = 0.05
    configs['kernel'] = 'gaussian'
    configs['sigma'] = 2.0
    configs['max_per_img'] = 30

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++[Training Configurations]+++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['epoch'] = 60
    configs['adam_alpha'] = 0.90
    configs['adam_beta'] = 0.999
    configs['epoch_decay'] = [20, 40]
    configs['lr'] = 0.001
    configs['patience'] = 3
    configs['val_dataset'] = 'test'
    configs['device'] = 'cuda' if platform_name != 'nano' else 'cuda'
    configs['seed'] = 3407
    configs['model_testing_data_split'] = 'test'
    configs['save_best_test_results_only'] = True

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++[Loss Functions Configurations]+++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['margin'] = 0.5
    configs['ranking_loss_alpha'] = 1.0
    configs['identity_loss_beta'] = 1.0

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++[Data Logging Configurations & Misc.]+++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['train_log_path'] = os.path.join(PARENT_DIR, 'logs', 'train.log')
    configs['test_log_path'] = os.path.join(PARENT_DIR, 'logs', 'test.log')
    configs['write_mode'] = 'overwrite'
    configs['model_save_path'] = os.path.join(PARENT_DIR, 'data', 'checkpoints')
    configs['plot_save_path'] = os.path.join(PARENT_DIR, 'data', 'plots')
    configs['log_config_parameters'] = True
    configs['project_parent_dir'] = PARENT_DIR

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++[Dataset Specific Configs]++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    configs['dataset_name'] = dataset_name
    configs['dataset_source'] = dataset_source

    if dataset_source == 'huggingface':
        configs['dataset_path'] = os.path.join(PARENT_DIR, 'data', 'CUHK-PEDES-HF')
        configs['mean'] = [0.4416847, 0.41812873, 0.4237452]
        configs['std'] = [0.3088255, 0.29743394, 0.301009]
    elif dataset_source == 'original':
        configs['dataset_path'] = os.path.join(PARENT_DIR, 'data', 'CUHK-PEDES')
        configs['mean'] = [0.4416847, 0.41812873, 0.4237452]
        configs['std'] = [0.3088255, 0.29743394, 0.301009]
    else:
        raise ValueError(f"dataset_source must be 'huggingface' or 'original', got '{dataset_source}'")

    # +++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++[Platform Specific Configs]+++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++
    if platform_name in ('PC-SIM', 'deeplearning', 'ultron', 'yrsn'):
        configs['num_workers'] = 4
        configs['batch_size'] = 8
    elif platform_name == 'dc-2019':
        configs['num_workers'] = 8
        configs['batch_size'] = 16
    elif platform_name == 'nano':
        configs['num_workers'] = 2
        configs['batch_size'] = 4
    else:
        configs['num_workers'] = 4
        configs['batch_size'] = 8

    return DotDict(configs)
