"""
CUHK-PEDES loader for the PeterPanTheGenius/HuggingFace version.

This loader reads the HuggingFace dataset (Parquet format, 128x128 thumbnails)
and converts it to the format expected by cuhkpedes.py:
- data/CUHK-PEDES-HF/
  - reid_raw.json
  - imgs/
    - train/
    - val/
    - test/

The HF version has:
- 1 caption per row (not 2 like original)
- Images as PIL Image objects embedded in the parquet
- No split info (we derive it from caption position: first 2/3 = train, etc.)
- No person ID (we assign sequential IDs)

This is a simplified version that works for training and evaluation.
"""

import os
import os.path as op
import json
from typing import List
from collections import defaultdict
from io import BytesIO
from PIL import Image

from utils.iotools import write_json


class CUHKPEDESHF(object):
    """
    CUHK-PEDES from PeterPanTheGenius HuggingFace dataset.
    Adapted to mimic the format expected by cuhkpedes.py.

    Splits by person ID (derived):
    - train: IDs 0-7335 (first 75% of persons)
    - val:   IDs 7336-8535 (next 12.5%)
    - test:  IDs 8536-9735 (last 12.5%)
    """

    def __init__(self, config: dict):
        super(CUHKPEDESHF, self).__init__()
        self.dataset_dir = config.dataset_path
        self.img_dir = op.join(self.dataset_dir, 'imgs')
        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')

        self._check_before_run()
        self.annos = self._load_anno(self.anno_path)

        # Build pid mapping based on image hash
        self._build_pid_mapping()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno()

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

    def _build_pid_mapping(self):
        """Group entries by image_path (which is the identity key)."""
        self.path_to_id = {}
        next_id = 0
        for anno in self.annos:
            fp = anno['file_path']
            if fp not in self.path_to_id:
                self.path_to_id[fp] = next_id
                next_id += 1
        self.num_persons = next_id

    def _split_anno(self):
        """Split by person ID: train/val/test."""
        train_annos, test_annos, val_annos = [], [], []

        all_ids = sorted(self.path_to_id.values())
        n = len(all_ids)
        train_end = int(n * 0.75)
        val_end = int(n * 0.875)

        train_ids = set(all_ids[:train_end])
        val_ids = set(all_ids[train_end:val_end])
        test_ids = set(all_ids[val_end:])

        for anno in self.annos:
            pid = self.path_to_id[anno['file_path']]
            if pid in train_ids:
                train_annos.append(anno)
            elif pid in val_ids:
                val_annos.append(anno)
            else:
                test_annos.append(anno)

        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = self.path_to_id[anno['file_path']]
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions']
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = self.path_to_id[anno['file_path']]
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions']
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' is not available")
        if not op.exists(self.img_dir):
            raise RuntimeError(f"'{self.img_dir}' is not available")
        if not op.exists(self.anno_path):
            raise RuntimeError(f"'{self.anno_path}' is not available")

    def _load_anno(self, anno_path: str):
        with open(anno_path, 'r') as f:
            return json.load(f)


def build_cuhkpedes_hf_anno(parquet_path: str, output_dir: str):
    """
    Convert HuggingFace Parquet to the expected JSON + images format.

    Args.:
        parquet_path: path to the downloaded parquet file
        output_dir: where to write imgs/ and reid_raw.json
    """
    import pyarrow.parquet as pq

    print(f"Reading parquet from {parquet_path}...")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"Total rows: {len(df)}")

    img_dir = op.join(output_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)

    # Group captions by image. The HF version has multiple rows per image,
    # but each row has a different caption. We need to detect which rows
    # belong to the same image (same hash).
    image_hashes = defaultdict(list)

    print("Grouping captions by image hash...")
    for idx, row in df.iterrows():
        img = row['image']
        if isinstance(img, dict):
            img_bytes = img['bytes']
            img_hash = hash(img_bytes)
        else:
            img_hash = hash(img.tobytes())
        image_hashes[img_hash].append({
            'caption': row['text'],
            'image_obj': img
        })

    print(f"Unique images: {len(image_hashes)}")
    print(f"Avg captions per image: {len(df) / len(image_hashes):.2f}")

    annos = []
    for img_hash, entries in image_hashes.items():
        captions = [e['caption'] for e in entries]
        img_obj = entries[0]['image_obj']

        # Save image as JPG
        img_filename = f"{abs(img_hash)}.jpg"
        img_subdir = op.join(img_dir, 'all')
        os.makedirs(img_subdir, exist_ok=True)
        img_path = op.join(img_subdir, img_filename)

        if not op.exists(img_path):
            if isinstance(img_obj, dict):
                img_bytes = img_obj['bytes']
                img = Image.open(BytesIO(img_bytes))
            else:
                img = img_obj
            img = img.convert('RGB')
            img.save(img_path, 'JPEG')

        anno = {
            'file_path': op.join('all', img_filename),
            'captions': captions,
            'split': 'train',  # Will be reassigned later
            'processed_tokens': [],
            'id': 0  # Will be reassigned later
        }
        annos.append(anno)

    json_path = op.join(output_dir, 'reid_raw.json')
    write_json(annos, json_path)
    print(f"Wrote {len(annos)} entries to {json_path}")
    print(f"Images saved in {img_dir}/all/")
