"""
Convierte el CUHK-PEDES original (caption_all.json) al formato del proyecto
(reid_raw.json) con el campo 'split'.

El archivo original `caption_all.json` trae entradas {id, file_path, captions}
sin el campo 'split'. Este script asigna el split por rango de identificador
de persona (igual que IRRA / el paper):

    train:  id 1  - 11003
    val:    id 11004 - 12003
    test:   id 12004 - 13003

Uso:
    python scripts/convert_original_dataset.py
    python scripts/convert_original_dataset.py --input data/CUHK-PEDES/caption_all.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.iotools import read_json, write_json

SPLIT_BY_ID = [
    (1, 11003, 'train'),
    (11004, 12003, 'val'),
    (12004, 13003, 'test'),
]


def split_for(ped_id: int) -> str:
    for lo, hi, name in SPLIT_BY_ID:
        if lo <= ped_id <= hi:
            return name
    raise ValueError(f'id {ped_id} fuera de rangos esperados (1-13003)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default=os.path.join('data', 'CUHK-PEDES', 'caption_all.json'))
    parser.add_argument('--output', type=str,
                        default=os.path.join('data', 'CUHK-PEDES', 'reid_raw.json'))
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f'No se encontró {args.input}')

    annos = read_json(args.input)
    print(f'Total entradas: {len(annos)}')

    out = []
    counts = {'train': 0, 'val': 0, 'test': 0}
    missing_files = 0
    total = len(annos)

    for i, anno in enumerate(annos):
        ped_id = int(anno['id'])
        split = split_for(ped_id)
        counts[split] += 1

        file_path = anno['file_path']
        img_path = os.path.join('data', 'CUHK-PEDES', 'imgs', file_path)
        if not os.path.exists(img_path):
            missing_files += 1

        out.append({
            'split': split,
            'captions': anno['captions'],
            'file_path': file_path,
            'processed_tokens': [],
            'id': ped_id,
        })

        if (i + 1) % 10000 == 0:
            print(f'  {i+1}/{total} ...')

    write_json(out, args.output)
    print(f'Escrito: {args.output}')
    print(f'Por split: {counts}')
    if missing_files:
        print(f'AVISO: {missing_files} imágenes faltantes (paths no resueltos)')
    else:
        print('Todas las imágenes resueltas correctamente.')


if __name__ == '__main__':
    main()