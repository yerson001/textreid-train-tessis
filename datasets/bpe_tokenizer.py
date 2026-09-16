"""
Doc.:   Tokenizador BPE bilingue (EN+ES) basado en la libreria `tokenizers`.
        El modelo no usa BERT: solo toma los ids que produce este tokenizador y
        aprende sus propias representaciones (nn.Embedding) desde esos ids.

        Ruta por defecto: data/tokenizer_bpe_en_es/config.json
        (generado por scripts/build_bilingual_bpe.py)
"""

import os
from tokenizers import Tokenizer

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(PARENT_DIR, '..', 'data', 'tokenizer_bpe_en_es', 'config.json')


class BPETokenizer(object):
    def __init__(self, config_path: str = None):
        super(BPETokenizer, self).__init__()
        path = config_path or DEFAULT_CFG
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Tokenizador BPE no encontrado en: {path}\n"
                f"Ejecuta: .venv/bin/python scripts/build_bilingual_bpe.py")
        self.tokenizer = Tokenizer.from_file(str(path))

    def __call__(self, text: str = None) -> list:
        tokens = self.tokenizer.encode(text)
        return tokens.ids