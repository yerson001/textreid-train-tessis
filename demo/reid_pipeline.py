"""
Etapa de inferencia de la demo: YOLO26n-pose (deteccion de personas + keypoints)
+ smart-crop a cuerpo completo + TextReIDNet (re-id) -> ranking texto<->persona.

Equivalente a model/unity.py del proyecto original, pero con detector publico
y libre, y normalizacion con las stats de CUHK-PEDES con las que se entreno
nuestro modelo (NO ImageNet).
"""

import os
import sys
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Root del repo (donde esta config.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import sys_configuration
from model.textreidnet import TextReIDNet
from datasets.bert_tokenizer import BERTTokenizer
from utils.miscellaneous_utils import pad_tokens

__all__ = ['ReIDPipeline']


class ReIDPipeline:
    """detectar personas en una escena -> recortes 384x128 -> texto -> ranking."""

    def __init__(self, checkpoint='data/checkpoints/TextReIDNet_latest.pth.tar',
                 dataset_source='original', model_name='yolo26n-pose.pt',
                 device=None, yolo_conf=0.30, top_k=6):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.config = sys_configuration(dataset_name='CUHK-PEDES', dataset_source=dataset_source)
        self.top_k = top_k
        self.yolo_conf = yolo_conf

        # Detector de personas (clase COCO 0 = person; yolo26n-pose da keypoints)
        from ultralytics import YOLO
        if not os.path.isabs(model_name):
            local = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_name)
            model_name = local if os.path.isfile(local) else model_name
        self.detector = YOLO(model_name)
        if self.device.startswith('cuda') and torch.cuda.is_available():
            self.detector.to(self.device)

        # Modelo de re-id
        self.model = TextReIDNet(self.config).to(self.device)
        ckpt_path = os.path.join(_REPO_ROOT, checkpoint)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
            self._ckpt_epoch = ckpt.get('epoch')
        else:
            self.model.load_state_dict(ckpt)
            self._ckpt_epoch = '?'
        self.model.eval()

        self.tokenizer = BERTTokenizer()
        self._init_transform()

    @property
    def checkpoint_epoch(self):
        return self._ckpt_epoch

    def _init_transform(self):
        import torchvision.transforms as T
        self.resize = T.Resize(tuple(self.config.CUHKPEDES_image_size),
                               T.InterpolationMode.BICUBIC)
        self.normalize = T.Normalize(mean=self.config.mean, std=self.config.std)

    # ------------------------------------------------------------------ ETAPA 1
    def detect_persons(self, img):
        """Devuelve [(box_xyxy, keypoints(N,17,2))] de personas detectadas."""
        import torch as T
        results = self.detector.predict(
            img, classes=[0], conf=self.yolo_conf, verbose=False)[0]
        boxes = results.boxes.xyxy
        if boxes is None or len(boxes) == 0:
            return []
        box_list = boxes.cpu().tolist()
        if getattr(results, 'keypoints', None) is not None and results.keypoints is not None:
            kpts = results.keypoints.data.cpu().numpy()          # (N,17,3)
        else:
            kpts = None
        out = []
        for i, b in enumerate(box_list):
            kp = kpts[i, :, :2] if kpts is not None else None
            out.append(([float(v) for v in b], kp))
        return out

    # ------------------------------------------------------------------ ETAPA 2
    def _full_body_box(self, box, kpt, img_w, img_h):
        """Expande la bbox para incluir cabeza y pies usando los keypoints."""
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        pad_top, pad_bot = 0.08 * h, 0.12 * h

        if kpt is not None:
            vis = kpt[(kpt[:, 0] > 0) | (kpt[:, 1] > 0)]
            if len(vis) > 0:
                top = vis[:, 1].min()
                bot = vis[:, 1].max()
                if top < y1 + 0.5 * h:      # keypoint por encima del box: es la cabeza
                    pad_top = (y1 - top) + 0.15 * h
                if bot > y2 - 0.2 * h:      # keypoint por debajo: son los pies
                    pad_bot = (bot - y2) + 0.10 * h
                if pad_top <= 0.08 * h:     # sin keypoints de cabeza -> estimar
                    pad_top = 0.30 * h
        # mantener centro horizontal y margen lateral
        new_x1 = x1 - 0.10 * w
        new_x2 = x2 + 0.10 * w
        new_y1 = y1 - pad_top
        new_y2 = y2 + pad_bot
        new_x1 = max(0, min(new_x1, img_w - 1))
        new_x2 = max(0, min(new_x2, img_w - 1))
        new_y1 = max(0, min(new_y1, img_h - 1))
        new_y2 = max(0, min(new_y2, img_h - 1))
        return new_x1, new_y1, new_x2, new_y2

    def _crop_to_tensor(self, img, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img.crop((x1, y1, x2, y2)).convert('RGB')
        crop = self.resize(crop)
        tensor = torch.from_numpy(np.asarray(crop).copy()).permute(2, 0, 1) / 255.0
        tensor = self.normalize(tensor)
        return tensor

    # ------------------------------------------------------------------ ETAPA 3
    @torch.no_grad()
    def _image_embeddings(self, img, boxes):
        tensors = [self._crop_to_tensor(img, b).to(self.device) for b in boxes]
        batch = torch.stack(tensors, 0)
        emb = self.model.image_embedding(batch).view(batch.size(0), -1)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    @torch.no_grad()
    def _text_embedding(self, text):
        tokens = self.tokenizer(text)
        token_ids, length = pad_tokens(tokens, self.config.tokens_length_max)
        token_ids = token_ids.to(torch.long).unsqueeze(0).to(self.device)
        length = torch.tensor([length]).to(self.device)
        emb = self.model.text_embedding(token_ids, length).view(1, -1)
        return torch.nn.functional.normalize(emb, p=2, dim=-1)

    # ------------------------------------------------------------------ ORQUESTADOR
    def do_reid(self, image, text):
        """Recibe imagen (ruta o PIL) y texto -> imagen anotada + lista de resultados."""
        if isinstance(image, (str, os.PathLike)):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')

        persons = self.detect_persons(img)
        if not persons:
            return img, []

        boxes_full = [self._full_body_box(b, kp, img.width, img.height)
                      for b, kp in persons]
        vis_emb = self._image_embeddings(img, boxes_full)
        txt_emb = self._text_embedding(text)

        sim = (vis_emb @ txt_emb.t()).squeeze(1)          # (N,) coseno
        probs = torch.softmax(sim, dim=0)                 # relativo dentro de la escena
        order = torch.argsort(sim, descending=True)

        results = []
        for rank in range(len(order)):
            i = int(order[rank])
            results.append({
                'rank': rank + 1,
                'box': [float(round(float(v), 1)) for v in boxes_full[i]],
                'box_px': [int(v) for v in persons[i][0]],
                'score': float(sim[i]),
                'prob': float(probs[i]),
            })

        annotated = self._draw(img, results)
        return annotated, results

    # ------------------------------------------------------------------ DIBUJO
    def _draw(self, img, results):
        img = img.convert('RGB').copy()
        draw = ImageDraw.Draw(img, 'RGBA')
        try:
            font = ImageFont.load_default(size=20)
        except TypeError:
            font = ImageFont.load_default()
        best = results[0] if results else None
        for r in results[:max(self.top_k, 1)]:
            x1, y1, x2, y2 = r['box']
            is_best = best is not None and r['rank'] == 1
            color = (34, 177, 76, 255) if is_best else (222, 41, 41, 255)
            fill = (34, 177, 76, 60) if is_best else (222, 41, 41, 30)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4, fill=fill)
            label = f"#{r['rank']}  {r['prob'] * 100:.0f}%"
            draw.rectangle([x1, y1 - 28, x1 + 190, y1 - 2], fill=color)
            draw.text((x1 + 6, y1 - 26), label, fill=(255, 255, 255, 255), font=font)
        return img

    def annotated_bytes(self, image, text, fmt='JPEG'):
        ann, results = self.do_reid(image, text)
        buf = io.BytesIO()
        ann.save(buf, format=fmt, quality=92)
        return buf.getvalue(), results