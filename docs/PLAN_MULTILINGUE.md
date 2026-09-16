# PLAN — Mejora de TextReIDNet: Multilingüe (EN–ES) + Edge eficiente + Paper

> Estado: **aprobado por el usuario (29 mar 2026)**. Backbone fijado: **EfficientNet-B0**.
> Objetivo edge: **ONNX + TensorRT (FP16/INT8)** en RTX 4070 SUPER, con reporte Jetson simulado.
> Alcance: **incremental** — validar con subconjunto barato, escalar solo si el resultado convence.

---

## 1. Diagnóstico de parámetros (medido en el código actual)

| Componente | Params | % |
|---|---|---|
| `nn.Embedding` de texto (vocab 29,610 × 512) | **15.16M** | 47% |
| BiGRU texto (1024 ocultas, bidir) | 9.44M | 29% |
| EfficientNet-B0 | 5.29M | 16% |
| 2× DSC (1280→1024, 1024→1024) | 2.38M | 8% |
| **TextReIDNet total** | **32.27M** | |
| Head IdentityLoss (Linear→11003) | +11.28M | (solo entrenamiento) |

**Hallazgos clave**
- El 47% del modelo es SOLO la tabla de embedding del texto.
- BERT **no está dentro del modelo**: solo se usan sus token ids; el embedding se aprende desde cero. → Cambiar de tokenizador NO toca la red, solo el vocab + reentrenar la tabla.

## 2. Aclaración del multilingüe (por qué no basta `bert-multilingual`)

- `bert-base-multilingual-uncased` tokeniza español gratis (un cambio de línea), pero su vocab = **119,547** → tabla de embedding ~**61M params** (hoy el modelo entero son 32M). Rompe objetivo edge.
- El problema real no es tokenizar sino **aprender**: las captions de CUHK-PEDES son 100% inglés, así que las palabras en español nunca se entrenan → embeddings al azar → el modelo tokeniza pero no entiende.
- **Decisión:** entrenar **BPE compacto propio EN+ES** (SentencePiece, ~16–24k tokens). Tabla de embedding ~4–8M y multilingüe.

## 3. Fases

### M1 — Tokenizador bilingüe (2–4h estimadas)
1. Entrenar BPE 16–24k EN+ES con SentencePiece sobre el corpus (captions EN + textos ES).
2. Traducir **subconjunto de 5,000 captions** EN→ES con `Helsinki-NLP/opus-mt-en-es` (~74M) en GPU (minutos). Dejar script listo para escalar las 68k.
3. Integrar `tokenizer_type="bpe_en_es"` en `datasets/bases.py`, `datasets/` nuevo tokenizer, `config.py`.
4. Smoke test del pipeline (tokenización + shapes) con `test_one_epoch.py`.

### M2 — Entrenamiento (7–8h GPU, background)
- Reentrenar 60 épocas sobre captions EN+ES mezclados (mismo pid en ambos idiomas → pares cross-lingual en el batch).
- Receta modernizada: **cosine schedule + warmup + TAL** con temperatura (quick-win +3–6 pts sobre 47% actual).
- Auto-retoma existente; guarda checkpoints por época.

### M3 — Evaluación bilingüe
- R@1/R@5/R@10 + mAP en EN, en ES, y **matriz cross-lingual 2×2** (query EN→galería EN/ES; query ES→galería EN/ES).
- Si el resultado convence → traducir corpus completo (68k) y repetir entrenamiento para el paper.
- Cross-dataset opcional: RSTPReid.

### M4 — Eficiencia edge
- Export **ONNX + FP16** y **INT8** (onnxruntime / TensorRT).
- Métricas: tamaño/MB, latencia imagen+texto, FPS, VRAM en RTX; **estimación Jetson** (simulada).
- Meta: ~10–12M params → **~10–12 MB en INT8**.
- Ablación de embedding/GRU (512→256/512) para el reporte.

### M5 — Paper (incremental)
- **Título (draft):** "Efficient Multilingual (EN–ES) Text-Based Person Re-Identification for Edge Devices".
- **Contribuciones:** (1) corpus bilingüe EN–ES sobre CUHK-PEDES; (2) BPE compacto EN+ES + alineación contrastiva cross-lingual por persona; (3) modelo edge 3× menor + FP16/INT8; (4) sistema interactivo bilingüe (YOLO + Flask).
- **Tablas:** EN/ES + cross-lingual; cross-dataset; eficiencia RTX/Jetson; comparativa SOTA 2025–2026 (`docs/SOTA_2025_2026.md`).
- **Figuras:** actualizar `docs/ARQUITECTURA.md` con la rama bilingüe + latencia/memoria.

## 4. Decisiones fijadas
- Backbone: **EfficientNet-B0 actual** (no se toca).
- Traducción: **opus-mt-en-es** para validar (subconjunto 5k); escalable a NLLB si se quiere mejor calidad.
- Edge: **RTX + ONNX/TensorRT**, reporte Jetson simulado.
- Alcance: **incremental** (validar barato → decidir profundidad).

## 5. Entregables y tiempos
| Hito | Tiempo | Estado |
|---|---|---|
| M1 tokenizer + 5k traducción + smoke test | 2–4h | pendiente |
| M2 entrenamiento 60 épocas EN+ES | 7–8h GPU (background) | pendiente |
| M3 evaluación EN/ES/cross-lingual | 1–2h | pendiente |
| M4 ONNX FP16/INT8 + métricas | 1–2h | pendiente |
| M5 borrador paper | 1–2 días | pendiente |