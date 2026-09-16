# ARQUITECTURA BILINGÜE (BPE_EN_ES + TextReIDNet)

TextoReIDNet **(Mismo modelo, "boca" nueva)**: la rama visual no cambia
(EfficientNet-B0 preentrenada). Solo la rama textual pasa del tokenizador BERT
a nuestro diccionario bilingüe BPE_EN_ES (24,000 tokens) y se re-entrena.

## Comparación Antes vs Ahora

| Componente | Antes (BERT, solo EN) | Ahora (BPE_EN_ES bilingüe) |
|---|---|---|
| Tokenización texto | BERT (29,610 tokens) | BPE propio (24,000) entrenado con EN+ES |
| `nn.Embedding` | 29,610 → 512 | 24,000 → 256 |
| BiGRU | 512 → 1024 | 256 → 512 |
| Espacio conjunto | 1024 dims | 512 dims |
| Head Identity | Linear 1024 → 11,003 | Linear 512 → 11,003 |
| Rama visual | EfficientNet-B0 → 1024 → 512 | EfficientNet-B0 → 1024 → 512 |
| Idioma | inglés | inglés + español |

## Gráfico: pipeline bilingüe

```mermaid
flowchart TD
    subgraph Datos["DATOS CUHK-PEDES"]
        CEN["Captions EN (originales)"]
        CES["Captions ES traducidas<br/>(opus-mt-en-es, 17,434)"]
        IMA["Imágenes (B, 3, 384×128)"]
    end

    subgraph BPE["BPE_EN_ES — diccionario bilingüe (M1, entrenado con EN+ES)"]
        T282["<b>vocab 24,000</b> | pad=0 unk=1<br/>texto EN o ES -> ids (B, N≤100)"]
    end

    subgraph RamaVis["RAMA VISUAL (igual que siempre)"]
        EBM["EfficientNet-B0 preentrenada<br/>conv stem + MBConv"]
        DSCV["DSC + MaxPool -> (B, 512)"]
        VE2["Embedding visual v (B,512)"]
        EBM --> DSCV --> VE2
    end

    subgraph RamaTex["RAMA TEXTUAL (reentrenada con el diccionario nuevo)"]
        EM2["nn.Embedding 24,000 -> 256<br/>(pesos nuevos, no heredados)"]
        GR2["BiGRU 256 -> 512 (1 capa)<br/>fwd+bwd promediadas"]
        MP2["MaxPool sobre tiempo<br/>(B, 512, 1, 1)"]
        DS2["text_final_convolution DSC 512->512"]
        TE2["Embedding textual t (B,512)"]
        EM2 --> GR2 --> MP2 --> DS2 --> TE2
    end

    JS2["Espacio común 512 dims<br/>coseno v·t"]
    ID2["IdentityLoss Linear 512 -> 11,003<br/>CE imagen + CE texto"]
    RK2["RankingLoss triplets semi-hard"]

    IMA --> EBM
    CEN --> T282
    CES --> T282
    T282 --> EM2
    VE2 --> JS2
    TE2 --> JS2
    JS2 --> ID2
    JS2 --> RK2

    ENT["RE-ENTRENAR TextReIDNet<br/>con diccionario bilingüe (épocas 1..60)<br/>+ evaluar EN y ES"]
    ID2 --> ENT
    RK2 --> ENT
```

## Checkpoints (no se pisan entre sí)

- **Baseline (original, BERT, EN, 60 épocas):** `data/checkpoints/`
  - `TextReIDNet_epoch60.pth.tar` / `TextReIDNet_latest.pth.tar`
  - Copia de seguridad: `data/checkpoints/TextReIDNet_baseline_epoch60.pth.tar`
- **Bilingüe (BPE_EN_ES, EN+ES):** `data/checkpoints/bpe24_256x512/`

## Por qué hay que re-entrenar

`nn.Embedding(vocab_size, embedding_dim)` es una tabla de pesos que depende del
vocabulario. Al cambiar BERT (29,610) por BPE_EN_ES (24,000) los pesos no son
transferibles → se re-entrena la rama textual completa (visual sin cambios).