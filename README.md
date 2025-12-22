# Ultrasound Needle CNN Trainer

Sistema completo para download de datasets e treinamento de CNN para deteccao de agulhas em ultrassom.

> **Projeto Relacionado**: Este treinador gera modelos para o plugin **NEEDLE PILOT v3.1** do [Aplicativo USG](https://github.com/petroscarvslho/aplicativo-usg-final)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## O que faz

1. **Download de Datasets**: Baixa datasets publicos de ultrassom
2. **Geracao Sintetica**: Cria dados de treinamento automaticamente
3. **Treinamento CNN**: Treina modelo para detectar ponta da agulha

## Instalacao

```bash
git clone https://github.com/petroscarvslho/ultrasound-needle-trainer.git
cd ultrasound-needle-trainer
pip install -r requirements.txt
```

## Uso Rapido

### 1. Gerar Dataset Sintetico

```bash
python download_datasets.py
# Escolha opcao 5 (Sintetico)
# Digite numero de amostras (ex: 5000)
```

### 2. Treinar Modelo

```bash
python train_vasst.py
# Escolha opcao 1 para treinar
```

O modelo sera salvo em: `models/vasst_needle.pt`

## Datasets Suportados

| Dataset | Imagens | Acesso |
|---------|---------|--------|
| Kaggle Nerve Segmentation | 5,635 | API publica |
| CAMUS Cardiac | 4,000+ | Registro gratis |
| Brachial Plexus | 41,000 | Busca academica |
| Breast Ultrasound | 780 | Kaggle API |
| **Sintetico** | **Ilimitado** | **Geracao local** |

## Estrutura do Projeto

```
ultrasound-needle-trainer/
├── download_datasets.py   # Download e geracao de dados
├── train_vasst.py         # Treinamento da CNN
├── requirements.txt       # Dependencias
├── README.md
├── LICENSE
│
├── synthetic_needle/      # Dataset sintetico (gerado)
├── kaggle_nerve/          # Dataset Kaggle (baixado)
├── camus_cardiac/         # Dataset CAMUS (baixado)
├── brachial_plexus/       # Dataset Brachial (manual)
├── processed/             # Dados processados
└── models/                # Modelos treinados
```

## Arquitetura da CNN

Baseada no VASST CNN, convertida para PyTorch:

```
Input: 256x256 grayscale
  |
  v
Conv2D(16) -> LeakyReLU -> MaxPool
Conv2D(32) -> LeakyReLU -> MaxPool
Conv2D(64) -> LeakyReLU -> MaxPool
Conv2D(128) -> LeakyReLU -> MaxPool
Conv2D(256) -> LeakyReLU -> MaxPool
  |
  v
Flatten -> Dense(1024) -> Dense(128) -> Dense(16) -> Dense(2)
  |
  v
Output: [y, x] coordenadas normalizadas [0,1]
```

## Parametros de Treinamento

- **Epocas**: 100 (com early stopping)
- **Batch size**: 32
- **Learning rate**: 0.0001 (com decay)
- **Loss**: MAE (Mean Absolute Error)
- **Optimizer**: Adam

## Links dos Datasets

- **Kaggle**: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- **CAMUS**: https://www.creatis.insa-lyon.fr/Challenge/camus/
- **IEEE DataPort**: https://ieee-dataport.org (buscar "ultrasound needle")
- **Zenodo**: https://zenodo.org/search?q=ultrasound
- **Papers With Code**: https://paperswithcode.com/datasets?q=ultrasound+needle

## Usando o Modelo Treinado

```python
import torch
import numpy as np
import cv2

# Carregar modelo
model = torch.load('models/vasst_needle.pt')
model.eval()

# Preparar imagem
img = cv2.imread('ultrasound.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
x = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0

# Predicao
with torch.no_grad():
    pred = model(x).numpy()[0]

# Coordenadas da ponta
tip_y = int(pred[0] * 256)
tip_x = int(pred[1] * 256)
print(f"Ponta da agulha: ({tip_x}, {tip_y})")
```

## Integracao com Aplicativo USG

Este projeto treina modelos para o plugin **NEEDLE PILOT v3.1** do aplicativo principal.

### Arquitetura dos Projetos

```
┌─────────────────────────────────────┐
│  ultrasound-needle-trainer          │
│  (Projeto de Treinamento)           │
│                                     │
│  - download_datasets.py             │
│  - train_vasst.py                   │
│  - models/vasst_needle.pt  ─────────┼──┐
└─────────────────────────────────────┘  │
                                         │ COPIAR
┌─────────────────────────────────────┐  │
│  aplicativo-usg-final               │  │
│  (Projeto Principal)                │  │
│                                     │  │
│  - main.py                          │  │
│  - src/ai_processor.py              │  │
│    ├── NEEDLE PILOT v3.1 ◄──────────┼──┘
│    ├── NERVE TRACK                  │
│    ├── CARDIAC AI                   │
│    └── ... (10 plugins)             │
│  - models/vasst_needle.pt           │
└─────────────────────────────────────┘
```

### Apos treinar, copie o modelo:

```bash
cp models/vasst_needle.pt /caminho/para/aplicativo-usg-final/models/
```

### O NEEDLE PILOT vai:
- Carregar o modelo automaticamente
- Usar a CNN para refinar deteccao da ponta da agulha
- Combinar com Kalman Filter e RANSAC para tracking suave

### Projeto Principal:
- **GitHub**: https://github.com/petroscarvslho/aplicativo-usg-final
- **Plugin**: NEEDLE PILOT v3.1 PREMIUM
- **Arquivo**: src/ai_processor.py

## Licenca

MIT License
