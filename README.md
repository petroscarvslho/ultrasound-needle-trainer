# Ultrasound Needle CNN Trainer

Sistema completo para download de datasets e treinamento de CNN para deteccao de agulhas em ultrassom.

> **Projeto Relacionado**: Este treinador gera modelos para o plugin **NEEDLE PILOT v3.1** do [Aplicativo USG](https://github.com/petroscarvslho/aplicativo-usg-final)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Funcionalidades

- **Download de Datasets**: Baixa datasets publicos de ultrassom
- **Geracao Sintetica**: Cria dados de treinamento automaticamente
- **Treinamento CNN**: Treina modelo VASST para detectar ponta da agulha
- **Inferencia**: Script standalone para deteccao em imagens/videos
- **Cross-Validation**: K-Fold CV para avaliacao robusta
- **Metricas Avancadas**: MAE, RMSE, Euclidean Distance, Accuracy@Threshold
- **Testes Unitarios**: Suite de testes com pytest
- **Docker**: Container para reproducibilidade
- **Benchmark**: Avaliacao de performance em diferentes devices

## Instalacao

### Opcao 1: Virtual Environment (Recomendado)

```bash
git clone https://github.com/petroscarvslho/ultrasound-needle-trainer.git
cd ultrasound-needle-trainer
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install pytest scikit-learn  # Para testes e CV
```

### Opcao 2: Docker

```bash
docker build -t needle-trainer .
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
# Treinar modelo
PYTHONUNBUFFERED=1 python -c "from train_vasst import train_model; train_model(epochs=100)"

# Ou de forma interativa
python train_vasst.py
```

O modelo sera salvo em: `models/vasst_needle.pt`

### 3. Inferencia

```bash
# Uma imagem
python inference.py --image path/to/image.png

# Pasta de imagens
python inference.py --folder path/to/images/ --output results/

# Video
python inference.py --video procedure.mp4 --output output.mp4

# Camera em tempo real
python inference.py --camera 0

# Benchmark de performance
python inference.py --benchmark
```

### 4. Cross-Validation

```bash
python cross_validation.py --folds 5 --epochs 50
```

### 5. Testes

```bash
pytest tests/ -v
```

### 6. Benchmark

```bash
python benchmark.py --iterations 100 --batch-sizes 1,4,8,16,32
```

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
├── train_vasst.py         # Treinamento da CNN VASST
├── inference.py           # Script de inferencia
├── metrics.py             # Metricas avancadas
├── cross_validation.py    # K-Fold Cross-Validation
├── benchmark.py           # Benchmark de performance
├── requirements.txt       # Dependencias Python
├── Dockerfile             # Container Docker
├── docker-compose.yml     # Orquestracao Docker
├── README.md
├── INSTRUCOES_CONTINUIDADE.md
├── LICENSE
│
├── tests/                 # Testes unitarios
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_inference.py
│
├── synthetic_needle/      # Dataset sintetico (gerado)
├── kaggle_nerve/          # Dataset Kaggle (baixado)
├── camus_cardiac/         # Dataset CAMUS (baixado)
├── brachial_plexus/       # Dataset Brachial (manual)
├── processed/             # Dados processados
└── models/                # Modelos treinados
    ├── vasst_needle.pt    # Modelo principal
    └── training_plot.png  # Grafico de treinamento
```

## Arquitetura da CNN VASST

Arquitetura baseada no VASST CNN, convertida para PyTorch:

```
Input: 256x256 grayscale
  |
  v
Conv2D(1→16, 3x3)  -> LeakyReLU(0.01) -> MaxPool(2x2)
Conv2D(16→32, 2x2) -> LeakyReLU(0.01) -> MaxPool(2x2)
Conv2D(32→64, 2x2) -> LeakyReLU(0.01) -> MaxPool(2x2)
Conv2D(64→128, 2x2) -> LeakyReLU(0.01) -> MaxPool(2x2)
Conv2D(128→256, 2x2) -> LeakyReLU(0.01) -> MaxPool(2x2)
  |
  v
Flatten -> Dense(1024) -> Dropout(0.3)
        -> Dense(128)  -> Dropout(0.3)
        -> Dense(16)
        -> Dense(2)
  |
  v
Output: [y, x] coordenadas normalizadas [0,1]
```

**Parametros**: ~13.1M

## Metricas de Avaliacao

O modulo `metrics.py` fornece:

- **MAE** (Mean Absolute Error) em pixels
- **MSE** (Mean Squared Error) em pixels
- **RMSE** (Root Mean Squared Error) em pixels
- **Euclidean Distance** (media, std, mediana, P95, max)
- **Accuracy@Threshold** (% predicoes dentro de X pixels)

Exemplo de uso:

```python
from metrics import evaluate_predictions

predictions = model_outputs  # [N, 2]
targets = ground_truth       # [N, 2]

results = evaluate_predictions(predictions, targets)
print(results)
```

## Parametros de Treinamento

| Parametro | Valor Padrao | Descricao |
|-----------|--------------|-----------|
| Epocas | 100 | Com early stopping (patience=15) |
| Batch size | 32 | Ajustar conforme GPU |
| Learning rate | 0.0001 | Com ReduceLROnPlateau |
| Loss | MAE | Mean Absolute Error |
| Optimizer | Adam | Com weight decay |

## Docker

### Build

```bash
# Com GPU (CUDA)
docker build -t needle-trainer .

# Apenas CPU
docker build --build-arg BASE_IMAGE=python:3.11-slim -t needle-trainer-cpu .
```

### Treinar

```bash
# GPU
docker run --gpus all \
    -v $(pwd)/synthetic_needle:/app/synthetic_needle \
    -v $(pwd)/models:/app/models \
    needle-trainer python -c "from train_vasst import train_model; train_model()"

# CPU
docker run \
    -v $(pwd)/synthetic_needle:/app/synthetic_needle \
    -v $(pwd)/models:/app/models \
    needle-trainer-cpu python -c "from train_vasst import train_model; train_model()"
```

### Docker Compose

```bash
# Treinar
docker-compose up trainer

# Cross-validation
docker-compose up cross-validation

# Testes
docker-compose up tests
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
│  - inference.py                     │
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

## Usando o Modelo Treinado

### Python Direto

```python
import torch
import cv2
from train_vasst import VASSTPyTorch

# Carregar modelo
model = VASSTPyTorch()
checkpoint = torch.load('models/vasst_needle.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
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

### Usando NeedleDetector

```python
from inference import NeedleDetector
import cv2

# Inicializar detector
detector = NeedleDetector()

# Detectar
image = cv2.imread('ultrasound.png')
result = detector.predict(image)

print(f"Ponta: ({result['tip_x']}, {result['tip_y']})")
print(f"Tempo: {result['inference_time_ms']:.1f}ms")

# Visualizar
annotated = detector.draw_prediction(image, result)
cv2.imwrite('output.png', annotated)
```

## Links dos Datasets

- **Kaggle**: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- **CAMUS**: https://www.creatis.insa-lyon.fr/Challenge/camus/
- **IEEE DataPort**: https://ieee-dataport.org (buscar "ultrasound needle")
- **Zenodo**: https://zenodo.org/search?q=ultrasound
- **Papers With Code**: https://paperswithcode.com/datasets?q=ultrasound+needle

## Resultados do Ultimo Treinamento

| Metrica | Valor |
|---------|-------|
| Val Loss | 0.012281 |
| Erro Medio | ~3.1 pixels |
| Epocas | 77 (early stopping) |
| Device | MPS (Apple Silicon) |

## Licenca

MIT License
