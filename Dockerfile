# NEEDLE PILOT - Ultrasound Needle Trainer
# Dockerfile para reproducibilidade do ambiente de treinamento

# Imagem base com CUDA (para GPU) ou CPU
ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM ${BASE_IMAGE}

# Metadata
LABEL maintainer="petroscarvslho"
LABEL description="NEEDLE PILOT - CNN Training for Ultrasound Needle Detection"
LABEL version="1.0"

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretorio de trabalho
WORKDIR /app

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest scikit-learn

# Copiar codigo fonte
COPY *.py ./
COPY tests/ ./tests/

# Criar diretorios
RUN mkdir -p models synthetic_needle processed

# Definir volume para dados e modelos
VOLUME ["/app/models", "/app/synthetic_needle", "/app/processed"]

# Porta para TensorBoard (opcional)
EXPOSE 6006

# Comando padrao: mostrar ajuda
CMD ["python", "train_vasst.py", "--help"]

# ============================================================
# COMANDOS DE USO:
# ============================================================
#
# BUILD:
#   docker build -t needle-trainer .
#
# BUILD para CPU apenas:
#   docker build --build-arg BASE_IMAGE=python:3.11-slim -t needle-trainer-cpu .
#
# GERAR DADOS SINTETICOS:
#   docker run -v $(pwd)/synthetic_needle:/app/synthetic_needle \
#       needle-trainer python download_datasets.py
#
# TREINAR MODELO:
#   docker run --gpus all \
#       -v $(pwd)/synthetic_needle:/app/synthetic_needle \
#       -v $(pwd)/models:/app/models \
#       needle-trainer python -c "from train_vasst import train_model; train_model(epochs=100)"
#
# CROSS-VALIDATION:
#   docker run --gpus all \
#       -v $(pwd)/synthetic_needle:/app/synthetic_needle \
#       -v $(pwd)/models:/app/models \
#       needle-trainer python cross_validation.py --folds 5 --epochs 50
#
# INFERENCIA:
#   docker run \
#       -v $(pwd)/models:/app/models \
#       -v $(pwd)/images:/app/images \
#       needle-trainer python inference.py --image /app/images/test.png
#
# TESTES:
#   docker run needle-trainer pytest tests/ -v
#
# SHELL INTERATIVO:
#   docker run -it --gpus all \
#       -v $(pwd)/synthetic_needle:/app/synthetic_needle \
#       -v $(pwd)/models:/app/models \
#       needle-trainer bash
# ============================================================
