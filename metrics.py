#!/usr/bin/env python3
"""
NEEDLE PILOT - Modulo de Metricas
Metricas avancadas para avaliacao de deteccao de agulhas.

Metricas implementadas:
- MAE (Mean Absolute Error) em pixels
- MSE (Mean Squared Error) em pixels
- RMSE (Root Mean Squared Error) em pixels
- Euclidean Distance Error
- Accuracy at threshold (porcentagem de predicoes dentro de X pixels)
- Precision/Recall para deteccao binaria
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class MetricsResult:
    """Resultado das metricas de avaliacao"""
    mae_pixels: float
    mse_pixels: float
    rmse_pixels: float
    euclidean_mean: float
    euclidean_std: float
    euclidean_median: float
    euclidean_p95: float
    euclidean_max: float
    accuracy_5px: float
    accuracy_10px: float
    accuracy_20px: float
    n_samples: int

    def to_dict(self) -> Dict:
        return {
            'mae_pixels': self.mae_pixels,
            'mse_pixels': self.mse_pixels,
            'rmse_pixels': self.rmse_pixels,
            'euclidean_mean': self.euclidean_mean,
            'euclidean_std': self.euclidean_std,
            'euclidean_median': self.euclidean_median,
            'euclidean_p95': self.euclidean_p95,
            'euclidean_max': self.euclidean_max,
            'accuracy_5px': self.accuracy_5px,
            'accuracy_10px': self.accuracy_10px,
            'accuracy_20px': self.accuracy_20px,
            'n_samples': self.n_samples
        }

    def __str__(self) -> str:
        return f"""
Metricas de Avaliacao ({self.n_samples} amostras)
{'=' * 50}
Erro Absoluto Medio (MAE):     {self.mae_pixels:.2f} pixels
Erro Quadratico Medio (MSE):   {self.mse_pixels:.2f} pixels²
Raiz do MSE (RMSE):            {self.rmse_pixels:.2f} pixels

Distancia Euclidiana:
  Media:    {self.euclidean_mean:.2f} ± {self.euclidean_std:.2f} pixels
  Mediana:  {self.euclidean_median:.2f} pixels
  P95:      {self.euclidean_p95:.2f} pixels
  Max:      {self.euclidean_max:.2f} pixels

Acuracia por Limiar:
  <= 5px:   {self.accuracy_5px:.1f}%
  <= 10px:  {self.accuracy_10px:.1f}%
  <= 20px:  {self.accuracy_20px:.1f}%
{'=' * 50}
"""


def compute_euclidean_distance(
    pred: np.ndarray,
    target: np.ndarray,
    image_size: int = 256
) -> np.ndarray:
    """
    Calcula distancia euclidiana entre predicoes e targets

    Args:
        pred: Predicoes normalizadas [N, 2] (y, x)
        target: Targets normalizados [N, 2] (y, x)
        image_size: Tamanho da imagem para desnormalizacao

    Returns:
        Array de distancias euclidianas em pixels
    """
    # Desnormalizar para pixels
    pred_px = pred * image_size
    target_px = target * image_size

    # Calcular distancia euclidiana
    distances = np.sqrt(np.sum((pred_px - target_px) ** 2, axis=1))

    return distances


def compute_mae(pred: np.ndarray, target: np.ndarray, image_size: int = 256) -> float:
    """Calcula Mean Absolute Error em pixels"""
    pred_px = pred * image_size
    target_px = target * image_size
    return np.mean(np.abs(pred_px - target_px))


def compute_mse(pred: np.ndarray, target: np.ndarray, image_size: int = 256) -> float:
    """Calcula Mean Squared Error em pixels"""
    pred_px = pred * image_size
    target_px = target * image_size
    return np.mean((pred_px - target_px) ** 2)


def compute_rmse(pred: np.ndarray, target: np.ndarray, image_size: int = 256) -> float:
    """Calcula Root Mean Squared Error em pixels"""
    return np.sqrt(compute_mse(pred, target, image_size))


def compute_accuracy_at_threshold(
    distances: np.ndarray,
    threshold: float
) -> float:
    """
    Calcula porcentagem de predicoes dentro do limiar

    Args:
        distances: Array de distancias euclidianas em pixels
        threshold: Limiar em pixels

    Returns:
        Porcentagem de predicoes dentro do limiar
    """
    return 100.0 * np.mean(distances <= threshold)


def evaluate_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_size: int = 256
) -> MetricsResult:
    """
    Avalia predicoes com todas as metricas

    Args:
        predictions: Array de predicoes [N, 2] normalizadas
        targets: Array de targets [N, 2] normalizados
        image_size: Tamanho da imagem

    Returns:
        MetricsResult com todas as metricas
    """
    # Calcular distancias euclidianas
    distances = compute_euclidean_distance(predictions, targets, image_size)

    # Calcular metricas
    mae = compute_mae(predictions, targets, image_size)
    mse = compute_mse(predictions, targets, image_size)
    rmse = compute_rmse(predictions, targets, image_size)

    return MetricsResult(
        mae_pixels=mae,
        mse_pixels=mse,
        rmse_pixels=rmse,
        euclidean_mean=np.mean(distances),
        euclidean_std=np.std(distances),
        euclidean_median=np.median(distances),
        euclidean_p95=np.percentile(distances, 95),
        euclidean_max=np.max(distances),
        accuracy_5px=compute_accuracy_at_threshold(distances, 5),
        accuracy_10px=compute_accuracy_at_threshold(distances, 10),
        accuracy_20px=compute_accuracy_at_threshold(distances, 20),
        n_samples=len(predictions)
    )


class MetricsTracker:
    """Rastreia metricas durante o treinamento"""

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.reset()

    def reset(self):
        """Reseta acumuladores"""
        self.predictions = []
        self.targets = []

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Adiciona batch de predicoes

        Args:
            pred: Predicoes do batch [B, 2]
            target: Targets do batch [B, 2]
        """
        self.predictions.append(pred)
        self.targets.append(target)

    def compute(self) -> MetricsResult:
        """Calcula metricas acumuladas"""
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)
        return evaluate_predictions(predictions, targets, self.image_size)


def compute_detection_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    has_needle_pred: np.ndarray,
    has_needle_true: np.ndarray
) -> Dict:
    """
    Calcula metricas de deteccao binaria (agulha presente/ausente)

    Args:
        predictions: Coordenadas preditas [N, 2]
        targets: Coordenadas reais [N, 2]
        has_needle_pred: Predicao de presenca de agulha [N]
        has_needle_true: Ground truth de presenca de agulha [N]

    Returns:
        Dict com precision, recall, f1
    """
    tp = np.sum((has_needle_pred == 1) & (has_needle_true == 1))
    fp = np.sum((has_needle_pred == 1) & (has_needle_true == 0))
    fn = np.sum((has_needle_pred == 0) & (has_needle_true == 1))
    tn = np.sum((has_needle_pred == 0) & (has_needle_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }


def save_metrics(metrics: MetricsResult, filepath: str):
    """Salva metricas em arquivo JSON"""
    with open(filepath, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)


def load_metrics(filepath: str) -> MetricsResult:
    """Carrega metricas de arquivo JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return MetricsResult(**data)


def compare_models(metrics_list: List[Tuple[str, MetricsResult]]) -> str:
    """
    Compara multiplos modelos

    Args:
        metrics_list: Lista de (nome_modelo, metricas)

    Returns:
        String formatada com comparacao
    """
    output = []
    output.append("\nComparacao de Modelos")
    output.append("=" * 80)

    # Header
    header = f"{'Modelo':<20} {'MAE':>8} {'RMSE':>8} {'Eucl.':>8} {'Acc@5':>8} {'Acc@10':>8}"
    output.append(header)
    output.append("-" * 80)

    # Dados
    for name, m in metrics_list:
        row = f"{name:<20} {m.mae_pixels:>8.2f} {m.rmse_pixels:>8.2f} {m.euclidean_mean:>8.2f} {m.accuracy_5px:>7.1f}% {m.accuracy_10px:>7.1f}%"
        output.append(row)

    output.append("=" * 80)

    return "\n".join(output)


# Exemplo de uso
if __name__ == "__main__":
    # Dados de exemplo
    np.random.seed(42)
    n_samples = 100

    # Simular predicoes (com algum erro)
    targets = np.random.rand(n_samples, 2)
    noise = np.random.randn(n_samples, 2) * 0.02  # ~5px de erro
    predictions = np.clip(targets + noise, 0, 1)

    # Avaliar
    results = evaluate_predictions(predictions, targets)
    print(results)

    # Tracker exemplo
    tracker = MetricsTracker()
    for i in range(0, n_samples, 10):
        batch_pred = predictions[i:i+10]
        batch_target = targets[i:i+10]
        tracker.update(batch_pred, batch_target)

    tracked_results = tracker.compute()
    print("\nResultados do Tracker:")
    print(tracked_results)
