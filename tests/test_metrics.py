#!/usr/bin/env python3
"""
Testes unitarios para o modulo de metricas
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from metrics import (
    compute_euclidean_distance,
    compute_mae,
    compute_mse,
    compute_rmse,
    compute_accuracy_at_threshold,
    evaluate_predictions,
    MetricsTracker,
    MetricsResult
)


class TestEuclideanDistance:
    """Testes para distancia euclidiana"""

    def test_zero_distance(self):
        """Testa distancia zero para predicoes perfeitas"""
        pred = np.array([[0.5, 0.5], [0.3, 0.7]])
        target = np.array([[0.5, 0.5], [0.3, 0.7]])

        distances = compute_euclidean_distance(pred, target, image_size=256)
        assert np.allclose(distances, 0)

    def test_known_distance(self):
        """Testa distancia conhecida"""
        # Distancia de (0, 0) para (1, 0) em imagem 256x256 = 256 pixels
        pred = np.array([[0.0, 0.0]])
        target = np.array([[1.0, 0.0]])

        distances = compute_euclidean_distance(pred, target, image_size=256)
        assert np.isclose(distances[0], 256.0)

    def test_diagonal_distance(self):
        """Testa distancia diagonal"""
        # Distancia de (0, 0) para (1, 1) = sqrt(2) * 256
        pred = np.array([[0.0, 0.0]])
        target = np.array([[1.0, 1.0]])

        distances = compute_euclidean_distance(pred, target, image_size=256)
        expected = np.sqrt(2) * 256
        assert np.isclose(distances[0], expected)

    def test_batch_distances(self):
        """Testa calculo em batch"""
        pred = np.random.rand(100, 2)
        target = np.random.rand(100, 2)

        distances = compute_euclidean_distance(pred, target, image_size=256)
        assert len(distances) == 100
        assert all(d >= 0 for d in distances)


class TestMAE:
    """Testes para Mean Absolute Error"""

    def test_zero_mae(self):
        """Testa MAE zero para predicoes perfeitas"""
        pred = np.array([[0.5, 0.5], [0.3, 0.7]])
        target = np.array([[0.5, 0.5], [0.3, 0.7]])

        mae = compute_mae(pred, target, image_size=256)
        assert np.isclose(mae, 0)

    def test_known_mae(self):
        """Testa MAE conhecido"""
        # Erro de 0.1 normalizado = 25.6 pixels
        pred = np.array([[0.5, 0.5]])
        target = np.array([[0.6, 0.6]])

        mae = compute_mae(pred, target, image_size=256)
        expected = 0.1 * 256  # 25.6
        assert np.isclose(mae, expected)


class TestMSEAndRMSE:
    """Testes para MSE e RMSE"""

    def test_zero_mse(self):
        """Testa MSE zero para predicoes perfeitas"""
        pred = np.array([[0.5, 0.5]])
        target = np.array([[0.5, 0.5]])

        mse = compute_mse(pred, target, image_size=256)
        assert np.isclose(mse, 0)

    def test_rmse_is_sqrt_mse(self):
        """Testa que RMSE = sqrt(MSE)"""
        pred = np.random.rand(50, 2)
        target = np.random.rand(50, 2)

        mse = compute_mse(pred, target, image_size=256)
        rmse = compute_rmse(pred, target, image_size=256)

        assert np.isclose(rmse, np.sqrt(mse))


class TestAccuracyAtThreshold:
    """Testes para accuracy por limiar"""

    def test_perfect_accuracy(self):
        """Testa 100% accuracy para distancias zero"""
        distances = np.zeros(100)
        acc = compute_accuracy_at_threshold(distances, threshold=5)
        assert acc == 100.0

    def test_zero_accuracy(self):
        """Testa 0% accuracy para distancias altas"""
        distances = np.full(100, 100.0)  # Todas > threshold
        acc = compute_accuracy_at_threshold(distances, threshold=5)
        assert acc == 0.0

    def test_partial_accuracy(self):
        """Testa accuracy parcial"""
        distances = np.array([1, 2, 3, 6, 7, 8, 9, 10, 11, 12])
        acc = compute_accuracy_at_threshold(distances, threshold=5)
        # 3 de 10 estao <= 5
        assert acc == 30.0


class TestEvaluatePredictions:
    """Testes para avaliacao completa"""

    def test_perfect_predictions(self):
        """Testa predicoes perfeitas"""
        pred = np.random.rand(100, 2)
        target = pred.copy()

        result = evaluate_predictions(pred, target)

        assert result.mae_pixels == 0
        assert result.mse_pixels == 0
        assert result.rmse_pixels == 0
        assert result.euclidean_mean == 0
        assert result.accuracy_5px == 100.0
        assert result.accuracy_10px == 100.0
        assert result.accuracy_20px == 100.0

    def test_result_structure(self):
        """Testa estrutura do resultado"""
        pred = np.random.rand(50, 2)
        target = np.random.rand(50, 2)

        result = evaluate_predictions(pred, target)

        assert isinstance(result, MetricsResult)
        assert result.n_samples == 50
        assert result.mae_pixels >= 0
        assert result.euclidean_std >= 0
        assert 0 <= result.accuracy_5px <= 100

    def test_result_to_dict(self):
        """Testa conversao para dicionario"""
        pred = np.random.rand(50, 2)
        target = np.random.rand(50, 2)

        result = evaluate_predictions(pred, target)
        d = result.to_dict()

        assert 'mae_pixels' in d
        assert 'euclidean_mean' in d
        assert 'accuracy_5px' in d


class TestMetricsTracker:
    """Testes para o MetricsTracker"""

    def test_tracker_empty(self):
        """Testa tracker vazio"""
        tracker = MetricsTracker()
        tracker.reset()
        # Nao deve dar erro ao resetar

    def test_tracker_accumulation(self):
        """Testa acumulacao de batches"""
        tracker = MetricsTracker()

        for _ in range(5):
            pred = np.random.rand(10, 2)
            target = np.random.rand(10, 2)
            tracker.update(pred, target)

        result = tracker.compute()
        assert result.n_samples == 50

    def test_tracker_reset(self):
        """Testa reset do tracker"""
        tracker = MetricsTracker()

        tracker.update(np.random.rand(10, 2), np.random.rand(10, 2))
        tracker.reset()
        tracker.update(np.random.rand(5, 2), np.random.rand(5, 2))

        result = tracker.compute()
        assert result.n_samples == 5


class TestMetricsResultString:
    """Testa representacao em string"""

    def test_str_representation(self):
        """Testa que __str__ funciona"""
        pred = np.random.rand(50, 2)
        target = np.random.rand(50, 2)

        result = evaluate_predictions(pred, target)
        s = str(result)

        assert 'MAE' in s
        assert 'RMSE' in s
        assert 'Euclidiana' in s
        assert '50 amostras' in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
