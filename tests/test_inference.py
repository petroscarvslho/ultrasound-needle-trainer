#!/usr/bin/env python3
"""
Testes unitarios para o script de inferencia
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

# Importar apenas se modelo existir
try:
    from inference import NeedleDetector, MODEL_PATH
    MODEL_EXISTS = MODEL_PATH.exists()
except ImportError:
    MODEL_EXISTS = False


@pytest.mark.skipif(not MODEL_EXISTS, reason="Modelo nao treinado")
class TestNeedleDetector:
    """Testes para o detector de agulhas"""

    @pytest.fixture
    def detector(self):
        """Cria detector para testes"""
        return NeedleDetector(device='cpu')

    def test_detector_creation(self, detector):
        """Testa criacao do detector"""
        assert detector is not None
        assert detector.model is not None

    def test_preprocess(self, detector):
        """Testa preprocessamento"""
        # Imagem BGR
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, original_size = detector.preprocess(image)

        assert tensor.shape == (1, 1, 256, 256)
        assert original_size == (480, 640)

    def test_preprocess_grayscale(self, detector):
        """Testa preprocessamento de imagem grayscale"""
        image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        tensor, original_size = detector.preprocess(image)

        assert tensor.shape == (1, 1, 256, 256)

    def test_predict(self, detector):
        """Testa predicao"""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = detector.predict(image)

        assert 'tip_x' in result
        assert 'tip_y' in result
        assert 'tip_norm' in result
        assert 'inference_time_ms' in result

        # Coordenadas devem estar na imagem
        assert 0 <= result['tip_x'] <= 256
        assert 0 <= result['tip_y'] <= 256

    def test_predict_different_sizes(self, detector):
        """Testa predicao em diferentes tamanhos"""
        for size in [(128, 128), (256, 256), (480, 640), (1080, 1920)]:
            image = np.random.randint(0, 255, size, dtype=np.uint8)
            result = detector.predict(image)

            # Coordenadas devem estar no tamanho original
            assert 0 <= result['tip_x'] <= size[1]
            assert 0 <= result['tip_y'] <= size[0]

    def test_draw_prediction(self, detector):
        """Testa desenho de predicao"""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        prediction = {
            'tip_x': 128,
            'tip_y': 128,
            'inference_time_ms': 10.5
        }

        output = detector.draw_prediction(image, prediction)

        assert output.shape == image.shape
        # Deve ter modificado a imagem
        assert not np.array_equal(output, image)


class TestNeedleDetectorWithoutModel:
    """Testes que nao precisam do modelo"""

    def test_missing_model_error(self, tmp_path):
        """Testa erro quando modelo nao existe"""
        fake_path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            NeedleDetector(model_path=fake_path)


class TestPreprocessingFunctions:
    """Testes para funcoes de preprocessamento isoladas"""

    def test_normalize_image(self):
        """Testa normalizacao de imagem"""
        import cv2

        # Criar imagem de teste
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Redimensionar
        resized = cv2.resize(image, (256, 256))

        # Normalizar
        normalized = resized.astype(np.float32) / 255.0

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_grayscale_conversion(self):
        """Testa conversao para grayscale"""
        import cv2

        # Imagem colorida
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        assert gray.shape == (100, 100)
        assert len(gray.shape) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
