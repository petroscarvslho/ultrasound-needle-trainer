#!/usr/bin/env python3
"""
Testes unitarios para o modelo VASST
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch

from train_vasst import VASSTPyTorch, NeedleDataset


class TestVASSTPyTorch:
    """Testes para a arquitetura do modelo"""

    def test_model_creation(self):
        """Testa criacao do modelo"""
        model = VASSTPyTorch(input_shape=(256, 256))
        assert model is not None

    def test_model_parameters(self):
        """Testa numero de parametros"""
        model = VASSTPyTorch(input_shape=(256, 256))
        total_params = sum(p.numel() for p in model.parameters())
        # Esperado: ~13M parametros
        assert 10_000_000 < total_params < 20_000_000

    def test_model_forward(self):
        """Testa forward pass"""
        model = VASSTPyTorch(input_shape=(256, 256))
        model.eval()

        # Input: batch de 4 imagens grayscale 256x256
        x = torch.randn(4, 1, 256, 256)
        output = model(x)

        # Output deve ser [batch_size, 2]
        assert output.shape == (4, 2)

    def test_model_output_range(self):
        """Testa que output esta em range razoavel"""
        model = VASSTPyTorch(input_shape=(256, 256))
        model.eval()

        x = torch.randn(10, 1, 256, 256)
        with torch.no_grad():
            output = model(x)

        # Output deve estar aproximadamente em [-1, 2] (antes de clamp)
        assert output.min() > -5
        assert output.max() < 5

    def test_model_deterministic(self):
        """Testa que modelo em eval mode e deterministico"""
        model = VASSTPyTorch(input_shape=(256, 256))
        model.eval()

        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2)

    def test_model_different_batch_sizes(self):
        """Testa diferentes tamanhos de batch"""
        model = VASSTPyTorch(input_shape=(256, 256))
        model.eval()

        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 1, 256, 256)
            output = model(x)
            assert output.shape == (batch_size, 2)

    def test_model_gradient_flow(self):
        """Testa que gradientes fluem corretamente"""
        model = VASSTPyTorch(input_shape=(256, 256))
        model.train()

        x = torch.randn(4, 1, 256, 256, requires_grad=True)
        target = torch.rand(4, 2)

        output = model(x)
        loss = torch.nn.L1Loss()(output, target)
        loss.backward()

        # Verificar que gradientes existem
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestNeedleDataset:
    """Testes para o dataset"""

    @pytest.fixture
    def sample_data(self):
        """Cria dados de exemplo"""
        n_samples = 100
        images = np.random.randint(0, 255, (n_samples, 256, 256, 1), dtype=np.uint8)
        labels = np.random.rand(n_samples, 2).astype(np.float32)
        return images, labels

    def test_dataset_creation(self, sample_data):
        """Testa criacao do dataset"""
        images, labels = sample_data
        dataset = NeedleDataset(images, labels)
        assert len(dataset) == 100

    def test_dataset_getitem(self, sample_data):
        """Testa acesso a items"""
        images, labels = sample_data
        dataset = NeedleDataset(images, labels)

        image, label = dataset[0]

        assert image.shape == (1, 256, 256)
        assert label.shape == (2,)
        assert 0 <= image.min() <= image.max() <= 1

    def test_dataset_augmentation(self, sample_data):
        """Testa data augmentation"""
        images, labels = sample_data
        dataset = NeedleDataset(images, labels, augment=True)

        # Pegar mesmo item multiplas vezes deve dar resultados diferentes
        results = []
        for _ in range(10):
            img, _ = dataset[0]
            results.append(img.numpy())

        # Pelo menos algumas devem ser diferentes
        unique_count = len(set(tuple(r.flatten()[:100]) for r in results))
        assert unique_count > 1

    def test_dataset_no_augmentation_deterministic(self, sample_data):
        """Testa que sem augmentation e deterministico"""
        images, labels = sample_data
        dataset = NeedleDataset(images, labels, augment=False)

        img1, lbl1 = dataset[0]
        img2, lbl2 = dataset[0]

        assert torch.allclose(img1, img2)
        assert torch.allclose(lbl1, lbl2)


class TestModelSaveLoad:
    """Testes para salvar/carregar modelo"""

    def test_save_and_load(self, tmp_path):
        """Testa salvar e carregar modelo"""
        # Criar e salvar modelo
        model1 = VASSTPyTorch(input_shape=(256, 256))
        save_path = tmp_path / "test_model.pt"

        torch.save({
            'model_state_dict': model1.state_dict(),
            'input_shape': (256, 256)
        }, save_path)

        # Carregar modelo
        model2 = VASSTPyTorch(input_shape=(256, 256))
        checkpoint = torch.load(save_path, weights_only=False)
        model2.load_state_dict(checkpoint['model_state_dict'])

        # Verificar que sao iguais
        model1.eval()
        model2.eval()

        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
