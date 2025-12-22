#!/usr/bin/env python3
"""
NEEDLE PILOT - K-Fold Cross-Validation
Treinamento com validacao cruzada para avaliacao robusta do modelo.

Uso:
    python cross_validation.py --folds 5 --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from train_vasst import VASSTPyTorch, NeedleDataset, load_data, MODELS_DIR
from metrics import MetricsTracker, MetricsResult, evaluate_predictions


class CrossValidator:
    """Executa K-Fold Cross-Validation para o modelo VASST"""

    def __init__(
        self,
        n_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        patience: int = 10,
        device: str = None
    ):
        self.n_folds = n_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience

        # Configurar device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.results: List[Dict] = []

    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[VASSTPyTorch, Dict]:
        """
        Treina um fold

        Args:
            fold: Numero do fold
            train_loader: DataLoader de treino
            val_loader: DataLoader de validacao

        Returns:
            Modelo treinado e metricas
        """
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1}/{self.n_folds}")
        print(f"{'=' * 60}")

        # Criar modelo
        model = VASSTPyTorch(input_shape=(256, 256))
        model = model.to(self.device)

        # Otimizador e loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.L1Loss()

        # Tracking
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            # TREINO
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # VALIDACAO
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    all_preds.append(outputs.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # Update scheduler
            scheduler.step(val_loss)

            # Log a cada 10 epocas ou no final
            if (epoch + 1) % 10 == 0 or epoch == 0:
                pixel_error = val_loss * 256
                print(f"Epoca {epoch + 1:3d}/{self.epochs} | "
                      f"Train: {train_loss:.6f} | "
                      f"Val: {val_loss:.6f} | "
                      f"~{pixel_error:.1f}px")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping na epoca {epoch + 1}")
                    break

        # Restaurar melhor modelo
        if best_state:
            model.load_state_dict(best_state)

        # Calcular metricas finais
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = model(images)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        predictions = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        metrics = evaluate_predictions(predictions, targets)

        fold_result = {
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics': metrics.to_dict()
        }

        print(f"\nFold {fold + 1} - Resultados:")
        print(f"  Val Loss: {best_val_loss:.6f}")
        print(f"  MAE: {metrics.mae_pixels:.2f}px")
        print(f"  Euclidean: {metrics.euclidean_mean:.2f} ± {metrics.euclidean_std:.2f}px")
        print(f"  Acc@5px: {metrics.accuracy_5px:.1f}%")
        print(f"  Acc@10px: {metrics.accuracy_10px:.1f}%")

        return model, fold_result

    def run(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """
        Executa cross-validation completa

        Args:
            X: Imagens [N, H, W, C]
            Y: Labels [N, 2]

        Returns:
            Resultados agregados
        """
        print("=" * 60)
        print("  NEEDLE PILOT - K-Fold Cross-Validation")
        print("=" * 60)
        print(f"\nConfiguracao:")
        print(f"  Folds: {self.n_folds}")
        print(f"  Epocas: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Amostras: {len(X)}")

        # Criar dataset completo
        full_dataset = NeedleDataset(X, Y, augment=False)

        # K-Fold split
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        self.results = []
        all_predictions = []
        all_targets = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            # Criar subsets
            train_dataset = NeedleDataset(X[train_idx], Y[train_idx], augment=True)
            val_dataset = NeedleDataset(X[val_idx], Y[val_idx], augment=False)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )

            # Treinar fold
            model, fold_result = self.train_fold(fold, train_loader, val_loader)
            self.results.append(fold_result)

            # Coletar predicoes para metricas agregadas
            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    outputs = model(images)
                    all_predictions.append(outputs.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())

        # Metricas agregadas
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        overall_metrics = evaluate_predictions(all_predictions, all_targets)

        # Resumo
        summary = self._generate_summary(overall_metrics)

        return summary

    def _generate_summary(self, overall_metrics: MetricsResult) -> Dict:
        """Gera resumo da cross-validation"""
        # Extrair metricas por fold
        val_losses = [r['best_val_loss'] for r in self.results]
        maes = [r['metrics']['mae_pixels'] for r in self.results]
        euclideans = [r['metrics']['euclidean_mean'] for r in self.results]
        acc5s = [r['metrics']['accuracy_5px'] for r in self.results]
        acc10s = [r['metrics']['accuracy_10px'] for r in self.results]

        summary = {
            'n_folds': self.n_folds,
            'config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'patience': self.patience
            },
            'per_fold': self.results,
            'aggregated': {
                'val_loss': {
                    'mean': float(np.mean(val_losses)),
                    'std': float(np.std(val_losses)),
                    'min': float(np.min(val_losses)),
                    'max': float(np.max(val_losses))
                },
                'mae_pixels': {
                    'mean': float(np.mean(maes)),
                    'std': float(np.std(maes)),
                    'min': float(np.min(maes)),
                    'max': float(np.max(maes))
                },
                'euclidean_mean': {
                    'mean': float(np.mean(euclideans)),
                    'std': float(np.std(euclideans)),
                    'min': float(np.min(euclideans)),
                    'max': float(np.max(euclideans))
                },
                'accuracy_5px': {
                    'mean': float(np.mean(acc5s)),
                    'std': float(np.std(acc5s)),
                    'min': float(np.min(acc5s)),
                    'max': float(np.max(acc5s))
                },
                'accuracy_10px': {
                    'mean': float(np.mean(acc10s)),
                    'std': float(np.std(acc10s)),
                    'min': float(np.min(acc10s)),
                    'max': float(np.max(acc10s))
                }
            },
            'overall_metrics': overall_metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        # Imprimir resumo
        print("\n" + "=" * 60)
        print("  RESUMO DA CROSS-VALIDATION")
        print("=" * 60)
        print(f"\n{self.n_folds}-Fold Cross-Validation Results:")
        print(f"\nVal Loss:    {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
        print(f"MAE:         {np.mean(maes):.2f} ± {np.std(maes):.2f} pixels")
        print(f"Euclidean:   {np.mean(euclideans):.2f} ± {np.std(euclideans):.2f} pixels")
        print(f"Acc@5px:     {np.mean(acc5s):.1f} ± {np.std(acc5s):.1f}%")
        print(f"Acc@10px:    {np.mean(acc10s):.1f} ± {np.std(acc10s):.1f}%")
        print("\nPor Fold:")
        print("-" * 60)

        for r in self.results:
            print(f"  Fold {r['fold']}: "
                  f"Loss={r['best_val_loss']:.6f}, "
                  f"MAE={r['metrics']['mae_pixels']:.2f}px, "
                  f"Acc@5={r['metrics']['accuracy_5px']:.1f}%")

        print("=" * 60)

        return summary

    def save_results(self, filepath: str):
        """Salva resultados em JSON"""
        # Converter train_losses e val_losses para listas simples
        results_copy = []
        for r in self.results:
            r_copy = r.copy()
            r_copy['train_losses'] = [float(x) for x in r['train_losses']]
            r_copy['val_losses'] = [float(x) for x in r['val_losses']]
            results_copy.append(r_copy)

        output = {
            'n_folds': self.n_folds,
            'results': results_copy
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResultados salvos em: {filepath}")

    def plot_results(self, save_path: str = None):
        """Plota resultados da cross-validation"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Loss por fold
        ax1 = axes[0, 0]
        for r in self.results:
            ax1.plot(r['val_losses'], label=f"Fold {r['fold']}", alpha=0.7)
        ax1.set_xlabel('Epoca')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Validation Loss por Fold')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Metricas por fold (barras)
        ax2 = axes[0, 1]
        folds = [r['fold'] for r in self.results]
        maes = [r['metrics']['mae_pixels'] for r in self.results]
        euclideans = [r['metrics']['euclidean_mean'] for r in self.results]

        x = np.arange(len(folds))
        width = 0.35

        ax2.bar(x - width/2, maes, width, label='MAE', color='steelblue')
        ax2.bar(x + width/2, euclideans, width, label='Euclidean', color='coral')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Erro (pixels)')
        ax2.set_title('Erro por Fold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(folds)
        ax2.legend()
        ax2.grid(True, axis='y')

        # Plot 3: Accuracy por fold
        ax3 = axes[1, 0]
        acc5s = [r['metrics']['accuracy_5px'] for r in self.results]
        acc10s = [r['metrics']['accuracy_10px'] for r in self.results]
        acc20s = [r['metrics']['accuracy_20px'] for r in self.results]

        ax3.bar(x - width, acc5s, width, label='@5px', color='green')
        ax3.bar(x, acc10s, width, label='@10px', color='orange')
        ax3.bar(x + width, acc20s, width, label='@20px', color='red')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('Accuracy por Limiar')
        ax3.set_xticks(x)
        ax3.set_xticklabels(folds)
        ax3.legend()
        ax3.grid(True, axis='y')

        # Plot 4: Boxplot de metricas
        ax4 = axes[1, 1]
        data = [
            [r['metrics']['mae_pixels'] for r in self.results],
            [r['metrics']['euclidean_mean'] for r in self.results],
            [r['metrics']['euclidean_p95'] for r in self.results]
        ]
        bp = ax4.boxplot(data, labels=['MAE', 'Euclidean Mean', 'Euclidean P95'])
        ax4.set_ylabel('Erro (pixels)')
        ax4.set_title('Distribuicao de Erros entre Folds')
        ax4.grid(True, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Grafico salvo em: {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="NEEDLE PILOT - K-Fold Cross-Validation"
    )
    parser.add_argument('--folds', '-k', type=int, default=5, help='Numero de folds (default: 5)')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Epocas por fold (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--device', '-d', type=str, choices=['cuda', 'mps', 'cpu'], help='Device')
    parser.add_argument('--output', '-o', type=str, default='cv_results', help='Nome base para outputs')

    args = parser.parse_args()

    # Carregar dados
    print("Carregando dados...")
    try:
        X_train, Y_train, X_val, Y_val = load_data()
        # Combinar treino e validacao para CV
        X = np.concatenate([X_train, X_val], axis=0)
        Y = np.concatenate([Y_train, Y_val], axis=0)
        print(f"Total de amostras: {len(X)}")
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        sys.exit(1)

    # Criar validador
    cv = CrossValidator(
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device
    )

    # Executar
    summary = cv.run(X, Y)

    # Salvar resultados
    output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.output}.json"
    plot_path = output_dir / f"{args.output}.png"

    # Salvar JSON
    with open(json_path, 'w') as f:
        # Converter para serializavel
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(summary, f, indent=2, default=convert)
    print(f"\nResultados salvos em: {json_path}")

    # Salvar plot
    cv.plot_results(str(plot_path))


if __name__ == "__main__":
    main()
