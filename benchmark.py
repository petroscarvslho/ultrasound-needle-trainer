#!/usr/bin/env python3
"""
NEEDLE PILOT - Benchmark de Performance
Avalia performance do modelo em diferentes configuracoes.

Metricas avaliadas:
- Latencia de inferencia (tempo por imagem)
- Throughput (imagens por segundo)
- Uso de memoria
- Performance por device (CPU, GPU, MPS)
"""

import argparse
import sys
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

import numpy as np
import torch
import torch.nn as nn

from train_vasst import VASSTPyTorch, MODELS_DIR


@dataclass
class BenchmarkResult:
    """Resultado de um benchmark"""
    device: str
    batch_size: int
    n_iterations: int
    warmup_iterations: int

    # Latencia (ms)
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

    # Throughput
    throughput_fps: float

    # Memoria
    memory_allocated_mb: float
    memory_reserved_mb: float

    def to_dict(self) -> Dict:
        return {
            'device': self.device,
            'batch_size': self.batch_size,
            'n_iterations': self.n_iterations,
            'latency_ms': {
                'mean': self.latency_mean,
                'std': self.latency_std,
                'min': self.latency_min,
                'max': self.latency_max,
                'p50': self.latency_p50,
                'p95': self.latency_p95,
                'p99': self.latency_p99
            },
            'throughput_fps': self.throughput_fps,
            'memory_mb': {
                'allocated': self.memory_allocated_mb,
                'reserved': self.memory_reserved_mb
            }
        }

    def __str__(self) -> str:
        return f"""
Benchmark Results ({self.device}, batch_size={self.batch_size})
{'=' * 50}
Latencia (ms):
  Mean:   {self.latency_mean:.2f} ± {self.latency_std:.2f}
  Min:    {self.latency_min:.2f}
  Max:    {self.latency_max:.2f}
  P50:    {self.latency_p50:.2f}
  P95:    {self.latency_p95:.2f}
  P99:    {self.latency_p99:.2f}

Throughput: {self.throughput_fps:.1f} FPS

Memoria:
  Allocated: {self.memory_allocated_mb:.1f} MB
  Reserved:  {self.memory_reserved_mb:.1f} MB
{'=' * 50}
"""


def get_memory_stats(device: torch.device) -> tuple:
    """Retorna estatisticas de memoria do device"""
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    elif device.type == 'mps':
        # MPS nao tem API de memoria detalhada
        allocated = 0
        reserved = 0
    else:
        # CPU - usar aproximacao
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            allocated = process.memory_info().rss / (1024 ** 2)
            reserved = allocated
        except ImportError:
            allocated = 0
            reserved = 0
    return allocated, reserved


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 1,
    n_iterations: int = 100,
    warmup: int = 10,
    input_size: tuple = (256, 256)
) -> BenchmarkResult:
    """
    Executa benchmark de inferencia

    Args:
        model: Modelo PyTorch
        device: Device para benchmark
        batch_size: Tamanho do batch
        n_iterations: Numero de iteracoes
        warmup: Iteracoes de warmup
        input_size: Tamanho da imagem de entrada

    Returns:
        BenchmarkResult com metricas
    """
    model = model.to(device)
    model.eval()

    # Criar input de teste
    test_input = torch.randn(batch_size, 1, input_size[0], input_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(test_input)

    # Sincronizar antes de medir
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    latencies = []

    for _ in range(n_iterations):
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(test_input)

        # Sincronizar para medicao precisa
        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)

    # Calcular memoria
    mem_allocated, mem_reserved = get_memory_stats(device)

    return BenchmarkResult(
        device=str(device),
        batch_size=batch_size,
        n_iterations=n_iterations,
        warmup_iterations=warmup,
        latency_mean=np.mean(latencies),
        latency_std=np.std(latencies),
        latency_min=np.min(latencies),
        latency_max=np.max(latencies),
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        throughput_fps=batch_size * 1000 / np.mean(latencies),
        memory_allocated_mb=mem_allocated,
        memory_reserved_mb=mem_reserved
    )


def benchmark_all_devices(
    model_path: Path = None,
    batch_sizes: List[int] = [1, 4, 8, 16, 32],
    n_iterations: int = 100
) -> List[BenchmarkResult]:
    """
    Executa benchmark em todos os devices disponiveis

    Args:
        model_path: Caminho do modelo (ou None para modelo aleatorio)
        batch_sizes: Lista de batch sizes para testar
        n_iterations: Iteracoes por teste

    Returns:
        Lista de resultados
    """
    results = []

    # Detectar devices disponiveis
    devices = [torch.device('cpu')]

    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))

    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))

    print(f"Devices disponiveis: {[str(d) for d in devices]}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Iteracoes: {n_iterations}")
    print()

    for device in devices:
        print(f"\n{'=' * 60}")
        print(f"  BENCHMARKING: {device}")
        print(f"{'=' * 60}")

        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}...", end=" ", flush=True)

            try:
                # Criar novo modelo para cada teste
                model = VASSTPyTorch(input_shape=(256, 256))

                # Carregar pesos se fornecido
                if model_path and model_path.exists():
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])

                result = benchmark_inference(
                    model,
                    device,
                    batch_size=batch_size,
                    n_iterations=n_iterations
                )
                results.append(result)

                print(f"OK - {result.latency_mean:.2f}ms, {result.throughput_fps:.1f} FPS")

                # Limpar memoria
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"ERRO: {e}")

    return results


def compare_results(results: List[BenchmarkResult]) -> str:
    """Gera tabela comparativa de resultados"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  COMPARACAO DE PERFORMANCE")
    lines.append("=" * 80)

    # Header
    header = f"{'Device':<10} {'Batch':>6} {'Latency (ms)':>15} {'Throughput':>12} {'Memory':>10}"
    lines.append(header)
    lines.append("-" * 80)

    # Agrupar por device
    current_device = None
    for r in sorted(results, key=lambda x: (x.device, x.batch_size)):
        if r.device != current_device:
            if current_device is not None:
                lines.append("-" * 80)
            current_device = r.device

        latency = f"{r.latency_mean:.2f} ± {r.latency_std:.2f}"
        throughput = f"{r.throughput_fps:.1f} FPS"
        memory = f"{r.memory_allocated_mb:.1f} MB"

        row = f"{r.device:<10} {r.batch_size:>6} {latency:>15} {throughput:>12} {memory:>10}"
        lines.append(row)

    lines.append("=" * 80)

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], filepath: str):
    """Salva resultados em JSON"""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': [r.to_dict() for r in results]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="NEEDLE PILOT - Benchmark de Performance"
    )
    parser.add_argument('--model', '-m', type=str, help='Caminho do modelo .pt')
    parser.add_argument('--iterations', '-n', type=int, default=100, help='Numero de iteracoes')
    parser.add_argument('--batch-sizes', '-b', type=str, default='1,4,8,16,32',
                        help='Batch sizes separados por virgula')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'cuda', 'mps', 'all'],
                        default='all', help='Device para benchmark')
    parser.add_argument('--output', '-o', type=str, help='Arquivo JSON de saida')

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Model path
    model_path = Path(args.model) if args.model else MODELS_DIR / "vasst_needle.pt"

    print("=" * 60)
    print("  NEEDLE PILOT - Benchmark de Performance")
    print("=" * 60)
    print(f"\nModelo: {model_path}")
    print(f"Modelo existe: {model_path.exists()}")

    if args.device == 'all':
        results = benchmark_all_devices(
            model_path=model_path,
            batch_sizes=batch_sizes,
            n_iterations=args.iterations
        )
    else:
        device = torch.device(args.device)
        model = VASSTPyTorch(input_shape=(256, 256))

        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

        results = []
        for batch_size in batch_sizes:
            result = benchmark_inference(
                model,
                device,
                batch_size=batch_size,
                n_iterations=args.iterations
            )
            results.append(result)
            print(result)

    # Comparacao
    print(compare_results(results))

    # Salvar resultados
    if args.output:
        save_results(results, args.output)
    else:
        output_path = MODELS_DIR / "benchmark_results.json"
        save_results(results, str(output_path))


if __name__ == "__main__":
    main()
