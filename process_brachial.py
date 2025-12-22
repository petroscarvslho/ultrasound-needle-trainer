#!/usr/bin/env python3
"""
Processa o dataset Brachial Plexus (Sonosite) para o formato VASST.

Fluxo:
1) Le videos em brachial_plexus/data/Sonosite/videos/
2) Le anotacoes em brachial_plexus/data/Sonosite/needle/needle_coordinates/
3) Usa apenas frames anotados (start = needle tip)
4) Redimensiona para 256x256 (grayscale)
5) Normaliza coordenadas para [0, 1] no formato [y, x]
6) Salva em processed/brachial_real/images.npy e labels.npy
7) (Opcional) Gera splits X_train/Y_train/X_val/Y_val/X_test/Y_test em processed/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


BASE_DIR = Path(__file__).parent
BRACHIAL_DIR = BASE_DIR / "brachial_plexus"
SONOSITE_DIR = BRACHIAL_DIR / "data" / "Sonosite"
VIDEOS_DIR = SONOSITE_DIR / "videos"
COORDS_DIR = SONOSITE_DIR / "needle" / "needle_coordinates"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "brachial_real"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Processa o Brachial Plexus (Sonosite) para o formato VASST."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Diretorio para salvar images.npy e labels.npy."
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROCESSED_DIR),
        help="Diretorio base para salvar splits X_*.npy e Y_*.npy."
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fracao de treino (restante vira validacao/teste)."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fracao de validacao (restante vira teste)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para embaralhamento dos splits."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero total de frames processados (para debug)."
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Nao gera X_train/Y_train/X_val/Y_val/X_test/Y_test."
    )
    return parser.parse_args()


def load_annotations(coord_path: Path) -> Dict[int, Tuple[float, float]]:
    data = json.loads(coord_path.read_text())
    annotations: Dict[int, Tuple[float, float]] = {}
    for frame_key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        start = entry.get("start")
        if not start or len(start) < 2:
            continue
        try:
            frame_idx = int(frame_key)
        except ValueError:
            continue
        x, y = float(start[0]), float(start[1])
        annotations[frame_idx] = (x, y)
    return annotations


def process_video(
    video_path: Path,
    annotations: Dict[int, Tuple[float, float]],
    limit: int = None
) -> Tuple[List[np.ndarray], List[List[float]], Dict[str, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], {"frames": 0, "processed": 0, "missing": len(annotations)}

    target_frames = sorted(annotations.keys())
    if limit is not None:
        target_frames = target_frames[:limit]

    images: List[np.ndarray] = []
    labels: List[List[float]] = []
    current_index = 0
    processed = 0

    target_iter = iter(target_frames)
    target_idx = next(target_iter, None)

    while target_idx is not None:
        ret, frame = cap.read()
        if not ret:
            break

        if current_index == target_idx:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            height, width = gray.shape[:2]
            x, y = annotations[target_idx]
            x = min(max(x, 0.0), float(width))
            y = min(max(y, 0.0), float(height))

            x_norm = x / width if width else 0.0
            y_norm = y / height if height else 0.0

            resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            images.append(resized)
            labels.append([y_norm, x_norm])
            processed += 1

            target_idx = next(target_iter, None)

        current_index += 1

    cap.release()

    missing = len(target_frames) - processed
    info = {
        "frames": len(target_frames),
        "processed": processed,
        "missing": max(missing, 0),
    }
    return images, labels, info


def build_splits(
    images: np.ndarray,
    labels: np.ndarray,
    train_split: float,
    val_split: float,
    seed: int,
    processed_dir: Path
) -> Dict[str, int]:
    if train_split <= 0 or val_split < 0 or train_split + val_split >= 1:
        raise ValueError("Splits invalidos: train_split + val_split deve ser < 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))

    train_end = int(train_split * len(images))
    val_end = train_end + int(val_split * len(images))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    processed_dir.mkdir(parents=True, exist_ok=True)

    images = np.expand_dims(images, axis=-1)

    np.save(processed_dir / "X_train.npy", images[train_idx])
    np.save(processed_dir / "Y_train.npy", labels[train_idx])
    np.save(processed_dir / "X_val.npy", images[val_idx])
    np.save(processed_dir / "Y_val.npy", labels[val_idx])
    np.save(processed_dir / "X_test.npy", images[test_idx])
    np.save(processed_dir / "Y_test.npy", labels[test_idx])

    return {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)

    if not VIDEOS_DIR.exists():
        print(f"ERROR: Diretorio de videos nao encontrado: {VIDEOS_DIR}")
        return 1
    if not COORDS_DIR.exists():
        print(f"ERROR: Diretorio de anotacoes nao encontrado: {COORDS_DIR}")
        return 1

    coord_files = sorted(COORDS_DIR.glob("s*.txt"))
    if not coord_files:
        print(f"ERROR: Nenhuma anotacao encontrada em: {COORDS_DIR}")
        return 1

    all_images: List[np.ndarray] = []
    all_labels: List[List[float]] = []
    video_stats: Dict[str, Dict[str, int]] = {}

    remaining_limit = args.limit

    for coord_path in coord_files:
        video_path = VIDEOS_DIR / f"{coord_path.stem}.mp4"
        if not video_path.exists():
            print(f"WARN: Video nao encontrado: {video_path.name}")
            continue

        annotations = load_annotations(coord_path)
        if not annotations:
            print(f"WARN: Sem anotacoes validas em: {coord_path.name}")
            continue

        local_limit = remaining_limit
        if remaining_limit is not None and remaining_limit <= 0:
            break

        images, labels, info = process_video(video_path, annotations, limit=local_limit)
        all_images.extend(images)
        all_labels.extend(labels)
        video_stats[video_path.stem] = info

        if remaining_limit is not None:
            remaining_limit -= info["processed"]

        print(
            f"OK: {video_path.name}: {info['processed']}/{info['frames']} frames processados"
        )

    if not all_images:
        print("ERROR: Nenhum frame processado.")
        return 1

    images_array = np.array(all_images, dtype=np.uint8)
    labels_array = np.array(all_labels, dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)

    images_normalized = images_array.astype(np.float32) / 255.0
    np.save(output_dir / "images.npy", images_normalized)
    np.save(output_dir / "labels.npy", labels_array)

    metadata = {
        "source": "Brachial Plexus - Sonosite",
        "video_count": len(video_stats),
        "total_frames": len(images_array),
        "image_shape": [256, 256],
        "label_format": "[y, x] normalized 0-1 (needle tip)",
        "videos": video_stats,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSalvo em: {output_dir}")
    print(f"   images.npy: {images_normalized.shape}")
    print(f"   labels.npy: {labels_array.shape}")

    if not args.no_splits:
        split_counts = build_splits(
            images_array,
            labels_array,
            train_split=args.train_split,
            val_split=args.val_split,
            seed=args.seed,
            processed_dir=processed_dir,
        )
        print("\nSplits gerados em processed/:")
        print(
            f"   Treino: {split_counts['train']} | "
            f"Val: {split_counts['val']} | "
            f"Teste: {split_counts['test']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
