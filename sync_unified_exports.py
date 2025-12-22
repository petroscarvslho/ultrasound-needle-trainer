#!/usr/bin/env python3
"""
Sincroniza exports do unified_dataset_manager para o trainer.
"""

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync unified exports -> processed/")
    parser.add_argument(
        "--source",
        default="/Users/priscoleao/aplicativo-usg-final/datasets/unified/exports/needle",
        help="Diretorio de exports do plugin NEEDLE.",
    )
    parser.add_argument(
        "--dest",
        default="processed",
        help="Destino (processed/) no ultrasound-needle-trainer.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)
    if not source_dir.exists():
        print(f"ERROR: source nao encontrado: {source_dir}")
        return 1

    dest_dir.mkdir(parents=True, exist_ok=True)
    files = ["X_train.npy", "Y_train.npy", "X_val.npy", "Y_val.npy", "X_test.npy", "Y_test.npy"]

    for name in files:
        src = source_dir / name
        if not src.exists():
            print(f"WARN: faltando {src}")
            continue
        dst = dest_dir / name
        shutil.copy2(src, dst)
        print(f"OK: {src} -> {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
