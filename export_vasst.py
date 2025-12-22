#!/usr/bin/env python3
"""
Exporta pesos VASST para inferencia (state_dict) com metadata.
"""

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description="Exportar VASST para inferencia")
    parser.add_argument(
        "--checkpoint",
        default="models/vasst_needle.pt",
        help="Checkpoint de treino (dict com model_state_dict).",
    )
    parser.add_argument(
        "--output",
        default="models/vasst_needle.pt",
        help="Arquivo de saida para state_dict.",
    )
    parser.add_argument(
        "--input-size",
        default="256,256",
        help="Input size (ex: 256,256).",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint nao encontrado: {checkpoint_path}")
        return 1

    data = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(data, dict) and "model_state_dict" in data:
        state_dict = data["model_state_dict"]
        # Preservar checkpoint se for sobrescrever
        if checkpoint_path.resolve() == output_path.resolve():
            backup_path = checkpoint_path.with_suffix(".ckpt.pt")
            torch.save(data, backup_path)
            print(f"OK: checkpoint preservado em {backup_path}")
    else:
        state_dict = data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)

    input_size = tuple(int(x) for x in args.input_size.split(","))
    meta = {
        "label_type": "point_yx",
        "label_order": "yx",
        "label_scale": "0_1",
        "input_size": input_size,
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"OK: exportado state_dict para {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
