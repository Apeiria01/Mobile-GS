#!/usr/bin/env python3
"""Export Mobile-GS tcnn opacity/phi weights for the SIBR gaussian viewer.

The current SIBR viewer expects a directory containing:
- params.bin: raw tiny-cuda-nn parameter buffer (fp16 or fp32)
- mlp_config.json: metadata used to rebuild the equivalent tcnn.Network

This script accepts either:
- a direct opacity_phi_nn.pt checkpoint path
- an iteration_xxxxx directory that contains opacity_phi_nn.pt

Examples:
    python export_sibr_tcnn_weights.py output/playroom/point_cloud/iteration_40000/opacity_phi_nn.pt
    python export_sibr_tcnn_weights.py output/playroom/point_cloud/iteration_40000 --dtype fp16
    python export_sibr_tcnn_weights.py output/playroom/point_cloud/iteration_40000 --output-dir mlp_weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch

DEFAULT_HIDDEN_DIM = 64
DEFAULT_HIDDEN_LAYERS = 3
DEFAULT_OUTPUT_DIM = 16
SUPPORTED_DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
SUPPORTED_NEURONS = {16, 32, 64, 128}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        help="Path to opacity_phi_nn.pt or an iteration_xxxxx directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to write params.bin and mlp_config.json. Defaults to <checkpoint_dir>/mlp_weights.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(SUPPORTED_DTYPES.keys()),
        default="fp16",
        help="Parameter dtype to write into params.bin.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Unpadded feature dimension. If omitted, infer from --sh-degree or nearby cfg_args.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="Use sh_degree to derive input_dim as 3*(degree+1)^2 + 3 + 3 + 4.",
    )
    parser.add_argument(
        "--pad-dim",
        type=int,
        default=None,
        help="Override padded input width. If omitted, infer from parameter count.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="FullyFusedMLP width. Defaults to 64.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=DEFAULT_HIDDEN_LAYERS,
        help="Number of hidden layers. Defaults to 3.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=DEFAULT_OUTPUT_DIM,
        help="Network output width. Defaults to 16.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        checkpoint = input_path / "opacity_phi_nn.pt"
        if checkpoint.is_file():
            return checkpoint

    raise FileNotFoundError(
        f"Could not find 'opacity_phi_nn.pt' from input path: {input_path}"
    )


def find_cfg_args(start_path: Path) -> Path | None:
    current = start_path.resolve()
    if current.is_file():
        current = current.parent

    while True:
        candidate = current / "cfg_args"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


def parse_cfg_args(cfg_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    for raw_part in text.replace("\n", ",").split(","):
        part = raw_part.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def input_dim_from_sh_degree(sh_degree: int) -> int:
    return 3 * (sh_degree + 1) * (sh_degree + 1) + 3 + 3 + 4


def align_to_16(value: int) -> int:
    return ((value + 15) // 16) * 16


def expected_param_count(pad_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> int:
    if hidden_layers < 1:
        raise ValueError("hidden_layers must be >= 1")
    return pad_dim * hidden_dim + (hidden_layers - 1) * hidden_dim * hidden_dim + hidden_dim * output_dim


def infer_pad_dim(n_params: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> int:
    remaining = n_params - ((hidden_layers - 1) * hidden_dim * hidden_dim) - (hidden_dim * output_dim)
    if remaining <= 0 or remaining % hidden_dim != 0:
        raise ValueError(
            "Cannot infer pad_dim from parameter count. Check hidden_dim, hidden_layers, and output_dim."
        )
    return remaining // hidden_dim


def determine_dims(
    args: argparse.Namespace,
    checkpoint_path: Path,
    n_params: int,
) -> Tuple[int, int, Path | None]:
    cfg_path = find_cfg_args(checkpoint_path.parent)

    input_dim = args.input_dim
    if input_dim is None and args.sh_degree is not None:
        input_dim = input_dim_from_sh_degree(args.sh_degree)

    if input_dim is None and cfg_path is not None:
        cfg_values = parse_cfg_args(cfg_path)
        sh_degree_value = cfg_values.get("sh_degree")
        if sh_degree_value is not None:
            input_dim = input_dim_from_sh_degree(int(sh_degree_value))

    pad_dim = args.pad_dim
    if pad_dim is None:
        pad_dim = infer_pad_dim(
            n_params=n_params,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            output_dim=args.output_dim,
        )

    if input_dim is None:
        input_dim = pad_dim

    if pad_dim < input_dim:
        raise ValueError(f"pad_dim ({pad_dim}) is smaller than input_dim ({input_dim}).")

    aligned_input_dim = align_to_16(input_dim)
    if pad_dim != aligned_input_dim:
        print(
            f"Warning: pad_dim={pad_dim}, but align_to_16(input_dim)={aligned_input_dim}.",
            file=sys.stderr,
        )

    expected = expected_param_count(
        pad_dim=pad_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        output_dim=args.output_dim,
    )
    if expected != n_params:
        raise ValueError(
            f"Parameter count mismatch: checkpoint has {n_params} params, but config implies {expected}."
        )

    return input_dim, pad_dim, cfg_path


def load_state_dict(checkpoint_path: Path) -> Dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def export_checkpoint(args: argparse.Namespace) -> None:
    if args.hidden_dim not in SUPPORTED_NEURONS:
        raise ValueError(
            f"hidden_dim must be one of {sorted(SUPPORTED_NEURONS)} for FullyFusedMLP."
        )

    checkpoint_path = resolve_checkpoint_path(Path(args.input_path))
    state = load_state_dict(checkpoint_path)

    if "net.params" not in state:
        available = ", ".join(sorted(state.keys()))
        raise KeyError(
            "Checkpoint does not contain 'net.params'. "
            f"Available keys: {available}"
        )

    params = state["net.params"].detach().cpu().contiguous().view(-1)
    input_dim, pad_dim, cfg_path = determine_dims(args, checkpoint_path, params.numel())

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "mlp_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "params.bin"
    config_path = output_dir / "mlp_config.json"

    export_dtype = SUPPORTED_DTYPES[args.dtype]
    params_to_write = params.to(export_dtype).contiguous().numpy()
    with open(params_path, "wb") as f:
        f.write(params_to_write.tobytes())

    config = {
        "input_dim": int(input_dim),
        "pad_dim": int(pad_dim),
        "n_input_dims": int(pad_dim),
        "output_dim": int(args.output_dim),
        "n_output_dims": int(args.output_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_neurons": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "n_hidden_layers": int(args.hidden_layers),
        "parameter_count": int(params.numel()),
        "parameter_type": args.dtype,
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": int(args.hidden_dim),
            "n_hidden_layers": int(args.hidden_layers),
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print("Export complete:")
    print(f"  checkpoint      : {checkpoint_path}")
    if cfg_path is not None:
        print(f"  cfg_args        : {cfg_path}")
    print(f"  input_dim       : {input_dim}")
    print(f"  pad_dim         : {pad_dim}")
    print(f"  hidden_dim      : {args.hidden_dim}")
    print(f"  hidden_layers   : {args.hidden_layers}")
    print(f"  output_dim      : {args.output_dim}")
    print(f"  n_params        : {params.numel()}")
    print(f"  params dtype    : {args.dtype}")
    print(f"  params.bin      : {params_path}")
    print(f"  mlp_config.json : {config_path}")


def main() -> int:
    args = parse_args()
    try:
        export_checkpoint(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
