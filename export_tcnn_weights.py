"""
Export tcnn-based OpaictyPhiNN weights to a binary file for C++ / SIBR viewer.

The trained state_dict contains a single key 'net.params' — a flat fp32 tensor
that is tiny-cuda-nn's internal weight buffer (column-major, no bias).

Binary layout:
    Header  (5 x uint32, little-endian):
        [0] pad_dim          — padded input width   (e.g. 32)
        [1] hidden_dim       — hidden layer width   (e.g. 64)
        [2] n_hidden_layers  — number of hidden layers (e.g. 3)
        [3] output_dim       — network output width (e.g. 16)
        [4] n_params         — total number of fp16 weight elements
    Body:
        n_params x float16   — the raw tcnn parameter buffer

Usage:
    python export_tcnn_weights.py <opacity_phi_nn.pt> [output.bin]

Example:
    python export_tcnn_weights.py output/playroom/point_cloud/iteration_40000/opacity_phi_nn.pt
"""

import torch
import struct
import sys
import os
import numpy as np


def export(pt_path: str, out_path: str,
           pad_dim: int = 32,
           hidden_dim: int = 64,
           n_hidden_layers: int = 3,
           output_dim: int = 16):

    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    if "net.params" not in state:
        print("Error: state_dict does not contain 'net.params'.")
        print(f"  Available keys: {list(state.keys())}")
        sys.exit(1)

    params = state["net.params"]
    print(f"  params dtype  = {params.dtype}")
    print(f"  params shape  = {params.shape}")
    print(f"  params numel  = {params.numel()}")

    # Verify expected param count
    # Layer sizes: pad_dim*hidden + (n_hidden-1)*hidden*hidden + hidden*output
    expected = (pad_dim * hidden_dim
                + (n_hidden_layers - 1) * hidden_dim * hidden_dim
                + hidden_dim * output_dim)
    print(f"  expected numel = {expected}")
    if params.numel() != expected:
        print(f"  WARNING: param count mismatch! "
              f"Got {params.numel()}, expected {expected}")
        print(f"  Check pad_dim/hidden_dim/n_hidden_layers/output_dim arguments.")

    # Convert to fp16 for tcnn C++ consumption
    params_fp16 = params.half().cpu().contiguous().numpy()
    n_params = params_fp16.size

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        # Header: 5 x uint32
        f.write(struct.pack("<IIIII",
                            pad_dim, hidden_dim, n_hidden_layers,
                            output_dim, n_params))
        # Body: fp16 raw buffer
        f.write(params_fp16.tobytes())

    file_size = os.path.getsize(out_path)
    print(f"\nExport complete:")
    print(f"  pad_dim         = {pad_dim}")
    print(f"  hidden_dim      = {hidden_dim}")
    print(f"  n_hidden_layers = {n_hidden_layers}")
    print(f"  output_dim      = {output_dim}")
    print(f"  n_params (fp16) = {n_params}")
    print(f"  file size       = {file_size} bytes ({file_size/1024:.1f} KB)")
    print(f"  output path     = {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <opacity_phi_nn.pt> [output.bin]")
        print(f"Example: python {sys.argv[0]} "
              f"output/playroom/point_cloud/iteration_40000/opacity_phi_nn.pt")
        sys.exit(1)

    pt_path = sys.argv[1]
    if not os.path.isfile(pt_path):
        print(f"Error: file not found: {pt_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = os.path.join(os.path.dirname(pt_path), "opacity_phi_tcnn.bin")

    export(pt_path, out_path)
