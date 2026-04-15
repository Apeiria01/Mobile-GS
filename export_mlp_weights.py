"""
将 Mobile-GS 训练好的 opacity_phi_nn.pt 导出为 UE5 可直接加载的二进制文件。

用法:
    python export_mlp_weights.py <opacity_phi_nn.pt路径> [输出目录]

输出:
    mlp_weights.bin — 包含头部(input_dim, total_floats) + 全部权重的单文件

权重拼接顺序 (PyTorch Linear weight 为 row-major [out_dim, in_dim]):
    main.0.weight   (256 × input_dim)
    main.0.bias     (256)
    main.2.weight   (128 × 256)
    main.2.bias     (128)
    main.4.weight   (64 × 128)
    main.4.bias     (64)
    phi_output.0.weight   (1 × 64)
    phi_output.0.bias     (1)
    opacity_output.0.weight (1 × 64)
    opacity_output.0.bias   (1)
"""

import torch
import struct
import sys
import os
import numpy as np

# 权重的固定拼接顺序
WEIGHT_KEYS = [
    "main.0.weight",
    "main.0.bias",
    "main.2.weight",
    "main.2.bias",
    "main.4.weight",
    "main.4.bias",
    "phi_output.0.weight",
    "phi_output.0.bias",
    "opacity_output.0.weight",
    "opacity_output.0.bias",
]


def export(pt_path: str, out_dir: str):
    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    # 推断 input_dim: main.0.weight 的 shape 为 (256, input_dim)
    w0 = state["main.0.weight"]
    assert w0.shape[0] == 256, f"期望 main.0.weight 第0维=256, 实际={w0.shape[0]}"
    input_dim = w0.shape[1]

    # 拼接全部权重为一个 float32 数组
    all_weights = []
    for key in WEIGHT_KEYS:
        t = state[key].float().cpu().contiguous().numpy().flatten()
        all_weights.append(t)
        print(f"  {key:35s}  shape={str(list(state[key].shape)):20s}  numel={t.size}")

    concat = np.concatenate(all_weights)
    total_floats = concat.size

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mlp_weights.bin")

    with open(out_path, "wb") as f:
        # 头部: uint32 input_dim, uint32 total_floats
        f.write(struct.pack("<II", input_dim, total_floats))
        # 权重数据: float32[]
        f.write(concat.tobytes())

    file_size = os.path.getsize(out_path)
    print(f"\n导出完成:")
    print(f"  input_dim   = {input_dim}")
    print(f"  total_floats = {total_floats}")
    print(f"  文件大小     = {file_size} bytes ({file_size/1024:.1f} KB)")
    print(f"  输出路径     = {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <opacity_phi_nn.pt> [输出目录]")
        print(f"示例: python {sys.argv[0]} output/playroom/point_cloud/iteration_40000/opacity_phi_nn.pt")
        sys.exit(1)

    pt_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = os.path.join(os.path.dirname(pt_path), "mlp_weights")

    if not os.path.isfile(pt_path):
        print(f"错误: 文件不存在 {pt_path}")
        sys.exit(1)

    export(pt_path, out_dir)
