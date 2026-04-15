"""
Bake a Mobile-GS trained asset (PLY + mlp_params.pt) into a plain INRIA-style PLY
that the UE5 Gaussian Splatting plugin can load directly.

What it does:
  1. Loads the raw PLY (xyz, stale f_dc/f_rest, opacity, scale, rot)
  2. Loads mlp_params.pt (mlp_cont / mlp_dc / mlp_view weights + features_static/view)
  3. Runs the three tinycudann MLPs to produce the REAL DC and SH-rest coefficients
  4. Writes a new PLY with those baked values, same field layout as the input

What it does NOT do:
  - Does NOT bake opacity or phi from OpacityPhiNN. Those are view-dependent
    and must be evaluated per-frame at runtime on the UE5 side. The opacity
    field in the output PLY is kept as the raw sigmoid-input value from the
    source PLY (mostly meaningless for Mobile-GS assets, but preserved as a
    placeholder so the PLY is structurally valid).

Usage:
    python bake_for_ue5.py \
        --source D:/Mobile-GS/output/playroom/point_cloud/iteration_40000 \
        --sh_degree 1 \
        --out D:/Mobile-GS/output/playroom/point_cloud/iteration_40000/point_cloud_baked.ply
"""

import argparse
import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from scene.gaussian_model import GaussianModel


def bake(source_dir: str, out_path: str, sh_degree: int):
    ply_path = os.path.join(source_dir, "point_cloud.ply")
    mlp_path = os.path.join(source_dir, "mlp_params.pt")

    assert os.path.exists(ply_path), f"missing {ply_path}"
    assert os.path.exists(mlp_path), f"missing {mlp_path}"

    print(f"[bake] source dir : {source_dir}")
    print(f"[bake] sh_degree  : {sh_degree}")
    print(f"[bake] output     : {out_path}")

    # --- 1. Build an empty GaussianModel, then load PLY + MLP weights ---
    gs = GaussianModel(sh_degree=sh_degree)
    gs.active_sh_degree = sh_degree

    # load PLY first so _xyz / _features_* / scale / rot / opacity are populated.
    # the f_dc and f_rest we read here are the STALE snapshot from the checkpoint
    # -- we will overwrite them below using the MLP outputs.
    gs.load_ply(ply_path)
    n_pts = gs._xyz.shape[0]
    print(f"[bake] loaded {n_pts} points from PLY")

    # construct the MLP skeletons without training-time init of
    # _features_static / _features_view (we load those from mlp_params.pt).
    gs.construct_net(train=False)

    mlp_data = torch.load(mlp_path, weights_only=False, map_location="cuda")
    gs.mlp_cont.load_state_dict(mlp_data["mlp_cont"])
    gs.mlp_dc.load_state_dict(mlp_data["mlp_dc"])
    gs.mlp_view.load_state_dict(mlp_data["mlp_view"])
    gs._features_static = mlp_data["features_static"].cuda()
    gs._features_view = mlp_data["features_view"].cuda()
    gs.net_enabled = True
    print(f"[bake] loaded mlp_params.pt")
    print(f"[bake]   features_static shape = {tuple(gs._features_static.shape)}")
    print(f"[bake]   features_view   shape = {tuple(gs._features_view.shape)}")

    # --- 2. Run the 3 static MLPs to get the real SH coefficients ---
    with torch.no_grad():
        aabb = torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device="cuda"
        )
        contracted = gs.contract_to_unisphere(
            gs.get_xyz.clone().detach(), aabb
        )                                                      # [N, 3]
        cont_feature = gs.mlp_cont(contracted)                 # [N, 13]

        space_feature = torch.cat(
            [cont_feature, gs._features_static], dim=-1
        )                                                      # [N, 16]
        view_feature = torch.cat(
            [cont_feature, gs._features_view], dim=-1
        )                                                      # [N, 16]

        # mlp_dc  : [N, 16] -> [N, 3]  -> reshape [N, 1, 3]
        # mlp_view: [N, 16] -> [N, 3*K] -> reshape [N, K, 3]
        # both are in per-coefficient RGB interleaved layout.
        dc_real = gs.mlp_dc(space_feature).reshape(-1, 1, 3).float()
        rest_real = (
            gs.mlp_view(view_feature)
            .reshape(-1, gs.max_sh_rest, 3)
            .float()
        )                                                      # [N, K, 3]

    print(f"[bake] mlp outputs:")
    print(f"[bake]   dc_real   shape = {tuple(dc_real.shape)}")
    print(f"[bake]   rest_real shape = {tuple(rest_real.shape)}")

    # --- 3. Convert to PLY channel-major layout ---
    # PLY stores f_dc_*, f_rest_* in channel-major order:
    #   f_dc_0  = R of dc_real[:, 0]
    #   f_dc_1  = G of dc_real[:, 0]
    #   f_dc_2  = B of dc_real[:, 0]
    #   f_rest_0..K-1   = R of rest_real[:, 0..K-1]
    #   f_rest_K..2K-1  = G of rest_real[:, 0..K-1]
    #   f_rest_2K..3K-1 = B of rest_real[:, 0..K-1]
    # transpose(1,2) takes [N, K, 3] -> [N, 3, K] then flatten gives that order.
    f_dc = (
        dc_real.transpose(1, 2).reshape(n_pts, -1).cpu().numpy()
    )                                                          # [N, 3]
    f_rest = (
        rest_real.transpose(1, 2).reshape(n_pts, -1).cpu().numpy()
    )                                                          # [N, 3*K]

    xyz = gs._xyz.detach().cpu().numpy().astype(np.float32)
    opacity = gs._opacity.detach().cpu().numpy().astype(np.float32)
    if opacity.ndim == 1:
        opacity = opacity[:, None]
    scale = gs._scaling.detach().cpu().numpy().astype(np.float32)
    rot = gs._rotation.detach().cpu().numpy().astype(np.float32)

    print(f"[bake] attribute shapes:")
    print(f"[bake]   xyz     {xyz.shape}")
    print(f"[bake]   f_dc    {f_dc.shape}")
    print(f"[bake]   f_rest  {f_rest.shape}")
    print(f"[bake]   opacity {opacity.shape}")
    print(f"[bake]   scale   {scale.shape}")
    print(f"[bake]   rot     {rot.shape}")

    # --- 4. Assemble the PLY structured array ---
    attr_names = ["x", "y", "z"]
    attr_names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    attr_names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    attr_names += ["opacity"]
    attr_names += [f"scale_{i}" for i in range(scale.shape[1])]
    attr_names += [f"rot_{i}" for i in range(rot.shape[1])]

    dtype_full = [(name, "f4") for name in attr_names]
    elements = np.empty(n_pts, dtype=dtype_full)

    attributes = np.concatenate(
        [xyz, f_dc.astype(np.float32), f_rest.astype(np.float32),
         opacity, scale, rot],
        axis=1,
    )
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    PlyData([el]).write(out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[bake] wrote {out_path}  ({size_mb:.2f} MB)")
    print(
        "[bake] NOTE: opacity in the baked PLY is still the raw checkpoint "
        "value; for faithful Mobile-GS rendering you must evaluate "
        "OpacityPhiNN per-frame on the UE5 side."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="directory containing point_cloud.ply and mlp_params.pt",
    )
    parser.add_argument(
        "--sh_degree",
        type=int,
        default=1,
        help="SH degree used during training (default 1 for Mobile-GS)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output PLY path (default: <source>/point_cloud_baked.ply)",
    )
    args = parser.parse_args()

    out = args.out or os.path.join(args.source, "point_cloud_baked.ply")
    bake(args.source, out, args.sh_degree)


if __name__ == "__main__":
    main()
