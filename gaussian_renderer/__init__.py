#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import struct
import os
import numpy as np
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_ms_nosorting import GaussianRasterizationSettings, GaussianRasterizer
import matplotlib.pyplot as plt



def render_teacher(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    from diff_gaussian_rasterization_msori import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rasterizer_depth = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[..., :1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1), p_hom)
    p_view = p_view[..., :3, :]
    depth = p_view.squeeze()[..., 2:3]
    depth = depth.repeat(1, 3)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)[0]

    rendered_depth = rasterizer_depth(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=depth,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)[0]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": rendered_depth,
            }




def _save_mlp_debug(save_dir, shs, scales, rotations, viewdirs, camera_center, phi, opacity):
    """Save OpaictyPhiNN inputs (UE5 binary) and outputs (human-readable CSV).

    Input binary (mlp_input.bin) — matches UE5 MLPForwardCS assembly order:
        Header (7 x uint32):
            magic(0x4D4C5044)  N  input_dim  shs_dim  viewdir=3  scale=3  rot=4
        Camera (3 x float32): camera_position
        Data (float32 arrays, contiguous):
            shs_raw     [N x shs_dim]    before L2-norm
            viewdirs    [N x 3]
            scales_raw  [N x 3]          before L2-norm
            rotations   [N x 4]
            feat_concat [N x input_dim]  after norm + concat (actual MLP input)

    Output CSV (mlp_output.csv):
            index, phi, opacity
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Reconstruct the assembled feature (same as OpaictyPhiNN.forward)
        shs_flat = shs.view(shs.size(0), -1)
        shs_norm = torch.nn.functional.normalize(shs_flat)
        scales_norm = torch.nn.functional.normalize(scales)
        feat = torch.cat([shs_norm, viewdirs, scales_norm, rotations], dim=1)

        shs_raw_np   = shs_flat.cpu().float().numpy()
        viewdirs_np  = viewdirs.cpu().float().numpy()
        scales_np    = scales.cpu().float().numpy()
        rot_np       = rotations.cpu().float().numpy()
        feat_np      = feat.cpu().float().numpy()
        phi_np       = phi.cpu().float().numpy().flatten()
        opacity_np   = opacity.cpu().float().numpy().flatten()
        cam_np       = camera_center.cpu().float().numpy().flatten()

        N          = feat_np.shape[0]
        input_dim  = feat_np.shape[1]
        shs_dim    = shs_raw_np.shape[1]

        # ---- Binary for UE5 ----
        bin_path = os.path.join(save_dir, "mlp_input.bin")
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<7I",
                0x4D4C5044, N, input_dim, shs_dim, 3, 3, 4))
            f.write(cam_np.astype(np.float32).tobytes())
            f.write(shs_raw_np.tobytes())
            f.write(viewdirs_np.tobytes())
            f.write(scales_np.tobytes())
            f.write(rot_np.tobytes())
            f.write(feat_np.tobytes())

        # ---- CSV for humans ----
        csv_path = os.path.join(save_dir, "mlp_output.csv")
        with open(csv_path, "w") as f:
            f.write("index,phi,opacity\n")
            for i in range(N):
                f.write(f"{i},{phi_np[i]:.6f},{opacity_np[i]:.6f}\n")

        sz = os.path.getsize(bin_path)
        print(f"\n[MLP Debug] {bin_path}  ({sz/1024/1024:.2f} MB)")
        print(f"            {csv_path}  ({N} rows)")
        print(f"  N={N}  input_dim={input_dim}  (shs={shs_dim}+viewdir=3+scale=3+rot=4)")


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, debug_save_dir = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if pc.net_enabled == False:
                shs = pc.get_features
            else:
                cont_feature = pc.mlp_cont(pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')))
                if pc.vq_enabled:
                    app_feature = pc.get_svq_appearance
                    space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
                    view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)
                else:
                    space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
                    view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)
                shs = pc.mlp_view(view_feature).reshape(-1,pc.max_sh_rest,3).float()
                dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
                shs = torch.cat([dc, shs], dim=1)
    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_xyz.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)


    phi, opacity = pc.opacity_phi_nn(shs, scales, pc.get_xyz, dir_pp_normalized, rotations)

    # ---- Save MLP debug data ----
    if debug_save_dir is not None:
        _save_mlp_debug(debug_save_dir, shs, scales, rotations,
                        dir_pp_normalized, viewpoint_camera.camera_center,
                        phi, opacity)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, kernerl_time = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        theta=torch.zeros_like(phi).cuda(),
        phi=phi,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "kernerl_time":kernerl_time
            }



def render_imp(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_ms import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rasterizer_depth = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # theta = pc._theta
    # phi = pc._phi

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None

    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    elif pc.vq_enabled:
        scales = pc.get_svq_scale
        rotations = pc.get_svq_rotation
        # theta = pc.get_svq_theta
        # phi = pc.get_svq_phi
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if pc.net_enabled == False:
                shs = pc.get_features
            else:
                cont_feature = pc.mlp_cont(pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda')))
                if pc.vq_enabled:
                    app_feature = pc.get_svq_appearance
                    space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
                    view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)
                else:
                    space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
                    view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)
                shs = pc.mlp_view(view_feature).reshape(-1,pc.max_sh_rest,3).float()
                dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
                shs = torch.cat([dc, shs], dim=1)

    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_xyz.shape[0], 1))
    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)


    phi, opacity = pc.opacity_phi_nn(shs, scales, pc.get_xyz, dir_pp_normalized, rotations)


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        theta=torch.zeros_like(phi).cuda(),
        phi=phi,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)
  
    p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[..., :1])], -1).unsqueeze(-1)
    p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1), p_hom)
    p_view = p_view[..., :3, :]
    depth = p_view.squeeze()[..., 2:3]
    depth = depth.repeat(1, 3)

    depth_img = rasterizer_depth(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=depth,
        opacities=opacity,
        theta=torch.zeros_like(phi).cuda(),
        phi=phi,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None)[0]
  

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": depth_img,
            "opacity": opacity,
            # "lod_mask": lod_mask,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }


def render_impori(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_msori import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, \
    accum_weights_ptr, accum_weights_count, accum_max_count  = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "accum_weights": accum_weights_ptr,
            "area_proj": accum_weights_count,
            "area_max": accum_max_count,
            }


def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    from diff_gaussian_rasterization_msori import GaussianRasterizationSettings, GaussianRasterizer
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    res  = rasterizer.render_depth(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return res