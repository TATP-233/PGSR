#!/usr/bin/env python3
"""
LiDAR渲染器
基于PGSR的平面化高斯渲染，扩展支持LiDAR物理属性
使用单次渲染调用，多通道输出避免梯度计算图问题
"""

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    """
    从深度图像计算法向量
    """
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                        intrinsic_matrix.to(depth.device), 
                                        extrinsic_matrix.to(depth.device), offset)
    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render_lidar(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, 
                return_plane=True, return_depth_normal=True):
    """
    LiDAR渲染函数，返回深度、intensity和raydrop
    
    使用单次渲染调用，将LiDAR属性作为多通道颜色渲染
    
    Args:
        viewpoint_camera: 虚拟相机视点
        pc: GaussianModel实例 
        pipe: 渲染管线配置
        bg_color: 背景颜色
        scaling_modifier: 缩放修改器
        return_plane: 是否返回平面信息
        return_depth_normal: 是否返回深度法向量
    
    Returns:
        dict: 包含 'depth', 'intensity', 'raydrop', 'rendered_normal', 'depth_normal' 等渲染结果
    """
    
    # 创建零缓冲区用于梯度计算
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # 设置光栅化配置
    # 对于LiDAR的360度视场角，需要特殊处理
    if abs(viewpoint_camera.FoVx - 2 * math.pi) < 1e-3:  # 接近360度
        # 对于全景LiDAR，使用线性映射而不是tan投影
        tanfovx = viewpoint_camera.image_width / (2 * math.pi)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # 获取缩放和旋转参数
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # 准备协方差
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    # 计算视线方向（用于球谐函数）
    viewdirs = (means3D - viewpoint_camera.camera_center)
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    # 计算LiDAR属性
    intensity_values = pc.get_intensity(viewdirs)  # (N, 1)
    raydrop_probs = pc.get_raydrop_prob(viewdirs)  # (N, 1)
    
    # 创建多通道颜色：[1, intensity, raydrop] 
    # 第一个通道用于深度渲染，后两个通道分别是intensity和raydrop
    lidar_colors = torch.cat([
        torch.ones_like(intensity_values),  # 深度通道
        intensity_values,                   # intensity通道
        raydrop_probs                      # raydrop通道
    ], dim=1)  # (N, 3)

    # 设置光栅化器
    raster_settings = PlaneGaussianRasterizationSettings(
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
        render_geo=return_plane,
        debug=pipe.debug
    )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
    
    results = {}

    # 单次渲染调用
    if return_plane:
        # 准备平面信息
        global_normal = pc.get_normal(viewpoint_camera)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        # 单次渲染调用：获取深度、intensity、raydrop和平面信息
        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=None,
            colors_precomp=lidar_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            all_map=input_all_map,
            cov3D_precomp=cov3D_precomp
        )
        
        # 解析多通道输出
        results["depth"] = plane_depth
        results["intensity"] = rendered_image[1:2]  # 第二个通道
        results["raydrop"] = rendered_image[2:3]    # 第三个通道
        results["rendered_normal"] = out_all_map[0:3]
        results["rendered_alpha"] = out_all_map[3:4]
        results["rendered_distance"] = out_all_map[4:5]
        results["radii"] = radii
        
        # 计算深度法向量
        if return_depth_normal:
            depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze())
            results["depth_normal"] = depth_normal
    else:
        # 简单渲染（不使用平面信息）- 注意返回值数量不同
        rendered_image, radii, out_observe, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=lidar_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )
        
        # 解析多通道输出
        results["depth"] = rendered_image[0:1]      # 第一个通道作为深度
        results["intensity"] = rendered_image[1:2]  # 第二个通道
        results["raydrop"] = rendered_image[2:3]    # 第三个通道
        results["radii"] = radii

    # 保存屏幕空间点用于梯度计算
    results["viewspace_points"] = screenspace_points
    results["viewspace_points_abs"] = screenspace_points_abs
    results["visibility_filter"] = radii > 0

    return results


def depth_to_range_image(depth, fov_horizontal, fov_vertical):
    """
    将深度图转换为Range Image格式
    
    Args:
        depth: (H, W) 深度图
        fov_horizontal: 水平视场角 (radians)
        fov_vertical: 垂直视场角 (radians)
    
    Returns:
        range_image: (H, W) Range Image
    """
    H, W = depth.shape
    
    # 创建像素坐标网格
    v, u = torch.meshgrid(torch.arange(H, device=depth.device), 
                          torch.arange(W, device=depth.device), indexing='ij')
    
    # 转换为角度
    # 水平角度：从-pi到pi
    azimuth = (u.float() / W - 0.5) * fov_horizontal
    # 垂直角度：从上到下
    inclination = (v.float() / H - 0.5) * fov_vertical
    
    # Range Image中的距离值就是深度值
    range_image = depth
    
    return range_image


def range_image_to_points(range_image, intensity_image=None, 
                         fov_horizontal=2*math.pi, fov_vertical=math.radians(26.9)):
    """
    将Range Image转换回3D点云
    
    Args:
        range_image: (H, W) Range Image
        intensity_image: (H, W) Intensity Image (可选)
        fov_horizontal: 水平视场角
        fov_vertical: 垂直视场角
    
    Returns:
        points: (N, 3) 3D点坐标
        intensities: (N,) 强度值（如果提供）
    """
    H, W = range_image.shape
    
    # 创建像素坐标网格
    v, u = torch.meshgrid(torch.arange(H, device=range_image.device), 
                          torch.arange(W, device=range_image.device), indexing='ij')
    
    # 转换为角度
    azimuth = (u.float() / W - 0.5) * fov_horizontal
    inclination = (v.float() / H - 0.5) * fov_vertical
    
    # 获取有效点（距离>0）
    valid_mask = range_image > 0
    valid_ranges = range_image[valid_mask]
    valid_azimuth = azimuth[valid_mask]
    valid_inclination = inclination[valid_mask]
    
    # 球坐标转笛卡尔坐标
    x = valid_ranges * torch.cos(valid_inclination) * torch.cos(valid_azimuth)
    y = valid_ranges * torch.cos(valid_inclination) * torch.sin(valid_azimuth)
    z = valid_ranges * torch.sin(valid_inclination)
    
    points = torch.stack([x, y, z], dim=-1)
    
    if intensity_image is not None:
        intensities = intensity_image[valid_mask]
        return points, intensities
    else:
        return points 