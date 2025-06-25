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

def render_lidar(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    LiDAR渲染函数，返回深度、intensity和raydrop
    
    使用PGSR的平面化高斯渲染，始终返回真实深度图
    
    Args:
        viewpoint_camera: 虚拟相机视点
        pc: GaussianModel实例 
        pipe: 渲染管线配置
        bg_color: 背景颜色
        scaling_modifier: 缩放修改器
        override_color: 覆盖颜色
    
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
        # 对于全景LiDAR，使用适度的tanfov值避免极端投影
        # 使用等效于60度的视场角，既能支持全景又不会导致数值问题
        tanfovx = math.tan(math.pi / 6)  # 相当于60度视场角
        print("[DEBUG] Using 360-degree panorama LiDAR mode")
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        print(f"[DEBUG] Using {math.degrees(viewpoint_camera.FoVx):.1f}-degree LiDAR mode")
    
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

    # 检查是否需要角度过滤（对于分割相机）
    if hasattr(viewpoint_camera, 'horizontal_fov_start') and hasattr(viewpoint_camera, 'horizontal_fov_end'):
        fov_start = viewpoint_camera.horizontal_fov_start
        fov_end = viewpoint_camera.horizontal_fov_end
        
        # 如果不是全景相机（360度），则需要过滤点
        if fov_end - fov_start < 359.0:  # 给1度的容差
            print(f"[DEBUG] Filtering points for camera FOV: {fov_start:.1f}° to {fov_end:.1f}°")
            
            # 获取点在相机坐标系中的位置
            world_to_camera = viewpoint_camera.world_view_transform[:3, :3]
            
            # 将点转换到相机坐标系
            points_cam = torch.matmul(means3D - viewpoint_camera.camera_center, world_to_camera.T)
            
            # 计算水平角度（相对于相机朝向）
            horizontal_angles = torch.atan2(points_cam[:, 0], -points_cam[:, 2])  # 相机Z轴向后
            horizontal_angles_deg = torch.rad2deg(horizontal_angles)
            
            # 处理角度换行（-180到180度 -> 0到360度）
            horizontal_angles_deg = (horizontal_angles_deg + 360.0) % 360.0
            
            # 处理跨越0度的FOV范围（例如300-60度）
            if fov_end < fov_start:  # 跨越0度
                angle_mask = (horizontal_angles_deg >= fov_start) | (horizontal_angles_deg <= fov_end)
            else:  # 正常范围
                angle_mask = (horizontal_angles_deg >= fov_start) & (horizontal_angles_deg <= fov_end)
            
            visible_count = angle_mask.sum().item()
            total_count = len(angle_mask)
            print(f"[DEBUG] Angle filtering: {visible_count}/{total_count} points visible")
            
            # 应用角度过滤到所有相关张量
            if visible_count > 0:
                filtered_means3D = means3D[angle_mask]
                filtered_means2D = means2D[angle_mask]
                filtered_means2D_abs = means2D_abs[angle_mask]
                filtered_opacity = opacity[angle_mask]
                filtered_scales = scales[angle_mask]
                filtered_rotations = rotations[angle_mask]
                filtered_shs = pc.get_features[angle_mask]
                filtered_lidar_colors = lidar_colors[angle_mask]
                filtered_cov3D_precomp = cov3D_precomp[angle_mask] if cov3D_precomp is not None else None
            else:
                # 没有可见点，创建空张量
                print("[DEBUG] No points visible in this camera's FOV")
                filtered_means3D = torch.empty((0, 3), device="cuda")
                filtered_means2D = torch.empty((0, 2), device="cuda")
                filtered_means2D_abs = torch.empty((0, 2), device="cuda")
                filtered_opacity = torch.empty((0, 1), device="cuda")
                filtered_scales = torch.empty((0, 3), device="cuda")
                filtered_rotations = torch.empty((0, 4), device="cuda")
                filtered_shs = torch.empty((0, pc.get_features.shape[1], pc.get_features.shape[2]), device="cuda")
                filtered_lidar_colors = torch.empty((0, 3), device="cuda")
                filtered_cov3D_precomp = None
        else:
            # 全景相机，使用所有点
            filtered_means3D = means3D
            filtered_means2D = means2D
            filtered_means2D_abs = means2D_abs
            filtered_opacity = opacity
            filtered_scales = scales
            filtered_rotations = rotations
            filtered_shs = pc.get_features
            filtered_lidar_colors = lidar_colors
            filtered_cov3D_precomp = cov3D_precomp
    else:
        # 默认使用所有点
        filtered_means3D = means3D
        filtered_means2D = means2D
        filtered_means2D_abs = means2D_abs
        filtered_opacity = opacity
        filtered_scales = scales
        filtered_rotations = rotations
        filtered_shs = pc.get_features
        filtered_lidar_colors = lidar_colors
        filtered_cov3D_precomp = cov3D_precomp

    # 为LiDAR适配增大高斯基元尺度
    lidar_scale_multiplier = 5.0  # 增大5倍尺度
    if len(filtered_means3D) > 0:
        adaptive_scale = torch.ones_like(filtered_scales[:, 0]) * lidar_scale_multiplier
        
        print(f"[DEBUG] Visible points: {len(filtered_means3D)}")
        print(f"[DEBUG] exp(scales)统计: min={torch.exp(filtered_scales).min():.6f}, max={torch.exp(filtered_scales).max():.6f}")
        
        # 应用尺度调整
        adjusted_scales = filtered_scales + torch.log(adaptive_scale).unsqueeze(-1)
        print(f"[DEBUG] 调整后exp(scales)统计: min={torch.exp(adjusted_scales).min():.6f}, max={torch.exp(adjusted_scales).max():.6f}")
    else:
        adjusted_scales = filtered_scales

    # 设置颜色
    if override_color is None:
        colors_precomp = filtered_lidar_colors
    else:
        colors_precomp = override_color

    # 设置光栅化器
    # 对于360度全景LiDAR，恢复使用标准的frustum culling，但调整投影参数
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
        prefiltered=False,  # 使用标准frustum culling
        render_geo=True,  # LiDAR渲染始终使用PGSR平面模式
        debug=pipe.debug
    )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)
    
    results = {}

    # LiDAR渲染：始终使用PGSR平面模式获取真实深度
    # 准备平面信息
    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = filtered_means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    
    input_all_map = torch.zeros((filtered_means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    # 单次渲染调用：获取真实深度、intensity、raydrop和平面信息
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D=filtered_means3D,
        means2D=filtered_means2D,
        means2D_abs=filtered_means2D_abs,
        shs=filtered_shs,
        colors_precomp=colors_precomp,
        opacities=filtered_opacity,
        scales=adjusted_scales,
        rotations=filtered_rotations,
        all_map=input_all_map,
        cov3D_precomp=filtered_cov3D_precomp
    )
    
    # 解析渲染结果
    results["depth"] = plane_depth  # PGSR计算的真实深度图
    results["intensity"] = rendered_image[1:2]  # intensity通道
    results["raydrop"] = rendered_image[2:3]    # raydrop通道
    results["rendered_normal"] = out_all_map[0:3]
    results["rendered_alpha"] = out_all_map[3:4]
    results["rendered_distance"] = out_all_map[4:5]
    results["radii"] = radii
    
    # 计算深度法向量
    depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze())
    results["depth_normal"] = depth_normal

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