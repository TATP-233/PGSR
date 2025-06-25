#!/usr/bin/env python3
"""
LiDAR渲染器
基于PGSR的平面化高斯渲染，扩展支持LiDAR物理属性
参考lidar-rt实现三通道输出：intensities、rayhit_logits、raydrop_logits
"""

import torch
import math
import torch.nn.functional as F
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

def render_lidar(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                scaling_modifier=1.0, override_color=None, use_rayhit=False):
    """
    LiDAR渲染函数，参考lidar-rt实现三通道输出
    
    Args:
        viewpoint_camera: 虚拟相机视点
        pc: GaussianModel实例 
        pipe: 渲染管线配置
        bg_color: 背景颜色 [intensity_bg, rayhit_bg, raydrop_bg]
        scaling_modifier: 缩放修改器
        override_color: 覆盖颜色
        use_rayhit: 是否使用rayhit+raydrop的softmax模式
    
    Returns:
        dict: 包含 'depth', 'intensity', 'rayhit_logits', 'raydrop_logits', 'raydrop', etc.
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
        # print("[DEBUG] Using 360-degree panorama LiDAR mode")
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        # print(f"[DEBUG] Using {math.degrees(viewpoint_camera.FoVx):.1f}-degree LiDAR mode")
    
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
    intensity_values = pc.get_intensity(viewdirs)      # (N, 1) - 强度值
    rayhit_logits = pc.get_rayhit_logits(viewdirs)    # (N, 1) - 光线命中逻辑值
    raydrop_logits = pc.get_raydrop_logits(viewdirs)  # (N, 1) - 光线丢失逻辑值
    
    # 创建三通道颜色：[intensity, rayhit_logits, raydrop_logits] 
    # 参考lidar-rt的rendered_tensor[:, :, 0:3]格式
    lidar_colors = torch.cat([
        intensity_values,   # 通道0：强度值
        rayhit_logits,      # 通道1：光线命中逻辑值  
        raydrop_logits      # 通道2：光线丢失逻辑值
    ], dim=1)  # (N, 3)

    # 检查是否需要角度过滤（对于分割相机）
    angle_mask = None  # 保存角度掩码以供后续使用
    is_split_camera = False
    
    if (hasattr(viewpoint_camera, 'horizontal_fov_start') and 
        hasattr(viewpoint_camera, 'horizontal_fov_end') and
        viewpoint_camera.horizontal_fov_start is not None and
        viewpoint_camera.horizontal_fov_end is not None):
        fov_start = viewpoint_camera.horizontal_fov_start
        fov_end = viewpoint_camera.horizontal_fov_end
        
        # 如果不是全景相机（360度），则需要过滤点
        if fov_end - fov_start < 359.0:  # 给1度的容差
            is_split_camera = True
            # print(f"[DEBUG] Filtering points for camera FOV: {fov_start:.1f}° to {fov_end:.1f}°")
            
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
            # print(f"[DEBUG] Angle filtering: {visible_count}/{total_count} points visible")
            
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
                # print("[DEBUG] No points visible in this camera's FOV")
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

    # 为LiDAR适配增大高斯基元尺度 - 移除重复缩放
    # 注意：尺度限制现在由gaussian_model.py中的safe_scaling_activation处理
    if len(filtered_means3D) > 0:
        # 移除额外的尺度缩放，避免重复放大
        # 之前的5倍缩放现在通过create_from_pcd初始化时完成
        adjusted_scales = filtered_scales
        
        # 仅在初始几次迭代显示调试信息
        if viewpoint_camera.image_name.endswith('_cam_0') and len(filtered_means3D) > 0:
            scales_exp = torch.exp(filtered_scales)
            # print(f"[DEBUG] Visible points: {len(filtered_means3D)}")
            # print(f"[DEBUG] Scales range: min={scales_exp.min():.3f}, max={scales_exp.max():.3f}, mean={scales_exp.mean():.3f}")
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
    # 准备平面信息 - 确保法向量计算使用过滤后的点云
    if len(filtered_means3D) > 0:
        # 为过滤后的点云计算法向量
        if is_split_camera and angle_mask is not None:
            # 分割相机：为过滤后的点计算法向量
            filtered_xyz = pc.get_xyz[angle_mask]
            filtered_scaling = pc.get_scaling[angle_mask]
            filtered_rotation = pc.get_rotation[angle_mask]
            
            # 计算最小轴向量作为法向量
            from pytorch3d.transforms import quaternion_to_matrix
            rotation_matrices = quaternion_to_matrix(filtered_rotation)
            smallest_axis_idx = filtered_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
            smallest_axis = rotation_matrices.gather(2, smallest_axis_idx).squeeze(dim=2)
            
            # 调整法向量方向
            gaussian_to_cam_global = viewpoint_camera.camera_center - filtered_xyz
            neg_mask = (smallest_axis * gaussian_to_cam_global).sum(-1) < 0.0
            smallest_axis[neg_mask] = -smallest_axis[neg_mask]
            
            global_normal = smallest_axis
        else:
            # 全景相机或没有角度限制：使用完整的法向量
            global_normal = pc.get_normal(viewpoint_camera)
        
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        pts_in_cam = filtered_means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    else:
        # 没有可见点，创建空的平面信息
        local_normal = torch.empty((0, 3), device="cuda")
        local_distance = torch.empty((0,), device="cuda")
    
    if len(filtered_means3D) > 0:
        input_all_map = torch.zeros((filtered_means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance
    else:
        input_all_map = torch.empty((0, 5)).cuda().float()

    # 单次渲染调用：获取真实深度、三通道LiDAR数据和平面信息
    # 在LiDAR模式下，使用预计算的颜色而不是球谐函数
    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D=filtered_means3D,
        means2D=filtered_means2D,
        means2D_abs=filtered_means2D_abs,
        shs=None,  # LiDAR模式不使用球谐函数
        colors_precomp=colors_precomp,
        opacities=filtered_opacity,
        scales=adjusted_scales,
        rotations=filtered_rotations,
        all_map=input_all_map,
        cov3D_precomp=filtered_cov3D_precomp
    )
    
    # 解析渲染结果 - 参考lidar-rt的三通道输出
    results["depth"] = plane_depth  # PGSR计算的真实深度图
    
    # 三个核心通道（参考lidar-rt的rendered_tensor格式）
    intensities = rendered_image[0:1]      # 通道0：强度值
    rayhit_logits = rendered_image[1:2]    # 通道1：光线命中逻辑值
    raydrop_logits = rendered_image[2:3]   # 通道2：光线丢失逻辑值
    
    results["intensity"] = intensities
    results["rayhit_logits"] = rayhit_logits  
    results["raydrop_logits"] = raydrop_logits
    
    # 计算raydrop概率 - 参考lidar-rt的处理逻辑
    if use_rayhit:
        # 使用rayhit+raydrop的softmax模式
        logits = torch.cat([rayhit_logits, raydrop_logits], dim=-1)
        prob = F.softmax(logits, dim=-1)
        raydrop_prob = prob[..., 1:2]  # 取raydrop的概率
    else:
        # 只对raydrop_logits使用sigmoid
        raydrop_prob = torch.sigmoid(raydrop_logits)
    
    results["raydrop"] = raydrop_prob
    
    # PGSR平面信息
    results["rendered_normal"] = out_all_map[0:3]
    results["rendered_alpha"] = out_all_map[3:4]
    results["rendered_distance"] = out_all_map[4:5]
    results["radii"] = radii
    
    # 计算深度法向量
    depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze())
    results["depth_normal"] = depth_normal

    # 保存屏幕空间点用于梯度计算
    # 确保viewspace点张量和可见性过滤器都对应完整的点云
    if is_split_camera and angle_mask is not None:
        # 分割相机模式：扩展到完整点云大小
        full_screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda")
        full_screenspace_points_abs = torch.zeros_like(means3D, requires_grad=True, device="cuda")
        
        # 将过滤后的屏幕空间点复制到完整张量中
        if len(filtered_means3D) > 0:
            # 注意：避免就地操作，使用clone和detach
            # 创建新的张量并复制数据
            temp_full_screenspace_points = full_screenspace_points.clone()
            temp_full_screenspace_points_abs = full_screenspace_points_abs.clone()
            temp_full_screenspace_points[angle_mask] = filtered_means2D.clone()
            temp_full_screenspace_points_abs[angle_mask] = filtered_means2D_abs.clone()
            
            full_screenspace_points = temp_full_screenspace_points
            full_screenspace_points_abs = temp_full_screenspace_points_abs
        
        try:
            full_screenspace_points.retain_grad()
            full_screenspace_points_abs.retain_grad()
        except:
            pass
            
        results["viewspace_points"] = full_screenspace_points
        results["viewspace_points_abs"] = full_screenspace_points_abs
        
        # 将过滤后的可见性映射回完整点云
        full_visibility_filter = torch.zeros(len(means3D), dtype=torch.bool, device="cuda")
        filtered_visibility = radii > 0
        if len(filtered_visibility) > 0:
            full_visibility_filter[angle_mask] = filtered_visibility
        results["visibility_filter"] = full_visibility_filter
    else:
        # 全景相机模式：直接使用原始张量
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