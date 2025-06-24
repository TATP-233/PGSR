#!/usr/bin/env python3
"""
LiDAR专用损失函数
包含深度损失、intensity损失、raydrop损失和几何正则化损失
"""

import torch
import torch.nn.functional as F
import math


def depth_loss(pred_depth, gt_depth, mask=None, loss_type='l1'):
    """
    深度损失函数
    
    Args:
        pred_depth: (H, W) 预测深度图
        gt_depth: (H, W) 真实深度图 
        mask: (H, W) 有效像素掩码（可选）
        loss_type: 损失类型 ('l1', 'l2', 'smooth_l1')
    
    Returns:
        loss: 深度损失值
    """
    if mask is not None:
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    
    # 过滤无效深度值（0值）
    valid_mask = (gt_depth > 0) & (pred_depth > 0)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device)
    
    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    
    if loss_type == 'l1':
        loss = F.l1_loss(pred_valid, gt_valid)
    elif loss_type == 'l2':
        loss = F.mse_loss(pred_valid, gt_valid)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred_valid, gt_valid)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def intensity_loss(pred_intensity, gt_intensity, mask=None, loss_type='l1'):
    """
    Intensity损失函数
    
    Args:
        pred_intensity: (H, W) 预测强度图
        gt_intensity: (H, W) 真实强度图
        mask: (H, W) 有效像素掩码（可选）
        loss_type: 损失类型 ('l1', 'l2', 'smooth_l1')
    
    Returns:
        loss: intensity损失值
    """
    if mask is not None:
        pred_intensity = pred_intensity[mask]
        gt_intensity = gt_intensity[mask]
    
    # 过滤无效强度值（通常intensity为-1表示无效）
    valid_mask = (gt_intensity >= 0) & (pred_intensity >= 0)
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_intensity.device)
    
    pred_valid = pred_intensity[valid_mask]
    gt_valid = gt_intensity[valid_mask]
    
    if loss_type == 'l1':
        loss = F.l1_loss(pred_valid, gt_valid)
    elif loss_type == 'l2':
        loss = F.mse_loss(pred_valid, gt_valid)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred_valid, gt_valid)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss


def raydrop_loss(pred_raydrop, gt_mask, mask=None):
    """
    Raydrop概率损失函数（二元交叉熵）
    
    Args:
        pred_raydrop: (H, W) 预测raydrop概率 [0,1]
        gt_mask: (H, W) 真实raydrop掩码 (1=dropped, 0=valid)
        mask: (H, W) 有效像素掩码（可选）
    
    Returns:
        loss: raydrop损失值
    """
    if mask is not None:
        pred_raydrop = pred_raydrop[mask]
        gt_mask = gt_mask[mask]
    
    # 将GT掩码转换为概率
    gt_prob = gt_mask.float()
    
    # 使用二元交叉熵损失
    loss = F.binary_cross_entropy(pred_raydrop, gt_prob, reduction='mean')
    
    return loss


def depth_smoothness_loss(depth, image=None, alpha=1.0):
    """
    深度平滑性损失 - 鼓励局部深度连续性
    
    Args:
        depth: (H, W) 深度图
        image: (H, W) 强度图（可选，用于边缘感知平滑）
        alpha: 平滑性权重
    
    Returns:
        loss: 平滑性损失值
    """
    # 计算深度梯度
    grad_depth_x = torch.abs(depth[:, :-1] - depth[:, 1:])
    grad_depth_y = torch.abs(depth[:-1, :] - depth[1:, :])
    
    if image is not None:
        # 边缘感知平滑 - 在图像边缘处减少平滑约束
        grad_img_x = torch.abs(image[:, :-1] - image[:, 1:])
        grad_img_y = torch.abs(image[:-1, :] - image[1:, :])
        
        weight_x = torch.exp(-alpha * grad_img_x)
        weight_y = torch.exp(-alpha * grad_img_y)
        
        grad_depth_x = grad_depth_x * weight_x
        grad_depth_y = grad_depth_y * weight_y
    
    smoothness_loss = grad_depth_x.mean() + grad_depth_y.mean()
    return smoothness_loss


def normal_consistency_loss(rendered_normal, depth_normal, mask=None):
    """
    法向量一致性损失 - 确保渲染法向量与深度计算的法向量一致
    
    Args:
        rendered_normal: (3, H, W) 渲染的法向量
        depth_normal: (3, H, W) 从深度计算的法向量
        mask: (H, W) 有效像素掩码（可选）
    
    Returns:
        loss: 法向量一致性损失
    """
    if mask is not None:
        rendered_normal = rendered_normal[:, mask]
        depth_normal = depth_normal[:, mask]
    
    # 计算法向量的角度差异
    cos_sim = F.cosine_similarity(rendered_normal, depth_normal, dim=0)
    
    # 转换为角度损失（鼓励cos_sim接近1）
    angle_loss = 1.0 - cos_sim.mean()
    
    return angle_loss


def planar_regularization_loss(points, normals, distances, scaling):
    """
    平面化正则化损失 - 鼓励高斯基元贴合平面表面
    
    Args:
        points: (N, 3) 高斯中心点
        normals: (N, 3) 局部法向量
        distances: (N,) 点到平面的距离
        scaling: (N, 3) 高斯缩放参数
    
    Returns:
        loss: 平面化正则化损失
    """
    # 鼓励最小缩放轴垂直于表面法向量
    min_scale_axis = scaling.min(dim=1)[1]
    
    # 这里需要从旋转矩阵中提取对应的轴向量
    # 简化版本：直接使用距离作为正则化
    planar_loss = distances.abs().mean()
    
    # 鼓励较小的最小缩放值（变得更"扁平"）
    min_scale_loss = scaling.min(dim=1)[0].mean()
    
    return planar_loss + 0.1 * min_scale_loss


def range_image_geometry_loss(pred_range, gt_range, fov_h=2*math.pi, fov_v=math.radians(26.9)):
    """
    Range Image几何一致性损失
    
    Args:
        pred_range: (H, W) 预测range图
        gt_range: (H, W) 真实range图
        fov_h: 水平视场角
        fov_v: 垂直视场角
    
    Returns:
        loss: 几何一致性损失
    """
    H, W = pred_range.shape
    
    # 转换为3D点云
    def range_to_points(range_img):
        v, u = torch.meshgrid(torch.arange(H, device=range_img.device), 
                             torch.arange(W, device=range_img.device), indexing='ij')
        
        azimuth = (u.float() / W - 0.5) * fov_h
        inclination = (v.float() / H - 0.5) * fov_v
        
        valid_mask = range_img > 0
        valid_ranges = range_img[valid_mask]
        valid_azimuth = azimuth[valid_mask]
        valid_inclination = inclination[valid_mask]
        
        x = valid_ranges * torch.cos(valid_inclination) * torch.cos(valid_azimuth)
        y = valid_ranges * torch.cos(valid_inclination) * torch.sin(valid_azimuth)
        z = valid_ranges * torch.sin(valid_inclination)
        
        return torch.stack([x, y, z], dim=-1)
    
    # 转换为点云并计算Chamfer距离
    pred_points = range_to_points(pred_range)
    gt_points = range_to_points(gt_range)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return torch.tensor(0.0, device=pred_range.device)
    
    # 简化的点云距离计算（可以用更精确的Chamfer距离）
    # 这里使用平均最近邻距离作为近似
    distances = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)
    min_distances, _ = distances.min(dim=1)
    geometry_loss = min_distances.mean()
    
    return geometry_loss


def compute_lidar_loss(render_pkg, gt_data, loss_weights=None):
    """
    计算完整的LiDAR损失
    
    Args:
        render_pkg: 渲染结果包，包含depth、intensity、raydrop等
        gt_data: 真实数据，包含range_image、intensity_image等
        loss_weights: 损失权重字典
    
    Returns:
        loss_dict: 各项损失的字典
        total_loss: 总损失
    """
    if loss_weights is None:
        loss_weights = {
            'depth': 1.0,
            'intensity': 0.5,
            'raydrop': 0.1,
            'smoothness': 0.01,
            'normal': 0.1,
            'planar': 0.05
        }
    
    loss_dict = {}
    total_loss = 0.0
    
    # 提取渲染结果
    rendered_depth = render_pkg.get("depth")
    rendered_intensity = render_pkg.get("intensity") 
    rendered_raydrop = render_pkg.get("raydrop")
    rendered_normal = render_pkg.get("rendered_normal")
    depth_normal = render_pkg.get("depth_normal")
    
    # 提取真实数据
    gt_range = gt_data.get("range_image")
    gt_intensity = gt_data.get("intensity_image")
    gt_mask = (gt_range > 0).float()  # 有效像素掩码
    
    # 深度损失
    if rendered_depth is not None and gt_range is not None:
        depth_l = depth_loss(rendered_depth.squeeze(), gt_range, mask=gt_mask > 0)
        loss_dict['depth'] = depth_l
        total_loss += loss_weights['depth'] * depth_l
    
    # Intensity损失
    if rendered_intensity is not None and gt_intensity is not None:
        intensity_l = intensity_loss(rendered_intensity.squeeze(), gt_intensity, mask=gt_mask > 0)
        loss_dict['intensity'] = intensity_l
        total_loss += loss_weights['intensity'] * intensity_l
    
    # Raydrop损失
    if rendered_raydrop is not None:
        raydrop_l = raydrop_loss(rendered_raydrop.squeeze(), 1.0 - gt_mask)
        loss_dict['raydrop'] = raydrop_l
        total_loss += loss_weights['raydrop'] * raydrop_l
    
    # 深度平滑性损失
    if rendered_depth is not None:
        smoothness_l = depth_smoothness_loss(rendered_depth.squeeze(), 
                                           rendered_intensity.squeeze() if rendered_intensity is not None else None)
        loss_dict['smoothness'] = smoothness_l
        total_loss += loss_weights['smoothness'] * smoothness_l
    
    # 法向量一致性损失
    if rendered_normal is not None and depth_normal is not None:
        normal_l = normal_consistency_loss(rendered_normal, depth_normal, mask=gt_mask > 0)
        loss_dict['normal'] = normal_l
        total_loss += loss_weights['normal'] * normal_l
    
    return loss_dict, total_loss 