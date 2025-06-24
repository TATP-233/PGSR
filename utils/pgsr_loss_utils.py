"""
PGSR 几何约束损失函数工具
实现PGSR论文中的几何正则化损失
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def get_image_gradient_weight(image):
    """
    计算图像梯度权重，用于边缘感知几何约束
    
    Args:
        image: (3, H, W) 或 (H, W) 图像张量
        
    Returns:
        grad_weight: (H, W) 归一化的梯度权重，范围[0,1]
    """
    if image.dim() == 3:
        # 转换为灰度图
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        gray = image
    
    # 计算Sobel梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    
    gray_pad = F.pad(gray.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(gray_pad, sobel_x)
    grad_y = F.conv2d(gray_pad, sobel_y)
    
    # 计算梯度幅度
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()
    
    # 归一化到[0,1]范围
    grad_weight = grad_magnitude / (grad_magnitude.max() + 1e-8)
    
    return grad_weight


def compute_depth_normal_from_depth(depth, fx, fy, cx, cy):
    """
    从深度图计算局部法向量
    
    Args:
        depth: (H, W) 深度图
        fx, fy, cx, cy: 相机内参
        
    Returns:
        normal: (3, H, W) 法向量图
    """
    H, W = depth.shape
    
    # 创建像素坐标网格
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype), 
                         torch.arange(W, device=depth.device, dtype=depth.dtype), 
                         indexing='ij')
    
    # 计算3D点坐标
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    # 计算相邻点之间的向量
    # 使用四邻域计算法向量：上下左右
    points_3d = torch.stack([x, y, z], dim=0)  # (3, H, W)
    
    # 计算梯度（使用有限差分）
    grad_x = torch.zeros_like(points_3d)
    grad_y = torch.zeros_like(points_3d)
    
    # X方向梯度
    grad_x[:, :, 1:-1] = (points_3d[:, :, 2:] - points_3d[:, :, :-2]) / 2.0
    grad_x[:, :, 0] = points_3d[:, :, 1] - points_3d[:, :, 0]
    grad_x[:, :, -1] = points_3d[:, :, -1] - points_3d[:, :, -2]
    
    # Y方向梯度
    grad_y[:, 1:-1, :] = (points_3d[:, 2:, :] - points_3d[:, :-2, :]) / 2.0
    grad_y[:, 0, :] = points_3d[:, 1, :] - points_3d[:, 0, :]
    grad_y[:, -1, :] = points_3d[:, -1, :] - points_3d[:, -2, :]
    
    # 叉积计算法向量
    normal = torch.cross(grad_x, grad_y, dim=0)
    
    # 归一化
    normal_norm = torch.norm(normal, dim=0, keepdim=True)
    normal = normal / (normal_norm + 1e-8)
    
    return normal


def single_view_geometry_loss(rendered_normal, depth_normal, image_grad=None):
    """
    PGSR单视图几何正则化损失
    
    Args:
        rendered_normal: (3, H, W) 渲染的法向量
        depth_normal: (3, H, W) 从深度计算的法向量
        image_grad: (H, W) 图像梯度权重（可选）
        
    Returns:
        loss: 单视图几何损失
    """
    if rendered_normal is None or depth_normal is None:
        return torch.tensor(0.0, device=rendered_normal.device if rendered_normal is not None else "cuda")
    
    # 计算法向量差异的L1损失
    normal_diff = torch.abs(rendered_normal - depth_normal).sum(0)  # (H, W)
    
    if image_grad is not None:
        # 边缘感知权重：在图像边缘处减少几何约束
        edge_weight = (1.0 - image_grad).clamp(0, 1) ** 2
        sv_loss = (edge_weight * normal_diff).mean()
    else:
        sv_loss = normal_diff.mean()
    
    return sv_loss


def multiview_photometric_loss(ref_patch, tgt_patch, valid_mask=None):
    """
    多视图光度一致性损失（使用NCC）
    
    Args:
        ref_patch: (N, C, patch_size, patch_size) 参考帧patch
        tgt_patch: (N, C, patch_size, patch_size) 目标帧patch
        valid_mask: (N,) 有效patch掩码
        
    Returns:
        loss: 多视图光度损失
    """
    if ref_patch is None or tgt_patch is None:
        return torch.tensor(0.0, device="cuda")
    
    # 计算NCC
    ref_mean = ref_patch.mean(dim=(-2, -1), keepdim=True)
    tgt_mean = tgt_patch.mean(dim=(-2, -1), keepdim=True)
    
    ref_centered = ref_patch - ref_mean
    tgt_centered = tgt_patch - tgt_mean
    
    numerator = (ref_centered * tgt_centered).sum(dim=(-2, -1))
    ref_var = (ref_centered ** 2).sum(dim=(-2, -1))
    tgt_var = (tgt_centered ** 2).sum(dim=(-2, -1))
    
    ncc = numerator / (torch.sqrt(ref_var * tgt_var) + 1e-8)
    
    # NCC损失：1 - NCC
    photo_loss = 1.0 - ncc.mean(dim=1)  # (N,)
    
    if valid_mask is not None:
        photo_loss = photo_loss[valid_mask]
    
    return photo_loss.mean()


def multiview_geometry_loss(ref_points, tgt_points, homography, valid_mask=None):
    """
    多视图几何一致性损失
    
    Args:
        ref_points: (N, 2) 参考帧像素坐标
        tgt_points: (N, 2) 目标帧像素坐标
        homography: (N, 3, 3) 单应性矩阵
        valid_mask: (N,) 有效点掩码
        
    Returns:
        loss: 多视图几何损失
    """
    if ref_points is None or tgt_points is None or homography is None:
        return torch.tensor(0.0, device="cuda")
    
    # 齐次坐标
    ref_homo = torch.cat([ref_points, torch.ones_like(ref_points[:, :1])], dim=1)  # (N, 3)
    
    # 前向变换：ref -> tgt
    tgt_pred = torch.bmm(homography, ref_homo.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    tgt_pred = tgt_pred[:, :2] / (tgt_pred[:, 2:3] + 1e-8)  # (N, 2)
    
    # 计算重投影误差
    forward_error = torch.norm(tgt_pred - tgt_points, dim=1)  # (N,)
    
    # 几何遮挡权重
    geo_weight = torch.where(forward_error < 1.0, 
                           1.0 / torch.exp(forward_error), 
                           torch.zeros_like(forward_error))
    
    if valid_mask is not None:
        forward_error = forward_error[valid_mask]
        geo_weight = geo_weight[valid_mask]
    
    # 加权几何损失
    geo_loss = (geo_weight.detach() * forward_error).mean()
    
    return geo_loss


def compute_homography_matrix(normal, distance, K_ref, K_tgt, R_rel, t_rel):
    """
    计算平面诱导的单应性矩阵
    
    Args:
        normal: (3,) 平面法向量
        distance: (1,) 平面到原点距离
        K_ref: (3, 3) 参考相机内参
        K_tgt: (3, 3) 目标相机内参
        R_rel: (3, 3) 相对旋转
        t_rel: (3,) 相对平移
        
    Returns:
        H: (3, 3) 单应性矩阵
    """
    # H = K_tgt * (R - t * n^T / d) * K_ref^(-1)
    normal = normal.unsqueeze(0)  # (1, 3)
    t_rel = t_rel.unsqueeze(-1)   # (3, 1)
    
    H = R_rel - torch.mm(t_rel, normal) / (distance + 1e-8)
    H = torch.mm(torch.mm(K_tgt, H), torch.inverse(K_ref))
    
    return H


def compute_pgsr_geometric_loss(render_pkg, gt_image, viewpoint_cam, 
                               lambda_sv_geom=0.015, lambda_mv_rgb=0.15, lambda_mv_geom=0.03):
    """
    计算完整的PGSR几何正则化损失
    
    Args:
        render_pkg: 渲染结果包
        gt_image: 真实图像
        viewpoint_cam: 当前相机
        lambda_sv_geom: 单视图几何损失权重
        lambda_mv_rgb: 多视图光度损失权重  
        lambda_mv_geom: 多视图几何损失权重
        
    Returns:
        loss_dict: 各项损失字典
        total_loss: 总几何损失
    """
    loss_dict = {}
    total_loss = torch.tensor(0.0, device="cuda")
    
    # 提取渲染结果
    rendered_normal = render_pkg.get("rendered_normal")
    depth_normal = render_pkg.get("depth_normal") 
    plane_depth = render_pkg.get("plane_depth")
    
    # 1. 单视图几何正则化损失
    if rendered_normal is not None and depth_normal is not None:
        # 计算图像梯度权重
        image_grad = get_image_gradient_weight(gt_image)
        
        # 单视图几何损失
        sv_geom_loss = single_view_geometry_loss(rendered_normal, depth_normal, image_grad)
        loss_dict["sv_geometry"] = sv_geom_loss
        total_loss += lambda_sv_geom * sv_geom_loss
    
    # 2. 多视图正则化损失（暂时简化实现）
    # 注意：完整的多视图损失需要相邻帧信息，这里先提供框架
    if hasattr(viewpoint_cam, 'nearest_id') and len(viewpoint_cam.nearest_id) > 0:
        # 多视图损失的完整实现需要：
        # - 相邻帧的渲染结果
        # - 单应性矩阵计算
        # - patch采样和NCC计算
        # 这里先占位，实际使用时需要完善
        mv_rgb_loss = torch.tensor(0.0, device="cuda")
        mv_geom_loss = torch.tensor(0.0, device="cuda")
        
        loss_dict["mv_photometric"] = mv_rgb_loss
        loss_dict["mv_geometry"] = mv_geom_loss
        total_loss += lambda_mv_rgb * mv_rgb_loss + lambda_mv_geom * mv_geom_loss
    
    return loss_dict, total_loss


def apply_exposure_compensation(rendered_image, gt_image, exp_a, exp_b):
    """
    应用曝光补偿
    
    Args:
        rendered_image: 渲染图像
        gt_image: 真实图像
        exp_a: 曝光系数a
        exp_b: 曝光系数b
        
    Returns:
        compensated_image: 曝光补偿后的图像
    """
    compensated = torch.exp(exp_a) * rendered_image + exp_b
    return compensated 