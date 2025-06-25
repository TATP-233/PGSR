#!/usr/bin/env python3
"""
LiDAR损失函数工具
参考lidar-rt的损失设计，使用现有的chamfer3D和lpipsPyTorch实现
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Union

from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from utils.lpipsPyTorch import lpips
from utils.loss_utils import l1_loss, l2_loss, ssim, BinaryCrossEntropyLoss

CHAMFER_AVAILABLE = True
LPIPS_AVAILABLE = True
print("Successfully imported lidar-rt modules: chamfer3D and lpipsPyTorch")

# def depth_smoothness_loss(depth, intensity=None, edge_weight=1.0):
#     """
#     深度平滑性损失，通过梯度惩罚不连续的深度变化
    
#     Args:
#         depth: (H, W) 深度图
#         intensity: (H, W) 强度图（可选，用于边缘感知）
#         edge_weight: 边缘权重
    
#     Returns:
#         loss: 平滑性损失值
#     """
#     if len(depth.shape) == 3:
#         depth = depth.squeeze()
    
#     # 计算深度梯度
#     grad_depth_x = torch.abs(depth[:, :-1] - depth[:, 1:])
#     grad_depth_y = torch.abs(depth[:-1, :] - depth[1:, :])
    
#     if intensity is not None:
#         if len(intensity.shape) == 3:
#             intensity = intensity.squeeze()
        
#         # 计算强度梯度作为边缘检测
#         grad_intensity_x = torch.abs(intensity[:, :-1] - intensity[:, 1:])
#         grad_intensity_y = torch.abs(intensity[:-1, :] - intensity[1:, :])
        
#         # 边缘权重：在强度变化大的地方减少深度平滑约束
#         weight_x = torch.exp(-edge_weight * grad_intensity_x)
#         weight_y = torch.exp(-edge_weight * grad_intensity_y)
        
#         smoothness_x = (weight_x * grad_depth_x).mean()
#         smoothness_y = (weight_y * grad_depth_y).mean()
#     else:
#         smoothness_x = grad_depth_x.mean()
#         smoothness_y = grad_depth_y.mean()
    
#     return smoothness_x + smoothness_y


def normal_consistency_loss(rendered_normal, depth_normal, mask=None, reduction='mean'):
    """
    法向量一致性损失，确保渲染法向量与深度法向量一致
    
    Args:
        rendered_normal: (3, H, W) 渲染的法向量
        depth_normal: (3, H, W) 从深度计算的法向量
        mask: (H, W) 有效区域掩码
        reduction: 归约方式
    
    Returns:
        loss: 法向量一致性损失
    """
    if rendered_normal is None or depth_normal is None:
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    # 确保法向量已归一化
    rendered_normal = F.normalize(rendered_normal, p=2, dim=0)
    depth_normal = F.normalize(depth_normal, p=2, dim=0)
    
    # 计算余弦相似度
    cosine_sim = (rendered_normal * depth_normal).sum(0)  # (H, W)
    
    # 转换为角度损失（1 - cos_sim）
    angular_loss = 1.0 - cosine_sim
    
    if mask is not None:
        angular_loss = angular_loss * mask
        if mask.sum() > 0:
            if reduction == 'mean':
                return angular_loss.sum() / mask.sum()
            else:
                return angular_loss.sum()
        else:
            return torch.tensor(0.0, device=angular_loss.device, requires_grad=True)
    
    if reduction == 'mean':
        return angular_loss.mean()
    else:
        return angular_loss.sum()


def compute_chamfer_distance_lidarrt(pred_points, gt_points):
    """
    使用lidar-rt的chamfer3D实现计算Chamfer距离
    
    Args:
        pred_points: 预测点云 (N, 3)
        gt_points: GT点云 (M, 3)
    
    Returns:
        chamfer_loss: Chamfer距离损失
    """
    if not CHAMFER_AVAILABLE:
        # Fallback to simple distance computation
        if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
            return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
        
        # 简化的点到点距离计算
        dist_matrix = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)
        dist_pred_to_gt = torch.min(dist_matrix, dim=1)[0]
        dist_gt_to_pred = torch.min(dist_matrix, dim=0)[0]
        return (dist_pred_to_gt.mean() + dist_gt_to_pred.mean()) * 0.5
    
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return torch.tensor(0.0, device=pred_points.device, requires_grad=True)
    
    # 使用lidar-rt的chamfer3D实现
    chamLoss = chamfer_3DDist()
    
    # 添加batch维度 (1, N, 3)
    pred_points_batch = pred_points.unsqueeze(0)
    gt_points_batch = gt_points.unsqueeze(0)
    
    # 计算Chamfer距离
    dist1, dist2, _, _ = chamLoss(pred_points_batch, gt_points_batch)
    
    # 返回平均Chamfer距离
    chamfer_loss = (dist1 + dist2).mean() * 0.5
    
    return chamfer_loss


def range_to_points_3d(range_image, fov_h=2*math.pi, fov_v=math.radians(26.9)):
    """
    将Range Image转换为3D点云（参考lidar-rt的inverse_projection）
    
    Args:
        range_image: (H, W) Range图像
        fov_h: 水平视场角
        fov_v: 垂直视场角
    
    Returns:
        points_3d: (N, 3) 3D点坐标
    """
    H, W = range_image.shape
    device = range_image.device
    
    # 创建像素坐标网格
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 计算角度
    azimuth = (u / W - 0.5) * fov_h
    inclination = (v / H - 0.5) * fov_v
    
    # 获取有效点
    valid_mask = range_image > 0.1
    valid_ranges = range_image[valid_mask]
    valid_azimuth = azimuth[valid_mask]
    valid_inclination = inclination[valid_mask]
    
    # 球坐标转笛卡尔坐标
    x = valid_ranges * torch.cos(valid_inclination) * torch.cos(valid_azimuth)
    y = valid_ranges * torch.cos(valid_inclination) * torch.sin(valid_azimuth)
    z = valid_ranges * torch.sin(valid_inclination)
    
    points_3d = torch.stack([x, y, z], dim=-1)
    
    return points_3d


def compute_3d_chamfer_distance(rendered_depth, gt_data, viewpoint_camera, 
                                max_points=5000, distance_threshold=2.0):
    """
    计算3D点云的Chamfer距离（使用lidar-rt实现）
    
    Args:
        rendered_depth: 渲染的深度图 (H, W)
        gt_data: GT数据字典，包含range_image
        viewpoint_camera: 相机对象
        max_points: 最大点数限制
        distance_threshold: 距离阈值（米）
    
    Returns:
        chamfer_loss: Chamfer距离损失
    """
    device = rendered_depth.device
    
    # 获取GT range image
    if 'range_image' not in gt_data:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    gt_range = gt_data['range_image']
    if gt_range is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 确保在同一设备上
    if isinstance(gt_range, np.ndarray):
        gt_range = torch.from_numpy(gt_range).float().to(device)
    else:
        gt_range = gt_range.to(device)
    
    try:
        # 检查相机是否为360度LiDAR
        is_panoramic = hasattr(viewpoint_camera, 'FoVx') and abs(viewpoint_camera.FoVx - 2 * math.pi) < 0.1
        
        if is_panoramic:
            # 360度LiDAR：使用球面投影
            fov_h = 2 * math.pi
            fov_v = viewpoint_camera.FoVy if hasattr(viewpoint_camera, 'FoVy') else math.radians(26.9)
            
            # 转换为3D点云
            pred_points = range_to_points_3d(rendered_depth, fov_h, fov_v)
            gt_points = range_to_points_3d(gt_range, fov_h, fov_v)
        else:
            # 普通透视投影（简化处理）
            H, W = rendered_depth.shape
            v, u = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            
            # 假设单位内参进行简化处理
            fx = fy = W / 2.0
            cx, cy = W / 2.0, H / 2.0
            
            # 预测点云
            pred_mask = rendered_depth > 0.1
            if pred_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            pred_z = rendered_depth[pred_mask]
            pred_x = (u[pred_mask] - cx) * pred_z / fx
            pred_y = (v[pred_mask] - cy) * pred_z / fy
            pred_points = torch.stack([pred_x, pred_y, pred_z], dim=-1)
            
            # GT点云
            gt_mask = gt_range > 0.1
            if gt_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            gt_z = gt_range[gt_mask]
            gt_x = (u[gt_mask] - cx) * gt_z / fx
            gt_y = (v[gt_mask] - cy) * gt_z / fy
            gt_points = torch.stack([gt_x, gt_y, gt_z], dim=-1)
        
        # 距离过滤
        pred_distances = torch.norm(pred_points, dim=1)
        gt_distances = torch.norm(gt_points, dim=1)
        
        pred_valid = pred_distances < 80.0
        gt_valid = gt_distances < 80.0
        
        pred_points = pred_points[pred_valid]
        gt_points = gt_points[gt_valid]
        
        if len(pred_points) == 0 or len(gt_points) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 修复尺寸不匹配问题：对两个点云进行统一的下采样
        min_points = min(len(pred_points), len(gt_points), max_points)
        
        # 对预测点云进行下采样
        if len(pred_points) > min_points:
            indices = torch.randperm(len(pred_points), device=device)[:min_points]
            pred_points = pred_points[indices]
        
        # 对GT点云进行下采样
        if len(gt_points) > min_points:
            indices = torch.randperm(len(gt_points), device=device)[:min_points]
            gt_points = gt_points[indices]
        
        # 进一步确保两个点云具有相同的点数（取较小者）
        final_points = min(len(pred_points), len(gt_points))
        if len(pred_points) > final_points:
            pred_points = pred_points[:final_points]
        if len(gt_points) > final_points:
            gt_points = gt_points[:final_points]
        
        if final_points == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 使用lidar-rt的Chamfer距离实现
        chamfer_loss = compute_chamfer_distance_lidarrt(pred_points, gt_points)
        
        # 距离阈值约束
        if distance_threshold > 0:
            chamfer_loss = torch.clamp(chamfer_loss, max=distance_threshold)
        
        return chamfer_loss
        
    except Exception as e:
        print(f"Chamfer distance computation failed: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_lpips_loss(pred_image, gt_image):
    """
    使用lidar-rt的lpipsPyTorch计算LPIPS损失
    
    Args:
        pred_image: 预测图像 (C, H, W) 或 (H, W)
        gt_image: GT图像 (C, H, W) 或 (H, W)
    
    Returns:
        lpips_loss: LPIPS损失值
    """
    if not LPIPS_AVAILABLE:
        return torch.tensor(0.0, device=pred_image.device, requires_grad=True)
    
    # 确保图像格式正确
    if len(pred_image.shape) == 2:
        pred_image = pred_image.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
    if len(gt_image.shape) == 2:
        gt_image = gt_image.unsqueeze(0).repeat(3, 1, 1)
    
    # 添加batch维度
    pred_image = pred_image.unsqueeze(0)  # (1, C, H, W)
    gt_image = gt_image.unsqueeze(0)
    
    # 归一化到[-1, 1]
    pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min() + 1e-8)
    gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min() + 1e-8)
    pred_image = 2.0 * pred_image - 1.0
    gt_image = 2.0 * gt_image - 1.0
    
    # 计算LPIPS
    try:
        lpips_loss = lpips(pred_image, gt_image, net_type='alex')
        return lpips_loss
    except Exception as e:
        print(f"Warning: LPIPS computation failed: {e}")
        return torch.tensor(0.0, device=pred_image.device, requires_grad=True)


def compute_lidar_loss(render_pkg, gt_data, loss_weights=None, use_rayhit=False):
    """
    计算完整的LiDAR损失（参考lidar-rt的三通道设计和现有实现）
    
    Args:
        render_pkg: 渲染结果包，包含depth、intensity、raydrop、rayhit_logits、raydrop_logits等
        gt_data: 真实数据，包含range_image、intensity_image等  
        loss_weights: 损失权重字典
        use_rayhit: 是否使用rayhit+raydrop的softmax模式
    
    Returns:
        loss_dict: 各项损失的字典
        total_loss: 总损失
    """
    if loss_weights is None:
        # 参考lidar-rt的权重设置
        loss_weights = {
            'depth_l1': 1.0,           # lambda_depth_l1
            'intensity_l1': 0.5,       # lambda_intensity_l1
            'intensity_l2': 0.1,       # lambda_intensity_l2  
            'intensity_dssim': 0.05,   # lambda_intensity_dssim
            'raydrop_bce': 0.1,        # lambda_raydrop_bce
            'cd': 0.01,                # lambda_cd (Chamfer Distance)
            'smoothness': 0.01,        # 深度平滑性损失
            'normal': 0.05,            # 法向量一致性损失
        }
    
    loss_dict = {}
    total_loss = 0.0
    device = next(iter(render_pkg.values())).device if render_pkg else torch.device('cuda')
    
    # 提取渲染结果
    rendered_depth = render_pkg.get("depth")
    rendered_intensity = render_pkg.get("intensity") 
    rendered_raydrop = render_pkg.get("raydrop")
    rendered_normal = render_pkg.get("rendered_normal")
    depth_normal = render_pkg.get("depth_normal")
    
    # 提取真实数据
    gt_range = gt_data.get("range_image") 
    gt_intensity = gt_data.get("intensity_image")
    gt_mask = gt_data.get("mask")
    
    # 确保数据类型和设备一致
    if gt_range is not None:
        if isinstance(gt_range, np.ndarray):
            gt_range = torch.from_numpy(gt_range).float().to(device)
        else:
            gt_range = gt_range.to(device)
    
    if gt_intensity is not None:
        if isinstance(gt_intensity, np.ndarray):
            gt_intensity = torch.from_numpy(gt_intensity).float().to(device)
        else:
            gt_intensity = gt_intensity.to(device)
    
    if gt_mask is not None:
        if isinstance(gt_mask, np.ndarray):
            gt_mask = torch.from_numpy(gt_mask).bool().to(device)
        else:
            gt_mask = gt_mask.bool().to(device)
    
    # === 深度L1损失（参考lidar-rt） ===
    if rendered_depth is not None and gt_range is not None and loss_weights.get('depth_l1', 0.0) > 0:
        if rendered_depth.dim() > 2:
            rendered_depth = rendered_depth.squeeze()
        if gt_range.dim() > 2:
            gt_range = gt_range.squeeze()
        
        if gt_mask is not None:
            if gt_mask.sum() > 0:
                depth_l1 = l1_loss(rendered_depth[gt_mask], gt_range[gt_mask])
            else:
                depth_l1 = torch.tensor(0.0, device=device)
        else:
            valid_mask = (rendered_depth > 0) & (gt_range > 0)
            if valid_mask.sum() > 0:
                depth_l1 = l1_loss(rendered_depth[valid_mask], gt_range[valid_mask])
            else:
                depth_l1 = torch.tensor(0.0, device=device)
        
        loss_dict['depth_l1'] = depth_l1
        total_loss += loss_weights['depth_l1'] * depth_l1
    
    # === 强度损失（参考lidar-rt的多项损失） ===
    if rendered_intensity is not None and gt_intensity is not None:
        if rendered_intensity.dim() > 2:
            rendered_intensity = rendered_intensity.squeeze()
        if gt_intensity.dim() > 2:
            gt_intensity = gt_intensity.squeeze()
        
        if gt_mask is not None:
            valid_mask = gt_mask
        else:
            valid_mask = (rendered_intensity > 0) & (gt_intensity > 0)
        
        if valid_mask.sum() > 0:
            # Intensity L1损失
            if loss_weights.get('intensity_l1', 0.0) > 0:
                intensity_l1 = l1_loss(rendered_intensity[valid_mask], gt_intensity[valid_mask])
                loss_dict['intensity_l1'] = intensity_l1
                total_loss += loss_weights['intensity_l1'] * intensity_l1
            
            # Intensity L2损失
            if loss_weights.get('intensity_l2', 0.0) > 0:
                intensity_l2 = l2_loss(rendered_intensity[valid_mask], gt_intensity[valid_mask])
                loss_dict['intensity_l2'] = intensity_l2
                total_loss += loss_weights['intensity_l2'] * intensity_l2
            
            # Intensity DSSIM损失
            if loss_weights.get('intensity_dssim', 0.0) > 0:
                try:
                    masked_pred = rendered_intensity * valid_mask.float()
                    masked_gt = gt_intensity * valid_mask.float()
                    intensity_ssim = ssim(masked_pred.unsqueeze(0), masked_gt.unsqueeze(0))
                    intensity_dssim = 1.0 - intensity_ssim
                    loss_dict['intensity_dssim'] = intensity_dssim
                    total_loss += loss_weights['intensity_dssim'] * intensity_dssim
                except:
                    # SSIM计算失败的fallback
                    pass
    
    # === Raydrop BCE损失（参考lidar-rt） ===
    if rendered_raydrop is not None and gt_mask is not None and loss_weights.get('raydrop_bce', 0.0) > 0:
        # gt_mask: True表示hit，False表示drop
        # labels: 0表示hit，1表示drop（与gt_mask相反）
        labels = (~gt_mask).float()  # 转换为raydrop标签
        
        if rendered_raydrop.dim() > 2:
            rendered_raydrop = rendered_raydrop.reshape(-1, 1)
        else:
            rendered_raydrop = rendered_raydrop.reshape(-1, 1)
        
        labels = labels.reshape(-1, 1)
        
        try:
            bce_loss_fn = BinaryCrossEntropyLoss()
            raydrop_bce = bce_loss_fn(labels, preds=rendered_raydrop)
            loss_dict['raydrop_bce'] = raydrop_bce
            total_loss += loss_weights['raydrop_bce'] * raydrop_bce
        except:
            # Fallback BCE
            bce_fn = torch.nn.BCELoss()
            raydrop_bce = bce_fn(rendered_raydrop, labels)
            loss_dict['raydrop_bce'] = raydrop_bce
            total_loss += loss_weights['raydrop_bce'] * raydrop_bce
    
    # === Chamfer距离损失（使用lidar-rt实现） ===
    if loss_weights.get('cd', 0.0) > 0 and rendered_depth is not None and gt_range is not None:
        try:
            cd_loss = compute_3d_chamfer_distance(
                rendered_depth.squeeze(), gt_data, None, 
                max_points=5000, distance_threshold=2.0
            )
            loss_dict['cd'] = cd_loss
            total_loss += loss_weights['cd'] * cd_loss
        except Exception as e:
            print(f"Warning: Chamfer distance computation failed: {e}")
            loss_dict['cd'] = torch.tensor(0.0, device=device)
    
    # # === 深度平滑性损失 ===
    # if rendered_depth is not None and loss_weights.get('smoothness', 0.0) > 0:
    #     smoothness_l = depth_smoothness_loss(
    #         rendered_depth.squeeze(), 
    #         rendered_intensity.squeeze() if rendered_intensity is not None else None
    #     )
    #     loss_dict['smoothness'] = smoothness_l
    #     total_loss += loss_weights['smoothness'] * smoothness_l
    
    # === 法向量一致性损失 ===
    if rendered_normal is not None and depth_normal is not None and loss_weights.get('normal', 0.0) > 0:
        normal_l = normal_consistency_loss(
            rendered_normal, 
            depth_normal, 
            mask=gt_mask if gt_mask is not None else None
        )
        loss_dict['normal'] = normal_l
        total_loss += loss_weights['normal'] * normal_l
    
    return loss_dict, total_loss 