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
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import cv2


class LiDARUtils:
    """
    简化的LiDAR工具类，仅提供基本的数据处理功能
    用于支持现有的相机系统，不进行复杂的数据管理
    """
    
    @staticmethod
    def spherical_to_range_image(points, H=64, W=1024, fov_up=15.0, fov_down=-25.0, max_range=80.0):
        """
        将3D点云投影到range image
        
        Args:
            points: 点云 (N, 3) 或 (N, 4)
            H, W: range image尺寸
            fov_up, fov_down: 垂直视场角范围（度）
            max_range: 最大距离
            
        Returns:
            depth_image: 深度图 (H, W)
            intensity_image: 强度图 (H, W) 
            pixel_coords: 像素坐标 (N, 2)
        """
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        
        if points.shape[1] >= 4:
            xyz = points[:, :3]
            intensity = points[:, 3]
        else:
            xyz = points
            intensity = np.ones(len(xyz))
        
        # 计算球坐标
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # 滤除过远的点
        valid_mask = (r > 0.1) & (r < max_range)
        xyz = xyz[valid_mask]
        intensity = intensity[valid_mask]
        r = r[valid_mask]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        
        # 计算角度
        yaw = np.arctan2(y, x)
        pitch = np.arcsin(z / r)
        
        # 转换为像素坐标
        u = (yaw + np.pi) / (2 * np.pi) * W
        v = (1.0 - (pitch - np.radians(fov_down)) / 
             (np.radians(fov_up) - np.radians(fov_down))) * H
        
        u = np.clip(u, 0, W-1).astype(np.int32)
        v = np.clip(v, 0, H-1).astype(np.int32)
        
        # 创建range image
        depth_image = np.zeros((H, W), dtype=np.float32)
        intensity_image = np.zeros((H, W), dtype=np.float32)
        
        # 处理重叠像素，保留最近的点
        for i in range(len(u)):
            if depth_image[v[i], u[i]] == 0 or r[i] < depth_image[v[i], u[i]]:
                depth_image[v[i], u[i]] = r[i]
                intensity_image[v[i], u[i]] = intensity[i]
        
        pixel_coords = np.column_stack([u, v])
        
        return depth_image, intensity_image, pixel_coords
    
    @staticmethod
    def range_image_to_pointcloud(depth_image, intensity_image=None, H=64, W=1024, 
                                 fov_up=15.0, fov_down=-25.0):
        """
        将range image转换回点云
        
        Args:
            depth_image: 深度图 (H, W)
            intensity_image: 强度图 (H, W)
            H, W: 图像尺寸
            fov_up, fov_down: 垂直视场角范围（度）
            
        Returns:
            points: 3D点云 (N, 3) 或 (N, 4)
        """
        if isinstance(depth_image, torch.Tensor):
            depth_image = depth_image.cpu().numpy()
        if intensity_image is not None and isinstance(intensity_image, torch.Tensor):
            intensity_image = intensity_image.cpu().numpy()
        
        # 获取有效像素
        valid_mask = depth_image > 0
        v_indices, u_indices = np.where(valid_mask)
        depths = depth_image[valid_mask]
        
        if intensity_image is not None:
            intensities = intensity_image[valid_mask]
        else:
            intensities = np.ones(len(depths))
        
        # 计算角度
        yaw = (u_indices / W) * 2 * np.pi - np.pi
        pitch = (1.0 - v_indices / H) * (np.radians(fov_up) - np.radians(fov_down)) + np.radians(fov_down)
        
        # 转换为笛卡尔坐标
        x = depths * np.cos(pitch) * np.cos(yaw)
        y = depths * np.cos(pitch) * np.sin(yaw)
        z = depths * np.sin(pitch)
        
        points = np.column_stack([x, y, z, intensities])
        return points
    
    @staticmethod
    def normalize_intensity(intensity_image, min_val=0.0, max_val=1.0):
        """标准化强度图像"""
        if isinstance(intensity_image, torch.Tensor):
            intensity_min = intensity_image[intensity_image > 0].min() if (intensity_image > 0).any() else 0.0
            intensity_max = intensity_image.max()
            if intensity_max > intensity_min:
                normalized = (intensity_image - intensity_min) / (intensity_max - intensity_min)
                normalized = normalized * (max_val - min_val) + min_val
                normalized[intensity_image == 0] = 0  # 保持无效像素为0
                return normalized
        else:
            valid_mask = intensity_image > 0
            if valid_mask.any():
                intensity_min = intensity_image[valid_mask].min()
                intensity_max = intensity_image.max()
                if intensity_max > intensity_min:
                    normalized = np.zeros_like(intensity_image)
                    normalized[valid_mask] = ((intensity_image[valid_mask] - intensity_min) / 
                                            (intensity_max - intensity_min) * (max_val - min_val) + min_val)
                    return normalized
        
        return intensity_image
    
    @staticmethod
    def compute_ray_directions(H, W, fov_up=15.0, fov_down=-25.0):
        """
        计算每个像素的光线方向
        
        Returns:
            rays: 光线方向 (H, W, 3)
        """
        v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # 计算角度
        yaw = (u_coords / W) * 2 * np.pi - np.pi
        pitch = (1.0 - v_coords / H) * (np.radians(fov_up) - np.radians(fov_down)) + np.radians(fov_down)
        
        # 转换为单位方向向量
        x = np.cos(pitch) * np.cos(yaw)
        y = np.cos(pitch) * np.sin(yaw)  
        z = np.sin(pitch)
        
        rays = np.stack([x, y, z], axis=-1)
        return rays


# 简化的别名类，保持向后兼容性
class LiDARSensor:
    """简化的LiDAR传感器类，仅提供静态工具方法"""
    
    def __init__(self, *args, **kwargs):
        # 为了兼容性保留构造函数，但不实际做任何事情
        pass
    
    @staticmethod
    def spherical_to_range_image(*args, **kwargs):
        return LiDARUtils.spherical_to_range_image(*args, **kwargs)
    
    @staticmethod  
    def range_image_to_pointcloud(*args, **kwargs):
        return LiDARUtils.range_image_to_pointcloud(*args, **kwargs)
    
    @staticmethod
    def normalize_intensity(*args, **kwargs):
        return LiDARUtils.normalize_intensity(*args, **kwargs)
    
    @staticmethod
    def compute_ray_directions(*args, **kwargs):
        return LiDARUtils.compute_ray_directions(*args, **kwargs) 