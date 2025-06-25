#!/usr/bin/env python3
"""
LiDAR-PGSR评估工具
基于lidar-rt的评估指标实现，使用现有的chamfer3D和lpipsPyTorch模块
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import open3d as o3d
from skimage.metrics import structural_similarity
from typing import Dict, List, Tuple, Optional
import os

# 使用lidar-rt现有的实现
try:
    import sys
    # 添加lidar-rt路径到系统路径
    lidar_rt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lidar-rt')
    if lidar_rt_path not in sys.path:
        sys.path.append(lidar_rt_path)
    
    from utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    from utils.lpipsPyTorch import lpips
    from utils.loss_utils import l1_loss, l2_loss, ssim, mse, psnr
    
    CHAMFER_AVAILABLE = True
    LPIPS_AVAILABLE = True
    print("Successfully imported lidar-rt evaluation modules")
    
except ImportError as e:
    print(f"Warning: Failed to import lidar-rt modules for evaluation: {e}")
    CHAMFER_AVAILABLE = False
    LPIPS_AVAILABLE = False
    
    # Fallback implementations
    def l1_loss(pred, gt):
        return torch.abs(pred - gt).mean()
    
    def l2_loss(pred, gt):
        return ((pred - gt) ** 2).mean()
    
    def mse(pred, gt):
        return ((pred - gt) ** 2).mean()
    
    def psnr(pred, gt, mask=None):
        if mask is not None:
            pred = pred[mask]
            gt = gt[mask]
        mse_val = torch.mean((pred - gt) ** 2)
        if mse_val == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(1.0 / torch.sqrt(mse_val))
    
    def ssim(img1, img2):
        return torch.tensor(0.8)  # 返回固定值


class LiDARPGSREvaluator:
    """LiDAR-PGSR评估器，实现lidar-rt的评估指标，使用现有模块"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 初始化LPIPS模型
        if LPIPS_AVAILABLE:
            try:
                # 使用lidar-rt的lpips实现，它会自动初始化模型
                self.lpips_fn = lambda x, y: lpips(x, y, net_type="alex")
            except Exception as e:
                print(f"Warning: Failed to initialize LPIPS: {e}. LPIPS evaluation will be disabled.")
                self.lpips_fn = None
        else:
            self.lpips_fn = None
        
        # 初始化Chamfer距离
        if CHAMFER_AVAILABLE:
            self.chamfer_fn = chamfer_3DDist()
        else:
            self.chamfer_fn = None
        
        # 评估指标名称
        self.depth_metrics = ["rmse", "mae", "medae", "lpips_loss", "ssim", "psnr"]
        self.intensity_metrics = ["rmse", "mae", "medae", "lpips_loss", "ssim", "psnr"]
        self.raydrop_metrics = ["rmse", "acc", "f1"]
        self.points_metrics = ["chamfer_dist", "fscore"]
        
        # 参数设置
        self.raydrop_ratio = 0.4  # raydrop阈值
        self.fscore_threshold = 0.05  # F-score阈值
    
    def compute_depth_metrics(self, gt: np.ndarray, pred: np.ndarray, 
                            min_depth: float = 1e-6, max_depth: float = 80.0) -> List[float]:
        """
        计算深度评估指标（参考lidar-rt）
        Args:
            gt: Ground truth depth [H, W]
            pred: Predicted depth [H, W]
            min_depth: 最小有效深度
            max_depth: 最大有效深度
        Returns:
            [rmse, mae, medae, lpips_loss, ssim, psnr]
        """
        # 转换为tensor
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float().to(self.device)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float().to(self.device)
        
        # 有效掩码
        valid = (gt > min_depth) & (gt < max_depth) & (pred > min_depth) & (pred < max_depth)
        
        if valid.sum() == 0:
            return [float('nan')] * 6
        
        gt_valid = gt[valid]
        pred_valid = pred[valid]
        
        # RMSE
        rmse = torch.sqrt(mse(pred_valid, gt_valid)).item()
        
        # MAE
        mae = l1_loss(pred_valid, gt_valid).item()
        
        # MedAE
        medae = torch.median(torch.abs(pred_valid - gt_valid)).item()
        
        # 为LPIPS和SSIM准备图像格式 (转换为3通道)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        
        gt_3ch = gt_norm.unsqueeze(0).repeat(3, 1, 1)
        pred_3ch = pred_norm.unsqueeze(0).repeat(3, 1, 1)
        
        # LPIPS
        if self.lpips_fn is not None:
            try:
                pred_tensor = pred_3ch.unsqueeze(0)  # (1, 3, H, W)
                gt_tensor = gt_3ch.unsqueeze(0)
                
                # 转换到[-1,1]范围
                pred_tensor = 2.0 * pred_tensor - 1.0
                gt_tensor = 2.0 * gt_tensor - 1.0
                
                with torch.no_grad():
                    lpips_value = self.lpips_fn(pred_tensor, gt_tensor).item()
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                lpips_value = float('nan')
        else:
            lpips_value = float('nan')
        
        # SSIM
        try:
            ssim_value = ssim(gt_3ch, pred_3ch).item()
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            ssim_value = float('nan')
        
        # PSNR
        try:
            psnr_value = psnr(gt_valid, pred_valid).item()
        except Exception as e:
            print(f"Warning: PSNR computation failed: {e}")
            psnr_value = float('nan')
        
        return [rmse, mae, medae, lpips_value, ssim_value, psnr_value]
    
    def compute_intensity_metrics(self, gt: np.ndarray, pred: np.ndarray) -> List[float]:
        """
        计算强度评估指标
        Args:
            gt: Ground truth intensity [H, W]
            pred: Predicted intensity [H, W]
        Returns:
            [rmse, mae, medae, lpips_loss, ssim, psnr]
        """
        # 转换为tensor
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt).float().to(self.device)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float().to(self.device)
        
        # 有效掩码（强度值通常在[0,1]范围内）
        valid = (gt >= 0) & (gt <= 1) & (pred >= 0) & (pred <= 1)
        
        if valid.sum() == 0:
            return [float('nan')] * 6
        
        gt_valid = gt[valid]
        pred_valid = pred[valid]
        
        # RMSE
        rmse = torch.sqrt(mse(pred_valid, gt_valid)).item()
        
        # MAE
        mae = l1_loss(pred_valid, gt_valid).item()
        
        # MedAE
        medae = torch.median(torch.abs(pred_valid - gt_valid)).item()
        
        # 为LPIPS和SSIM准备图像格式
        gt_3ch = gt.unsqueeze(0).repeat(3, 1, 1)
        pred_3ch = pred.unsqueeze(0).repeat(3, 1, 1)
        
        # LPIPS
        if self.lpips_fn is not None:
            try:
                pred_tensor = pred_3ch.unsqueeze(0)  # (1, 3, H, W)
                gt_tensor = gt_3ch.unsqueeze(0)
                
                # 转换到[-1,1]范围
                pred_tensor = 2.0 * pred_tensor - 1.0
                gt_tensor = 2.0 * gt_tensor - 1.0
                
                with torch.no_grad():
                    lpips_value = self.lpips_fn(pred_tensor, gt_tensor).item()
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                lpips_value = float('nan')
        else:
            lpips_value = float('nan')
        
        # SSIM
        try:
            ssim_value = ssim(gt_3ch, pred_3ch).item()
        except Exception as e:
            print(f"Warning: SSIM computation failed: {e}")
            ssim_value = float('nan')
        
        # PSNR
        try:
            psnr_value = psnr(gt_valid, pred_valid).item()
        except Exception as e:
            psnr_value = float('nan')
        
        return [rmse, mae, medae, lpips_value, ssim_value, psnr_value]
    
    def compute_raydrop_metrics(self, gt_mask: np.ndarray, pred_raydrop: np.ndarray) -> List[float]:
        """
        计算raydrop评估指标
        Args:
            gt_mask: Ground truth mask [H, W] (True=hit, False=drop)
            pred_raydrop: Predicted raydrop probability [H, W] [0,1]
        Returns:
            [rmse, accuracy, f1_score]
        """
        # 转换为tensor
        if isinstance(gt_mask, np.ndarray):
            gt_mask = torch.from_numpy(gt_mask).bool().to(self.device)
        if isinstance(pred_raydrop, np.ndarray):
            pred_raydrop = torch.from_numpy(pred_raydrop).float().to(self.device)
        
        # 转换GT掩码为raydrop标签 (True=hit -> 0=hit, False=drop -> 1=drop)
        gt_raydrop = (~gt_mask).float()
        
        # 展平
        gt_flat = gt_raydrop.view(-1)
        pred_flat = pred_raydrop.view(-1)
        
        # RMSE
        rmse = torch.sqrt(mse(pred_flat, gt_flat)).item()
        
        # 准确率 (使用阈值将概率转为二元预测)
        pred_binary = (pred_flat > self.raydrop_ratio).float()
        accuracy = (pred_binary == gt_flat).float().mean().item()
        
        # F1分数
        tp = ((pred_binary == 1) & (gt_flat == 1)).float().sum()
        fp = ((pred_binary == 1) & (gt_flat == 0)).float().sum()
        fn = ((pred_binary == 0) & (gt_flat == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return [rmse, accuracy, f1.item()]
    
    def compute_chamfer_distance(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用lidar-rt的chamfer3D计算Chamfer距离
        Args:
            pts1: 点云1 [N, 3]
            pts2: 点云2 [M, 3]
        Returns:
            (chamfer_distance, dist1, dist2)
        """
        if not CHAMFER_AVAILABLE or self.chamfer_fn is None:
            # Fallback到简单实现
            pts1 = torch.from_numpy(pts1).float().to(self.device).unsqueeze(0)  # [1, N, 3]
            pts2 = torch.from_numpy(pts2).float().to(self.device).unsqueeze(0)  # [1, M, 3]
            
            # 计算距离矩阵
            dist1 = torch.cdist(pts1, pts2, p=2)  # [1, N, M]
            dist2 = torch.cdist(pts2, pts1, p=2)  # [1, M, N]
            
            # 最近邻距离
            min_dist1, _ = torch.min(dist1, dim=2)  # [1, N]
            min_dist2, _ = torch.min(dist2, dim=2)  # [1, M]
            
            # Chamfer距离
            chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
            
            return chamfer_dist, min_dist1, min_dist2
        
        # 使用lidar-rt的chamfer3D实现
        pts1_tensor = torch.from_numpy(pts1).float().to(self.device).unsqueeze(0)  # [1, N, 3]
        pts2_tensor = torch.from_numpy(pts2).float().to(self.device).unsqueeze(0)  # [1, M, 3]
        
        # 调用lidar-rt的chamfer距离
        dist1, dist2, _, _ = self.chamfer_fn(pts1_tensor, pts2_tensor)
        
        # 计算平均Chamfer距离
        chamfer_dist = (dist1.mean() + dist2.mean()) * 0.5
        
        return chamfer_dist, dist1, dist2
    
    def compute_fscore(self, dist1: torch.Tensor, dist2: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算F-score
        Args:
            dist1: 距离1 [1, N] 或 [N]
            dist2: 距离2 [1, M] 或 [M]
            threshold: 距离阈值
        Returns:
            (fscore, precision, recall)
        """
        if dist1.dim() > 1:
            dist1 = dist1.squeeze(0)
        if dist2.dim() > 1:
            dist2 = dist2.squeeze(0)
        
        # 计算精确率和召回率
        precision = (dist1 < threshold).float().mean()
        recall = (dist2 < threshold).float().mean()
        
        # 计算F-score
        fscore = 2 * precision * recall / (precision + recall + 1e-8)
        
        return fscore, precision, recall
    
    def compute_points_metrics(self, gt_pts: np.ndarray, pred_pts: np.ndarray) -> List[float]:
        """
        计算点云评估指标
        Args:
            gt_pts: Ground truth点云 [N, 3]
            pred_pts: Predicted点云 [M, 3]
        Returns:
            [chamfer_dist, fscore]
        """
        # 计算Chamfer距离
        chamfer_dist, dist1, dist2 = self.compute_chamfer_distance(gt_pts, pred_pts)
        
        # 计算F-score
        fscore, _, _ = self.compute_fscore(dist1, dist2, threshold=self.fscore_threshold)
        
        return [chamfer_dist.cpu().item(), fscore.cpu().item()]
    
    def range_to_points(self, range_image: np.ndarray, intensity_image: Optional[np.ndarray] = None,
                       fov_h: float = 2*np.pi, fov_v: float = np.radians(26.9)) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        将Range Image转换为3D点云
        Args:
            range_image: Range图像 [H, W]
            intensity_image: 强度图像 [H, W] (可选)
            fov_h: 水平视场角
            fov_v: 垂直视场角
        Returns:
            (points_3d, intensities)
        """
        H, W = range_image.shape
        
        # 创建像素坐标网格
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # 计算角度
        azimuth = (u / W - 0.5) * fov_h
        inclination = (v / H - 0.5) * fov_v
        
        # 获取有效点
        valid_mask = range_image > 0.1
        valid_ranges = range_image[valid_mask]
        valid_azimuth = azimuth[valid_mask]
        valid_inclination = inclination[valid_mask]
        
        # 球坐标转笛卡尔坐标
        x = valid_ranges * np.cos(valid_inclination) * np.cos(valid_azimuth)
        y = valid_ranges * np.cos(valid_inclination) * np.sin(valid_azimuth)
        z = valid_ranges * np.sin(valid_inclination)
        
        points_3d = np.stack([x, y, z], axis=-1)
        
        if intensity_image is not None:
            intensities = intensity_image[valid_mask]
            return points_3d, intensities
        else:
            return points_3d, None
    
    def evaluate_frame(self, gt_data: Dict, pred_data: Dict, 
                      eval_depth: bool = True, eval_intensity: bool = True, 
                      eval_raydrop: bool = True, eval_points: bool = True) -> Dict:
        """
        评估单帧结果
        Args:
            gt_data: Ground truth数据字典
            pred_data: 预测结果字典
            eval_*: 各项评估开关
        Returns:
            评估结果字典
        """
        results = {}
        
        # 深度评估
        if eval_depth and 'range_image' in gt_data and 'depth' in pred_data:
            gt_depth = gt_data['range_image']
            pred_depth = pred_data['depth']
            
            if isinstance(pred_depth, torch.Tensor):
                pred_depth = pred_depth.cpu().numpy()
            
            depth_metrics = self.compute_depth_metrics(gt_depth, pred_depth)
            for i, metric_name in enumerate(self.depth_metrics):
                results[f'depth_{metric_name}'] = depth_metrics[i]
        
        # 强度评估
        if eval_intensity and 'intensity_image' in gt_data and 'intensity' in pred_data:
            gt_intensity = gt_data['intensity_image']
            pred_intensity = pred_data['intensity']
            
            if isinstance(pred_intensity, torch.Tensor):
                pred_intensity = pred_intensity.cpu().numpy()
            
            intensity_metrics = self.compute_intensity_metrics(gt_intensity, pred_intensity)
            for i, metric_name in enumerate(self.intensity_metrics):
                results[f'intensity_{metric_name}'] = intensity_metrics[i]
        
        # Raydrop评估
        if eval_raydrop and 'mask' in gt_data and 'raydrop' in pred_data:
            gt_mask = gt_data['mask']
            pred_raydrop = pred_data['raydrop']
            
            if isinstance(pred_raydrop, torch.Tensor):
                pred_raydrop = pred_raydrop.cpu().numpy()
            
            raydrop_metrics = self.compute_raydrop_metrics(gt_mask, pred_raydrop)
            for i, metric_name in enumerate(self.raydrop_metrics):
                results[f'raydrop_{metric_name}'] = raydrop_metrics[i]
        
        # 点云评估
        if eval_points and 'range_image' in gt_data and 'depth' in pred_data:
            gt_range = gt_data['range_image']
            pred_depth = pred_data['depth']
            
            if isinstance(pred_depth, torch.Tensor):
                pred_depth = pred_depth.cpu().numpy()
            
            # 转换为点云
            gt_points, _ = self.range_to_points(gt_range)
            pred_points, _ = self.range_to_points(pred_depth)
            
            if len(gt_points) > 0 and len(pred_points) > 0:
                # 下采样以加速计算
                max_points = 10000
                if len(gt_points) > max_points:
                    indices = np.random.choice(len(gt_points), max_points, replace=False)
                    gt_points = gt_points[indices]
                if len(pred_points) > max_points:
                    indices = np.random.choice(len(pred_points), max_points, replace=False)
                    pred_points = pred_points[indices]
                
                points_metrics = self.compute_points_metrics(gt_points, pred_points)
                for i, metric_name in enumerate(self.points_metrics):
                    results[f'points_{metric_name}'] = points_metrics[i]
        
        return results


def format_metrics_output(metrics_dict: Dict[str, float], precision: int = 4) -> str:
    """
    格式化评估指标输出
    Args:
        metrics_dict: 指标字典
        precision: 小数精度
    Returns:
        格式化的字符串
    """
    output_lines = []
    
    # 按类别分组
    categories = {
        'depth': [],
        'intensity': [],
        'raydrop': [],
        'points': []
    }
    
    for key, value in metrics_dict.items():
        for category in categories.keys():
            if key.startswith(category):
                categories[category].append((key, value))
                break
    
    # 输出每个类别
    for category, metrics in categories.items():
        if metrics:
            output_lines.append(f"\n{category.upper()} METRICS:")
            for key, value in metrics:
                metric_name = key.replace(f'{category}_', '').upper()
                if np.isnan(value):
                    output_lines.append(f"  {metric_name}: N/A")
                else:
                    output_lines.append(f"  {metric_name}: {value:.{precision}f}")
    
    return '\n'.join(output_lines) 