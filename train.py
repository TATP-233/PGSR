#!/usr/bin/env python3
"""
LiDAR-PGSR 统一训练脚本
支持配置文件模式和传统命令行参数模式
基于PGSR的平面化高斯，使用LiDAR数据进行监督训练
"""

import os
import torch
import torchvision
import random
from random import randint
from datetime import datetime
import numpy as np
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.loss_utils import l1_loss, ssim
from utils.lidar_loss_utils import compute_lidar_loss
from utils.pgsr_loss_utils import compute_pgsr_geometric_loss
from gaussian_renderer.lidar_renderer import render_lidar
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from utils.config_utils import load_config, save_config, override_config
from utils.evaluation_utils import LiDARPGSREvaluator, format_metrics_output
import sys

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_config_mode(config):
    """基于配置文件的训练模式"""
    print("Optimizing:", config.exp_name)
    print("Scene:", getattr(config, 'scene_id', 'unknown'))  
    print("Iterations:", config.opt.iterations)
    
    # 初始化高斯模型
    gaussians = GaussianModel(config.model.sh_degree)
    
    # 准备输出目录和日志
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from = prepare_output_and_logger_config(config)
    
    # 读取数据集
    scene = Scene(dataset, gaussians, shuffle=False)
    
    # 设置训练优化器
    gaussians.training_setup(opt)
    
    # 获取相机数量，在三相机模式下每帧有3个相机
    train_cameras = scene.getTrainCameras()
    total_cameras = len(train_cameras)
    frames_count = total_cameras // 3 if hasattr(config, 'cam_split_mode') and config.cam_split_mode == "triple" else total_cameras
    print(f"Training with {total_cameras} cameras ({frames_count} frames)")
    
    # 创建Tensorboard日志 - 使用统一的writer
    disable_tb = getattr(config, 'disable_tensorboard', False)
    if not disable_tb:
        if TENSORBOARD_FOUND:
            tensorboard_path = os.path.join(config.model_path, "tensorboard")
            writer = SummaryWriter(log_dir=tensorboard_path)
            print(f"Tensorboard logging to: {tensorboard_path}")
        else:
            writer = None
            print("Tensorboard not available: not logging progress")
    else:
        writer = None
        print("Tensorboard disabled by config")

    # 如果从检查点恢复
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, config)

    # 设置背景颜色：LiDAR三通道 [intensity, rayhit_logits, raydrop_logits]
    # 参考lidar-rt：背景设为 [0强度, 0命中逻辑值, 1丢失逻辑值]
    if dataset.white_background:
        bg_color = [1, 1, 1]  # 白色背景模式（少用）
    else:
        bg_color = [0, 0, 1]  # LiDAR模式：无强度、无命中、高丢失概率
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置迭代器
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # === 密集化统计记录（参考lidar-rt）===
    densification_log = {
        "iteration": [],
        "total_points": [],
        "points_change": [],
        "clone_count": [],
        "split_count": [],
        "prune_scale_count": [],
        "prune_opacity_count": []
    }
    
    # === 初始化评估器（参考lidar-rt评估指标）===
    evaluator = LiDARPGSREvaluator()
    eval_interval = getattr(config.opt, 'eval_interval', 500)  # 默认每500次迭代评估一次
    eval_results_log = []
    
    # 为多相机模式准备相机循环
    progress_bar = tqdm(range(first_iter, config.opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 初始化训练日志系统
    training_log = {
        "config": {
            "data_type": getattr(config, 'data_type', 'range_image'),
            "cam_split_mode": getattr(config.model, 'cam_split_mode', 'single'),
            "iterations": opt.iterations,
            "learning_rates": {
                "position_lr_init": opt.position_lr_init,
                "position_lr_final": opt.position_lr_final, 
                "feature_lr": opt.feature_lr,
                "opacity_lr": opt.opacity_lr,
                "scaling_lr": opt.scaling_lr,
                "rotation_lr": opt.rotation_lr
            }
        },
        "loss_weights": {},
        "iterations": [],
        "losses": [],
        "metrics": [],
        "densification": [],
        "evaluation": [],
        "errors": []
    }
    
    debug_log = {
        "start_time": str(datetime.now()),
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
        },
        "iterations_detail": []
    }
    
    for iteration in range(first_iter, config.opt.iterations + 1):
        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH的度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 选择训练相机
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 对多相机模式的特殊处理
        if (hasattr(viewpoint_camera, 'horizontal_fov_start') and 
            hasattr(viewpoint_camera, 'horizontal_fov_end') and
            viewpoint_camera.horizontal_fov_start is not None and
            viewpoint_camera.horizontal_fov_end is not None):
            fov_info = f"[FOV: {viewpoint_camera.horizontal_fov_start:.0f}°-{viewpoint_camera.horizontal_fov_end:.0f}°]"
        else:
            fov_info = "[Full 360°]"
            
        if iteration % 100 == 1:
            print(f"Iteration {iteration}: Training with camera {viewpoint_camera.image_name} {fov_info}")

        # 渲染 - 使用新的三通道输出格式
        use_rayhit = getattr(config.opt, 'use_rayhit', False)  # 可配置是否使用rayhit+raydrop的softmax模式
        render_pkg = render_lidar(viewpoint_camera, gaussians, pipe, background, use_rayhit=use_rayhit)
        rendered_depth, rendered_intensity, rendered_raydrop, viewspace_point_tensor = render_pkg["depth"], render_pkg["intensity"], render_pkg["raydrop"], render_pkg["viewspace_points"]
        
        # 提取新的三通道数据
        rendered_rayhit_logits = render_pkg.get("rayhit_logits", None)
        rendered_raydrop_logits = render_pkg.get("raydrop_logits", None)
        
        # 获取渲染中的其他变量
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
        update_filter = visibility_filter

        # LiDAR损失计算
        # 构建渲染包 - 包含新的三通道数据
        render_pkg_loss = {
            "depth": rendered_depth,
            "intensity": rendered_intensity,
            "raydrop": rendered_raydrop,
            "rayhit_logits": rendered_rayhit_logits,
            "raydrop_logits": rendered_raydrop_logits,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "viewspace_points": viewspace_point_tensor,
            # 添加PGSR几何约束所需的法向量信息
            "rendered_normal": render_pkg.get("rendered_normal"),
            "rendered_alpha": render_pkg.get("rendered_alpha"), 
            "rendered_distance": render_pkg.get("rendered_distance"),
            "depth_normal": render_pkg.get("depth_normal")
        }
        
        # 获取真实LiDAR数据（参考lidar-rt数据格式）
        gt_data = {}
        
        # 构建GT数据字典
        if hasattr(viewpoint_camera, 'range_image') and viewpoint_camera.range_image is not None:
            gt_data["range_image"] = viewpoint_camera.range_image
            if iteration <= 5 or iteration % 1000 == 0:
                print(f"[Iter {iteration}] GT range_image loaded: {gt_data['range_image'].shape}, min={gt_data['range_image'].min():.3f}, max={gt_data['range_image'].max():.3f}")
        else:
            if iteration <= 10:
                print(f"[Iter {iteration}] Warning: viewpoint_camera.range_image is None or missing!")
                if hasattr(viewpoint_camera, 'is_lidar_camera'):
                    print(f"  -> is_lidar_camera: {viewpoint_camera.is_lidar_camera}")
                if hasattr(viewpoint_camera, 'lidar_data'):
                    print(f"  -> lidar_data keys: {list(viewpoint_camera.lidar_data.keys()) if viewpoint_camera.lidar_data else 'None'}")
        
        if hasattr(viewpoint_camera, 'intensity_image') and viewpoint_camera.intensity_image is not None:
            gt_data["intensity_image"] = viewpoint_camera.intensity_image
            if iteration <= 5 or iteration % 1000 == 0:
                print(f"[Iter {iteration}] GT intensity_image loaded: {gt_data['intensity_image'].shape}")
        else:
            if iteration <= 10:
                print(f"[Iter {iteration}] Warning: viewpoint_camera.intensity_image is None or missing!")
        
        # 生成或获取有效像素掩码
        if hasattr(viewpoint_camera, 'mask') and viewpoint_camera.mask is not None:
            gt_data["mask"] = viewpoint_camera.mask
        elif "range_image" in gt_data:
            # 从range_image自动生成掩码：深度>0的像素为有效
            gt_data["mask"] = (gt_data["range_image"] > 0).float()
            if iteration <= 5 or iteration % 1000 == 0:
                valid_pixels = gt_data["mask"].sum().item()
                total_pixels = gt_data["mask"].numel()
                print(f"[Iter {iteration}] Generated mask: {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")
        
        # 检查是否有足够的GT数据进行训练
        if not gt_data or "range_image" not in gt_data:
            # 如果没有LiDAR数据，跳过此次迭代（避免虚拟数据）
            if iteration <= 10:
                print(f"[Iter {iteration}] Warning: No valid LiDAR data found, skipping iteration")
                print(f"  -> Camera: {viewpoint_camera.image_name}")
                print(f"  -> GT data keys: {list(gt_data.keys())}")
            continue
        
        # 设置损失权重（参考lidar-rt的命名约定，优化权重配置）
        loss_weights = {
            'depth_l1': getattr(config.opt, 'lambda_depth_l1', 3.0),          # 增加深度损失权重，提升几何精度
            'intensity_l1': getattr(config.opt, 'lambda_intensity_l1', 1.0),  # 增加强度损失权重
            'intensity_l2': getattr(config.opt, 'lambda_intensity_l2', 0.2),  # 增加L2损失
            'intensity_dssim': getattr(config.opt, 'lambda_intensity_dssim', 0.1),  # 增加结构相似性损失
            'raydrop_bce': getattr(config.opt, 'lambda_raydrop_bce', 0.2),    # 增加raydrop分类损失
            'raydrop_logits': getattr(config.opt, 'lambda_raydrop_logits', 0.1),
            'rayhit_raydrop': getattr(config.opt, 'lambda_rayhit_raydrop', 0.1),
            'cd': getattr(config.opt, 'lambda_cd', 0.05),                     # 增加Chamfer距离权重，改善点云质量
            'normal': getattr(config.opt, 'lambda_normal', 0.1)               # 增加法向量一致性权重
        }
        
        # 记录损失权重（只在第一次迭代时记录）
        if iteration == first_iter:
            training_log["loss_weights"] = loss_weights
        
        loss_dict, total_lidar_loss = compute_lidar_loss(render_pkg_loss, gt_data, loss_weights, use_rayhit=use_rayhit)
        
        # 调试信息：验证深度损失是否被计算
        if iteration <= 10 or iteration % 500 == 0:
            if 'depth_l1' in loss_dict:
                print(f"[Iter {iteration}] Depth L1 loss: {loss_dict['depth_l1']:.6f}")
            else:
                print(f"[Iter {iteration}] Warning: depth_l1 loss not computed!")
                print(f"Available losses: {list(loss_dict.keys())}")
                if "range_image" not in gt_data:
                    print("  -> Missing range_image in gt_data")
                if "mask" not in gt_data:
                    print("  -> Missing mask in gt_data")
                if rendered_depth is None:
                    print("  -> rendered_depth is None")
                    
                # 记录错误到debug日志
                debug_log["iterations_detail"].append({
                    "iteration": iteration,
                    "error": "depth_l1 loss not computed",
                    "available_losses": list(loss_dict.keys()),
                    "gt_data_keys": list(gt_data.keys()),
                    "rendered_depth_is_none": rendered_depth is None
                })
        
        # PGSR几何约束损失
        # === 使用LiDAR intensity图像进行几何正则化 ===
        # 原理：LiDAR intensity包含真实边缘信息，适合PGSR单视图几何约束的边缘感知计算
        # intensity_image格式：KITTI-360数据集中为2D张量(H, W)，PGSR已支持单通道图像处理
        gt_image = viewpoint_camera.intensity_image  # 保持原始(H, W)格式
        
        # PGSR几何约束损失 - 增加权重改善几何质量
        pgsr_loss_dict, pgsr_total_loss = compute_pgsr_geometric_loss(
            render_pkg, gt_image, viewpoint_camera,
            getattr(config.opt, 'lambda_sv_geom', 0.05),   # 增加单视图几何约束权重
            0.0, 0.0  # 在LiDAR模式下关闭多视图损失
        )
        
        # 合并损失
        total_loss = total_lidar_loss + pgsr_total_loss
        loss_dict.update(pgsr_loss_dict)

        # 反向传播计算梯度
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 进度报告
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == config.opt.iterations:
                progress_bar.close()
            
            # 记录详细的训练信息到日志
            if iteration % 100 == 0 or iteration <= 10:  # 前10次迭代每次记录，之后每100次记录
                # 计算梯度统计
                grad_norms = {}
                
                # 直接获取GaussianModel的参数组
                for group in gaussians.optimizer.param_groups:
                    param_name = group["name"]
                    param = group["params"][0]
                    if param.grad is not None:
                        grad_norms[param_name] = {
                            "norm": float(torch.norm(param.grad).item()),
                            "mean": float(param.grad.mean().item()),
                            "std": float(param.grad.std().item())
                        }
                
                # 记录当前迭代信息
                iter_log = {
                    "iteration": iteration,
                    "timestamp": str(datetime.now()),
                    "total_loss": float(total_loss.item()),
                    "lidar_loss": float(total_lidar_loss.item()),
                    "pgsr_loss": float(pgsr_total_loss.item()),
                    "ema_loss": float(ema_loss_for_log),
                    "losses": {k: float(v.item()) if torch.is_tensor(v) else float(v) for k, v in loss_dict.items()},
                    "visible_points": int(visibility_filter.sum().item()) if visibility_filter is not None else 0,
                    "total_points": int(gaussians.get_xyz.shape[0]),
                    "gradient_norms": grad_norms,
                    "camera_info": {
                        "image_name": viewpoint_camera.image_name,
                        "fov_info": fov_info if 'fov_info' in locals() else "N/A"
                    }
                }
                
                # 添加尺度统计
                scales = gaussians.get_scaling
                iter_log["scale_stats"] = {
                    "min": float(scales.min().item()),
                    "max": float(scales.max().item()),
                    "mean": float(scales.mean().item()),
                    "std": float(scales.std().item())
                }
                
                # 添加不透明度统计
                opacities = gaussians.get_opacity
                iter_log["opacity_stats"] = {
                    "min": float(opacities.min().item()),
                    "max": float(opacities.max().item()),
                    "mean": float(opacities.mean().item()),
                    "std": float(opacities.std().item())
                }
                
                training_log["iterations"].append(iter_log)

            # 记录
            training_report_config(writer, iteration, loss_dict, total_loss, iter_start.elapsed_time(iter_end), config, scene, render_lidar, pipe, background, densification_log)
            
            # === 定期评估（参考lidar-rt）===
            if iteration % eval_interval == 0 and iteration > 0:
                print(f"\n[ITER {iteration}] Running LiDAR evaluation...")
                torch.cuda.empty_cache()  # 清理GPU内存
                
                # 选择测试相机进行评估
                test_cameras = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else train_cameras[:min(10, len(train_cameras))]
                frame_results = []
                
                # 切换到评估模式（GaussianModel没有eval方法）
                
                for eval_idx, eval_camera in enumerate(test_cameras[:5]):  # 限制评估相机数量以节省时间
                    if not hasattr(eval_camera, 'range_image') or eval_camera.range_image is None:
                        continue
                        
                    try:
                        with torch.no_grad():
                            # 渲染评估数据
                            eval_render = render_lidar(eval_camera, gaussians, pipe, background, use_rayhit=use_rayhit)
                            
                            # 构建渲染数据字典，确保维度正确
                            def fix_dimensions(tensor_data):
                                """修正tensor维度为[H, W]或[H, W, 1]"""
                                if len(tensor_data.shape) == 3:
                                    if tensor_data.shape[0] == 1:  # [1, H, W] -> [H, W]
                                        return tensor_data.squeeze(0)
                                    elif tensor_data.shape[2] == 1:  # [H, W, 1] -> [H, W]
                                        return tensor_data.squeeze(2)
                                return tensor_data
                            
                            rendered_data = {
                                'depth': fix_dimensions(eval_render["depth"]).cpu().numpy(),
                                'intensity': fix_dimensions(eval_render["intensity"]).cpu().numpy(), 
                                'raydrop': fix_dimensions(eval_render["raydrop"]).cpu().numpy()
                            }
                            
                            # 构建GT数据字典
                            gt_depth = eval_camera.range_image.cpu().numpy() if hasattr(eval_camera, 'range_image') else None
                            gt_intensity = eval_camera.intensity_image.cpu().numpy() if hasattr(eval_camera, 'intensity_image') and eval_camera.intensity_image is not None else None
                            
                            # 确保GT数据维度正确
                            if gt_depth is not None:
                                if len(gt_depth.shape) == 3:
                                    if gt_depth.shape[0] == 1:
                                        gt_depth = gt_depth.squeeze(0)
                                    elif gt_depth.shape[2] == 1:
                                        gt_depth = gt_depth.squeeze(2)
                            
                            if gt_intensity is not None:
                                if len(gt_intensity.shape) == 3:
                                    if gt_intensity.shape[0] == 1:
                                        gt_intensity = gt_intensity.squeeze(0)
                                    elif gt_intensity.shape[2] == 1:
                                        gt_intensity = gt_intensity.squeeze(2)
                            
                            gt_data = {
                                'depth': gt_depth,
                                'intensity': gt_intensity
                            }
                            
                            # 生成掩码
                            if hasattr(eval_camera, 'mask') and eval_camera.mask is not None:
                                gt_data['rayhit_mask'] = eval_camera.mask.cpu().numpy()
                            else:
                                gt_data['rayhit_mask'] = (gt_data['depth'] > 0).astype('float32')
                            
                            # 评估单帧
                            frame_result = evaluator.evaluate_frame(rendered_data, gt_data)
                            if frame_result:
                                frame_results.append(frame_result)
                                
                    except Exception as e:
                        print(f"  Warning: Evaluation failed for camera {eval_camera.image_name}: {str(e)}")
                        continue
                
                # 恢复训练模式（GaussianModel没有train方法）
                
                # 聚合评估结果
                if frame_results:
                    aggregated_results = evaluator.aggregate_results(frame_results)
                    
                    # 输出评估结果
                    print(format_metrics_output(aggregated_results, f"Evaluation Results @ Iter {iteration}"))
                    
                    # 记录到Tensorboard
                    if writer:
                        for data_type, metrics in aggregated_results.items():
                            for metric_name, metric_stats in metrics.items():
                                writer.add_scalar(f'eval_{data_type}/{metric_name}_mean', metric_stats['mean'], iteration)
                                writer.add_scalar(f'eval_{data_type}/{metric_name}_std', metric_stats['std'], iteration)
                    
                    # 保存评估结果到日志
                    eval_log_entry = {
                        'iteration': iteration,
                        'results': aggregated_results,
                        'num_frames': len(frame_results)
                    }
                    eval_results_log.append(eval_log_entry)
                    
                    # 提取关键指标进行简化显示
                    key_metrics = {}
                    if 'depth' in aggregated_results:
                        key_metrics['depth_rmse'] = aggregated_results['depth']['rmse']['mean']
                        key_metrics['depth_psnr'] = aggregated_results['depth']['psnr']['mean']
                    if 'intensity' in aggregated_results:
                        key_metrics['intensity_psnr'] = aggregated_results['intensity']['psnr']['mean']
                        key_metrics['intensity_ssim'] = aggregated_results['intensity']['ssim']['mean']
                    if 'raydrop' in aggregated_results:
                        key_metrics['raydrop_f1'] = aggregated_results['raydrop']['f1']['mean']
                    
                    print(f"[ITER {iteration}] Key metrics: " + 
                          " | ".join([f"{k}: {v:.4f}" for k, v in key_metrics.items()]))
                    
                    # 在重要的评估节点保存PLY模型
                    if iteration % (eval_interval * 4) == 0 or iteration in [5000, 10000, 15000]:  # 每2000次迭代或特定节点
                        print(f"[ITER {iteration}] Saving PLY model...")
                        ply_save_path = os.path.join(config.model_path, f"point_cloud_iter_{iteration}.ply")
                        gaussians.save_ply(ply_save_path)
                        print(f"PLY model saved to: {ply_save_path}")
                else:
                    print(f"[ITER {iteration}] No valid evaluation frames")
                
                torch.cuda.empty_cache()  # 再次清理GPU内存
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密度自适应（参考lidar-rt的densification策略）
            if iteration <= opt.densify_until_iter:
                # 统计可见点信息（仅在早期迭代打印调试信息）
                visible_count = visibility_filter.sum().item()
                if visible_count > 0:
                    gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)
                elif iteration <= 5:
                    # print(f"[Iter {iteration}] No visible points for densification stats")
                    pass

                # 密集化和修剪操作（参考lidar-rt的密集化策略）
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    points_before = gaussians.get_xyz.shape[0]
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 执行密集化和修剪操作
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 
                        0.01, 
                        opt.min_opacity, 
                        scene.cameras_extent, 
                        size_threshold
                    )
                    
                    points_after = gaussians.get_xyz.shape[0]
                    
                    # 记录密集化统计信息
                    densification_log["iteration"].append(iteration)
                    densification_log["total_points"].append(points_after)
                    densification_log["points_change"].append(points_after - points_before)
                    # 添加详细的操作统计（简化版本）
                    densification_log["clone_count"].append(max(0, points_after - points_before) if points_after > points_before else 0)
                    densification_log["split_count"].append(0)  # 需要从gaussian_model中获取具体数据
                    densification_log["prune_scale_count"].append(0)  # 需要从gaussian_model中获取具体数据
                    densification_log["prune_opacity_count"].append(max(0, points_before - points_after) if points_before > points_after else 0)
                    
                    # 定期打印密集化信息
                    if iteration % (opt.densification_interval * 10) == 0:
                        print(f"[Iter {iteration}] Densification: {points_before} -> {points_after} points "
                              f"({points_after - points_before:+d})")
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < config.opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # === 训练结束后保存密集化统计日志（参考lidar-rt）===
    if densification_log and len(densification_log["iteration"]) > 0:
        import json
        log_path = os.path.join(config.model_path, "densification_log.json")
        with open(log_path, 'w') as f:
            json.dump(densification_log, f, indent=4)
        print(f"Densification log saved to: {log_path}")
        
        # 打印最终统计
        if densification_log["total_points"]:
            initial_points = gaussians.get_xyz.shape[0] if len(densification_log["total_points"]) == 0 else densification_log["total_points"][0]
            final_points = densification_log["total_points"][-1]
            total_change = sum(densification_log["points_change"])
            
            print(f"\n=== Final Densification Statistics ===")
            print(f"Points: {initial_points} -> {final_points} (total change: {total_change:+d})")
            print(f"Densification operations: {len(densification_log['iteration'])}")
    
    # === 保存详细的训练和调试日志 ===
    import json
    import numpy as np
    
    # 递归转换numpy类型为Python原生类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # 完善训练日志的最终信息
    training_log["end_time"] = str(datetime.now())
    training_log["final_stats"] = {
        "total_iterations": len(training_log["iterations"]),
        "final_loss": training_log["iterations"][-1]["total_loss"] if training_log["iterations"] else 0,
        "final_points": int(gaussians.get_xyz.shape[0]),
        "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    }
    
    # 完善debug日志的最终信息
    debug_log["end_time"] = str(datetime.now())
    debug_log["final_model_stats"] = {
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "intensity_sh_shape": list(gaussians._intensity_sh.shape) if hasattr(gaussians, '_intensity_sh') else None,
        "raydrop_sh_shape": list(gaussians._raydrop_sh.shape) if hasattr(gaussians, '_raydrop_sh') else None,
        "scale_range": [float(gaussians.get_scaling.min()), float(gaussians.get_scaling.max())],
        "opacity_range": [float(gaussians.get_opacity.min()), float(gaussians.get_opacity.max())]
    }
    
    # 转换并保存训练日志
    converted_training_log = convert_numpy_types(training_log)
    training_log_path = os.path.join(config.model_path, "training_log.json")
    with open(training_log_path, 'w') as f:
        json.dump(converted_training_log, f, indent=2)
    print(f"Training log saved to: {training_log_path}")
    
    # 转换并保存debug日志
    converted_debug_log = convert_numpy_types(debug_log)
    debug_log_path = os.path.join(config.model_path, "debug_log.json")
    with open(debug_log_path, 'w') as f:
        json.dump(converted_debug_log, f, indent=2)
    print(f"Debug log saved to: {debug_log_path}")
    
    # === 保存评估结果日志（参考lidar-rt）===
    if eval_results_log:
        import json
        import numpy as np
        
        # 递归转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 转换评估日志
        converted_log = convert_numpy_types(eval_results_log)
        
        eval_log_path = os.path.join(config.model_path, "evaluation_log.json")
        with open(eval_log_path, 'w') as f:
            json.dump(converted_log, f, indent=4)
        print(f"Evaluation log saved to: {eval_log_path}")
        
        # 输出最终评估结果摘要
        if eval_results_log:
            print(f"\n=== Final Evaluation Summary ===")
            print(f"Total evaluations: {len(eval_results_log)}")
            
            # 显示最后一次评估的关键指标
            last_eval = eval_results_log[-1]
            last_results = last_eval['results']
            print(f"Final evaluation @ iteration {last_eval['iteration']}:")
            
            final_metrics = {}
            if 'depth' in last_results:
                final_metrics.update({
                    'Depth RMSE': last_results['depth']['rmse']['mean'],
                    'Depth PSNR': last_results['depth']['psnr']['mean'],
                    'Depth SSIM': last_results['depth']['ssim']['mean']
                })
            if 'intensity' in last_results:
                final_metrics.update({
                    'Intensity PSNR': last_results['intensity']['psnr']['mean'],
                    'Intensity SSIM': last_results['intensity']['ssim']['mean']
                })
            if 'raydrop' in last_results:
                final_metrics.update({
                    'Raydrop F1': last_results['raydrop']['f1']['mean'],
                    'Raydrop Acc': last_results['raydrop']['acc']['mean']
                })
            
            for metric_name, metric_value in final_metrics.items():
                print(f"  {metric_name}: {metric_value:.6f}")
        
        if writer:
            writer.close()

def prepare_output_and_logger_config(config):
    """配置模式的输出目录和日志记录器准备"""
    if not hasattr(config, 'model_path') or not config.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            # 使用日期-时间格式：YYYYMMDD-HHMMSS
            unique_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        config.model_path = os.path.join(
            config.model_dir, 
            config.task_name, 
            config.exp_name, 
            unique_str
        )
        
    # 创建输出文件夹
    print(f"Output folder: {config.model_path}")
    os.makedirs(config.model_path, exist_ok=True)
    
    # 保存配置文件
    config_save_path = os.path.join(config.model_path, "config.yaml")
    save_config(config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # 注意：TensorBoard writer现在在training_config_mode中统一创建，这里不再重复创建
    
    # 从配置中提取参数到Namespace对象
    dataset = Namespace()
    dataset.source_path = getattr(config, 'source_dir', getattr(config.model, 'source_path', ''))
    dataset.model_path = config.model_path 
    dataset.images = getattr(config.data, 'images', 'images')
    dataset.resolution = getattr(config.data, 'resolution', -1)
    dataset.white_background = getattr(config.data, 'white_background', config.model.white_background)
    dataset.data_device = getattr(config.data, 'data_device', "cuda")
    dataset.eval = getattr(config.data, 'eval', False)
    dataset.sh_degree = getattr(config.model, 'sh_degree', 3)
    dataset.preload_img = getattr(config.model, 'preload_img', True)
    dataset.ncc_scale = getattr(config.model, 'ncc_scale', 1.0)
    
    dataset.multi_view_num = getattr(config.model, 'multi_view_num', 8)
    dataset.multi_view_max_angle = getattr(config.model, 'multi_view_max_angle', 30)
    dataset.multi_view_min_dis = getattr(config.model, 'multi_view_min_dis', 0.01)
    dataset.multi_view_max_dis = getattr(config.model, 'multi_view_max_dis', 1.5)
    
    dataset.frame_length = getattr(config.model, 'frame_length', [0, 100])
    dataset.seq = getattr(config.model, 'seq', "0000")
    dataset.data_type = getattr(config, 'data_type', "range_image")
    dataset.cam_split_mode = getattr(config.model, 'cam_split_mode', "single")
    
    opt = Namespace()
    opt.iterations = config.opt.iterations
    # 优化学习率配置以改善LiDAR训练收敛
    opt.position_lr_init = getattr(config.opt, 'position_lr_init', 0.0002)    # 适度提高初始位置学习率
    opt.position_lr_final = getattr(config.opt, 'position_lr_final', 0.000002) # 适度提高最终位置学习率
    opt.position_lr_delay_mult = getattr(config.opt, 'position_lr_delay_mult', 0.01)
    opt.position_lr_max_steps = getattr(config.opt, 'position_lr_max_steps', 30000)
    opt.feature_lr = getattr(config.opt, 'feature_lr', 0.003)     # 适度提高特征学习率，加速intensity/raydrop学习
    opt.opacity_lr = getattr(config.opt, 'opacity_lr', 0.05)
    opt.scaling_lr = getattr(config.opt, 'scaling_lr', 0.005)
    opt.rotation_lr = getattr(config.opt, 'rotation_lr', 0.001)
    opt.percent_dense = getattr(config.opt, 'percent_dense', 0.01)
    opt.lambda_dssim = getattr(config.opt, 'lambda_dssim', 0.2)
    opt.densification_interval = getattr(config.opt, 'densification_interval', 100)
    opt.opacity_reset_interval = getattr(config.opt, 'opacity_reset_interval', 3000)
    opt.densify_from_iter = getattr(config.opt, 'densify_from_iter', 500)
    opt.densify_until_iter = getattr(config.opt, 'densify_until_iter', 15000)
    opt.densify_grad_threshold = getattr(config.opt, 'densify_grad_threshold', 0.0002)
    opt.min_opacity = getattr(config.opt, 'min_opacity', 0.005)
    opt.random_background = getattr(config.opt, 'random_background', False)
    opt.abs_split_radii2D_threshold = getattr(config.opt, 'abs_split_radii2D_threshold', 20)
    opt.max_abs_split_points = getattr(config.opt, 'max_abs_split_points', 50000)
    opt.max_all_points = getattr(config.opt, 'max_all_points', 6000000)
    
    pipe = Namespace()
    pipe.convert_SHs_python = getattr(config.pipe, 'convert_SHs_python', False)
    pipe.compute_cov3D_python = getattr(config.pipe, 'compute_cov3D_python', False)
    pipe.debug = getattr(config.pipe, 'debug', False)
    
    testing_iterations = getattr(config, 'testing_iterations', [7_000, 30_000])
    saving_iterations = getattr(config, 'saving_iterations', [7_000, 30_000])
    checkpoint_iterations = getattr(config, 'checkpoint_iterations', [])
    checkpoint = getattr(config, 'start_checkpoint', None)
    debug_from = getattr(config.pipe, 'debug_from', -1)
    
    return dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from


def training_report_config(tb_writer, iteration, loss_dict, total_loss, elapsed, config, scene, renderFunc, pipe, bg, densification_log=None):
    """配置模式的训练报告和日志记录"""
    if tb_writer:
        # === 1. 损失记录 ===
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        
        # 分类记录不同类型的损失，便于在TensorBoard中查看
        lidar_losses = ['depth_l1', 'intensity_l1', 'intensity_l2', 'intensity_dssim', 'raydrop_bce', 'raydrop_logits', 'rayhit_raydrop', 'cd']
        pgsr_losses = ['sv_geom', 'mv_geom', 'planar']
        
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                loss_value = value.item()
            else:
                loss_value = value
            
            # 根据损失类型分类记录
            if key in lidar_losses:
                tb_writer.add_scalar(f'train_loss_lidar/{key}', loss_value, iteration)
            elif key in pgsr_losses:
                tb_writer.add_scalar(f'train_loss_pgsr/{key}', loss_value, iteration)
            else:
                tb_writer.add_scalar(f'train_loss_other/{key}', loss_value, iteration)
            
            # 同时记录到总损失组中（向后兼容）
            tb_writer.add_scalar(f'train_loss_patches/{key}', loss_value, iteration)
        
        # 每100次迭代输出损失调试信息
        if iteration % 100 == 0:
            loss_info = []
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    loss_info.append(f"{key}:{value.item():.6f}")
                else:
                    loss_info.append(f"{key}:{value:.6f}")
            print(f"[Iter {iteration}] Losses: {', '.join(loss_info)}")
        
        # === 2. 高斯基元统计 ===
        gaussians = scene.gaussians
        tb_writer.add_scalar('gaussian_stats/num_points', gaussians.get_xyz.shape[0], iteration)
        
        if gaussians.get_xyz.shape[0] > 0:
            # 位置统计
            xyz = gaussians.get_xyz
            tb_writer.add_scalar('gaussian_stats/xyz_std', torch.std(xyz).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/xyz_mean_x', torch.mean(xyz[:, 0]).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/xyz_mean_y', torch.mean(xyz[:, 1]).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/xyz_mean_z', torch.mean(xyz[:, 2]).item(), iteration)
            
            # 缩放统计  
            scaling = gaussians.get_scaling
            tb_writer.add_scalar('gaussian_stats/scaling_mean', torch.mean(scaling).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/scaling_std', torch.std(scaling).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/scaling_max', torch.max(scaling).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/scaling_min', torch.min(scaling).item(), iteration)
            
            # 不透明度统计
            opacity = gaussians.get_opacity
            tb_writer.add_scalar('gaussian_stats/opacity_mean', torch.mean(opacity).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/opacity_std', torch.std(opacity).item(), iteration)
            
            # PGSR平面化程度
            min_scales = scaling.min(dim=1)[0]
            tb_writer.add_scalar('gaussian_stats/planarity_mean', torch.mean(min_scales).item(), iteration)
            tb_writer.add_scalar('gaussian_stats/planarity_std', torch.std(min_scales).item(), iteration)
        
        # === 3. 学习率记录 ===
        for param_group in gaussians.optimizer.param_groups:
            tb_writer.add_scalar(f'learning_rates/{param_group["name"]}', param_group['lr'], iteration)
        
        # === 4. 性能指标 ===
        tb_writer.add_scalar('performance/iteration_time_ms', elapsed, iteration)
        if torch.cuda.is_available():
            tb_writer.add_scalar('performance/gpu_memory_gb', torch.cuda.max_memory_allocated() / 1e9, iteration)
        
        # === 5. 密集化统计（参考lidar-rt）===
        if densification_log and len(densification_log.get("iteration", [])) > 0:
            # 记录最新的密集化统计
            latest_stats = {
                "total_points": densification_log["total_points"][-1],
                "clone_count": sum(densification_log.get("clone_count", [])),
                "split_count": sum(densification_log.get("split_count", [])),
                "prune_scale_count": sum(densification_log.get("prune_scale_count", [])),
                "prune_opacity_count": sum(densification_log.get("prune_opacity_count", []))
            }
            
            tb_writer.add_scalar('densification_stats/total_points', latest_stats["total_points"], iteration)
            tb_writer.add_scalar('densification_stats/cumulative_clone', latest_stats["clone_count"], iteration)
            tb_writer.add_scalar('densification_stats/cumulative_split', latest_stats["split_count"], iteration)
            tb_writer.add_scalar('densification_stats/cumulative_prune_scale', latest_stats["prune_scale_count"], iteration)
            tb_writer.add_scalar('densification_stats/cumulative_prune_opacity', latest_stats["prune_opacity_count"], iteration)
            
            # 如果有新的密集化记录，记录单次操作的统计
            if densification_log["iteration"][-1] == iteration:
                tb_writer.add_scalar('densification_per_iter/clone_count', densification_log.get("clone_count", [0])[-1], iteration)
                tb_writer.add_scalar('densification_per_iter/split_count', densification_log.get("split_count", [0])[-1], iteration)
                tb_writer.add_scalar('densification_per_iter/prune_scale_count', densification_log.get("prune_scale_count", [0])[-1], iteration)
                tb_writer.add_scalar('densification_per_iter/prune_opacity_count', densification_log.get("prune_opacity_count", [0])[-1], iteration)
        
        # === 6. 每1000次迭代进行测试和可视化 ===
        if iteration % 1000 == 0 and len(scene.getTrainCameras()) > 0:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                                {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config_test in validation_configs:
                if config_test['cameras'] and len(config_test['cameras']) > 0:
                    render_results = []
                    # === 参考lidar-rt的评价指标计算 ===
                    total_depth_psnr = 0.0
                    total_intensity_psnr = 0.0
                    total_depth_mse = 0.0
                    total_chamfer_distance = 0.0
                    valid_eval_count = 0
                    
                    for idx, viewpoint in enumerate(config_test['cameras']):
                        if idx >= 5:  # 增加测试样本数量以获得更稳定的指标
                            break
                        
                        try:
                            render_result = renderFunc(viewpoint, scene.gaussians, pipe, bg)
                            
                            # LiDAR评价指标 - 参考lidar-rt实现
                            if hasattr(viewpoint, 'range_image') and render_result.get("depth") is not None:
                                gt_depth = viewpoint.range_image
                                pred_depth = render_result["depth"].squeeze()
                                
                                # 获取真实掩码
                                if hasattr(viewpoint, 'intensity_image'):
                                    gt_mask = (gt_depth > 0)  # 真实的有效像素掩码
                                else:
                                    gt_mask = torch.ones_like(gt_depth, dtype=torch.bool)
                                
                                # Raydrop概率掩码（类似lidar-rt中的处理）
                                if render_result.get("raydrop") is not None:
                                    raydrop_prob = render_result["raydrop"].squeeze()
                                    pred_mask = raydrop_prob < 0.5  # hit概率高的像素
                                else:
                                    pred_mask = (pred_depth > 0)
                                
                                # 组合掩码
                                combined_mask = gt_mask & pred_mask
                                
                                if combined_mask.sum() > 0:
                                    # === 深度PSNR（参考lidar-rt计算方法）===
                                    # 归一化到0-1范围，lidar-rt中除以80米
                                    max_depth = 80.0  # 参考lidar-rt的最大深度
                                    gt_depth_norm = (gt_depth * combined_mask) / max_depth
                                    pred_depth_norm = (pred_depth * combined_mask) / max_depth
                                    
                                    depth_mse = torch.mean((gt_depth_norm - pred_depth_norm) ** 2)
                                    if depth_mse > 0:
                                        depth_psnr = 20 * torch.log10(1.0 / torch.sqrt(depth_mse))
                                        total_depth_psnr += depth_psnr.item()
                                        total_depth_mse += depth_mse.item()
                                    
                                    # === 深度其他指标 ===
                                    # 深度RMSE（原始尺度）
                                    depth_rmse = torch.sqrt(torch.mean((gt_depth[combined_mask] - pred_depth[combined_mask]) ** 2))
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics_detail/depth_rmse', depth_rmse.item(), iteration)
                                    
                                    # 深度相对误差
                                    relative_error = torch.mean(torch.abs(gt_depth[combined_mask] - pred_depth[combined_mask]) / (gt_depth[combined_mask] + 1e-8))
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics_detail/depth_relative_error', relative_error.item(), iteration)
                                    
                                    # === 倒角距离（Chamfer Distance）- 参考lidar-rt ===
                                    try:
                                        # 将深度图反投影到3D点云（简化版本）
                                        H, W = gt_depth.shape
                                        # 创建像素坐标网格
                                        u, v = torch.meshgrid(torch.arange(W, device=gt_depth.device), 
                                                            torch.arange(H, device=gt_depth.device))
                                        u, v = u.float(), v.float()
                                        
                                        # 简化的反投影（假设单位相机内参）
                                        # 真实数据点
                                        valid_gt = combined_mask
                                        if valid_gt.sum() > 100:  # 确保有足够的点
                                            gt_x = (u[valid_gt] - W/2) * gt_depth[valid_gt] / (W/2)
                                            gt_y = (v[valid_gt] - H/2) * gt_depth[valid_gt] / (H/2)
                                            gt_z = gt_depth[valid_gt]
                                            gt_points = torch.stack([gt_x, gt_y, gt_z], dim=1)
                                            
                                            # 预测数据点
                                            pred_x = (u[valid_gt] - W/2) * pred_depth[valid_gt] / (W/2)
                                            pred_y = (v[valid_gt] - H/2) * pred_depth[valid_gt] / (H/2)
                                            pred_z = pred_depth[valid_gt]
                                            pred_points = torch.stack([pred_x, pred_y, pred_z], dim=1)
                                            
                                            # 简化的倒角距离计算
                                            if gt_points.shape[0] > 0 and pred_points.shape[0] > 0:
                                                # 子采样以加速计算
                                                max_points = 1000
                                                if gt_points.shape[0] > max_points:
                                                    indices = torch.randperm(gt_points.shape[0])[:max_points]
                                                    gt_points = gt_points[indices]
                                                    pred_points = pred_points[indices]
                                                
                                                # 计算最近邻距离
                                                dist_matrix = torch.cdist(pred_points, gt_points)
                                                dist1 = dist_matrix.min(dim=1)[0].mean()  # pred到gt的距离
                                                dist2 = dist_matrix.min(dim=0)[0].mean()  # gt到pred的距离
                                                chamfer_dist = (dist1 + dist2) / 2
                                                total_chamfer_distance += chamfer_dist.item()
                                                
                                    except Exception as e:
                                        print(f"Error computing chamfer distance: {e}")
                            
                            # === Intensity评价指标 - 参考lidar-rt ===
                            if hasattr(viewpoint, 'intensity_image') and render_result.get("intensity") is not None:
                                gt_intensity = viewpoint.intensity_image
                                pred_intensity = render_result["intensity"].squeeze()
                                
                                # 获取掩码
                                if hasattr(viewpoint, 'range_image'):
                                    gt_mask = (viewpoint.range_image > 0)
                                else:
                                    gt_mask = torch.ones_like(gt_intensity, dtype=torch.bool)
                                
                                if render_result.get("raydrop") is not None:
                                    raydrop_prob = render_result["raydrop"].squeeze()
                                    pred_mask = raydrop_prob < 0.5
                                else:
                                    pred_mask = torch.ones_like(pred_intensity, dtype=torch.bool)
                                
                                combined_mask = gt_mask & pred_mask
                                
                                if combined_mask.sum() > 0:
                                    # 强度值裁剪到[0,1]范围（参考lidar-rt）
                                    gt_intensity_clamped = torch.clamp(gt_intensity, 0, 1)
                                    pred_intensity_clamped = torch.clamp(pred_intensity, 0, 1)
                                    
                                    # === 强度PSNR（参考lidar-rt计算方法）===
                                    masked_gt = gt_intensity_clamped * combined_mask
                                    masked_pred = pred_intensity_clamped * combined_mask
                                    
                                    intensity_mse = torch.mean((masked_gt - masked_pred) ** 2)
                                    if intensity_mse > 0:
                                        intensity_psnr = 20 * torch.log10(1.0 / torch.sqrt(intensity_mse))
                                        total_intensity_psnr += intensity_psnr.item()
                                    
                                    # Intensity SSIM
                                    try:
                                        gt_3ch = masked_gt.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
                                        pred_3ch = masked_pred.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
                                        intensity_ssim = ssim(gt_3ch, pred_3ch)
                                        tb_writer.add_scalar(f'{config_test["name"]}_metrics_detail/intensity_ssim', intensity_ssim.item(), iteration)
                                    except:
                                        pass
                            
                            render_results.append(render_result)
                            valid_eval_count += 1
                            
                        except Exception as e:
                            print(f"Error during {config_test['name']} evaluation at iteration {iteration}: {e}")
                            continue
                    
                    # === 记录平均评价指标（参考lidar-rt的mix_metric概念）===
                    if valid_eval_count > 0:
                        avg_depth_psnr = total_depth_psnr / valid_eval_count
                        avg_intensity_psnr = total_intensity_psnr / valid_eval_count
                        avg_depth_mse = total_depth_mse / valid_eval_count
                        avg_chamfer_distance = total_chamfer_distance / valid_eval_count
                        
                        # 主要指标
                        tb_writer.add_scalar(f'{config_test["name"]}_metrics/depth_psnr', avg_depth_psnr, iteration)
                        tb_writer.add_scalar(f'{config_test["name"]}_metrics/intensity_psnr', avg_intensity_psnr, iteration)
                        tb_writer.add_scalar(f'{config_test["name"]}_metrics/depth_mse', avg_depth_mse, iteration)
                        tb_writer.add_scalar(f'{config_test["name"]}_metrics/chamfer_distance', avg_chamfer_distance, iteration)
                        
                        # === 混合指标（参考lidar-rt的mix_metric）===
                        mix_metric = avg_depth_psnr + avg_intensity_psnr
                        tb_writer.add_scalar(f'{config_test["name"]}_metrics/mix_metric', mix_metric, iteration)
                        
                        # 打印重要指标
                        if config_test["name"] == "test":
                            print(f"[Iter {iteration}] Test Metrics: Depth PSNR={avg_depth_psnr:.2f}, "
                                  f"Intensity PSNR={avg_intensity_psnr:.2f}, Mix={mix_metric:.2f}, "
                                  f"Chamfer={avg_chamfer_distance:.4f}")
                    
                    # 记录图像（每3000次迭代）
                    if iteration % 3000 == 0 and render_results:
                        try:
                            # 深度图可视化
                            if render_results[0].get("depth") is not None:
                                depth_img = render_results[0]["depth"].squeeze()
                                depth_img_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-8)
                                tb_writer.add_image(f'{config_test["name"]}_images/depth', depth_img_normalized.unsqueeze(0), iteration)
                            
                            # Intensity图可视化
                            if render_results[0].get("intensity") is not None:
                                intensity_img = render_results[0]["intensity"].squeeze()
                                intensity_img_normalized = (intensity_img - intensity_img.min()) / (intensity_img.max() - intensity_img.min() + 1e-8)
                                tb_writer.add_image(f'{config_test["name"]}_images/intensity', intensity_img_normalized.unsqueeze(0), iteration)
                            
                            # Raydrop可视化
                            if render_results[0].get("raydrop") is not None:
                                raydrop_img = render_results[0]["raydrop"].squeeze()
                                tb_writer.add_image(f'{config_test["name"]}_images/raydrop', raydrop_img.unsqueeze(0), iteration)
                                
                        except Exception as e:
                            print(f"Error saving images to tensorboard: {e}")
            
            torch.cuda.empty_cache()

def main():
    """主函数 - 支持配置文件和传统命令行两种模式"""
    parser = argparse.ArgumentParser(description="LiDAR-PGSR Training")
    
    # 添加配置文件选项
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file (config mode)")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    parser.add_argument("--exp_name", type=str, default=None, help="Override experiment name")
    parser.add_argument("--scene_id", type=str, default=None, help="Override scene ID")
    
    args = parser.parse_args()
    
    # 配置文件模式
    print(f"Running in CONFIG mode with config: {args.config}")
    config = load_config(args.config)
    
    # 命令行参数覆盖
    overrides = {}
    if args.start_checkpoint:
        overrides['start_checkpoint'] = args.start_checkpoint
    if args.scene_id:
        overrides['scene_id'] = args.scene_id
    if args.exp_name:
        overrides['exp_name'] = args.exp_name
    if args.debug_from >= 0:
        overrides['pipe.debug_from'] = args.debug_from
    if args.detect_anomaly:
        overrides['detect_anomaly'] = True
    
    if overrides:
        config = override_config(config, overrides)
    
    print(f"Optimizing: {config.exp_name}")
    print(f"Scene: {getattr(config, 'scene_id', 'default')}")
    print(f"Iterations: {config.opt.iterations}")

    # 初始化系统状态
    safe_state(args.quiet)

    # 启动配置模式训练
    torch.autograd.set_detect_anomaly(getattr(config, 'detect_anomaly', False))
    training_config_mode(config)
    
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
