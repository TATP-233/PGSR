#!/usr/bin/env python3
"""
LiDAR-PGSR配置化训练脚本
基于YAML配置文件的训练管理
"""

import os
import torch
import random
import sys
import uuid
import argparse
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.lidar_renderer import render_lidar
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.config_utils import load_config, save_config, override_config
from tqdm import tqdm
from utils.image_utils import psnr
from utils.lidar_loss_utils import compute_lidar_loss
from utils.pgsr_loss_utils import compute_pgsr_geometric_loss
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(config):
    """
    基于配置文件的LiDAR-PGSR训练主循环
    """
    first_iter = 0
    
    # 初始化输出目录和日志记录器
    tb_writer = prepare_output_and_logger(config)
    
    # 创建高斯模型和场景
    gaussians = GaussianModel(config.model.sh_degree, enable_lidar=config.model.enable_lidar)
    scene = Scene(config, gaussians)
    gaussians.training_setup(config)
    
    # 加载检查点（如果有）
    if hasattr(config, 'model_path') and config.model_path:
        (model_params, first_iter) = torch.load(config.model_path)
        gaussians.restore(model_params, config)

    # 设置背景
    bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 计时器
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 训练状态
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, config.opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, config.opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数阶数
        if iteration % config.opt.sh_increase_interval == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 渲染
        if hasattr(config.pipe, 'debug_from') and (iteration - 1) == config.pipe.debug_from:
            config.pipe.debug = True

        bg = torch.rand((3), device="cuda") if hasattr(config.opt, 'random_background') and config.opt.random_background else background

        render_pkg = render_lidar(viewpoint_cam, gaussians, config, bg)
        
        # 获取真实LiDAR数据
        gt_lidar_data = viewpoint_cam.get_lidar_data()
        if not gt_lidar_data:
            print(f"Warning: No LiDAR data for iteration {iteration}")
            continue

        # 计算损失
        loss_dict, total_loss = compute_lidar_loss(render_pkg, gt_lidar_data, config)

        # 添加PGSR的几何正则化损失
        if hasattr(config.model, 'geometric_regularization') and config.model.geometric_regularization:
            if iteration > config.opt.densify_from_iter:
                # 平面化损失
                if hasattr(config.model, 'planar_constraint') and config.model.planar_constraint:
                    planar_loss = gaussians.compute_planar_loss()
                    total_loss += config.opt.lambda_planar * planar_loss
                    loss_dict["planar"] = planar_loss
                
                # PGSR几何正则化损失（单视图+多视图）
                if hasattr(config.opt, 'lambda_sv_geom') and config.opt.lambda_sv_geom > 0:
                    gt_image = viewpoint_cam.original_image
                    pgsr_loss_dict, pgsr_total_loss = compute_pgsr_geometric_loss(
                        render_pkg, 
                        gt_image, 
                        viewpoint_cam,
                        lambda_sv_geom=config.opt.lambda_sv_geom,
                        lambda_mv_rgb=getattr(config.opt, 'lambda_mv_rgb', 0.0),
                        lambda_mv_geom=getattr(config.opt, 'lambda_mv_geom', 0.0)
                    )
                    
                    # 合并PGSR损失
                    total_loss += pgsr_total_loss
                    for key, value in pgsr_loss_dict.items():
                        loss_dict[f"pgsr_{key}"] = value

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 进度条更新
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Points": gaussians.get_xyz.shape[0],
                    "Exp": config.exp_name
                })
                progress_bar.update(10)
            if iteration == config.opt.iterations:
                progress_bar.close()

            # 记录和保存
            training_report(tb_writer, iteration, loss_dict, total_loss, 
                          iter_start.elapsed_time(iter_end), config, scene, render_lidar, background)
            
            # 保存检查点
            if iteration in config.saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # 密化操作
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
            
            if iteration < config.opt.densify_until_iter:
                # 记录最大半径用于修剪
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > config.opt.densify_from_iter and iteration % config.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > config.opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(config.opt.densify_grad_threshold, 
                                              config.opt.thresh_opa_prune, 
                                              config.opt.thresh_opa_prune, 
                                              scene.cameras_extent, 
                                              size_threshold)
                
                if iteration % config.opt.opacity_reset_interval == 0 or \
                   (config.model.white_background and iteration == config.opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < config.opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(config):
    """准备输出目录和日志记录器"""
    if not hasattr(config, 'model_path') or not config.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        
        config.model_path = os.path.join(
            config.model_dir, 
            config.task_name, 
            config.exp_name, 
            unique_str[0:10]
        )
        
    # 创建输出文件夹
    print(f"Output folder: {config.model_path}")
    os.makedirs(config.model_path, exist_ok=True)
    
    # 保存配置文件
    config_save_path = os.path.join(config.model_path, "config.yaml")
    save_config(config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # 创建Tensorboard记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(config.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    return tb_writer


def training_report(tb_writer, iteration, loss_dict, total_loss, elapsed, config, scene, renderFunc, bg):
    """训练报告和日志记录"""
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        for key, value in loss_dict.items():
            tb_writer.add_scalar(f'train_loss_patches/{key}', value.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 测试和验证
    if iteration % config.testing_iterations == 0:
        torch.cuda.empty_cache()
        
        validation_configs = [
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                        for idx in range(5, min(30, len(scene.getTrainCameras())), 5)]}
        ]

        for val_config in validation_configs:
            if val_config['cameras'] and len(val_config['cameras']) > 0:
                total_l1_test = 0.0
                total_depth_test = 0.0
                total_intensity_test = 0.0
                
                for idx, viewpoint in enumerate(val_config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, config, bg)
                    gt_lidar = viewpoint.get_lidar_data()
                    if gt_lidar:
                        test_loss_dict, _ = compute_lidar_loss(render_pkg, gt_lidar, config)
                        total_l1_test += test_loss_dict.get("depth", torch.tensor(0.0)).item()
                        total_depth_test += test_loss_dict.get("depth", torch.tensor(0.0)).item()
                        total_intensity_test += test_loss_dict.get("intensity", torch.tensor(0.0)).item()
                        
                l1_test = total_l1_test / len(val_config['cameras'])
                depth_test = total_depth_test / len(val_config['cameras'])
                intensity_test = total_intensity_test / len(val_config['cameras'])
                
                print(f"\n[ITER {iteration}] Evaluating {val_config['name']}: "
                      f"L1 {l1_test:.6f} Depth {depth_test:.6f} Intensity {intensity_test:.6f}")
                
                if tb_writer:
                    tb_writer.add_scalar(f"{val_config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{val_config['name']}/loss_viewpoint - depth_loss", depth_test, iteration)
                    tb_writer.add_scalar(f"{val_config['name']}/loss_viewpoint - intensity_loss", intensity_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LiDAR-PGSR Training with Config Files")
    parser.add_argument("-c", "--config", type=str, required=True, 
                       help="Path to the configuration file")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to a checkpoint file")
    parser.add_argument("--scene_id", type=str, default=None,
                       help="Override scene ID")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Override experiment name")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    parser.add_argument("--debug_from", type=int, default=-1,
                       help="Start debugging from iteration")
    parser.add_argument("--detect_anomaly", action="store_true",
                       help="Enable autograd anomaly detection")
    
    args = parser.parse_args()
    
    # 加载配置文件
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # 命令行参数覆盖
    overrides = {}
    if args.model_path:
        overrides['model_path'] = args.model_path
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

    # 启动训练
    torch.autograd.set_detect_anomaly(getattr(config, 'detect_anomaly', False))
    training(config)

    print("\nTraining complete.")


if __name__ == "__main__":
    main() 