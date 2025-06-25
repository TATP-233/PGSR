#!/usr/bin/env python3
"""
LiDAR-PGSR 统一训练脚本
支持配置文件模式和传统命令行参数模式
基于PGSR的平面化高斯，使用LiDAR数据进行监督训练
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
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time
from torch.utils.tensorboard import SummaryWriter

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
    scene = Scene(dataset, gaussians, config, shuffle=False)
    
    # 获取相机数量，在三相机模式下每帧有3个相机
    train_cameras = scene.getTrainCameras()
    total_cameras = len(train_cameras)
    frames_count = total_cameras // 3 if hasattr(config, 'cam_split_mode') and config.cam_split_mode == "triple" else total_cameras
    print(f"Training with {total_cameras} cameras ({frames_count} frames)")
    
    # 创建Tensorboard日志
    if not config.disable_tensorboard:
        writer = SummaryWriter(log_dir=f"runs/{config.model_path.split('/')[-1]}")
        print(f"Tensorboard logging to: runs/{config.model_path.split('/')[-1]}")
    else:
        writer = None

    # 如果从检查点恢复
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, config)

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置迭代器
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # 为多相机模式准备相机循环
    progress_bar = tqdm(range(first_iter, config.opt.iterations), desc="Training progress")
    first_iter += 1
    
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
        if hasattr(viewpoint_camera, 'horizontal_fov_start') and hasattr(viewpoint_camera, 'horizontal_fov_end'):
            fov_info = f"[FOV: {viewpoint_camera.horizontal_fov_start:.0f}°-{viewpoint_camera.horizontal_fov_end:.0f}°]"
        else:
            fov_info = "[Full 360°]"
            
        if iteration % 100 == 1:
            print(f"Iteration {iteration}: Training with camera {viewpoint_camera.image_name} {fov_info}")

        # 渲染
        render_pkg = render_lidar(viewpoint_camera, gaussians, pipe, background)
        rendered_depth, rendered_intensity, rendered_raydrop, viewspace_point_tensor = render_pkg["depth"], render_pkg["intensity"], render_pkg["raydrop"], render_pkg["viewspace_points"]
        
        # 获取渲染中的其他变量
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
        update_filter = visibility_filter

        # LiDAR损失计算
        # 构建渲染包
        render_pkg_loss = {
            "depth": rendered_depth,
            "intensity": rendered_intensity,
            "raydrop": rendered_raydrop,
            "visibility_filter": visibility_filter,
            "radii": radii,
            "viewspace_points": viewspace_point_tensor
        }
        
        # 获取真实LiDAR数据
        gt_data = {
            "range_image": viewpoint_camera.range_image,
            "intensity_image": viewpoint_camera.intensity_image
        }
        
        # 设置损失权重
        loss_weights = {
            'depth': getattr(config.opt, 'lambda_depth', 1.0),
            'intensity': getattr(config.opt, 'lambda_intensity', 0.5),
            'raydrop': getattr(config.opt, 'lambda_raydrop', 0.1),
            'smoothness': 0.01,
            'normal': 0.1,
            'planar': 0.05
        }
        
        loss_dict, total_lidar_loss = compute_lidar_loss(render_pkg_loss, gt_data, loss_weights)
        
        # PGSR几何约束损失
        # === 使用LiDAR intensity图像进行几何正则化 ===
        # 原理：LiDAR intensity包含真实边缘信息，适合PGSR单视图几何约束的边缘感知计算
        # intensity_image格式：KITTI-360数据集中为2D张量(H, W)，PGSR已支持单通道图像处理
        gt_image = viewpoint_camera.intensity_image  # 保持原始(H, W)格式
        
        pgsr_loss_dict, pgsr_total_loss = compute_pgsr_geometric_loss(
            render_pkg, gt_image, viewpoint_camera,
            getattr(config.opt, 'lambda_sv_geom', 0.015), 0.0, 0.0  # 在LiDAR模式下关闭多视图损失
        )
        
        # 合并损失
        total_loss = total_lidar_loss + pgsr_total_loss
        loss_dict.update(pgsr_loss_dict)

        # 调试：检查backward前的梯度状态
        if iteration <= 3:
            print(f"[DEBUG Iter {iteration}] Before backward:")
            print(f"  viewspace_point_tensor.grad: {viewspace_point_tensor.grad.shape if viewspace_point_tensor.grad is not None else 'None'}")
            print(f"  viewspace_point_tensor_abs.grad: {viewspace_point_tensor_abs.grad.shape if viewspace_point_tensor_abs.grad is not None else 'None'}")

        total_loss.backward()

        # 调试：检查backward后的梯度状态
        if iteration <= 3:
            print(f"[DEBUG Iter {iteration}] After backward:")
            print(f"  viewspace_point_tensor.grad: {viewspace_point_tensor.grad.shape if viewspace_point_tensor.grad is not None else 'None'}")
            print(f"  viewspace_point_tensor_abs.grad: {viewspace_point_tensor_abs.grad.shape if viewspace_point_tensor_abs.grad is not None else 'None'}")

        iter_end.record()

        with torch.no_grad():
            # 进度报告
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == config.opt.iterations:
                progress_bar.close()

            # 记录
            training_report_config(writer, iteration, loss_dict, total_loss, iter_start.elapsed_time(iter_end), config, scene, render_lidar, pipe, background)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密度自适应
            if iteration <= opt.densify_until_iter:
                # 调试可见点信息
                if iteration <= 10:
                    total_points = len(visibility_filter)
                    visible_points = visibility_filter.sum().item()
                    radii_stats = radii[radii > 0]
                    print(f"[DEBUG Iter {iteration}] Visibility: {visible_points}/{total_points} points visible")
                    if len(radii_stats) > 0:
                        print(f"[DEBUG Iter {iteration}] Radii stats: min={radii_stats.min():.3f}, max={radii_stats.max():.3f}, mean={radii_stats.float().mean():.3f}")
                    else:
                        print(f"[DEBUG Iter {iteration}] All radii are 0")
                
                # 可见点存在时进行密集化统计
                if visibility_filter.sum() > 0:
                    gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)
                elif iteration <= 5:
                    print(f"[DEBUG Iter {iteration}] Skipping densification stats due to no visible points")

                # 重新启用密集化
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, opt.min_opacity, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < config.opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger_config(config):
    """配置模式的输出目录和日志记录器准备"""
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
    
    # 从配置中提取参数到Namespace对象
    dataset = Namespace()
    dataset.source_path = getattr(config, 'source_dir', getattr(config.data, 'source_path', ''))
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
    
    dataset.frame_length = getattr(config, 'frame_length', [0, 100])
    dataset.seq = getattr(config, 'seq', "0000")
    dataset.data_type = getattr(config, 'data_type', "range_image")
    
    opt = Namespace()
    opt.iterations = config.opt.iterations
    opt.position_lr_init = getattr(config.opt, 'position_lr_init', 0.00016)
    opt.position_lr_final = getattr(config.opt, 'position_lr_final', 0.0000016)
    opt.position_lr_delay_mult = getattr(config.opt, 'position_lr_delay_mult', 0.01)
    opt.position_lr_max_steps = getattr(config.opt, 'position_lr_max_steps', 30000)
    opt.feature_lr = getattr(config.opt, 'feature_lr', 0.0025)
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


def training_report_config(tb_writer, iteration, loss_dict, total_loss, elapsed, config, scene, renderFunc, pipe, bg):
    """配置模式的训练报告和日志记录"""
    if tb_writer:
        # === 1. 损失记录 ===
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                tb_writer.add_scalar(f'train_loss_patches/{key}', value.item(), iteration)
            else:
                tb_writer.add_scalar(f'train_loss_patches/{key}', value, iteration)
        
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
        
        # === 5. 每1000次迭代进行测试和可视化 ===
        if iteration % 1000 == 0 and len(scene.getTrainCameras()) > 0:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                                {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

            for config_test in validation_configs:
                if config_test['cameras'] and len(config_test['cameras']) > 0:
                    render_results = []
                    for idx, viewpoint in enumerate(config_test['cameras']):
                        if idx >= 3:  # 限制测试样本数量
                            break
                        
                        try:
                            render_result = renderFunc(viewpoint, scene.gaussians, pipe, bg)
                            
                            # LiDAR评价指标
                            if hasattr(viewpoint, 'range_image') and render_result.get("depth") is not None:
                                gt_depth = viewpoint.range_image
                                pred_depth = render_result["depth"].squeeze()
                                
                                # 深度RMSE
                                valid_mask = (gt_depth > 0) & (pred_depth > 0)
                                if valid_mask.sum() > 0:
                                    depth_rmse = torch.sqrt(torch.mean((gt_depth[valid_mask] - pred_depth[valid_mask]) ** 2))
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics/depth_rmse', depth_rmse.item(), iteration)
                                    
                                    # 深度相对误差
                                    relative_error = torch.mean(torch.abs(gt_depth[valid_mask] - pred_depth[valid_mask]) / gt_depth[valid_mask])
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics/depth_relative_error', relative_error.item(), iteration)
                            
                            # Intensity评价指标
                            if hasattr(viewpoint, 'intensity_image') and render_result.get("intensity") is not None:
                                gt_intensity = viewpoint.intensity_image
                                pred_intensity = render_result["intensity"].squeeze()
                                
                                # Intensity PSNR
                                intensity_mse = torch.mean((gt_intensity - pred_intensity) ** 2)
                                if intensity_mse > 0:
                                    intensity_psnr = 20 * torch.log10(1.0 / torch.sqrt(intensity_mse))
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics/intensity_psnr', intensity_psnr.item(), iteration)
                                
                                # Intensity SSIM
                                if hasattr(ssim, '__call__'):
                                    gt_3ch = gt_intensity.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
                                    pred_3ch = pred_intensity.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
                                    intensity_ssim = ssim(gt_3ch, pred_3ch)
                                    tb_writer.add_scalar(f'{config_test["name"]}_metrics/intensity_ssim', intensity_ssim.item(), iteration)
                            
                            render_results.append(render_result)
                            
                        except Exception as e:
                            print(f"Error during {config_test['name']} evaluation at iteration {iteration}: {e}")
                            continue
                    
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
                                
                        except Exception as e:
                            print(f"Error saving images to tensorboard: {e}")
            
            torch.cuda.empty_cache()

def main():
    """主函数 - 支持配置文件和传统命令行两种模式"""
    parser = argparse.ArgumentParser(description="LiDAR-PGSR Training")
    
    # 添加配置文件选项
    parser.add_argument("-c", "--config", type=str, required=True,
                       help="Path to the configuration file (config mode)")
    
    # 传统命令行参数
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
