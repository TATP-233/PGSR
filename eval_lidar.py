#!/usr/bin/env python3
"""
LiDAR-PGSR独立评估脚本
参考lidar-rt的eval.py实现，提供完整的LiDAR评估功能
"""

import os
import argparse
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gaussian_renderer.lidar_renderer import render_lidar
from scene import Scene, GaussianModel
from utils.evaluation_utils import LiDARPGSREvaluator, format_metrics_output
from utils.config_utils import load_config
from argparse import Namespace


def load_model(model_path, config):
    """加载训练好的模型"""
    gaussians = GaussianModel(config.model.sh_degree)
    
    # 查找模型文件
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    if os.path.isdir(model_path):
        # 如果是目录，查找最新的模型文件
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pth')]
        if not model_files:
            raise ValueError(f"No .pth files found in {model_path}")
        
        # 优先选择包含"good"的文件，否则选择最新的
        good_files = [f for f in model_files if 'good' in f.lower()]
        if good_files:
            model_file = os.path.join(model_path, good_files[0])
        else:
            # 按修改时间排序，选择最新的
            model_files_with_time = [(f, os.path.getmtime(os.path.join(model_path, f))) for f in model_files]
            model_files_with_time.sort(key=lambda x: x[1], reverse=True)
            model_file = os.path.join(model_path, model_files_with_time[0][0])
    else:
        model_file = model_path
    
    print(f"Loading model from: {model_file}")
    
    # 加载模型参数
    checkpoint = torch.load(model_file)
    if isinstance(checkpoint, tuple):
        model_params, iteration = checkpoint
        print(f"Loaded model from iteration: {iteration}")
    else:
        model_params = checkpoint
        iteration = "unknown"
    
    gaussians.restore(model_params, config)
    return gaussians, iteration


def evaluate_lidar_pgsr(args):
    """执行LiDAR-PGSR评估"""
    # 加载配置
    config = load_config(args.config)
    
    # 初始化评估器
    evaluator = LiDARPGSREvaluator(device=args.device)
    
    # 加载模型
    gaussians, iteration = load_model(args.model_path, config)
    gaussians.eval()
    
    # 加载场景数据
    dataset = Namespace()
    dataset.source_path = getattr(config, 'source_path', config.data.source_path)
    dataset.model_path = args.model_path
    dataset.images = getattr(config.data, 'images', 'images')
    dataset.resolution = getattr(config.data, 'resolution', -1)
    dataset.white_background = getattr(config.data, 'white_background', False)
    dataset.data_device = getattr(config.data, 'data_device', 'cuda')
    dataset.eval = True
    
    scene = Scene(dataset, gaussians, shuffle=False)
    
    # 设置渲染参数
    pipe = Namespace()
    pipe.convert_SHs_python = getattr(config.pipe, 'convert_SHs_python', False)
    pipe.compute_cov3D_python = getattr(config.pipe, 'compute_cov3D_python', False)
    pipe.debug = getattr(config.pipe, 'debug', False)
    
    # 设置背景
    bg_color = [0, 0, 1] if not dataset.white_background else [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    # 获取评估相机
    if args.eval_type == 'train':
        cameras = scene.getTrainCameras()
        print(f"Evaluating on {len(cameras)} training cameras")
    elif args.eval_type == 'test':
        cameras = scene.getTestCameras()
        print(f"Evaluating on {len(cameras)} test cameras")
    else:  # 'all'
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        cameras = train_cameras + test_cameras
        print(f"Evaluating on {len(cameras)} cameras (train: {len(train_cameras)}, test: {len(test_cameras)})")
    
    if len(cameras) == 0:
        print("No cameras found for evaluation!")
        return
    
    # 创建保存目录
    save_dir = Path(args.save_path) if args.save_path else Path(args.model_path) / "evaluation" / str(iteration)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行评估
    frame_results = []
    failed_count = 0
    
    print(f"Starting evaluation on {len(cameras)} cameras...")
    
    use_rayhit = getattr(config.opt, 'use_rayhit', False)
    
    for idx, camera in enumerate(tqdm(cameras, desc="Evaluating cameras")):
        # 检查相机是否有LiDAR数据
        if not hasattr(camera, 'range_image') or camera.range_image is None:
            failed_count += 1
            continue
        
        try:
            with torch.no_grad():
                # 渲染
                render_result = render_lidar(camera, gaussians, pipe, background, use_rayhit=use_rayhit)
                
                # 构建渲染数据
                rendered_data = {
                    'depth': render_result["depth"].cpu().numpy(),
                    'intensity': render_result["intensity"].cpu().numpy(),
                    'raydrop': render_result["raydrop"].cpu().numpy()
                }
                
                # 构建GT数据
                gt_data = {
                    'depth': camera.range_image.cpu().numpy(),
                }
                
                if hasattr(camera, 'intensity_image') and camera.intensity_image is not None:
                    gt_data['intensity'] = camera.intensity_image.cpu().numpy()
                
                # 生成掩码
                if hasattr(camera, 'mask') and camera.mask is not None:
                    gt_data['rayhit_mask'] = camera.mask.cpu().numpy()
                else:
                    gt_data['rayhit_mask'] = (gt_data['depth'] > 0).astype(np.float32)
                
                # 评估
                frame_result = evaluator.evaluate_frame(rendered_data, gt_data)
                if frame_result:
                    frame_result['camera_name'] = camera.image_name
                    frame_result['frame_idx'] = idx
                    frame_results.append(frame_result)
                
                # 保存可视化结果（可选）
                if args.save_images and idx < args.max_save_images:
                    save_frame_results(save_dir, camera.image_name, rendered_data, gt_data, frame_result)
                    
        except Exception as e:
            print(f"Error evaluating camera {camera.image_name}: {str(e)}")
            failed_count += 1
            continue
    
    print(f"Evaluation completed. Success: {len(frame_results)}, Failed: {failed_count}")
    
    if not frame_results:
        print("No valid evaluation results!")
        return
    
    # 聚合结果
    aggregated_results = evaluator.aggregate_results(frame_results)
    
    # 输出结果
    print(format_metrics_output(aggregated_results, f"Final Evaluation Results"))
    
    # 保存结果
    results_summary = {
        'model_path': args.model_path,
        'iteration': iteration,
        'eval_type': args.eval_type,
        'num_cameras': len(cameras),
        'num_successful': len(frame_results),
        'num_failed': failed_count,
        'aggregated_metrics': aggregated_results,
        'per_frame_results': frame_results if args.save_per_frame else None,
        'config': {
            'use_rayhit': use_rayhit,
            'evaluator_settings': {
                'raydrop_ratio': evaluator.raydrop_ratio,
                'fscore_threshold': evaluator.fscore_threshold
            }
        }
    }
    
    # 保存JSON结果
    results_file = save_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=4, default=str)
    print(f"Results saved to: {results_file}")
    
    # 生成简化报告
    report_file = save_dir / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"LiDAR-PGSR Evaluation Report\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Evaluation Type: {args.eval_type}\n")
        f.write(f"Cameras Evaluated: {len(frame_results)}/{len(cameras)}\n\n")
        
        f.write(format_metrics_output(aggregated_results, "Metrics Summary"))
        
        # 添加关键指标对比表
        f.write(f"\n\nKey Metrics Summary:\n")
        f.write(f"{'-' * 30}\n")
        
        key_metrics = {}
        if 'depth' in aggregated_results:
            key_metrics.update({
                'Depth RMSE': aggregated_results['depth']['rmse']['mean'],
                'Depth MAE': aggregated_results['depth']['mae']['mean'],
                'Depth PSNR': aggregated_results['depth']['psnr']['mean'],
            })
        if 'intensity' in aggregated_results:
            key_metrics.update({
                'Intensity PSNR': aggregated_results['intensity']['psnr']['mean'],
                'Intensity SSIM': aggregated_results['intensity']['ssim']['mean'],
            })
        if 'raydrop' in aggregated_results:
            key_metrics.update({
                'Raydrop Accuracy': aggregated_results['raydrop']['acc']['mean'],
                'Raydrop F1-Score': aggregated_results['raydrop']['f1']['mean'],
            })
        
        for metric_name, value in key_metrics.items():
            f.write(f"{metric_name:<20}: {value:>10.6f}\n")
    
    print(f"Report saved to: {report_file}")
    
    return aggregated_results


def save_frame_results(save_dir, camera_name, rendered_data, gt_data, metrics):
    """保存单帧的可视化结果"""
    frame_dir = save_dir / "frames" / camera_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存渲染结果
    np.save(frame_dir / "rendered_depth.npy", rendered_data['depth'])
    np.save(frame_dir / "rendered_intensity.npy", rendered_data['intensity'])
    np.save(frame_dir / "rendered_raydrop.npy", rendered_data['raydrop'])
    
    # 保存GT数据
    np.save(frame_dir / "gt_depth.npy", gt_data['depth'])
    if 'intensity' in gt_data and gt_data['intensity'] is not None:
        np.save(frame_dir / "gt_intensity.npy", gt_data['intensity'])
    if 'rayhit_mask' in gt_data:
        np.save(frame_dir / "gt_mask.npy", gt_data['rayhit_mask'])
    
    # 保存指标
    with open(frame_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4, default=str)


def main():
    parser = argparse.ArgumentParser(description="LiDAR-PGSR Evaluation Script")
    parser.add_argument('--model_path', '-m', type=str, required=True, 
                        help='Path to trained model (.pth file or directory)')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--eval_type', type=str, choices=['train', 'test', 'all'], 
                        default='test', help='Evaluation dataset type')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for evaluation')
    parser.add_argument('--save_images', action='store_true',
                        help='Save rendered images and GT data')
    parser.add_argument('--max_save_images', type=int, default=10,
                        help='Maximum number of images to save')
    parser.add_argument('--save_per_frame', action='store_true',
                        help='Save per-frame evaluation results')
    
    args = parser.parse_args()
    
    # 设置设备
    torch.cuda.set_device(args.device)
    
    # 执行评估
    results = evaluate_lidar_pgsr(args)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 