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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from utils.evaluation_utils import LiDARPGSREvaluator, format_metrics_output
from argparse import ArgumentParser
import numpy as np

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readLiDARData(lidar_dir):
    """读取LiDAR数据（深度、强度、掩码）"""
    lidar_data = {}
    
    # 读取深度数据
    depth_dir = lidar_dir / "depth"
    if depth_dir.exists():
        depth_data = []
        depth_names = []
        for fname in sorted(os.listdir(depth_dir)):
            if fname.endswith(('.npy', '.npz')):
                depth = np.load(depth_dir / fname)
                depth_data.append(depth)
                depth_names.append(fname)
        lidar_data['depth'] = (depth_data, depth_names)
    
    # 读取强度数据
    intensity_dir = lidar_dir / "intensity"
    if intensity_dir.exists():
        intensity_data = []
        intensity_names = []
        for fname in sorted(os.listdir(intensity_dir)):
            if fname.endswith(('.npy', '.npz')):
                intensity = np.load(intensity_dir / fname)
                intensity_data.append(intensity)
                intensity_names.append(fname)
        lidar_data['intensity'] = (intensity_data, intensity_names)
    
    # 读取掩码数据
    mask_dir = lidar_dir / "mask"
    if mask_dir.exists():
        mask_data = []
        mask_names = []
        for fname in sorted(os.listdir(mask_dir)):
            if fname.endswith(('.npy', '.npz')):
                mask = np.load(mask_dir / fname)
                mask_data.append(mask)
                mask_names.append(fname)
        lidar_data['mask'] = (mask_data, mask_names)
    
    return lidar_data

def evaluate(model_paths, eval_lidar=False):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    
    # 初始化LiDAR评估器
    lidar_evaluator = None
    if eval_lidar:
        lidar_evaluator = LiDARPGSREvaluator()
    
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                
                # === 传统图像评估 ===
                if (renders_dir / "image").exists() and (gt_dir / "image").exists():
                    renders, gts, image_names = readImages(renders_dir / "image", gt_dir / "image")

                    ssims = []
                    psnrs = []
                    lpipss = []

                    for idx in tqdm(range(len(renders)), desc="Image metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    print("  Image SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                    print("  Image PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                    print("  Image LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))

                    full_dict[scene_dir][method].update({
                        "Image_SSIM": torch.tensor(ssims).mean().item(),
                        "Image_PSNR": torch.tensor(psnrs).mean().item(),
                        "Image_LPIPS": torch.tensor(lpipss).mean().item()
                    })
                    per_view_dict[scene_dir][method].update({
                        "Image_SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                        "Image_PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                        "Image_LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}
                    })
                
                # === LiDAR评估 ===
                if eval_lidar and lidar_evaluator:
                    lidar_gt_dir = gt_dir / "lidar"
                    lidar_renders_dir = renders_dir / "lidar"
                    
                    if lidar_gt_dir.exists() and lidar_renders_dir.exists():
                        print("  Evaluating LiDAR metrics...")
                        
                        # 读取LiDAR数据
                        gt_lidar_data = readLiDARData(lidar_gt_dir)
                        rendered_lidar_data = readLiDARData(lidar_renders_dir)
                        
                        frame_results = []
                        
                        # 确定帧数
                        num_frames = 0
                        if 'depth' in gt_lidar_data:
                            num_frames = len(gt_lidar_data['depth'][0])
                        elif 'intensity' in gt_lidar_data:
                            num_frames = len(gt_lidar_data['intensity'][0])
                        
                        for frame_idx in tqdm(range(num_frames), desc="LiDAR metric evaluation progress"):
                            # 构建GT数据字典
                            gt_data = {}
                            if 'depth' in gt_lidar_data:
                                gt_data['depth'] = gt_lidar_data['depth'][0][frame_idx]
                            if 'intensity' in gt_lidar_data:
                                gt_data['intensity'] = gt_lidar_data['intensity'][0][frame_idx]
                            if 'mask' in gt_lidar_data:
                                gt_data['rayhit_mask'] = gt_lidar_data['mask'][0][frame_idx]
                            
                            # 构建渲染数据字典
                            rendered_data = {}
                            if 'depth' in rendered_lidar_data:
                                rendered_data['depth'] = rendered_lidar_data['depth'][0][frame_idx]
                            if 'intensity' in rendered_lidar_data:
                                rendered_data['intensity'] = rendered_lidar_data['intensity'][0][frame_idx]
                            if 'mask' in rendered_lidar_data:
                                # 假设渲染的mask是raydrop概率
                                rendered_data['raydrop'] = rendered_lidar_data['mask'][0][frame_idx]
                            
                            # 评估单帧
                            if gt_data and rendered_data:
                                frame_result = lidar_evaluator.evaluate_frame(rendered_data, gt_data)
                                if frame_result:
                                    frame_results.append(frame_result)
                        
                        # 聚合结果
                        if frame_results:
                            aggregated_results = lidar_evaluator.aggregate_results(frame_results)
                            
                            # 输出LiDAR评估结果
                            print(format_metrics_output(aggregated_results, "LiDAR Metrics"))
                            
                            # 保存到结果字典
                            lidar_summary = {}
                            for data_type, metrics in aggregated_results.items():
                                for metric_name, metric_stats in metrics.items():
                                    key = f"LiDAR_{data_type}_{metric_name}"
                                    lidar_summary[key] = metric_stats['mean']
                            
                            full_dict[scene_dir][method].update(lidar_summary)
                            
                            # 保存每帧结果
                            lidar_per_view = {}
                            for frame_idx, frame_result in enumerate(frame_results):
                                frame_key = f"frame_{frame_idx:06d}"
                                lidar_per_view[frame_key] = frame_result
                            
                            per_view_dict[scene_dir][method]["LiDAR"] = lidar_per_view

                print("")

            # 保存结果
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
                
        except Exception as e:
            print(f"Unable to compute metrics for model {scene_dir}: {str(e)}")


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--eval_lidar', action='store_true', help='Enable LiDAR evaluation')
    args = parser.parse_args()
    
    evaluate(args.model_paths, eval_lidar=args.eval_lidar)
