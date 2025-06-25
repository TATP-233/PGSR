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

import os
import numpy as np
from typing import NamedTuple
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, BasicPointCloud
from utils.sh_utils import SH2RGB

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FoVY: float
    FoVX: float
    image: object
    image_path: str
    image_name: str
    width: int
    height: int
    lidar_path: str = None
    horizontal_fov_start: float = 0.0
    horizontal_fov_end: float = 360.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    lidar_data: dict = None  # 添加LiDAR数据字段

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readKITTI360Info(path, images=None, eval=False, args=None, cam_split_mode="triple"):
    """
    读取KITTI-360数据集信息
    
    Args:
        path: 数据集根目录
        images: 图像路径（LiDAR模式下不使用）
        eval: 是否为评估模式
        args: 参数对象，包含seq、frame_length等
        cam_split_mode: 相机分割模式，"single"为单个360度相机，"triple"为三个120度相机
    """
    print(f"Found data_3d_raw folder, assuming KITTI-360 dataset!")
    
    import math
    from pathlib import Path
    
    # 从args获取参数
    frames = args.frame_length if hasattr(args, 'frame_length') else [0, 100]
    seq = args.seq if hasattr(args, 'seq') else "0000"
    
    if hasattr(args, 'seq') and args.seq:
        seq = args.seq
    else:
        seq = "0000"  # 默认序列
    
    print(f"Using frame range: {frames}, sequence: {seq}")
    full_seq = f"2013_05_28_drive_{seq}_sync"
    
    print(f"Loading KITTI-360 sequence: {full_seq}, frames: {frames}")
    
    # LiDAR传感器参数（来自LiDAR-RT）
    W, H = 1030, 66
    inc_bottom, inc_top = math.radians(-24.9), math.radians(2.0)
    azimuth_left, azimuth_right = np.pi, -np.pi
    max_depth = 80.0
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_bottom - inc_top) / H
    
    # 加载位姿数据
    pose_file = os.path.join(path, "data_pose", full_seq, "poses.txt")
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
        
    ego2world = {}
    with open(pose_file, "r") as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.split()
        frame = int(parts[0])
        if frames[0] <= frame <= frames[1]:  # 只加载指定帧范围
            values = [float(x) for x in parts[1:]]
            matrix = np.array(values).reshape(3, 4)  # (3, 4)
            # 转换为4x4齐次变换矩阵
            matrix_4x4 = np.eye(4)
            matrix_4x4[:3, :] = matrix
            ego2world[frame] = matrix_4x4
    
    # LiDAR到ego的变换矩阵（来自LiDAR-RT）
    cam2velo = np.array([
        0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
        -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574,
        -0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824,
        0.0, 0.0, 0.0, 1.0
    ]).reshape(4, 4)
    
    cam2ego = np.array([
        0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039,
        0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093,
        0.0090621821, 0.9951109327, 0.0983468786, -0.6500000000,
        0.0, 0.0, 0.0, 1.0
    ]).reshape(4, 4)
    
    lidar2ego = cam2ego @ np.linalg.inv(cam2velo)
    
    # 创建虚拟相机信息（用于适配PGSR的相机系统）
    # 每一帧LiDAR数据对应一个虚拟相机
    cam_infos = []
    lidar_data = {}  # 存储LiDAR数据
    
    lidar_dir = os.path.join(path, "data_3d_raw", full_seq, "velodyne_points", "data")
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR data directory not found: {lidar_dir}")
    
    # 获取LiDAR文件列表
    lidar_files = {}
    for frame in range(frames[0], frames[1] + 1):
        lidar_file = os.path.join(lidar_dir, f"{str(frame).zfill(10)}.bin")
        if os.path.exists(lidar_file):
            lidar_files[frame] = lidar_file
    
    valid_frames = []
    for frame in range(frames[0], frames[1] + 1):
        if frame in lidar_files and frame in ego2world:
            valid_frames.append(frame)
            
            # 加载并存储LiDAR数据
            with open(lidar_files[frame], "rb") as f:
                points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            
            lidar_data[frame] = {
                'raw_points': points,
                'lidar2ego': lidar2ego,
                'ego2world': ego2world[frame]
            }
    
    print(f"Found {len(valid_frames)} valid frames with both LiDAR and pose data")
    
    train_cam_infos = []
    test_cam_infos = []
    
    # 为每一帧创建相机
    for frame in valid_frames:
        # 获取该帧的ego2world变换矩阵
        ego2world_matrix = ego2world[frame]
        lidar2world_matrix = ego2world_matrix @ lidar2ego
        
        # 从lidar2world矩阵提取LiDAR传感器的世界坐标系位置和朝向
        lidar_world_pos = lidar2world_matrix[:3, 3]  # LiDAR在世界坐标系中的位置
        lidar_world_rot = lidar2world_matrix[:3, :3]  # LiDAR在世界坐标系中的旋转
        
        print(f"[DEBUG] Frame {frame}: LiDAR world position = {lidar_world_pos}")
        print(f"[DEBUG] Frame {frame}: LiDAR world rotation =\n{lidar_world_rot}")
        
        if cam_split_mode == "triple":
            # 创建三个120度相机，分别覆盖不同的水平角度范围
            for cam_idx in range(3):
                # 计算该相机的水平角度偏移（0度、120度、240度）
                horizontal_offset = cam_idx * 120.0  # 度
                horizontal_offset_rad = np.radians(horizontal_offset)
                
                # 创建围绕Z轴的旋转矩阵（水平旋转）
                cos_h = np.cos(horizontal_offset_rad)
                sin_h = np.sin(horizontal_offset_rad)
                horizontal_rotation = np.array([
                    [cos_h, -sin_h, 0],
                    [sin_h,  cos_h, 0],
                    [0,      0,     1]
                ])
                
                # 应用水平旋转到LiDAR坐标系
                rotated_lidar_rot = lidar_world_rot @ horizontal_rotation
                
                # 坐标系变换：LiDAR(X前Y左Z上) -> 相机(X右Y上Z后)
                lidar_to_camera_transform = np.array([
                    [0, -1,  0],  # LiDAR的Y(左) -> 相机的-X(左)
                    [0,  0,  1],  # LiDAR的Z(上) -> 相机的Y(上)  
                    [-1, 0,  0]   # LiDAR的X(前) -> 相机的-Z(前)
                ])
                
                # 计算最终的相机到世界的旋转矩阵
                camera_to_world_rotation = rotated_lidar_rot @ lidar_to_camera_transform.T
                
                # PGSR需要的参数
                R = camera_to_world_rotation  # camera-to-world旋转矩阵
                
                # 计算world-to-camera的translation
                # 对于旋转偏移的相机，位置保持在同一个LiDAR中心
                world_to_camera_translation = -camera_to_world_rotation.T @ lidar_world_pos
                T = world_to_camera_translation
                
                print(f"[DEBUG] Frame {frame} Cam {cam_idx}: Camera R (camera-to-world) =\n{R}")
                print(f"[DEBUG] Frame {frame} Cam {cam_idx}: Camera T (world-to-camera translation) = {T}")
                
                # 验证相机中心是否正确
                expected_camera_center = lidar_world_pos  # 所有分割相机共享同一中心
                print(f"[DEBUG] Frame {frame} Cam {cam_idx}: Expected camera center = {expected_camera_center}")
                print(f"[DEBUG] Frame {frame} Cam {cam_idx}: Should match LiDAR world position = {lidar_world_pos}")
                
                # 计算相机中心误差用于验证
                # 从world-to-view变换矩阵中提取相机中心
                world_view_transform = getWorld2View2(R, T)
                C2W = np.linalg.inv(world_view_transform)
                camera_center_from_transform = C2W[:3, 3]
                camera_center_error = np.linalg.norm(camera_center_from_transform - expected_camera_center)
                print(f"[DEBUG] Frame {frame} Cam {cam_idx}: Camera center error = {camera_center_error:.6f}")
                
                # 120度水平FOV，保持原有的垂直FOV
                FoVx = np.radians(120.0)  # 120度水平视场角
                FoVy = np.radians(26.9)   # 保持原有垂直视场角
                
                # 相机图像尺寸：水平分辨率为原来的1/3，垂直保持不变
                image_width = 1030 // 3  # 约343像素
                image_height = 66
                
                # 创建相机信息
                cam_info = CameraInfo(
                    uid=len(train_cam_infos),
                    R=R,
                    T=T,
                    FoVx=FoVx,
                    FoVy=FoVy,
                    image=None,  # LiDAR模式下不需要图像
                    image_path=None,
                    image_name=f"frame_{frame:06d}_cam_{cam_idx}",
                    width=image_width,
                    height=image_height,
                    lidar_path=lidar_files[frame] if frame < len(lidar_files) else None,
                    # 添加水平角度范围信息，用于LiDAR点过滤
                    horizontal_fov_start=horizontal_offset,
                    horizontal_fov_end=(horizontal_offset + 120.0) % 360.0
                )
                
                train_cam_infos.append(cam_info)
        else:
            # 原来的单相机模式（保持360度）
            # 坐标系变换：LiDAR(X前Y左Z上) -> 相机(X右Y上Z后)
            lidar_to_camera_transform = np.array([
                [0, -1,  0],  # LiDAR的Y(左) -> 相机的-X(左)
                [0,  0,  1],  # LiDAR的Z(上) -> 相机的Y(上)  
                [-1, 0,  0]   # LiDAR的X(前) -> 相机的-Z(前)
            ])
            
            # 计算最终的相机到世界的旋转矩阵
            camera_to_world_rotation = lidar_world_rot @ lidar_to_camera_transform.T
            
            # PGSR需要的参数
            R = camera_to_world_rotation  # camera-to-world旋转矩阵
            
            # 计算world-to-camera的translation
            world_to_camera_translation = -camera_to_world_rotation.T @ lidar_world_pos
            T = world_to_camera_translation
            
            print(f"[DEBUG] Frame {frame}: Camera R (camera-to-world) =\n{R}")
            print(f"[DEBUG] Frame {frame}: Camera T (world-to-camera translation) = {T}")
            
            # 验证相机中心是否正确
            expected_camera_center = lidar_world_pos
            print(f"[DEBUG] Frame {frame}: Expected camera center = {expected_camera_center}")
            print(f"[DEBUG] Frame {frame}: Should match LiDAR world position = {lidar_world_pos}")
            
            # 计算相机中心误差用于验证
            # 从world-to-view变换矩阵中提取相机中心
            world_view_transform = getWorld2View2(R, T)
            C2W = np.linalg.inv(world_view_transform)
            camera_center_from_transform = C2W[:3, 3]
            camera_center_error = np.linalg.norm(camera_center_from_transform - expected_camera_center)
            print(f"[DEBUG] Frame {frame}: Camera center error = {camera_center_error:.6f}")
            
            # 360度全景相机
            FoVx = 2 * np.pi  # 360度水平视场角
            FoVy = np.radians(26.9)   # 保持原有垂直视场角
            
            # 创建相机信息
            cam_info = CameraInfo(
                uid=len(train_cam_infos),
                R=R,
                T=T,
                FoVx=FoVx,
                FoVy=FoVy,
                image=None,  # LiDAR模式下不需要图像
                image_path=None,
                image_name=f"frame_{frame:06d}",
                width=1030,
                height=66,
                lidar_path=lidar_files[frame] if frame < len(lidar_files) else None,
                horizontal_fov_start=0.0,
                horizontal_fov_end=360.0
            )
            
            train_cam_infos.append(cam_info)
    
    # 划分训练和测试集
    if eval:
        # 简单的8:2划分
        split_idx = int(len(train_cam_infos) * 0.8)
        train_cam_infos = train_cam_infos[:split_idx]
        test_cam_infos = train_cam_infos[split_idx:]
    else:
        train_cam_infos = train_cam_infos
        test_cam_infos = []
    
    print(f"Training cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")
    
    # 计算场景归一化参数
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # 从LiDAR数据生成初始点云
    ply_path = os.path.join(path, "lidar_points3d.ply")
    
    all_points = []
    all_colors = []
    
    # 从部分LiDAR帧采样点云
    sample_frames = valid_frames[::max(1, len(valid_frames)//10)]  # 最多采样10帧
    
    for frame in sample_frames:
        points = lidar_data[frame]['raw_points']
        ego2world_matrix = lidar_data[frame]['ego2world']
        lidar2ego_matrix = lidar_data[frame]['lidar2ego']
        
        # 转换到世界坐标系
        xyzs = points[:, :3]
        intensities = points[:, 3]
        
        # 添加齐次坐标
        xyzs_homo = np.hstack([xyzs, np.ones((xyzs.shape[0], 1))])
        
        # 变换到世界坐标系
        world_points = (ego2world_matrix @ lidar2ego_matrix @ xyzs_homo.T).T[:, :3]
        
        # 根据强度生成颜色
        colors = np.stack([intensities, intensities, intensities], axis=1)
        colors = np.clip(colors / np.max(intensities), 0, 1)  # 归一化
        
        # 下采样以减少点数
        step = max(1, len(world_points) // 10000)  # 每帧最多1万个点
        all_points.append(world_points[::step])
        all_colors.append(colors[::step])
    
    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        print(f"Generated point cloud with {len(all_points)} points")
        storePly(ply_path, all_points, (all_colors * 255).astype(np.uint8))
        pcd = BasicPointCloud(points=all_points, colors=all_colors, normals=np.zeros_like(all_points))
    else:
        print("Warning: No valid LiDAR points found, creating random point cloud")
        # 创建随机点云作为fallback
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 20 - 10  # 适合KITTI场景的范围
        colors = np.random.random((num_pts, 3))
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, (colors * 255).astype(np.uint8))
    
    # 创建包含LiDAR数据的SceneInfo
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        lidar_data=lidar_data
    )
    
    return scene_info

sceneLoadTypeCallbacks = {
    "Kitti360": readKITTI360Info
}