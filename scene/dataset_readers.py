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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    global_id: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float

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

def load_poses(pose_path, num):
    poses = []
    with open(pose_path, "r") as f:
        lines = f.readlines()
    for i in range(num):
        line = lines[i]
        c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
        c2w[:3,3] = c2w[:3,3] * 10.0
        w2c = np.linalg.inv(c2w)
        w2c = w2c
        poses.append(w2c)
    poses = np.stack(poses, axis=0)
    return poses

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        cam_info = CameraInfo(uid=uid, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, fx=focal_length_x, fy=focal_length_y)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('_')[-1]))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    js_file = f"{path}/split.json"
    train_list = None
    test_list = None
    if os.path.exists(js_file):
        with open(js_file) as file:
            meta = json.load(file)
            train_list = meta["train"]
            test_list = meta["test"]
            print(f"train_list {len(train_list)}, test_list {len(test_list)}")

    if train_list is not None:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in train_list]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name in test_list]
        print(f"train_cam_infos {len(train_cam_infos)}, test_cam_infos {len(test_cam_infos)}")
    elif eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/points3D.ply")
    bin_path = os.path.join(path, "sparse/points3D.bin")
    txt_path = os.path.join(path, "sparse/points3D.txt")
    if not os.path.exists(ply_path) or True:
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            print(f"xyz {xyz.shape}")
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           lidar_data=None)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, global_id=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           lidar_data=None)
    return scene_info

def readKitti360SceneInfo(path, images=None, eval=False, args=None):
    """
    读取KITTI-360数据集，包含LiDAR点云和位姿信息
    适配LiDAR-RT的数据格式到PGSR的Scene结构
    """
    import math
    from pathlib import Path
    
    # 设置默认参数
    if args is None:
        class DefaultArgs:
            seq = "0000"
            frame_length = [0, 100]  # 默认帧范围
            data_type = "range_image"
        args = DefaultArgs()
    
    # 获取序列信息
    seq = getattr(args, 'seq', "0000")
    frames = getattr(args, 'frame_length', [0, 100])
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
    
    valid_frames = []
    for frame in range(frames[0], frames[1] + 1):
        lidar_file = os.path.join(lidar_dir, f"{str(frame).zfill(10)}.bin")
        if os.path.exists(lidar_file) and frame in ego2world:
            valid_frames.append(frame)
    
    print(f"Found {len(valid_frames)} valid frames with both LiDAR and pose data")
    
    for idx, frame in enumerate(valid_frames):
        # 加载LiDAR点云数据
        lidar_file = os.path.join(lidar_dir, f"{str(frame).zfill(10)}.bin")
        with open(lidar_file, "rb") as f:
            points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
        
        xyzs, intensities = points[:, :3], points[:, 3]
        dists = np.linalg.norm(xyzs, axis=1)
        
        # 转换到Range Image
        azimuth = np.arctan2(xyzs[:, 1], xyzs[:, 0])
        inclination = np.arctan2(xyzs[:, 2], np.sqrt(xyzs[:, 0]**2 + xyzs[:, 1]**2))
        
        w_idx = np.round((azimuth - azimuth_left) / h_res).astype(int)
        h_idx = np.round((inclination - inc_top) / v_res).astype(int)
        
        valid_mask = (dists <= max_depth) & (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
        
        # 创建range image
        range_map = np.ones((H, W)) * -1
        intensity_map = np.ones((H, W)) * -1
        
        if np.any(valid_mask):
            valid_h = h_idx[valid_mask]
            valid_w = w_idx[valid_mask]
            valid_dists = dists[valid_mask]
            valid_intensities = intensities[valid_mask]
            
            # 对于重复像素，保留最近的点
            indices = np.lexsort((valid_dists, valid_h, valid_w))
            valid_h = valid_h[indices]
            valid_w = valid_w[indices]
            valid_dists = valid_dists[indices]
            valid_intensities = valid_intensities[indices]
            
            _, unique_idx = np.unique(np.column_stack((valid_h, valid_w)), axis=0, return_index=True)
            
            range_map[valid_h[unique_idx], valid_w[unique_idx]] = valid_dists[unique_idx]
            intensity_map[valid_h[unique_idx], valid_w[unique_idx]] = valid_intensities[unique_idx]
        
        # 将-1替换为0
        range_map[range_map == -1] = 0
        intensity_map[intensity_map == -1] = 0
        
        # 存储LiDAR数据
        lidar_data[frame] = {
            'range_image': range_map,
            'intensity_image': intensity_map,
            'raw_points': points,
            'lidar2ego': lidar2ego,
            'ego2world': ego2world[frame]
        }
        
        # 创建虚拟相机参数
        # 计算世界坐标下的LiDAR位置
        ego2world_matrix = ego2world[frame]
        lidar2world = ego2world_matrix @ lidar2ego
        
        # 提取旋转和平移
        R = lidar2world[:3, :3].T  # PGSR expects transposed rotation
        T = lidar2world[:3, 3]
        
        # 虚拟相机内参（基于Range Image的分辨率）
        fx = W / (2 * np.pi)  # 水平方向的焦距
        fy = H / (inc_top - inc_bottom)  # 垂直方向的焦距
        
        FovX = 2 * np.pi  # 水平视场角360度
        FovY = inc_top - inc_bottom  # 垂直视场角
        
        # 创建CameraInfo
        cam_info = CameraInfo(
            uid=idx,
            global_id=frame,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image_path=f"frame_{frame:06d}",  # 虚拟路径
            image_name=f"frame_{frame:06d}",
            width=W,
            height=H,
            fx=fx,
            fy=fy
        )
        cam_infos.append(cam_info)
    
    # 划分训练和测试集
    if eval:
        # 简单的8:2划分
        split_idx = int(len(cam_infos) * 0.8)
        train_cam_infos = cam_infos[:split_idx]
        test_cam_infos = cam_infos[split_idx:]
    else:
        train_cam_infos = cam_infos
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
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Kitti360": readKitti360SceneInfo
}