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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, lidar_data=None):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    # 修复LiDAR数据传递：支持KITTI-360数据集的帧号模式
    cam_lidar_data = None
    if lidar_data is not None:
        # 支持两种模式：
        # 1. 基于global_id的索引（原始模式）
        if hasattr(cam_info, 'global_id') and cam_info.global_id in lidar_data:
            cam_lidar_data = lidar_data[cam_info.global_id]
        # 2. 基于帧号的索引（KITTI-360模式）
        elif hasattr(cam_info, 'image_name'):
            # 从相机名称中提取帧号
            frame_str = cam_info.image_name.split('_')[1] if '_' in cam_info.image_name else None
            if frame_str and frame_str.isdigit():
                frame = int(frame_str)
                if frame in lidar_data:
                    # 为该相机准备LiDAR数据
                    from scene.lidar_sensor import LiDARUtils
                    raw_points = lidar_data[frame]['raw_points']
                    
                    # 转换为range image格式（KITTI-360参数）
                    H, W = 66, 1030 // 3 if hasattr(cam_info, 'horizontal_fov_start') else 1030
                    range_img, intensity_img, _ = LiDARUtils.spherical_to_range_image(
                        raw_points, H=H, W=W, fov_up=2.0, fov_down=-24.9, max_range=80.0
                    )
                    
                    cam_lidar_data = {
                        'range_image': range_img,
                        'intensity_image': intensity_img,
                        'raw_points': raw_points,
                        'pose': lidar_data[frame]['ego2world'],
                        'lidar2ego': lidar_data[frame]['lidar2ego']
                    }
                    
                    # 如果是分割相机，添加FOV信息
                    if hasattr(cam_info, 'horizontal_fov_start'):
                        cam_lidar_data['horizontal_fov_start'] = cam_info.horizontal_fov_start
                        cam_lidar_data['horizontal_fov_end'] = cam_info.horizontal_fov_end
                    
                    print(f"[LiDAR Data] Loaded for {cam_info.image_name}: "
                          f"frame={frame}, range_img={range_img.shape}, "
                          f"intensity_img={intensity_img.shape}")

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FoVX, FoVy=cam_info.FoVY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.uid, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  lidar_data=cam_lidar_data)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, lidar_data=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, lidar_data))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVY, camera.height),
        'fx' : fov2focal(camera.FoVX, camera.width)
    }
    return camera_entry
