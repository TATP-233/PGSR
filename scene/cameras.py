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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, getProjectionMatrixCenterShift
import copy
from PIL import Image
from utils.general_utils import PILtoTorch
import os, cv2
import torch.nn.functional as F

def dilate(bin_img, ksize=6):
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=12):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

def process_image(image_path, resolution, ncc_scale):
    image = Image.open(image_path)
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image_width, image_height,
                 image_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 ncc_scale=1.0,
                 preload_img=True, data_device = "cuda",
                 # LiDAR特定参数
                 lidar_data=None
                 ):
        super(Camera, self).__init__()
        self.uid = uid
        self.nearest_id = []
        self.nearest_names = []
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.resolution = (image_width, image_height)
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image, self.image_gray, self.mask = None, None, None
        self.preload_img = preload_img
        self.ncc_scale = ncc_scale
        
        # === LiDAR数据处理（参考lidar-rt的设计）===
        self.lidar_data = lidar_data
        self.is_lidar_camera = False
        
        # LiDAR特定属性
        self.range_image = None
        self.intensity_image = None
        self.valid_mask = None
        self.horizontal_fov_start = None
        self.horizontal_fov_end = None
        
        if self.lidar_data is not None:
            self.is_lidar_camera = True
            try:
                # 将LiDAR数据转移到device
                if 'range_image' in self.lidar_data:
                    range_data = self.lidar_data['range_image']
                    if isinstance(range_data, np.ndarray):
                        self.range_image = torch.from_numpy(range_data).float().to(self.data_device)
                    else:
                        self.range_image = torch.tensor(range_data, dtype=torch.float32).to(self.data_device)
                
                if 'intensity_image' in self.lidar_data:
                    intensity_data = self.lidar_data['intensity_image']
                    if isinstance(intensity_data, np.ndarray):
                        self.intensity_image = torch.from_numpy(intensity_data).float().to(self.data_device)
                    else:
                        self.intensity_image = torch.tensor(intensity_data, dtype=torch.float32).to(self.data_device)
                
                # 预计算有效掩码
                if self.range_image is not None:
                    self.valid_mask = (self.range_image > 0).to(self.data_device)
                
                # 处理分割相机的FOV信息
                if 'horizontal_fov_start' in self.lidar_data:
                    self.horizontal_fov_start = self.lidar_data['horizontal_fov_start']
                if 'horizontal_fov_end' in self.lidar_data:
                    self.horizontal_fov_end = self.lidar_data['horizontal_fov_end']
                
                # 处理姿态信息
                if 'pose' in self.lidar_data:
                    self.lidar_pose = self.lidar_data['pose']
                
                print(f"[LiDAR Camera] {self.image_name}: "
                      f"Range {self.range_image.shape if self.range_image is not None else 'None'}, "
                      f"Intensity {self.intensity_image.shape if self.intensity_image is not None else 'None'}")
                
            except Exception as e:
                print(f"[Warning] Error processing LiDAR data for {self.image_name}: {e}")
                self.is_lidar_camera = False
                self.range_image = None
                self.intensity_image = None
                self.valid_mask = None
        
        # 如果没有LiDAR数据但是是特定的相机模式，创建虚拟数据
        if not self.is_lidar_camera:
            # 检查是否为120度分割相机
            if hasattr(self, 'FoVx') and abs(self.FoVx - 2 * np.pi / 3) < 0.1:  # 约120度
                print(f"[Virtual LiDAR] Creating for 120° camera {self.image_name}")
                self.range_image = torch.zeros((self.image_height, self.image_width), dtype=torch.float32).to(self.data_device)
                self.intensity_image = torch.zeros((self.image_height, self.image_width), dtype=torch.float32).to(self.data_device)
                self.valid_mask = torch.zeros((self.image_height, self.image_width), dtype=torch.bool).to(self.data_device)
                self.is_lidar_camera = True  # 标记为LiDAR相机（虽然是虚拟的）
        
        # 检查是否为LiDAR-only模式（没有真实图像文件）
        image_exists = os.path.exists(self.image_path) if self.image_path else False
        is_lidar_only = not image_exists  # 简化：如果没有有效的图像路径就认为是LiDAR模式
        
        if self.preload_img and not is_lidar_only:
            gt_image, gray_image, loaded_mask = process_image(self.image_path, self.resolution, ncc_scale)
            self.original_image = gt_image.to(self.data_device)
            self.original_image_gray = gray_image.to(self.data_device)
            self.mask = loaded_mask
        else:
            # LiDAR-only模式或不预加载：创建虚拟图像数据
            if is_lidar_only:
                print(f"LiDAR-only mode for camera {self.image_name}, skipping image loading")
            # 创建黑色虚拟图像用于兼容性
            self.original_image = torch.zeros((3, self.image_height, self.image_width), dtype=torch.float32).to(self.data_device)
            self.original_image_gray = torch.zeros((1, self.image_height, self.image_width), dtype=torch.float32).to(self.data_device)
            self.mask = None

        # 为LiDAR相机设置更大的远平面以适应KITTI-360数据集
        if is_lidar_only:
            self.zfar = 10000.0  # 10公里远平面，适应KITTI-360的距离范围
            self.znear = 0.1     # 稍大的近平面避免精度问题
        else:
            self.zfar = 100.0    # 普通相机使用标准远平面
            self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.plane_mask, self.non_plane_mask = None, None

    def get_lidar_data(self):
        """
        获取LiDAR数据（参考lidar-rt的设计）
        
        Returns:
            dict: 包含range_image、intensity_image等LiDAR数据
        """
        if not self.is_lidar_camera:
            return {}
        
        lidar_dict = {
            'range_image': self.range_image,
            'intensity_image': self.intensity_image,
            'valid_mask': self.valid_mask,
            'is_lidar_camera': self.is_lidar_camera
        }
        
        # 添加FOV信息
        if self.horizontal_fov_start is not None:
            lidar_dict['horizontal_fov_start'] = self.horizontal_fov_start
        if self.horizontal_fov_end is not None:
            lidar_dict['horizontal_fov_end'] = self.horizontal_fov_end
        
        # 添加原始LiDAR数据中的额外字段
        if self.lidar_data is not None:
            if 'points' in self.lidar_data:
                lidar_dict['points'] = self.lidar_data['points']
            if 'pose' in self.lidar_data:
                lidar_dict['pose'] = self.lidar_data['pose']
            if 'timestamp' in self.lidar_data:
                lidar_dict['timestamp'] = self.lidar_data['timestamp']
            
        return lidar_dict
    
    def get_lidar_rays(self, scale=1.0):
        """
        获取LiDAR光线（参考lidar-rt的设计）
        
        Args:
            scale: 缩放因子
            
        Returns:
            ray_origins: 光线起点 (H, W, 3)
            ray_directions: 光线方向 (H, W, 3)
        """
        if not self.is_lidar_camera:
            return None, None
        
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        
        # 创建像素坐标网格
        u, v = torch.meshgrid(
            torch.arange(W, device=self.data_device, dtype=torch.float32),
            torch.arange(H, device=self.data_device, dtype=torch.float32),
            indexing='xy'
        )
        
        # 检查是否是360度LiDAR
        if abs(self.FoVx - 2 * np.pi) < 0.1:  # 360度
            # 球坐标投影
            azimuth = (u / W) * 2 * np.pi - np.pi  # -π 到 π
            inclination = (v / H) * self.FoVy - (self.FoVy / 2)  # 根据垂直FOV计算
            
            # 计算光线方向
            ray_d_x = torch.cos(inclination) * torch.cos(azimuth)
            ray_d_y = torch.cos(inclination) * torch.sin(azimuth)
            ray_d_z = torch.sin(inclination)
            
            ray_directions = torch.stack([ray_d_x, ray_d_y, ray_d_z], -1)
        else:
            # 普通透视投影
            ray_d_x = (u - W/2) / (W/2) * torch.tan(torch.tensor(self.FoVx/2))
            ray_d_y = (v - H/2) / (H/2) * torch.tan(torch.tensor(self.FoVy/2))
            ray_d_z = torch.ones_like(u)
            
            ray_directions = torch.stack([ray_d_x, ray_d_y, ray_d_z], -1)
            ray_directions = torch.nn.functional.normalize(ray_directions, dim=-1)
        
        # 光线起点（相机中心）
        ray_origins = self.camera_center.expand(H, W, 3)
        
        return ray_origins, ray_directions
    
    def is_valid_lidar_pixel(self, u, v):
        """
        检查像素是否为有效的LiDAR像素
        
        Args:
            u, v: 像素坐标
            
        Returns:
            bool: 是否有效
        """
        if not self.is_lidar_camera or self.valid_mask is None:
            return False
        
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return self.valid_mask[v, u].item()
        return False
    
    def get_depth_at_pixel(self, u, v):
        """
        获取指定像素的深度值
        
        Args:
            u, v: 像素坐标
            
        Returns:
            float: 深度值，如果无效返回0
        """
        if not self.is_lidar_camera or self.range_image is None:
            return 0.0
        
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return self.range_image[v, u].item()
        return 0.0
    
    def get_intensity_at_pixel(self, u, v):
        """
        获取指定像素的强度值
        
        Args:
            u, v: 像素坐标
            
        Returns:
            float: 强度值，如果无效返回0
        """
        if not self.is_lidar_camera or self.intensity_image is None:
            return 0.0
        
        if 0 <= u < self.image_width and 0 <= v < self.image_height:
            return self.intensity_image[v, u].item()
        return 0.0

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.original_image_gray.cuda()
        else:
            gt_image, gray_image, _ = process_image(self.image_path, self.resolution, self.ncc_scale)
            return gt_image.cuda(), gray_image.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def sample_cam(cam_l: Camera, cam_r: Camera):
    cam = copy.copy(cam_l)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam_l.R.transpose()
    Rt[:3, 3] = cam_l.T
    Rt[3, 3] = 1.0

    Rt2 = np.zeros((4, 4))
    Rt2[:3, :3] = cam_r.R.transpose()
    Rt2[:3, 3] = cam_r.T
    Rt2[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    C2W2 = np.linalg.inv(Rt2)
    w = np.random.rand()
    pose_c2w_at_unseen =  w * C2W + (1 - w) * C2W2
    Rt = np.linalg.inv(pose_c2w_at_unseen)
    cam.R = Rt[:3, :3]
    cam.T = Rt[:3, 3]

    cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam
