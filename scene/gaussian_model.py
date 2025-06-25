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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_scaling
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix

def dilate(bin_img, ksize=5):
    pad = (ksize - 1) // 2
    bin_img = torch.nn.functional.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
    out = torch.nn.functional.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
    return out

def erode(bin_img, ksize=5):
    out = 1 - dilate(1 - bin_img, ksize)
    return out

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._knn_f = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_abs = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.knn_dists = None
        self.knn_idx = None
        
        # LiDAR专用属性 - 使用球谐函数表示intensity和raydrop的方向相关性
        self.lidar_sh_degree = 2  # 使用2阶球谐函数
        self._intensity_sh = torch.empty(0)  # intensity的球谐系数 
        self._raydrop_sh = torch.empty(0)    # raydrop概率的球谐系数
        
        self.setup_functions()
        self.use_app = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._knn_f,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.max_weight,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.denom_abs,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._knn_f,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        self.max_weight,
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        denom,
        denom_abs,
        opt_dict, 
        self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.denom_abs = denom_abs
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
        
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property 
    def get_intensity_sh(self):
        return self._intensity_sh
            
    @property
    def get_raydrop_sh(self):
        return self._raydrop_sh
    
    def get_intensity(self, viewdirs):
        """
        计算给定视线方向的intensity值
        viewdirs: (N, 3) 归一化的视线方向
        """
        from utils.sh_utils import eval_sh
        # 添加通道维度，使其与球谐函数兼容
        # intensity_sh形状从 (N, 9) 变为 (N, 1, 9)
        intensity_sh = self._intensity_sh.unsqueeze(1)  # (N, 1, 9)
        
        # 使用球谐函数计算intensity
        intensity = eval_sh(self.lidar_sh_degree, intensity_sh, viewdirs)  # (N, 1)
        return torch.sigmoid(intensity)  # 确保intensity在[0,1]范围内
    
    def get_raydrop_prob(self, viewdirs):
        """
        计算给定视线方向的raydrop概率
        viewdirs: (N, 3) 归一化的视线方向
        """
        from utils.sh_utils import eval_sh
        # 添加通道维度，使其与球谐函数兼容
        # raydrop_sh形状从 (N, 9) 变为 (N, 1, 9)
        raydrop_sh = self._raydrop_sh.unsqueeze(1)  # (N, 1, 9)
        
        # 使用球谐函数计算raydrop概率
        raydrop = eval_sh(self.lidar_sh_degree, raydrop_sh, viewdirs)  # (N, 1)
        return torch.sigmoid(raydrop)  # 确保概率在[0,1]范围内
    
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    
    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = torch.sqrt(torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001))
        
        # 计算点到原点的距离（作为深度的代理）
        point_depths = torch.norm(fused_point_cloud, dim=1)
        
        # 为KITTI-360数据集设置更合适的最小尺度
        # 原问题：torch.log(dist)如果dist很小会产生很大的负值，导致exp(scales)极小
        # 新方案：基于深度自适应设置尺度，确保远距离点有足够大的投影尺度
        min_dist = 0.1  # 最小距离设为0.1米
        dist = torch.clamp(dist, min=min_dist)
        
        # 自适应尺度：对于距离很远的点，增加基础尺度
        # 目标：在远距离处保证投影半径至少1-2像素
        # 假设焦距约500像素，目标投影半径2像素
        # scale * focal / depth >= 2
        # scale >= 2 * depth / focal
        target_radius_pixels = 2.0
        focal_approx = 500.0  # 近似焦距
        
        # 计算所需的最小尺度
        required_scale = target_radius_pixels * point_depths / focal_approx
        
        # 使用距离和深度的较大值作为基础尺度
        adaptive_scale = torch.maximum(dist, required_scale * 0.3)  # 使用30%的所需尺度作为基础
        
        print(f"点云距离统计: min={dist.min():.4f}, max={dist.max():.4f}, mean={dist.mean():.4f}")
        print(f"点云深度统计: min={point_depths.min():.4f}, max={point_depths.max():.4f}, mean={point_depths.mean():.4f}")
        print(f"所需尺度统计: min={required_scale.min():.4f}, max={required_scale.max():.4f}, mean={required_scale.mean():.4f}")
        print(f"自适应尺度统计: min={adaptive_scale.min():.4f}, max={adaptive_scale.max():.4f}, mean={adaptive_scale.mean():.4f}")
        
        # 为LiDAR增加额外的尺度缩放因子，解决可见点太少的问题
        lidar_scale_multiplier = 5.0  # 增大5倍尺度
        scales = torch.log(adaptive_scale * lidar_scale_multiplier)[...,None].repeat(1, 3)
        print(f"初始尺度统计: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")
        print(f"exp(scales)统计: min={torch.exp(scales).min():.6f}, max={torch.exp(scales).max():.6f}")
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        knn_f = torch.randn((fused_point_cloud.shape[0], 6)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._knn_f = nn.Parameter(knn_f.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 初始化LiDAR属性
        num_points = fused_point_cloud.shape[0]
        # intensity和raydrop的球谐系数数量
        lidar_sh_coeffs = (self.lidar_sh_degree + 1) ** 2
        
        # 初始化intensity球谐系数 (默认中等强度)
        intensity_sh = torch.zeros((num_points, lidar_sh_coeffs), device="cuda")
        intensity_sh[:, 0] = inverse_sigmoid(torch.ones(num_points, device="cuda") * 0.5)  # DC分量设为0.5
        self._intensity_sh = nn.Parameter(intensity_sh.requires_grad_(True))
        
        # 初始化raydrop球谐系数 (默认低概率)
        raydrop_sh = torch.zeros((num_points, lidar_sh_coeffs), device="cuda") 
        raydrop_sh[:, 0] = inverse_sigmoid(torch.ones(num_points, device="cuda") * 0.1)  # DC分量设为0.1
        self._raydrop_sh = nn.Parameter(raydrop_sh.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.abs_split_radii2D_threshold = training_args.abs_split_radii2D_threshold
        self.max_abs_split_points = training_args.max_abs_split_points
        self.max_all_points = training_args.max_all_points
        
        # 确保所有参数都在CUDA设备上
        if not self._xyz.is_cuda:
            print(f"[WARNING] Moving _xyz from {self._xyz.device} to cuda")
            self._xyz = self._xyz.cuda()
        if not self._knn_f.is_cuda:
            print(f"[WARNING] Moving _knn_f from {self._knn_f.device} to cuda")
            self._knn_f = self._knn_f.cuda()
        if not self._scaling.is_cuda:
            print(f"[WARNING] Moving _scaling from {self._scaling.device} to cuda")
            self._scaling = self._scaling.cuda()
        if not self._rotation.is_cuda:
            print(f"[WARNING] Moving _rotation from {self._rotation.device} to cuda")
            self._rotation = self._rotation.cuda()
        if not self._opacity.is_cuda:
            print(f"[WARNING] Moving _opacity from {self._opacity.device} to cuda")
            self._opacity = self._opacity.cuda()
        if not self._intensity_sh.is_cuda:
            print(f"[WARNING] Moving _intensity_sh from {self._intensity_sh.device} to cuda")
            self._intensity_sh = self._intensity_sh.cuda()
        if not self._raydrop_sh.is_cuda:
            print(f"[WARNING] Moving _raydrop_sh from {self._raydrop_sh.device} to cuda")
            self._raydrop_sh = self._raydrop_sh.cuda()
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._knn_f], 'lr': 0.01, "name": "knn_f"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._intensity_sh], 'lr': training_args.feature_lr / 20.0, "name": "intensity_sh"},
            {'params': [self._raydrop_sh], 'lr': training_args.feature_lr / 20.0, "name": "raydrop_sh"}
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    
    def clip_grad(self, norm=1.0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, mask=None):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._intensity_sh = optimizable_tensors["intensity_sh"]
        self._raydrop_sh = optimizable_tensors["raydrop_sh"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            
            # 确保extension_tensor在正确的设备上
            existing_param = group["params"][0]
            if extension_tensor.device != existing_param.device:
                extension_tensor = extension_tensor.to(existing_param.device)
            
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor).to(stored_state["exp_avg"].device)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor).to(stored_state["exp_avg_sq"].device)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_knn_f, new_scaling, new_rotation, new_opacity, new_intensity_sh, new_raydrop_sh):
        # 确保所有张量都在正确的设备上
        device = self.get_xyz.device
        d = {"xyz": new_xyz.to(device),
        "knn_f": new_knn_f.to(device),
        "scaling" : new_scaling.to(device),
        "rotation" : new_rotation.to(device),
        "opacity": new_opacity.to(device),
        "intensity_sh": new_intensity_sh.to(device),
        "raydrop_sh": new_raydrop_sh.to(device)}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._knn_f = optimizable_tensors["knn_f"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._opacity = optimizable_tensors["opacity"]
        self._intensity_sh = optimizable_tensors["intensity_sh"]
        self._raydrop_sh = optimizable_tensors["raydrop_sh"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device="cuda")
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device="cuda")
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3), device=self.get_xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_knn_f = self._knn_f[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_intensity_sh = self._intensity_sh[selected_pts_mask].repeat(N,1)
        new_raydrop_sh = self._raydrop_sh[selected_pts_mask].repeat(N,1)

        # 如果有要分裂的点，才进行densification
        if len(new_xyz) > 0:
            self.densification_postfix(new_xyz, new_knn_f, new_scaling, new_rotation, new_opacity, new_intensity_sh, new_raydrop_sh)

            # 修复：densification_postfix之后，模型包含原始点+新增点
            # prune_filter应该标记要删除的原始点（选中的分裂点）
            # 而新增的点保留（不删除）
            current_n_points = self.get_xyz.shape[0]
            expected_new_points = len(new_xyz)
            
            if current_n_points == n_init_points + expected_new_points:
                # 创建prune filter：删除被分裂的原始点，保留新增点
                prune_filter = torch.cat((selected_pts_mask, torch.zeros(expected_new_points, device=selected_pts_mask.device, dtype=bool)))
                self.prune_points(prune_filter)
            else:
                print(f"[DEBUG] Unexpected point count after densification: {current_n_points}, expected {n_init_points + expected_new_points}")
        else:
            print(f"[DEBUG] No points selected for splitting")

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3), device=self.get_xyz.device)
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_knn_f = self._knn_f[selected_pts_mask]
            new_opacity = self._opacity[selected_pts_mask]
            new_intensity_sh = self._intensity_sh[selected_pts_mask]
            new_raydrop_sh = self._raydrop_sh[selected_pts_mask]

            self.densification_postfix(new_xyz, new_knn_f, new_scaling, new_rotation, new_opacity, new_intensity_sh, new_raydrop_sh)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size):
        # 检查梯度累积器是否已初始化
        if len(self.xyz_gradient_accum) == 0 or len(self.denom) == 0:
            print(f"[DEBUG] Skipping densify_and_prune: gradient accumulators not initialized")
            return
            
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        # 重要：在densify_and_clone和densify_and_split之间，点的数量可能会变化
        # 因此需要分别计算初始点数
        self.densify_and_clone(grads, max_grad, extent)
        
        # 重新获取当前点的数量用于densify_and_split
        # 因为densify_and_clone可能已经改变了点的数量
        if len(self.xyz_gradient_accum) != self.get_xyz.shape[0]:
            # 如果梯度累积器的大小与当前点数不匹配，重新计算梯度
            print(f"[DEBUG] Point count changed after clone, skipping split")
        else:
            self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        # 如果梯度累积器为空，重新初始化（第一次有可见点时）
        if len(self.xyz_gradient_accum) == 0 and update_filter.sum() > 0:
            num_points = self.get_xyz.shape[0]
            print(f"[DEBUG] Reinitializing gradient accumulators for {num_points} points")
            self.xyz_gradient_accum = torch.zeros((num_points, 1), device="cuda")
            self.xyz_gradient_accum_abs = torch.zeros((num_points, 1), device="cuda")
            self.denom = torch.zeros((num_points, 1), device="cuda")
            self.denom_abs = torch.zeros((num_points, 1), device="cuda")
        
        # 检查形状是否匹配
        if len(self.xyz_gradient_accum) == 0 or len(self.xyz_gradient_accum_abs) == 0:
            print(f"[DEBUG] Skipping densification stats due to empty gradient accumulators")
            return
        
        if viewspace_point_tensor.grad is None or viewspace_point_tensor_abs.grad is None:
            print(f"[DEBUG] Skipping densification stats due to None gradients")
            return
        
        if update_filter.sum() == 0:
            print(f"[DEBUG] Skipping densification stats due to no visible points")
            return
        
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts
    
    def compute_planar_loss(self):
        """
        计算PGSR平面化约束损失
        鼓励高斯基元压缩为平面状（最小缩放值接近0）
        
        Returns:
            loss: 平面化损失值
        """
        scaling = self.get_scaling  # (N, 3)
        
        # 计算最小缩放值的L1损失
        min_scales = scaling.min(dim=1)[0]  # (N,)
        planar_loss = min_scales.mean()
        
        return planar_loss
    
    def compute_sv_geometry_loss(self, rendered_normal, depth_normal, image_grad=None):
        """
        计算PGSR单视图几何正则化损失
        
        Args:
            rendered_normal: (3, H, W) 渲染的法向量
            depth_normal: (3, H, W) 从深度计算的法向量  
            image_grad: (H, W) 图像梯度（可选，用于边缘感知）
            
        Returns:
            loss: 单视图几何损失
        """
        if rendered_normal is None or depth_normal is None:
            return torch.tensor(0.0, device="cuda")
            
        # 计算法向量差异
        normal_diff = torch.abs(rendered_normal - depth_normal).sum(0)  # (H, W)
        
        if image_grad is not None:
            # 边缘感知权重：在图像边缘处减少几何约束
            edge_weight = (1.0 - image_grad).clamp(0, 1) ** 2
            sv_loss = (edge_weight * normal_diff).mean()
        else:
            sv_loss = normal_diff.mean()
            
        return sv_loss
    