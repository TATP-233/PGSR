# PGSR 核心技术要点

## 方法概述

PGSR (Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction) 是一种基于平面化高斯的表面重建方法，主要解决3DGS在几何重建方面的不足。

## 核心创新

### 1. 平面化高斯表示 (Planar-based Gaussian Splatting)

**问题**: 原始3DGS的无序高斯点云难以保证几何精度和多视图一致性

**解决方案**: 将3D高斯压缩为2D平面表示
```python
# 压缩最小尺度因子
Ls = ||min(s1, s2, s3)||1

# 法向量计算
ni = 最小尺度因子对应的方向

# 平面距离
di = (Rc^T(μi - Tc))^T(Rc^T ni)
```

### 2. 无偏深度渲染 (Unbiased Depth Rendering)

**核心思想**: 渲染平面参数而非直接深度，避免curved surface问题

**实现流程**:
1. 渲染法向量图: `N = Σ(Rc^T ni αi Ti)`
2. 渲染距离图: `D = Σ(di αi Ti)`  
3. 计算深度: `Depth(p) = D / (N(p) K^-1 p~)`

**优势**:
- 深度点位于高斯平面上，与几何一致
- 消除权重累积的偏差影响

### 3. 几何正则化

#### 3.1 单视图正则化
**局部平面假设**: 相邻像素属于同一平面

```python
# 计算局部法向量
Nd(p) = (P1 - P0) × (P3 - P2) / ||(P1 - P0) × (P3 - P2)||

# 边缘感知损失
Lsvgeom = (1/W) Σ (1 - ∇I)^2 ||Nd(p) - N(p)||1
```

#### 3.2 多视图正则化
**几何一致性**: 通过单应性变换确保多视图几何一致性

```python
# 单应性矩阵
Hrn = Kn(Rrn - Trn*nr^T/dr) * Kr^-1

# 几何一致性损失
Lmvgeom = (1/V) Σ w(pr) * φ(pr)
其中 φ(pr) = ||pr - Hnr*Hrn*pr||
```

**光度一致性**: 使用NCC确保patch一致性
```python
Lmvrgb = (1/V) Σ w(pr) * (1 - NCC(Ir(pr), In(Hrn*pr)))
```

### 4. 曝光补偿模型

```python
# 每张图像的曝光系数
Ii_a = exp(ai) * Ii_r + bi

# 自适应损失选择
I~ = Ii_a if LSSIM(Ii_r - Ii) < 0.5 else Ii_r
```

## 损失函数设计

```python
# 总损失
L = Lrgb + λ1*Ls + Lgeom

# 几何损失
Lgeom = λ2*Lsvgeom + λ3*Lmvrgb + λ4*Lmvgeom

# RGB损失  
Lrgb = (1-λ)*L1(I~ - Ii) + λ*LSSIM(Ii_r - Ii)
```

**参数设置**:
- λ1 = 100 (平面化权重)
- λ = 0.2 (SSIM权重)
- λ2 = 0.015 (单视图几何)
- λ3 = 0.15 (多视图光度)
- λ4 = 0.03 (多视图几何)

## 训练流程

1. **初始化**: 使用SfM点云初始化高斯
2. **平面化**: 通过Ls损失压缩高斯到平面
3. **深度监督**: 使用真实深度指导平面参数学习
4. **几何正则化**: 单视图和多视图约束确保一致性
5. **表面提取**: 使用TSDF融合提取mesh

## 关键代码模块

### 核心渲染函数
```python
def render_depth_normal(viewpoint_camera, pc, pipe, bg_color):
    """渲染深度和法向量"""
    # 1. 计算平面参数
    normals = compute_plane_normals(pc)
    distances = compute_plane_distances(pc, viewpoint_camera)
    
    # 2. α-blending渲染
    rendered_normal = alpha_blend(normals, alphas, Ts)
    rendered_distance = alpha_blend(distances, alphas, Ts)
    
    # 3. 计算无偏深度
    depth = rendered_distance / (rendered_normal @ K_inv @ p_tilde)
    
    return depth, rendered_normal, rendered_distance
```

### 几何损失函数
```python
def compute_geometric_loss(depth, normal, image_grad):
    """计算几何正则化损失"""
    # 单视图损失
    local_normal = compute_local_normal_from_depth(depth)
    sv_loss = torch.mean((1 - image_grad)**2 * torch.abs(local_normal - normal))
    
    # 多视图损失 (需要相邻帧)
    mv_loss = compute_multiview_consistency(...)
    
    return sv_loss, mv_loss
```

## 实现要点

1. **内存优化**: 深度图渲染比RGB渲染内存消耗更大
2. **数值稳定性**: 平面参数计算需要注意除零问题
3. **边缘处理**: 使用图像梯度加权避免边缘区域的几何约束
4. **多视图选择**: 相邻帧选择策略影响几何一致性效果

## 与原始3DGS的区别

| 方面 | 3DGS | PGSR |
|------|------|------|
| 几何表示 | 3D椭球高斯 | 2D平面高斯 |
| 深度渲染 | 直接Z值混合 | 平面参数计算 |
| 几何约束 | 无 | 单/多视图正则化 |
| 表面质量 | 较差 | 高精度 |
| 训练复杂度 | 简单 | 较复杂 | 