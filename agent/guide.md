# LiDAR-PGSR 项目指南

## 项目概述

本项目的目标是将PGSR（Planar-based Gaussian Splatting for Surface Reconstruction）与LiDAR-RT（Gaussian-based Ray Tracing for Dynamic LiDAR Re-simulation）两篇文章的核心技术相结合，实现基于Gaussian Splatting的高精度LiDAR模拟系统。

### 核心思想

LiDAR-RT原有方法存在高斯基元"飘在空中"的问题，导致模拟精度不足。通过引入PGSR的平面化高斯表示和几何约束，可以有效提升LiDAR模拟的几何精度和真实感。

### 技术架构

```
PGSR (基础架构)
├── Planar-based Gaussian Splatting
├── Unbiased Depth Rendering  
├── Geometric Regularization
└── Multi-view Consistency

+

LiDAR-RT (功能模块)
├── LiDAR数据集读取
├── Ray-drop和Intensity建模
├── 深度监督训练
└── Spherical Harmonics表示
```

## 主要贡献点

1. **几何精度提升**: 使用PGSR的平面化高斯和几何约束替代原始3D高斯，解决"飘浮"问题
2. **深度监督**: 将RGB图像监督替换为基于LiDAR的深度图监督
3. **物理建模**: 保留LiDAR-RT的ray-drop和intensity物理建模能力
4. **静态场景重建**: 首先专注于KITTI-360数据集的静态场景

## 项目结构

```
PGSR/ (主代码库)
├── agent/                  # 项目管理文件
│   ├── guide.md            # 本文件 - 项目指南
│   ├── pgsr_simplified.md  # PGSR核心技术要点
│   ├── lidar_rt_simplified.md # LiDAR-RT核心技术要点
│   ├── tasks.md            # 任务清单
│   └── log.md              # 工作日志
├── scene/                  # 场景表示相关
├── gaussian_renderer/      # 渲染器相关
├── utils/                  # 工具函数
├── train.py               # 主训练脚本
└── lidar-rt/              # LiDAR-RT原始代码库
```

## 技术要点

### PGSR核心技术
- **平面化高斯**: 将3D高斯压缩为2D平面表示
- **无偏深度渲染**: 通过平面参数计算精确深度
- **几何正则化**: 单视图和多视图几何一致性约束
- **表面重建**: 高保真度表面重建能力

### LiDAR-RT核心技术
- **物理建模**: Ray-drop概率和反射强度建模
- **球谐函数**: 用SH系数表示方向相关的物理属性
- **深度监督**: 基于LiDAR点云的深度监督训练
- **KITTI-360数据**: 处理64线激光雷达数据

## 开发阶段

### 阶段1: 环境搭建和代码理解
- 搭建开发环境
- 分析两个代码库的核心模块
- 理解数据格式和处理流程

### 阶段2: 核心功能迁移
- 将LiDAR数据读取模块集成到PGSR
- 实现深度监督训练替代RGB监督
- 集成ray-drop和intensity建模

### 阶段3: 几何约束集成
- 将PGSR的平面化约束应用到LiDAR场景
- 实现几何正则化损失函数
- 优化表面重建质量

### 阶段4: 测试和优化
- 在KITTI-360数据集上测试
- 性能优化和参数调节
- 结果分析和对比

## 关键挑战

1. **数据格式转换**: 将LiDAR点云转换为适合PGSR训练的深度图
2. **损失函数设计**: 平衡几何约束和LiDAR物理建模
3. **内存管理**: 处理大规模点云数据的内存优化
4. **渲染管线**: 集成两种不同的渲染方法

## 成功指标

1. **几何精度**: 相比原始LiDAR-RT提升表面重建精度
2. **物理真实性**: 保持ray-drop和intensity建模能力
3. **训练效率**: 合理的训练时间和内存占用
4. **渲染质量**: 高质量的LiDAR仿真结果

## 下一步行动

请查看`tasks.md`文件了解详细的任务分解和当前进度，查看`log.md`了解最新的工作状态。开始开发前，建议先阅读`pgsr_simplified.md`和`lidar_rt_simplified.md`了解两个方法的核心技术细节。 