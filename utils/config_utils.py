#!/usr/bin/env python3
"""
配置文件解析工具
支持YAML配置的层次继承和参数合并
"""

import yaml
import os
from types import SimpleNamespace
from typing import Dict, Any, Optional


class ConfigParser:
    """配置文件解析器，支持parent_config继承"""
    
    def __init__(self):
        self._loaded_configs = {}  # 缓存已加载的配置
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件，支持parent_config继承
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            合并后的配置字典
        """
        if config_path in self._loaded_configs:
            return self._loaded_configs[config_path]
        
        # 加载当前配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f) or {}
        
        # 检查是否有父配置
        if 'parent_config' in current_config:
            parent_path = current_config['parent_config']
            
            # 处理相对路径
            if not os.path.isabs(parent_path):
                config_dir = os.path.dirname(config_path)
                parent_path = os.path.join(config_dir, parent_path)
                parent_path = os.path.normpath(parent_path)
            
            # 递归加载父配置
            parent_config = self.load_config(parent_path)
            
            # 合并配置（当前配置覆盖父配置）
            merged_config = self._deep_merge(parent_config, current_config)
            
            # 移除parent_config键
            if 'parent_config' in merged_config:
                del merged_config['parent_config']
        else:
            merged_config = current_config
        
        # 缓存结果
        self._loaded_configs[config_path] = merged_config
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并两个配置字典
        
        Args:
            base: 基础配置（父配置）
            override: 覆盖配置（当前配置）
            
        Returns:
            合并后的配置字典
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并嵌套字典
                merged[key] = self._deep_merge(merged[key], value)
            else:
                # 直接覆盖
                merged[key] = value
        
        return merged
    
    def to_namespace(self, config: Dict[str, Any]) -> SimpleNamespace:
        """
        将配置字典转换为SimpleNamespace对象，支持点式访问
        
        Args:
            config: 配置字典
            
        Returns:
            SimpleNamespace对象
        """
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(item) for item in d]
            else:
                return d
        
        return dict_to_namespace(config)


def load_config(config_path: str) -> SimpleNamespace:
    """
    便捷函数：加载配置文件并返回SimpleNamespace对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置的SimpleNamespace对象
    """
    parser = ConfigParser()
    config_dict = parser.load_config(config_path)
    return parser.to_namespace(config_dict)


def save_config(config: SimpleNamespace, output_path: str):
    """
    保存配置到YAML文件
    
    Args:
        config: 配置对象
        output_path: 输出文件路径
    """
    def namespace_to_dict(obj):
        if isinstance(obj, SimpleNamespace):
            return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [namespace_to_dict(item) for item in obj]
        else:
            return obj
    
    config_dict = namespace_to_dict(config)
    
    # 只在有目录路径时创建目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def override_config(config: SimpleNamespace, overrides: Dict[str, Any]) -> SimpleNamespace:
    """
    使用命令行参数覆盖配置
    
    Args:
        config: 原始配置
        overrides: 覆盖参数字典
        
    Returns:
        更新后的配置
    """
    def set_nested_attr(obj, key_path: str, value: Any):
        """设置嵌套属性"""
        keys = key_path.split('.')
        for key in keys[:-1]:
            if not hasattr(obj, key):
                setattr(obj, key, SimpleNamespace())
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)
    
    # 创建配置副本
    config_dict = config.__dict__.copy()
    new_config = SimpleNamespace(**config_dict)
    
    # 应用覆盖
    for key, value in overrides.items():
        if '.' in key:
            set_nested_attr(new_config, key, value)
        else:
            setattr(new_config, key, value)
    
    return new_config


# 示例用法
if __name__ == "__main__":
    # 测试配置加载
    try:
        config = load_config("configs/quick_test.yaml")
        print("配置加载成功！")
        print(f"实验名称: {config.exp_name}")
        print(f"训练轮数: {config.opt.iterations}")
        print(f"LiDAR模式: {config.model.enable_lidar}")
    except Exception as e:
        print(f"配置加载失败: {e}") 