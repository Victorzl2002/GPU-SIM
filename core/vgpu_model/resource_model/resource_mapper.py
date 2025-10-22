"""
资源映射器

实现任务资源需求到vGPU资源的映射
"""

from typing import Dict, Any, List
from .vgpu_resource import VGPUResource


class ResourceMapper:
    """资源映射器"""
    
    def __init__(self):
        self._mapping_rules: Dict[str, Dict[str, float]] = {}
        self._load_default_mappings()
    
    def _load_default_mappings(self):
        """加载默认映射规则"""
        # 深度学习任务映射规则
        self._mapping_rules['deep_learning'] = {
            'compute_ratio': 0.8,    # 算力需求比例
            'memory_ratio': 0.9,     # 显存需求比例
            'bandwidth_ratio': 0.7   # 带宽需求比例
        }
        
        # 推理任务映射规则
        self._mapping_rules['inference'] = {
            'compute_ratio': 0.6,
            'memory_ratio': 0.5,
            'bandwidth_ratio': 0.8
        }
        
        # 训练任务映射规则
        self._mapping_rules['training'] = {
            'compute_ratio': 0.9,
            'memory_ratio': 0.95,
            'bandwidth_ratio': 0.6
        }
    
    def map_task_to_resource(self, task_type: str, 
                           available_resource: VGPUResource) -> VGPUResource:
        """
        将任务类型映射到资源需求
        
        Args:
            task_type: 任务类型
            available_resource: 可用资源
            
        Returns:
            映射后的资源需求
        """
        if task_type not in self._mapping_rules:
            raise ValueError(f"不支持的任务类型: {task_type}")
        
        rules = self._mapping_rules[task_type]
        
        return VGPUResource(
            compute=available_resource.compute * rules['compute_ratio'],
            memory=available_resource.memory * rules['memory_ratio'],
            bandwidth=available_resource.bandwidth * rules['bandwidth_ratio'],
            resource_id=f"{task_type}_mapped",
            vendor=available_resource.vendor,
            model=available_resource.model
        )
    
    def add_mapping_rule(self, task_type: str, 
                        compute_ratio: float, 
                        memory_ratio: float, 
                        bandwidth_ratio: float) -> None:
        """添加新的映射规则"""
        self._mapping_rules[task_type] = {
            'compute_ratio': compute_ratio,
            'memory_ratio': memory_ratio,
            'bandwidth_ratio': bandwidth_ratio
        }
    
    def list_task_types(self) -> List[str]:
        """列出所有支持的任务类型"""
        return list(self._mapping_rules.keys())
