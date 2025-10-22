"""
vGPU资源模型定义

实现vGPU三维资源模型 ⟨Compute, Memory, Bandwidth⟩
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import math


class ResourceType(Enum):
    """资源类型枚举"""
    COMPUTE = "compute"      # 算力资源
    MEMORY = "memory"        # 显存资源  
    BANDWIDTH = "bandwidth"  # 带宽资源


@dataclass
class VGPUResource:
    """
    vGPU三维资源模型
    
    定义vGPU的三种核心资源：Compute、Memory、Bandwidth
    支持资源需求映射和跨厂商标准化
    """
    
    # 三维资源定义
    compute: float      # 算力资源 (TFLOPS)
    memory: float       # 显存资源 (GB)
    bandwidth: float    # 带宽资源 (GB/s)
    
    # 资源元数据
    resource_id: str = ""
    vendor: str = ""    # 厂商类型 (nvidia, huawei)
    model: str = ""     # GPU型号 (A100, Ascend910B)
    
    def __post_init__(self):
        """初始化后验证"""
        if self.compute < 0 or self.memory < 0 or self.bandwidth < 0:
            raise ValueError("资源值不能为负数")
        
        # 资源合理性验证
        if self.memory > 1000:  # 显存超过1TB不合理
            raise ValueError(f"显存大小不合理: {self.memory}GB")
        if self.compute > 1000:  # 算力超过1000 TFLOPS不合理
            raise ValueError(f"算力大小不合理: {self.compute} TFLOPS")
        if self.bandwidth > 10000:  # 带宽超过10TB/s不合理
            raise ValueError(f"带宽大小不合理: {self.bandwidth} GB/s")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'compute': self.compute,
            'memory': self.memory, 
            'bandwidth': self.bandwidth,
            'resource_id': self.resource_id,
            'vendor': self.vendor,
            'model': self.model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VGPUResource':
        """从字典创建实例"""
        return cls(
            compute=data['compute'],
            memory=data['memory'],
            bandwidth=data['bandwidth'],
            resource_id=data.get('resource_id', ''),
            vendor=data.get('vendor', ''),
            model=data.get('model', '')
        )
    
    def normalize(self, alpha: float, beta: float, gamma: float) -> 'VGPUResource':
        """
        应用软件栈折算系数进行标准化
        
        Args:
            alpha: Compute折算系数
            beta: Memory折算系数  
            gamma: Bandwidth折算系数
            
        Returns:
            标准化后的资源模型
        """
        return VGPUResource(
            compute=self.compute / alpha,
            memory=self.memory / beta,
            bandwidth=self.bandwidth / gamma,
            resource_id=self.resource_id,
            vendor=self.vendor,
            model=self.model
        )
    
    def calculate_score(self, alpha: float, beta: float, gamma: float, 
                       performance_weight: float = 1.0) -> float:
        """
        计算跨厂商平台得分
        
        Args:
            alpha: Compute折算系数
            beta: Memory折算系数
            gamma: Bandwidth折算系数
            performance_weight: 性能权重
            
        Returns:
            标准化得分: score = perf / Σ(c/α + m/β + b/γ)
        """
        normalized_resource = (self.compute / alpha + 
                             self.memory / beta + 
                             self.bandwidth / gamma)
        if normalized_resource == 0:
            return 0.0
        return performance_weight / normalized_resource
    
    def __add__(self, other: 'VGPUResource') -> 'VGPUResource':
        """资源相加"""
        return VGPUResource(
            compute=self.compute + other.compute,
            memory=self.memory + other.memory,
            bandwidth=self.bandwidth + other.bandwidth,
            resource_id=f"{self.resource_id}+{other.resource_id}",
            vendor=self.vendor,
            model=self.model
        )
    
    def __sub__(self, other: 'VGPUResource') -> 'VGPUResource':
        """资源相减"""
        return VGPUResource(
            compute=max(0, self.compute - other.compute),
            memory=max(0, self.memory - other.memory),
            bandwidth=max(0, self.bandwidth - other.bandwidth),
            resource_id=f"{self.resource_id}-{other.resource_id}",
            vendor=self.vendor,
            model=self.model
        )
    
    def __mul__(self, factor: float) -> 'VGPUResource':
        """资源缩放"""
        return VGPUResource(
            compute=self.compute * factor,
            memory=self.memory * factor,
            bandwidth=self.bandwidth * factor,
            resource_id=self.resource_id,
            vendor=self.vendor,
            model=self.model
        )
    
    def __truediv__(self, factor: float) -> 'VGPUResource':
        """资源分割"""
        if factor == 0:
            raise ValueError("除数不能为零")
        return VGPUResource(
            compute=self.compute / factor,
            memory=self.memory / factor,
            bandwidth=self.bandwidth / factor,
            resource_id=self.resource_id,
            vendor=self.vendor,
            model=self.model
        )
    
    def __str__(self) -> str:
        return f"VGPUResource(compute={self.compute:.2f}, memory={self.memory:.2f}, bandwidth={self.bandwidth:.2f})"
    
    def __repr__(self) -> str:
        return f"VGPUResource(compute={self.compute}, memory={self.memory}, bandwidth={self.bandwidth}, vendor={self.vendor}, model={self.model})"
