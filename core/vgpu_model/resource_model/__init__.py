"""
vGPU三维资源模型核心模块

实现统一的vGPU资源抽象，支持Compute、Memory、Bandwidth三维资源建模
"""

from .vgpu_resource import VGPUResource, ResourceType
from .resource_mapper import ResourceMapper
from .resource_allocator import ResourceAllocator

__all__ = [
    'VGPUResource',
    'ResourceType', 
    'ResourceMapper',
    'ResourceAllocator'
]
