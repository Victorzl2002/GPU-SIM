"""
vGPU三维资源模型核心模块

目前仿真所需仅依赖 `VGPUResource` 和 `ResourceType`，其余旧版
映射/分配器已删除，避免与 GLB/Sandbox 的资源流重复。
"""

from .vgpu_resource import VGPUResource, ResourceType

__all__ = ["VGPUResource", "ResourceType"]
