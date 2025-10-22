"""
资源分配器

实现vGPU资源的分配和管理
"""

from typing import List, Optional, Dict, Any
from .vgpu_resource import VGPUResource


class ResourceAllocator:
    """资源分配器"""
    
    def __init__(self):
        self._allocated_resources: Dict[str, VGPUResource] = {}
        self._available_resources: Dict[str, VGPUResource] = {}
    
    def add_available_resource(self, resource: VGPUResource) -> None:
        """添加可用资源"""
        self._available_resources[resource.resource_id] = resource
    
    def allocate_resource(self, resource_id: str, 
                         required_resource: VGPUResource) -> Optional[VGPUResource]:
        """
        分配资源
        
        Args:
            resource_id: 资源ID
            required_resource: 所需资源
            
        Returns:
            分配的资源，如果无法分配则返回None
        """
        if resource_id not in self._available_resources:
            return None
        
        available = self._available_resources[resource_id]
        
        # 检查资源是否足够
        if (available.compute >= required_resource.compute and
            available.memory >= required_resource.memory and
            available.bandwidth >= required_resource.bandwidth):
            
            # 分配资源
            allocated = VGPUResource(
                compute=required_resource.compute,
                memory=required_resource.memory,
                bandwidth=required_resource.bandwidth,
                resource_id=f"{resource_id}_allocated",
                vendor=available.vendor,
                model=available.model
            )
            
            # 更新可用资源
            remaining = available - allocated
            self._available_resources[resource_id] = remaining
            self._allocated_resources[allocated.resource_id] = allocated
            
            return allocated
        
        return None
    
    def release_resource(self, allocated_resource_id: str) -> bool:
        """
        释放资源
        
        Args:
            allocated_resource_id: 已分配资源的ID
            
        Returns:
            是否成功释放
        """
        if allocated_resource_id not in self._allocated_resources:
            return False
        
        allocated = self._allocated_resources[allocated_resource_id]
        
        # 找到原始资源ID
        original_id = allocated_resource_id.replace('_allocated', '')
        if original_id in self._available_resources:
            # 恢复可用资源
            self._available_resources[original_id] = (
                self._available_resources[original_id] + allocated
            )
        
        # 移除已分配资源
        del self._allocated_resources[allocated_resource_id]
        
        return True
    
    def get_available_resources(self) -> Dict[str, VGPUResource]:
        """获取所有可用资源"""
        return self._available_resources.copy()
    
    def get_allocated_resources(self) -> Dict[str, VGPUResource]:
        """获取所有已分配资源"""
        return self._allocated_resources.copy()
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        utilization = {}
        
        for resource_id, available in self._available_resources.items():
            # 计算已分配的资源
            allocated_compute = sum(
                r.compute for r in self._allocated_resources.values() 
                if r.vendor == available.vendor and r.model == available.model
            )
            allocated_memory = sum(
                r.memory for r in self._allocated_resources.values() 
                if r.vendor == available.vendor and r.model == available.model
            )
            allocated_bandwidth = sum(
                r.bandwidth for r in self._allocated_resources.values() 
                if r.vendor == available.vendor and r.model == available.model
            )
            
            # 计算利用率
            total_compute = available.compute + allocated_compute
            total_memory = available.memory + allocated_memory
            total_bandwidth = available.bandwidth + allocated_bandwidth
            
            utilization[resource_id] = {
                'compute_utilization': allocated_compute / total_compute if total_compute > 0 else 0,
                'memory_utilization': allocated_memory / total_memory if total_memory > 0 else 0,
                'bandwidth_utilization': allocated_bandwidth / total_bandwidth if total_bandwidth > 0 else 0
            }
        
        return utilization
