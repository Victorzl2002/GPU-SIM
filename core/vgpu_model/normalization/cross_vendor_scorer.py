"""
跨厂商得分计算器

实现跨厂商平台的得分计算（基础计算能力）
注意：调度决策相关功能请使用 core.scheduling.vendor_select.VendorSelector
"""

from typing import List, Dict, Any, Tuple
from .normalization_coefficients import CoefficientManager
from ..resource_model.vgpu_resource import VGPUResource


class CrossVendorScorer:
    """
    跨厂商得分计算器
    
    提供基础的跨厂商得分计算能力，不包含调度决策逻辑。
    调度决策请使用 core.scheduling.vendor_select.VendorSelector
    """
    
    def __init__(self):
        self.coefficient_manager = CoefficientManager()
    
    def calculate_cross_vendor_score(self, task: VGPUResource, 
                                   gpu: VGPUResource,
                                   performance_weight: float = 1.0) -> float:
        """
        计算跨厂商得分
        
        根据公式：score_{i,v} = perf_v / cost_{i,v}
        其中 cost_{i,v} = c_i/C^eff + m_i/M^eff + b_i/B^eff
        C^eff = αC, M^eff = βM, B^eff = γB
        
        注意：performance_weight 统一为 1.0，因为软件栈折算系数（α, β, γ）
        已经考虑了不同厂商GPU的性能差异。通过折算系数计算有效可用容量后，
        所有平台已标准化，因此 perf_v = 1.0。
        
        Args:
            task: 任务资源需求 d_i = (c_i, m_i, b_i)
            gpu: GPU资源
            performance_weight: 性能权重 perf_v（统一为1.0，因为折算系数已考虑性能差异）
            
        Returns:
            标准化得分 score_{i,v}
        """
        # 获取折算系数
        coeff = self.coefficient_manager.get_coefficients(
            gpu.vendor, gpu.model
        )
        
        if coeff is None:
            raise ValueError(f"未找到厂商 {gpu.vendor} 型号 {gpu.model} 的折算系数")
        
        # 计算有效可用容量 D^eff=(αC, βM, γB)
        effective_capacity = VGPUResource(
            compute=gpu.compute * coeff.alpha,
            memory=gpu.memory * coeff.beta,
            bandwidth=gpu.bandwidth * coeff.gamma,
            resource_id=f"{gpu.resource_id}_effective",
            vendor=gpu.vendor,
            model=gpu.model
        )
        
        # 计算得分 score = perf / Σ(c/C^eff + m/M^eff + b/B^eff)
        return task.calculate_score(effective_capacity, performance_weight)
    
    def compare_resources(self, task: VGPUResource,
                         resources: List[VGPUResource], 
                         performance_weight: float = 1.0) -> List[Tuple[VGPUResource, float]]:
        """
        比较多个资源的得分
        
        Args:
            task: 任务资源需求
            resources: GPU资源列表
            performance_weight: 性能权重
            
        Returns:
            (资源, 得分) 的列表，按得分降序排列
            
        Note:
            此方法用于基础比较，调度决策请使用 VendorSelector
        """
        scored_resources = []
        
        for resource in resources:
            try:
                score = self.calculate_cross_vendor_score(task, resource, performance_weight)
                scored_resources.append((resource, score))
            except ValueError as e:
                print(f"警告: 跳过资源 {resource.resource_id}: {e}")
                continue
        
        # 按得分降序排列
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        return scored_resources
    
    def find_best_resource(self, task: VGPUResource,
                          resources: List[VGPUResource], 
                          performance_weight: float = 1.0) -> Tuple[VGPUResource, float]:
        """
        找到最佳资源
        
        Args:
            task: 任务资源需求
            resources: GPU资源列表
            performance_weight: 性能权重
            
        Returns:
            (最佳资源, 最高得分)
            
        Note:
            此方法用于基础选择，调度决策请使用 VendorSelector
        """
        if not resources:
            raise ValueError("资源列表不能为空")
        
        scored_resources = self.compare_resources(task, resources, performance_weight)
        
        if not scored_resources:
            raise ValueError("没有可用的资源")
        
        return scored_resources[0]
    
    def get_resource_ranking(self, task: VGPUResource,
                           resources: List[VGPUResource], 
                           performance_weight: float = 1.0) -> Dict[str, Any]:
        """
        获取资源排名
        
        Args:
            task: 任务资源需求
            resources: GPU资源列表
            performance_weight: 性能权重
            
        Returns:
            排名信息
            
        Note:
            此方法用于基础排名，调度决策请使用 VendorSelector
        """
        scored_resources = self.compare_resources(task, resources, performance_weight)
        
        ranking = {
            'total_resources': len(resources),
            'valid_resources': len(scored_resources),
            'ranking': []
        }
        
        for i, (resource, score) in enumerate(scored_resources):
            ranking['ranking'].append({
                'rank': i + 1,
                'resource_id': resource.resource_id,
                'vendor': resource.vendor,
                'model': resource.model,
                'score': score,
                'compute': resource.compute,
                'memory': resource.memory,
                'bandwidth': resource.bandwidth
            })
        return ranking
