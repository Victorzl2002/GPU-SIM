"""
跨厂商选址决策器

实现一级调度：基于折算系数与平台得分选择最佳GPU平台
根据公式：score_{i,v} = perf_v / cost_{i,v}
其中：cost_{i,v} = c_i/α_v + m_i/β_v + b_i/γ_v
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.vgpu_model.normalization.cross_vendor_scorer import CrossVendorScorer


@dataclass
class Platform:
    """
    GPU平台表示
    
    包含平台的GPU资源和性能权重信息
    
    Note:
        perf_v 统一为 1.0，因为软件栈折算系数（α, β, γ）已经考虑了性能差异
        通过折算系数计算有效可用容量 D^eff = (αC, βM, γB) 后，所有平台已标准化
    """
    platform_id: str          # 平台ID（如 "nvidia_a100_pool", "huawei_ascend910b_pool"）
    vendor: str               # 厂商 (nvidia, huawei)
    model: str                # GPU型号 (A100, Ascend910B)
    gpu: VGPUResource         # GPU资源
    perf_v: float = 1.0       # 平台性能权重（统一为1.0，因为折算系数已考虑性能差异）
    pool_capacity: int = 1    # 资源池容量（GPU数量）
    
    def __post_init__(self):
        """初始化后验证"""
        if self.perf_v <= 0:
            raise ValueError("平台性能权重必须大于0")
        if self.pool_capacity <= 0:
            raise ValueError("资源池容量必须大于0")


@dataclass
class VendorSelectionResult:
    """
    跨厂商选址结果
    """
    selected_platform: Platform    # 选中的平台
    score: float                   # 平台得分
    cost: float                    # 等效成本
    all_scores: List[Tuple[Platform, float]]  # 所有平台的得分（用于调试）
    task: VGPUResource             # 任务需求


class VendorSelector:
    """
    跨厂商选址决策器
    
    实现一级调度：基于折算系数与平台得分选择最佳GPU平台
    """
    
    def __init__(self, scorer: Optional[CrossVendorScorer] = None):
        """
        初始化选址器
        
        Args:
            scorer: 跨厂商得分计算器，如果为None则创建新实例
        """
        self.scorer = scorer if scorer is not None else CrossVendorScorer()
    
    def calculate_cost(self, task: VGPUResource, platform: Platform) -> float:
        """
        计算等效成本 cost_{i,v} = c_i/α_v + m_i/β_v + b_i/γ_v
        
        Args:
            task: 任务资源需求 d_i = (c_i, m_i, b_i)
            platform: GPU平台
            
        Returns:
            等效成本
        """
        # 获取折算系数
        from core.vgpu_model.normalization.normalization_coefficients import CoefficientManager
        coeff_manager = CoefficientManager()
        coeff = coeff_manager.get_coefficients(platform.vendor, platform.model)
        
        if coeff is None:
            raise ValueError(f"未找到厂商 {platform.vendor} 型号 {platform.model} 的折算系数")
        
        # 计算有效可用容量 D^eff = (αC, βM, γB)
        c_eff = platform.gpu.compute * coeff.alpha
        m_eff = platform.gpu.memory * coeff.beta
        b_eff = platform.gpu.bandwidth * coeff.gamma
        
        # 计算等效成本 cost_{i,v} = c_i/α_v + m_i/β_v + b_i/γ_v
        # 注意：这里使用的是有效容量，所以实际上是 c_i/C^eff + m_i/M^eff + b_i/B^eff
        if c_eff == 0 or m_eff == 0 or b_eff == 0:
            return float('inf')
        
        cost = (task.compute / c_eff + 
                task.memory / m_eff + 
                task.bandwidth / b_eff)
        
        return cost
    
    def calculate_score(self, task: VGPUResource, platform: Platform) -> float:
        """
        计算平台得分 score_{i,v} = perf_v / cost_{i,v}
        
        注意：perf_v 统一为 1.0，因为软件栈折算系数（α, β, γ）已经考虑了性能差异
        通过折算系数计算有效可用容量后，所有平台已标准化，因此 perf_v = 1.0
        
        Args:
            task: 任务资源需求
            platform: GPU平台
            
        Returns:
            平台得分
        """
        cost = self.calculate_cost(task, platform)
        
        if cost == 0:
            return float('inf')
        
        # 使用 CrossVendorScorer 计算得分
        # perf_v 统一为 1.0，因为折算系数已经考虑了性能差异
        score = self.scorer.calculate_cross_vendor_score(
            task, platform.gpu, performance_weight=1.0  # 统一为1.0
        )
        
        return score
    
    def select_vendor(self, task: VGPUResource, 
                     candidate_platforms: List[Platform],
                     filter_available: bool = True) -> VendorSelectionResult:
        """
        选择最佳平台（一级选址调度）
        
        Args:
            task: 任务资源需求 d_i
            candidate_platforms: 候选平台列表
            filter_available: 是否过滤资源不足的平台
            
        Returns:
            选址结果，包含选中的平台和得分信息
            
        Raises:
            ValueError: 如果没有可用的平台
        """
        if not candidate_platforms:
            raise ValueError("候选平台列表不能为空")
        
        # 计算所有平台的得分
        platform_scores = []
        
        for platform in candidate_platforms:
            try:
                # 检查资源是否充足
                if filter_available:
                    if (platform.gpu.compute < task.compute or
                        platform.gpu.memory < task.memory or
                        platform.gpu.bandwidth < task.bandwidth):
                        continue  # 跳过资源不足的平台
                
                # 计算得分
                score = self.calculate_score(task, platform)
                cost = self.calculate_cost(task, platform)
                
                platform_scores.append((platform, score, cost))
                
            except (ValueError, ZeroDivisionError) as e:
                # 跳过无法计算得分的平台
                continue
        
        if not platform_scores:
            raise ValueError("没有可用的平台或所有平台资源不足")
        
        # 选择得分最高的平台
        best_platform, best_score, best_cost = max(
            platform_scores, 
            key=lambda x: x[1]  # 按得分排序
        )
        
        # 构建结果
        all_scores = [(platform, score) for platform, score, _ in platform_scores]
        
        return VendorSelectionResult(
            selected_platform=best_platform,
            score=best_score,
            cost=best_cost,
            all_scores=all_scores,
            task=task
        )
    
    def rank_platforms(self, task: VGPUResource,
                      candidate_platforms: List[Platform],
                      filter_available: bool = True) -> List[Tuple[Platform, float, float]]:
        """
        对所有平台进行排名
        
        Args:
            task: 任务资源需求
            candidate_platforms: 候选平台列表
            filter_available: 是否过滤资源不足的平台
            
        Returns:
            (平台, 得分, 成本) 的列表，按得分降序排列
        """
        platform_scores = []
        
        for platform in candidate_platforms:
            try:
                # 检查资源是否充足
                if filter_available:
                    if (platform.gpu.compute < task.compute or
                        platform.gpu.memory < task.memory or
                        platform.gpu.bandwidth < task.bandwidth):
                        continue
                
                # 计算得分和成本
                score = self.calculate_score(task, platform)
                cost = self.calculate_cost(task, platform)
                
                platform_scores.append((platform, score, cost))
                
            except (ValueError, ZeroDivisionError):
                continue
        
        # 按得分降序排列
        platform_scores.sort(key=lambda x: x[1], reverse=True)
        
        return platform_scores
    
    def get_selection_info(self, result: VendorSelectionResult) -> Dict[str, Any]:
        """
        获取选址信息的详细字典
        
        Args:
            result: 选址结果
            
        Returns:
            详细信息字典
        """
        return {
            'selected_platform': {
                'platform_id': result.selected_platform.platform_id,
                'vendor': result.selected_platform.vendor,
                'model': result.selected_platform.model,
                'perf_v': result.selected_platform.perf_v,
                'pool_capacity': result.selected_platform.pool_capacity,
                'gpu': {
                    'compute': result.selected_platform.gpu.compute,
                    'memory': result.selected_platform.gpu.memory,
                    'bandwidth': result.selected_platform.gpu.bandwidth,
                }
            },
            'score': result.score,
            'cost': result.cost,
            'task': {
                'compute': result.task.compute,
                'memory': result.task.memory,
                'bandwidth': result.task.bandwidth,
            },
            'all_platform_scores': [
                {
                    'platform_id': platform.platform_id,
                    'vendor': platform.vendor,
                    'model': platform.model,
                    'score': score
                }
                for platform, score in result.all_scores
            ]
        }

