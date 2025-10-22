"""
软件栈折算系数定义

实现CUDA与CANN平台的折算系数 (α, β, γ)
用于跨厂商平台的可比性标准化
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
import json


@dataclass
class NormalizationCoefficients:
    """
    软件栈折算系数
    
    用于将不同厂商GPU的性能指标标准化到统一基准
    α: Compute折算系数 (算力标准化)
    β: Memory折算系数 (显存标准化)  
    γ: Bandwidth折算系数 (带宽标准化)
    """
    
    # 折算系数
    alpha: float  # Compute折算系数
    beta: float   # Memory折算系数
    gamma: float  # Bandwidth折算系数
    
    # 基准信息
    vendor: str   # 厂商 (nvidia, huawei)
    model: str    # GPU型号
    baseline: str # 基准平台
    
    def __post_init__(self):
        """初始化后验证"""
        if self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0:
            raise ValueError("折算系数必须大于0")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'vendor': self.vendor,
            'model': self.model,
            'baseline': self.baseline
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalizationCoefficients':
        """从字典创建实例"""
        return cls(
            alpha=data['alpha'],
            beta=data['beta'],
            gamma=data['gamma'],
            vendor=data['vendor'],
            model=data['model'],
            baseline=data['baseline']
        )
    
    def save_to_file(self, filepath: str) -> None:
        """保存到文件"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as e:
            raise RuntimeError(f"保存文件失败: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NormalizationCoefficients':
        """从文件加载"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (IOError, OSError) as e:
            raise RuntimeError(f"加载文件失败: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"文件格式错误: {e}")


class CoefficientManager:
    """折算系数管理器"""
    
    def __init__(self):
        self._coefficients: Dict[str, NormalizationCoefficients] = {}
        self._load_default_coefficients()
    
    def _load_default_coefficients(self):
        """加载默认折算系数"""
        # NVIDIA A100 基准系数 (以自身为基准)
        self._coefficients['nvidia_a100'] = NormalizationCoefficients(
            alpha=1.0, beta=1.0, gamma=1.0,
            vendor='nvidia', model='A100', baseline='nvidia_a100'
        )
        
        # 华为 Ascend 910B 折算系数 (相对于NVIDIA A100)
        # 这些系数需要通过实际基准测试获得
        self._coefficients['huawei_ascend910b'] = NormalizationCoefficients(
            alpha=0.85,  # 算力相对系数 (需要实际测试)
            beta=0.90,   # 显存相对系数 (需要实际测试)
            gamma=0.80,  # 带宽相对系数 (需要实际测试)
            vendor='huawei', model='Ascend910B', baseline='nvidia_a100'
        )
    
    def get_coefficients(self, vendor: str, model: str) -> Optional[NormalizationCoefficients]:
        """获取指定厂商和型号的折算系数"""
        key = f"{vendor}_{model.lower()}"
        return self._coefficients.get(key)
    
    def add_coefficients(self, coefficients: NormalizationCoefficients) -> None:
        """添加新的折算系数"""
        key = f"{coefficients.vendor}_{coefficients.model.lower()}"
        self._coefficients[key] = coefficients
    
    def list_available_platforms(self) -> list:
        """列出所有可用的平台"""
        return list(self._coefficients.keys())
    
    def update_coefficients(self, vendor: str, model: str, 
                          alpha: float, beta: float, gamma: float) -> None:
        """更新折算系数"""
        key = f"{vendor}_{model.lower()}"
        if key in self._coefficients:
            self._coefficients[key].alpha = alpha
            self._coefficients[key].beta = beta
            self._coefficients[key].gamma = gamma
        else:
            self.add_coefficients(NormalizationCoefficients(
                alpha=alpha, beta=beta, gamma=gamma,
                vendor=vendor, model=model, baseline='nvidia_a100'
            ))


# 全局系数管理器实例
coefficient_manager = CoefficientManager()
