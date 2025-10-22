"""
软件栈折算系数模块

实现CUDA与CANN平台的折算系数 (α, β, γ)
建立跨厂商平台的可比性标准
"""

from .normalization_coefficients import NormalizationCoefficients
from .cross_vendor_scorer import CrossVendorScorer
from .platform_benchmark import PlatformBenchmark

__all__ = [
    'NormalizationCoefficients',
    'CrossVendorScorer', 
    'PlatformBenchmark'
]
