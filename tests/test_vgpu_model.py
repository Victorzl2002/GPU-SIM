"""
vGPU资源模型测试

测试VGPUResource和NormalizationCoefficients的功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource, ResourceType
from core.vgpu_model.normalization.normalization_coefficients import (
    NormalizationCoefficients, CoefficientManager
)


def test_vgpu_resource_basic():
    """测试VGPUResource基本功能"""
    print("=== 测试VGPUResource基本功能 ===")
    
    # 创建NVIDIA A100资源
    a100 = VGPUResource(
        compute=312.0,  # TFLOPS
        memory=80.0,    # GB
        bandwidth=2039.0,  # GB/s
        resource_id="a100_001",
        vendor="nvidia",
        model="A100"
    )
    
    print(f"A100资源: {a100}")
    print(f"A100字典: {a100.to_dict()}")
    
    # 创建华为Ascend 910B资源
    ascend = VGPUResource(
        compute=280.0,  # TFLOPS
        memory=64.0,    # GB
        bandwidth=1600.0,  # GB/s
        resource_id="ascend_001",
        vendor="huawei",
        model="Ascend910B"
    )
    
    print(f"Ascend资源: {ascend}")
    
    # 测试资源运算
    combined = a100 + ascend
    print(f"资源相加: {combined}")
    
    # 测试资源缩放
    half_a100 = a100 * 0.5
    print(f"A100一半: {half_a100}")


def test_normalization():
    """测试标准化功能"""
    print("\n=== 测试标准化功能 ===")
    
    # 创建华为Ascend资源
    ascend = VGPUResource(
        compute=280.0,
        memory=64.0,
        bandwidth=1600.0,
        vendor="huawei",
        model="Ascend910B"
    )
    
    # 应用折算系数标准化
    normalized = ascend.normalize(alpha=0.85, beta=0.90, gamma=0.80)
    print(f"原始资源: {ascend}")
    print(f"标准化后: {normalized}")
    
    # 计算得分
    score = normalized.calculate_score(alpha=0.85, beta=0.90, gamma=0.80)
    print(f"标准化得分: {score:.4f}")


def test_coefficient_manager():
    """测试折算系数管理器"""
    print("\n=== 测试折算系数管理器 ===")
    
    manager = CoefficientManager()
    
    # 列出可用平台
    platforms = manager.list_available_platforms()
    print(f"可用平台: {platforms}")
    
    # 获取NVIDIA A100系数
    a100_coeff = manager.get_coefficients("nvidia", "A100")
    print(f"A100系数: {a100_coeff}")
    
    # 获取华为Ascend系数
    ascend_coeff = manager.get_coefficients("huawei", "Ascend910B")
    print(f"Ascend系数: {ascend_coeff}")


def test_cross_vendor_scoring():
    """测试跨厂商得分计算"""
    print("\n=== 测试跨厂商得分计算 ===")
    
    # 创建不同厂商的GPU资源
    a100 = VGPUResource(
        compute=312.0, memory=80.0, bandwidth=2039.0,
        vendor="nvidia", model="A100"
    )
    
    ascend = VGPUResource(
        compute=280.0, memory=64.0, bandwidth=1600.0,
        vendor="huawei", model="Ascend910B"
    )
    
    # 获取折算系数
    manager = CoefficientManager()
    a100_coeff = manager.get_coefficients("nvidia", "A100")
    ascend_coeff = manager.get_coefficients("huawei", "Ascend910B")
    
    # 标准化并计算得分
    a100_normalized = a100.normalize(a100_coeff.alpha, a100_coeff.beta, a100_coeff.gamma)
    ascend_normalized = ascend.normalize(ascend_coeff.alpha, ascend_coeff.beta, ascend_coeff.gamma)
    
    a100_score = a100_normalized.calculate_score(a100_coeff.alpha, a100_coeff.beta, a100_coeff.gamma)
    ascend_score = ascend_normalized.calculate_score(ascend_coeff.alpha, ascend_coeff.beta, ascend_coeff.gamma)
    
    print(f"A100标准化得分: {a100_score:.4f}")
    print(f"Ascend标准化得分: {ascend_score:.4f}")
    print(f"得分比较: {'A100' if a100_score > ascend_score else 'Ascend'} 更优")


if __name__ == "__main__":
    try:
        test_vgpu_resource_basic()
        test_normalization()
        test_coefficient_manager()
        test_cross_vendor_scoring()
        print("\n✅ 所有测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
