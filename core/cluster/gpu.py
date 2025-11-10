"""
集群中单卡的抽象。

该模块负责描述参与仿真的物理 GPU：既保留芯片原始能力，也能结合
折算系数 (α, β, γ) 得出跨厂商可比较的等效容量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.vgpu_model.normalization.normalization_coefficients import (
    NormalizationCoefficients,
    coefficient_manager,
)


@dataclass
class GPUDevice:
    """
    单块 GPU 的表示。

    同时维护“原始容量”与“折算后的等效容量”，便于在离散仿真里做跨厂商对比。
    """

    gpu_id: str
    vendor: str
    model: str
    capacity: VGPUResource
    link_bandwidth: float
    coefficients: Optional[NormalizationCoefficients] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capacity.vendor = self.vendor
        self.capacity.model = self.model

    def _get_coefficients(self) -> NormalizationCoefficients:
        if self.coefficients:
            return self.coefficients
        coeff = coefficient_manager.get_coefficients(self.vendor, self.model)
        if coeff is None:
            raise ValueError(f"缺少 {self.vendor}:{self.model} 的折算系数")
        self.coefficients = coeff
        return coeff

    def effective_capacity(self) -> VGPUResource:
        """返回等效容量向量 D^eff = (αC, βM, γB)。"""
        coeff = self._get_coefficients()
        return VGPUResource(
            compute=self.capacity.compute * coeff.alpha,
            memory=self.capacity.memory * coeff.beta,
            bandwidth=self.capacity.bandwidth * coeff.gamma,
            resource_id=f"{self.gpu_id}_effective",
            vendor=self.vendor,
            model=self.model,
        )

    def clone(self) -> "GPUDevice":
        """生成一个浅拷贝，复用折算系数句柄用于调试或快照。"""
        return GPUDevice(
            gpu_id=self.gpu_id,
            vendor=self.vendor,
            model=self.model,
            capacity=VGPUResource(
                compute=self.capacity.compute,
                memory=self.capacity.memory,
                bandwidth=self.capacity.bandwidth,
                resource_id=self.capacity.resource_id,
                vendor=self.vendor,
                model=self.model,
            ),
            link_bandwidth=self.link_bandwidth,
            coefficients=self.coefficients,
            metadata=self.metadata.copy(),
        )
