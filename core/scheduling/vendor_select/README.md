# 跨厂商选址模块

## 概述

`vendor_select` 模块实现了一级调度：跨厂商/资源池选址。基于折算系数与平台得分选择最佳GPU平台。

## 架构设计

### 职责划分

- **资源模型层** (`core/vgpu_model/normalization/`)：
  - `CrossVendorScorer`：提供基础的跨厂商得分计算能力
  - 职责：提供可复用的计算工具，不包含调度决策逻辑

- **调度系统层** (`core/scheduling/vendor_select/`)：
  - `VendorSelector`：实现跨厂商选址决策
  - `Platform`：表示GPU平台（包含资源池信息）
  - 职责：使用资源模型层的工具完成调度决策，处理调度上下文

## 核心类

### Platform

表示一个GPU平台，包含：
- `platform_id`: 平台ID
- `vendor`: 厂商 (nvidia, huawei)
- `model`: GPU型号 (A100, Ascend910B)
- `gpu`: GPU资源 (`VGPUResource`)
- `perf_v`: 平台性能权重
- `pool_capacity`: 资源池容量（GPU数量）

### VendorSelector

跨厂商选址决策器，主要方法：
- `calculate_cost()`: 计算等效成本 `cost_{i,v} = c_i/C^eff + m_i/M^eff + b_i/B^eff`
- `calculate_score()`: 计算平台得分 `score_{i,v} = perf_v / cost_{i,v}`
- `select_vendor()`: 选择最佳平台
- `rank_platforms()`: 对所有平台进行排名

## 使用示例

```python
from core.scheduling.vendor_select import VendorSelector, Platform
from core.vgpu_model.resource_model.vgpu_resource import VGPUResource

# 1. 创建任务需求
task = VGPUResource(
    compute=200.0,    # TFLOPS
    memory=50.0,      # GB
    bandwidth=800.0,  # GB/s
    resource_id="task_001"
)

# 2. 创建候选平台
nvidia_a100_gpu = VGPUResource(
    compute=312.0,
    memory=80.0,
    bandwidth=2039.0,
    resource_id="a100_gpu",
    vendor="nvidia",
    model="A100"
)

huawei_ascend_gpu = VGPUResource(
    compute=280.0,
    memory=64.0,
    bandwidth=1600.0,
    resource_id="ascend_gpu",
    vendor="huawei",
    model="Ascend910B"
)

platforms = [
    Platform(
        platform_id="nvidia_a100_pool",
        vendor="nvidia",
        model="A100",
        gpu=nvidia_a100_gpu,
        perf_v=1.0,
        pool_capacity=10
    ),
    Platform(
        platform_id="huawei_ascend910b_pool",
        vendor="huawei",
        model="Ascend910B",
        gpu=huawei_ascend_gpu,
        perf_v=0.9,
        pool_capacity=8
    )
]

# 3. 创建选址器并选择最佳平台
selector = VendorSelector()
result = selector.select_vendor(task, platforms)

# 4. 查看结果
print(f"选中的平台: {result.selected_platform.platform_id}")
print(f"平台得分: {result.score:.6f}")
print(f"等效成本: {result.cost:.6f}")

# 5. 查看所有平台的排名
rankings = selector.rank_platforms(task, platforms)
for i, (platform, score, cost) in enumerate(rankings, 1):
    print(f"第{i}名: {platform.platform_id}, 得分={score:.6f}, 成本={cost:.6f}")
```

## 公式说明

### 等效成本计算

```
cost_{i,v} = c_i/C^eff + m_i/M^eff + b_i/B^eff
```

其中：
- `c_i, m_i, b_i`: 任务对 Compute、Memory、Bandwidth 的需求
- `C^eff = α_v · C_v`: 有效算力容量
- `M^eff = β_v · M_v`: 有效显存容量
- `B^eff = γ_v · B_v`: 有效带宽容量
- `α_v, β_v, γ_v`: 平台 v 的折算系数

### 平台得分计算

```
score_{i,v} = perf_v / cost_{i,v}
```

其中：
- `perf_v`: 平台性能权重
- `cost_{i,v}`: 等效成本

### 决策规则

选择 `score_{i,v}` 最大的平台作为任务入口池。

## 注意事项

1. **资源充足性检查**：默认会过滤资源不足的平台，可通过 `filter_available=False` 禁用
2. **折算系数**：确保平台的折算系数已正确配置在 `CoefficientManager` 中
3. **性能权重**：`perf_v` 应根据实际平台性能设置，通常以基准平台（如 A100）为 1.0

