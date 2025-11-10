# GPU-SIM：异构 vGPU 池离散仿真平台

本仓库依据《仿真实验思路.md》完成了体系化重构，落地“多厂商 GPU 统一 vGPU 池 + GLB 两级调度 + API 沙盒三门限流”离散时间仿真框架。所有实验策略（A1–A5）均可在同一工作负载上复现 GPU 利用率、SLO 满足率、干扰率与稳定性指标。

## 目录结构

```
gpu-sim/
├── core/
│   ├── cluster/           # GPU/节点模型 (Step 2.4)
│   ├── workload/          # 任务属性与生成器 (Step 3)
│   ├── scheduling/        # GLB 两级调度 (Step 4)
│   ├── isolation/         # API 沙盒三门 + SLO 守护 (Step 5)
│   ├── simulation/        # 离散时间主循环 (Step 6)
│   └── vgpu_model/        # 资源归一化与折算系数 (Step 2.1–2.3)
├── experiments/           # A1–A5 场景定义 + runner (Step 7)
├── evaluation/            # 指标收集与分析 (Step 8)
└── 仿真实验思路.md        # 设计蓝图 (Step 0)
```

## 仿真流水线

1. **集群定义**（`core/cluster`）  
   - 以 `GPUDevice`+`ClusterNode` 表达多厂商异构拓扑；节点容量由折算系数 (α, β, γ) 归一。
2. **任务与工作负载**（`core/workload`）  
   - `TaskProfile` 给出 dᵢ=(cᵢ,mᵢ,bᵢ)、兼容性、SLO、工作量；`WorkloadGenerator` 生成到达序列。
3. **GLB 两级调度**（`core/scheduling`）  
   - `VendorSelector` 依据跨厂商得分锁定最优平台，`NodeSelector` 用 LRP+BRA 贪心，`GLBScheduler` 输出静态配额 Qᵢ。
4. **API 沙盒三门**（`core/isolation`）  
   - 显存配额门、带宽令牌桶、算力节流门构成 `APISandbox`；`SLOGuard` 在尾部抖动时提升 compute gate。
5. **离散主循环**（`core/simulation`）  
   - 每 tick 执行“任务到达 → 调度 → 沙盒限流 → 进度推进 → 度量采样”，并在 deadline 触发 SLO/drop。
6. **指标收集**（`evaluation/metrics/collector.py`）  
   - 输出 SLO rate、平均 IR、节点利用率/稳定性、三门触发次数等，供 README/论文 6.x 章节引用。

## 实验场景（A1–A5）

| 场景 | 说明 | 配置入口 |
| ---- | ---- | -------- |
| A1   | 无沙盒共享基线 | `experiments/baseline/scenario.py` |
| A2   | MIG 近似硬切分 | `experiments/hard_split/scenario.py` |
| A2P  | ParvaGPU 固定切片 + 局部共享 | `experiments/parvagpu/scenario.py` |
| A3   | GLB + API 沙盒 + SLO 守护 | `experiments/sandbox/scenario.py` |
| A4   | A3 去链路门消融 | `experiments/abl_no_link_gate/scenario.py` |
| A5   | A3 去 SLO 守护消融 | `experiments/abl_no_slo_guard/scenario.py` |

所有场景共享 `experiments/base.py`：  
- `ExperimentProfile` 绑定 `SimulationConfig`、默认集群（A100+910B）和任务谱；  
- `run()` 会自动创建 `SimulationEngine` 并返回 `MetricCollector.summarize()` 的结果。  
通过 `experiments/runner.py` 可快速调用：

```python
from experiments.runner import run

summary = run("A3-sandbox", seed=42)
print(summary["slo_rate"], summary["avg_interference"])
```

## 后续扩展建议

1. **工作负载**：在 `core/workload/generator.py` 中增加真实 traces / Poisson 参数，即可覆盖更多 burst、混部场景。  
2. **闭环控制**：扩充 `SandboxConfig.slo_guard`，加入全局指标 S (方差 + IR) 反馈，即可模拟论文 4.3/6.5 变体。  
3. **监控/可视化**：将 `MetricCollector` 输出写入 `evaluation/analysis`，生成负载扫描曲线、尾延迟 CDF。  
4. **单元测试**：在 `tests/` 中为调度与限流模块补充 deterministic case，确保 refactoring 稳定。
