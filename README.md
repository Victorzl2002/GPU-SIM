<!--
 * @Author: Victorzl
 * @Date: 2025-11-10 15:24:23
 * @LastEditors: Victorzl
 * @LastEditTime: 2025-11-10 23:32:44
 * @Description: 请填写简介
-->

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

| 场景 | 说明                         | 配置入口                                   |
| ---- | ---------------------------- | ------------------------------------------ |
| A1   | 无沙盒共享基线               | `experiments/baseline/scenario.py`         |
| A2   | MIG 近似硬切分               | `experiments/hard_split/scenario.py`       |
| A2P  | ParvaGPU 固定切片 + 局部共享 | `experiments/parvagpu/scenario.py`         |
| A3   | GLB + API 沙盒 + SLO 守护    | `experiments/sandbox/scenario.py`          |
| A4   | A3 去链路门消融              | `experiments/abl_no_link_gate/scenario.py` |
| A5   | A3 去 SLO 守护消融           | `experiments/abl_no_slo_guard/scenario.py` |

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

思路本身有明确的研究目标：异构 GPU 资源统一建模、两级 GLB 调度、API 沙盒三门限流，并在 A1–A5 多场景下比对 SLO/利用率/干扰。这些要素具备“可发表”潜质，但要达到核心期刊/会议要求，还需补强几方面：

实验严谨性：目前仿真还缺关键环节，例如：需求突发/动态行为、沙盒限流真实触发、SLO 差异显著的对照，以及指标统计的可信区间（多 seed、方差）。这些都是评价系统方案必备的数据。
参数合理性：任务谱、负载强度、到达模式需与真实 workload 对齐；否则 reviewer 会质疑实验代表性。前面讨论的“理想耗时小于 deadline”“利用率偏低”都要解决。
对比充分：除了 A1–A5，最好还引入其它公开方案或 ablation（例如不同 oversub 策略、无需沙盒的自适应调度），才能凸显方法优势。
理论支撑：GLB + sandbox 的算法或模型最好有一定分析（复杂度、收敛、SLO 保证等），不仅仅是工程描述。
只要补齐上述环节，再结合仿真和（若能的话）真实系统的验证，论文的完整度和说服力就会大幅提升，进入核心期刊/会议的可能性也会更高。

差距：
负载模型真实性不足：任务需求是静态常量、到达过程简单，缺少真实 trace 或动态突发建模，导致沙盒几乎不触发，A3 与基线差异有限。需引入时变需求、burst/波动统计或真实业务采样，证明方法在复杂负载下仍有效。
资源压力与对照场景不充分：当前集群算力远超任务需求，平均利用率仅 20%–40%，干扰率 <1。需要加压（更多任务、更紧配额、缩短时长或缩减 GPU 规模）并在多档负载下扫描，观察临界点与退化行为。
指标与统计可靠性：目前只报告单次运行结果，缺少多 seed/置信区间、尾部数据（p95/p99）与波动分析，也未展示 limiter 触发、SLO 漂移曲线等关键证据。核心期刊通常要求完整的统计分析。
理论与实现细节欠缺：GLB/Sandbox 算法没有复杂度或最优性分析，难以说服审稿人其普适性；同时缺乏实际系统验证或开销评估（例如沙盒延迟、调度开销）。
对比面狭窄：A1–A5 基本是自家方案的变体，不含外部基准（如 MIG/MPS/其它学术方案），缺少说服力。应选择业界/论文常见策略作为 baseline，并给出同等条件下的对比。

核心模型

core/vgpu_model/resource_model/vgpu_resource.py (lines 14-118) 定义统一的 vGPU 三维向量（Compute/Memory/Bandwidth）及算分、加减操作，是所有模块共享的资源抽象。
core/vgpu_model/normalization/normalization_coefficients.py (lines 11-132) 和 cross_vendor_scorer.py (lines 9-108) 负责折算系数与跨厂商得分，CoefficientManager 提供 α/β/γ，CrossVendorScorer.calculate_cross_vendor_score() 用等效容量对任务–平台组合打分。
core/cluster/gpu.py (lines 1-86)、core/cluster/node.py (lines 1-103) 表示物理 GPU 与节点：GPU 记录原始/折算容量，节点聚合同厂商 GPU、维护可用配额/利用率，提供 allocate()/release() 等接口，供调度器调用。
工作负载与任务

core/workload/task.py (lines 1-78) 描述单个任务的生命周期，包含资源需求、SLO、进度、状态机以及 update_progress()、interference_ratio 计算。
core/workload/generator.py (lines 1-82) 的 WorkloadGenerator.generate() 根据 TaskProfile 列表（默认定义在 experiments/base.py (lines 65-112)）生成离散时间仿真用的任务序列，支持 Poisson/Burst/Wave 到达模式。
调度系统

core/scheduling/config.py (lines 1-13) 定义调度配置（GLB 周期、是否允许共享、超量倍率等）。
core/scheduling/vendor_selector.py (lines 1-76) 执行一级 GLB：对兼容厂商聚合容量并用 CrossVendorScorer 选最优平台。
core/scheduling/node_selector.py (lines 1-63) 实现二级 GLB（LRP + BRA 节点评分），select() 返回得分最高节点。
core/scheduling/glb.py (lines 1-80) 整合上述组件：schedule() 遍历等待任务、调用两级选择并在 ClusterNode.allocate() 成功后生成 TaskAllocation；release() 在任务完成/掉线时归还配额。
API 沙盒与隔离

core/isolation/limiters/**init**.py (lines 1-66) 包含显存配额门（MemoryQuotaGate）、带宽令牌桶（BandwidthTokenBucket）、算力节流门（ComputeThrottle）及其输出结构。
core/isolation/policies/**init**.py (lines 1-35) 定义 SandboxConfig/SLOGuardConfig，控制三门开启、令牌补充速率、算力上限、守护调参等。
core/isolation/**init**.py (lines 1-64) 的 APISandbox.apply() 在每个 tick 执行限流逻辑：根据配置调用三门，并通过 hooks.registry.emit() 发布事件；release() 清理令牌桶/算力状态。
core/isolation/hooks/**init**.py (lines 1-32) 提供沙盒事件注册/触发机制，方便调试或监控。
仿真主循环

core/simulation/config.py (lines 1-17) 聚合仿真参数（tick、时长、调度/sandbox 配置）。
core/simulation/engine.py (lines 1-117) 是离散时间仿真核心：
初始化 GLB 调度器、APISandbox、MetricCollector。
每个 tick：将到达任务加入队列、按调度周期调用 GLBScheduler.schedule() 分配配额。
对 running_tasks 调用沙盒限流、推进进度、检查完成/超时、记录节点使用量。
结束后将剩余任务标记为 drop，返回 MetricCollector.summarize() 的指标（利用率、稳定性、SLO、干扰率与 limiter 触发次数）。
指标收集

evaluation/metrics/collector.py (lines 1-83) 维护节点容量、采样值与 limiter 计数；summarize() 计算平均利用率、标准差（稳定性）、全局 SLO rate、平均干扰率等。
实验管理

experiments/base.py (lines 1-118) 提供 build_reference_cluster()（3×A100 + 2×910B）和高压任务谱（llm-surge 等），ExperimentProfile.run() 将工作负载、集群和 SimulationConfig 组合后调用 SimulationEngine 获取结果。
各实验场景（experiments/baseline/scenario.py, experiments/hard_split/scenario.py, …）通过不同的 SimulationConfig/SandboxConfig/任务数量构成 A1–A5 对照：
A1：无沙盒的共享池基线。
A2：静态分片（MIG 近似）。
A2P：ParvaGPU 式固定切片 + 局部共享。
A3：GLB + API 沙盒 + SLO 守护。
A4/A5：A3 的消融（去带宽门 / 去 SLO 守护）。
experiments/runner.py (lines 1-34) 暴露 run(name, seed)，按名称选择场景并执行。
命令行入口

run_experiment.py (lines 1-37) 提供简单 CLI，可 python3 run_experiment.py A3-sandbox --seed 42 --pretty 直接运行指定场景并输出指标。
调用流程总结

选择场景（或通过 CLI），实例化 ExperimentProfile，加载默认任务谱/集群。
run() 生成任务 → 创建 SimulationEngine，持有 GLBScheduler 与 APISandbox。
主循环按 tick 调度任务、执行沙盒、收集节点使用量与 limiter 事件。
仿真结束后由 MetricCollector 输出各类指标，供 README/论文分析。
