# 基于GPU API沙盒机制的异构GPU池化系统

## 项目结构

```
gpu-sim/
├── adapters/                   # 硬件适配器层
│   ├── cuda/                   # NVIDIA CUDA适配器
│   └── cann/                   # 华为CANN适配器
├── core/                       # 核心功能模块
│   ├── vgpu_model/            # vGPU三维资源模型
│   │   ├── resource_model/    # 资源模型定义 ⟨Compute, Memory, Bandwidth⟩
│   │   └── normalization/     # 软件栈折算系数 (α, β, γ)
│   ├── scheduling/            # 调度系统
│   │   ├── vendor_select/    # 一级：跨厂商/资源池选址 (基于折算系数与平台得分)
│   │   ├── node_glb/         # 二级：池内节点选择 (LRP + BRA 负载均衡)
│   │   └── scoring/          # 通用得分函数与负载计算工具
│   └── isolation/            # API沙盒机制
│       ├── api_model/        # API抽象模型
│       ├── hooks/            # 软隔离控制钩子
│       ├── limiters/         # 配额门、令牌桶、链路门
│       └── policies/         # 隔离策略配置
├── experiments/               # 实验策略对比
│   ├── baseline/             # A1: 无隔离基线
│   ├── hard_split/           # A2: 硬切分（MIG近似，三维静态）
│   ├── parvagpu/             # A2P: ParvaGPU 近似策略（本文新增）
│   ├── sandbox/              # A3: 本文 API 沙盒 + GLB 调度（开环/闭环变体）
│   ├── abl_no_link_gate/     # A4: 去链路门（消融）
│   └── abl_no_slo_guard/     # A5: 去SLO守护（消融）
├── evaluation/               # 评估分析
│   ├── metrics/              # 评估指标 (GPU利用率、干扰率、SLO满足率)
│   └── analysis/             # 结果分析
├── monitoring/               # 监控系统
│   ├── slo/                  # SLO监控与动态调度
│   └── telemetry/            # 遥测数据收集
│       └── dashboards/       # 监控仪表板
├── utils/                    # 工具模块
│   ├── config/               # 配置文件
│   └── scripts/              # 脚本工具
├── tests/                    # 测试用例
└── docs/                     # 文档
```

## 核心设计理念

### 1. 统一抽象层 (adapters/)
- 支持NVIDIA A100 (CUDA) 和华为Ascend 910B (CANN)
- 通过软件栈折算系数实现跨厂商可比性

### 2. vGPU三维资源模型 (core/vgpu_model/)
- 资源模型: ⟨Compute, Memory, Bandwidth⟩
- 跨厂商得分: `score = perf / Σ(c/α + m/β + b/γ)`

### 3. 调度系统 (core/scheduling/)

#### 3.1 两级 GLB 调度架构

**一级（跨厂商/资源池选址）** - `core/scheduling/vendor_select/`
- 实现位置：`VendorSelector` 类
- 功能：对每个候选平台 v，基于 vGPU 三维需求 d_i 与折算系数 (α_v, β_v, γ_v) 计算等效成本
  - 计算公式：`cost_{i,v} = c_i/C^eff + m_i/M^eff + b_i/B^eff`
    - 其中 `C^eff = α_v·C_v`, `M^eff = β_v·M_v`, `B^eff = γ_v·B_v`
  - 平台得分：`score_{i,v} = perf_v / cost_{i,v}`
    - **注意**：`perf_v` 统一为 1.0，因为软件栈折算系数（α, β, γ）已经考虑了性能差异
    - 通过折算系数计算有效可用容量后，所有平台已标准化，因此 `perf_v = 1.0`
  - 决策：选择 score 最大的平台作为任务入口池
- 依赖：调用 `core/vgpu_model/normalization/cross_vendor_scorer.py` 中的 `CrossVendorScorer` 进行基础得分计算

**二级（池内节点选择）** - `core/scheduling/node_glb/`
- 功能：在选定平台内，对每个候选节点 n_j 计算
  - LRP 分数 `scoreL_{i,j}`：放置后剩余资源越多越高
  - BRA 分数 `scoreB_{i,j}`：放置后三维负载越均衡越高
  - 合成得分：`score_{i,j} = λ·scoreL_{i,j} + (1-λ)·scoreB_{i,j}`
  - 决策：选择得分最高节点执行任务

#### 3.2 调度机制

- **调度周期**：采用多时间尺度控制，每隔 T_sched 对"已到达未完成任务"重算上述得分与配额，支持任务动态到达
- **输出**：调度层输出每任务在选定节点上的三维配额上限 (Compute/Memory/Bandwidth)
- **执行**：由 API 沙盒在毫秒级 tick 上通过配额门、令牌桶与计算节流门将其落实为运行时限流

#### 3.3 模块职责划分

- **资源模型层** (`core/vgpu_model/normalization/`)：
  - `CrossVendorScorer`：提供基础的跨厂商得分计算能力
  - `NormalizationCoefficients`：管理折算系数
  - 职责：提供可复用的计算工具，不包含调度决策逻辑

- **调度系统层** (`core/scheduling/vendor_select/`)：
  - `VendorSelector`：实现跨厂商选址决策
  - `Platform`：表示GPU平台（包含资源池信息）
  - 职责：使用资源模型层的工具完成调度决策，处理调度上下文

### 4. API沙盒机制 (core/isolation/)
- 软隔离控制
- 配额门、令牌桶、链路门限流
- kernel执行、显存访问、通信带宽控制

### 5. SLO监控与动态调度 (monitoring/slo/)
- p95延迟滑窗监测
- 动态Stream调度
- 扩/缩卡策略

### 6. 实验评估 (experiments/ + evaluation/)
- **A1**: 无隔离基线
- **A2**: 硬切分（MIG近似，三维静态）
- **A2P**: ParvaGPU 近似策略（本文新增）
- **A3**: 本文 API 沙盒 + GLB 调度（开环/闭环变体）
- **A4**: 去链路门（消融实验）
- **A5**: 去SLO守护（消融实验）

上述实验均基于两级 GLB 调度：首先按折算系数进行跨厂商资源池选址，再在池内通过 LRP+BRA 节点打分选择具体节点；沙盒机制在此基础上实施三维软隔离，用于对比不同隔离/调度组合对 GPU 利用率、干扰率与 SLO 的影响。

## 技术特点

- **异构统一**: 支持不同厂商GPU的统一抽象
- **三维建模**: Compute、Memory、Bandwidth三维资源模型
- **软隔离**: API沙盒机制实现安全共享
- **动态调度**: 基于SLO的动态扩缩容
- **负载均衡**: GLB 调度兼顾多任务需求与节点利用率，减少资源碎片



Phase 1: vGPU资源模型 (当前步骤)
Phase 2: 硬件适配器层 (adapters/)
Phase 3: 调度系统 (core/scheduling/)
Phase 4: API沙盒机制 (core/isolation/)
Phase 5: 实验策略实现 (experiments/)
Phase 6: 监控与评估 (monitoring/ + evaluation/)

## 实验流程

详细的实验步骤、实验方法、模块调用关系请参考：
- [实验流程文档](docs/experiment_workflow.md) - 完整的实验步骤和调用关系说明

### 快速开始

运行完整实验流程：
```bash
python tests/test_vgpu_model.py
```

运行统一测试场景：
```bash
python tests/test_unified_scenario.py
```