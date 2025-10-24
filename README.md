<!--
 * @Author: Victorzl
 * @Date: 2025-10-22 14:46:19
 * @LastEditors: Victorzl
 * @LastEditTime: 2025-10-24 10:13:38
 * @Description: 请填写简介
-->
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
│   │   ├── placement/         # 一级选址调度
│   │   ├── drf/              # DRF三维配额分配算法
│   │   └── scoring/          # 跨厂商平台得分计算
│   └── isolation/            # API沙盒机制
│       ├── api_model/        # API抽象模型
│       ├── hooks/            # 软隔离控制钩子
│       ├── limiters/         # 配额门、令牌桶、链路门
│       └── policies/         # 隔离策略配置
├── experiments/               # 实验策略对比
│   ├── baseline/             # A1: 无隔离基线
│   ├── hard_split/           # A2: 硬切分（MIG近似，三维静态）
│   ├── parvagpu/             # A2P: ParvaGPU 近似策略（本文新增）
│   ├── sandbox/              # A3: 本文 API 沙盒 + DRF + SLO
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
- 一级选址调度确定任务运行节点
- DRF算法进行三维配额分配
- 实现公平占额分配

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
- **A3**: 本文 API 沙盒 + DRF + SLO
- **A4**: 去链路门（消融实验）
- **A5**: 去SLO守护（消融实验）
- 评估指标: GPU利用率、干扰率(IR)、SLO满足率

## 技术特点

- **异构统一**: 支持不同厂商GPU的统一抽象
- **三维建模**: Compute、Memory、Bandwidth三维资源模型
- **软隔离**: API沙盒机制实现安全共享
- **动态调度**: 基于SLO的动态扩缩容
- **公平分配**: DRF算法保证资源公平性



Phase 1: vGPU资源模型 (当前步骤)
Phase 2: 硬件适配器层 (adapters/)
Phase 3: 调度系统 (core/scheduling/)
Phase 4: API沙盒机制 (core/isolation/)
Phase 5: 实验策略实现 (experiments/)
Phase 6: 监控与评估 (monitoring/ + evaluation/)