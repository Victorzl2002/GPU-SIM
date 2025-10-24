"""
任务和GPU卡使用VGPUResource的对比示例

展示厂商GPU卡和任务在vGPU资源模型中的不同使用方式
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource
from core.vgpu_model.resource_model.resource_mapper import ResourceMapper
from core.vgpu_model.resource_model.resource_allocator import ResourceAllocator


def demonstrate_gpu_vs_task_usage():
    """演示GPU卡和任务的不同使用方式"""
    
    print("=" * 60)
    print("厂商GPU卡 vs 任务使用VGPUResource对比")
    print("=" * 60)
    
    # 1. 厂商GPU卡 - 表示物理硬件资源
    print("\n1. 厂商GPU卡 - 物理硬件资源")
    print("-" * 40)
    
    # NVIDIA A100 物理GPU
    a100_gpu = VGPUResource(
        compute=312.0,    # 物理GPU的峰值算力 (TFLOPS)
        memory=80.0,      # 物理GPU的显存容量 (GB)
        bandwidth=2039.0, # 物理GPU的内存带宽 (GB/s)
        resource_id="nvidia_a100_physical",
        vendor="nvidia",
        model="A100"
    )
    
    print(f"NVIDIA A100 物理GPU:")
    print(f"  算力: {a100_gpu.compute} TFLOPS")
    print(f"  显存: {a100_gpu.memory} GB")
    print(f"  带宽: {a100_gpu.bandwidth} GB/s")
    print(f"  含义: 这是GPU硬件的完整资源容量")
    
    # 华为 Ascend 910B 物理GPU
    ascend_gpu = VGPUResource(
        compute=280.0,    # 物理GPU的峰值算力
        memory=64.0,      # 物理GPU的显存容量
        bandwidth=1600.0, # 物理GPU的内存带宽
        resource_id="huawei_ascend910b_physical",
        vendor="huawei",
        model="Ascend910B"
    )
    
    print(f"\n华为 Ascend 910B 物理GPU:")
    print(f"  算力: {ascend_gpu.compute} TFLOPS")
    print(f"  显存: {ascend_gpu.memory} GB")
    print(f"  带宽: {ascend_gpu.bandwidth} GB/s")
    print(f"  含义: 这是GPU硬件的完整资源容量")
    
    # 2. 任务 - 表示资源需求
    print("\n2. 任务 - 资源需求")
    print("-" * 40)
    
    # 创建资源映射器
    mapper = ResourceMapper()
    
    # 深度学习任务需求
    dl_task = mapper.map_task_to_resource("deep_learning", a100_gpu)
    print(f"深度学习任务需求:")
    print(f"  算力需求: {dl_task.compute:.1f} TFLOPS ({dl_task.compute/a100_gpu.compute*100:.1f}%)")
    print(f"  显存需求: {dl_task.memory:.1f} GB ({dl_task.memory/a100_gpu.memory*100:.1f}%)")
    print(f"  带宽需求: {dl_task.bandwidth:.1f} GB/s ({dl_task.bandwidth/a100_gpu.bandwidth*100:.1f}%)")
    print(f"  含义: 这是任务从GPU中需要的资源配额")
    
    # 推理任务需求
    inference_task = mapper.map_task_to_resource("inference", a100_gpu)
    print(f"\n推理任务需求:")
    print(f"  算力需求: {inference_task.compute:.1f} TFLOPS ({inference_task.compute/a100_gpu.compute*100:.1f}%)")
    print(f"  显存需求: {inference_task.memory:.1f} GB ({inference_task.memory/a100_gpu.memory*100:.1f}%)")
    print(f"  带宽需求: {inference_task.bandwidth:.1f} GB/s ({inference_task.bandwidth/a100_gpu.bandwidth*100:.1f}%)")
    print(f"  含义: 这是任务从GPU中需要的资源配额")
    
    # 3. 资源分配场景
    print("\n3. 资源分配场景")
    print("-" * 40)
    
    # 创建资源分配器
    allocator = ResourceAllocator()
    
    # 添加物理GPU到可用资源池
    allocator.add_available_resource(a100_gpu)
    print(f"添加物理GPU到资源池: {a100_gpu.resource_id}")
    
    # 尝试分配深度学习任务
    allocated_dl = allocator.allocate_resource(a100_gpu.resource_id, dl_task)
    if allocated_dl:
        print(f"成功分配深度学习任务: {allocated_dl.resource_id}")
        print(f"  分配算力: {allocated_dl.compute} TFLOPS")
        print(f"  分配显存: {allocated_dl.memory} GB")
        print(f"  分配带宽: {allocated_dl.bandwidth} GB/s")
    
    # 尝试分配推理任务
    allocated_inference = allocator.allocate_resource(a100_gpu.resource_id, inference_task)
    if allocated_inference:
        print(f"成功分配推理任务: {allocated_inference.resource_id}")
        print(f"  分配算力: {allocated_inference.compute} TFLOPS")
        print(f"  分配显存: {allocated_inference.memory} GB")
        print(f"  分配带宽: {allocated_inference.bandwidth} GB/s")
    
    # 查看剩余资源
    available = allocator.get_available_resources()
    for resource_id, resource in available.items():
        print(f"\n剩余可用资源 ({resource_id}):")
        print(f"  剩余算力: {resource.compute:.1f} TFLOPS")
        print(f"  剩余显存: {resource.memory:.1f} GB")
        print(f"  剩余带宽: {resource.bandwidth:.1f} GB/s")
    
    # 4. 关键差异总结
    print("\n4. 关键差异总结")
    print("-" * 40)
    print("厂商GPU卡使用VGPUResource:")
    print("  ✓ 表示物理硬件的完整资源容量")
    print("  ✓ 是资源分配的上限和基准")
    print("  ✓ 包含厂商和型号信息")
    print("  ✓ 用于资源池管理")
    
    print("\n任务使用VGPUResource:")
    print("  ✓ 表示任务对资源的需求量")
    print("  ✓ 是从物理GPU中分配的部分资源")
    print("  ✓ 包含任务类型和分配信息")
    print("  ✓ 用于资源分配和调度")


if __name__ == "__main__":
    demonstrate_gpu_vs_task_usage()
