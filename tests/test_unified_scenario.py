"""
统一测试场景：异构GPU池化系统

测试场景：
- 2个节点：NVIDIA节点、华为节点
- 每个节点5张GPU卡
- 预定义的任务集合
- 统一的测试流程
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vgpu_model.resource_model.vgpu_resource import VGPUResource, ResourceType
from core.vgpu_model.resource_model.resource_mapper import ResourceMapper
from core.vgpu_model.resource_model.resource_allocator import ResourceAllocator
from core.vgpu_model.normalization.normalization_coefficients import (
    NormalizationCoefficients, CoefficientManager
)
from core.vgpu_model.normalization.cross_vendor_scorer import CrossVendorScorer
from core.vgpu_model.normalization.platform_benchmark import PlatformBenchmark


class UnifiedTestScenario:
    """统一测试场景类"""
    
    def __init__(self):
        """初始化测试场景"""
        self.nodes = {}
        self.tasks = []
        self.coeff_manager = CoefficientManager()
        self.scorer = CrossVendorScorer()
        self.mapper = ResourceMapper()
        self.allocator = ResourceAllocator()
        
        # 初始化数据
        self._init_nodes()
        self._init_tasks()
        self._init_coefficients()
        self._init_allocator()
    
    def _init_nodes(self):
        """初始化节点和GPU卡"""
        print("初始化节点和GPU卡...")
        
        # 节点1：NVIDIA节点
        nvidia_node = {
            'node_id': 'nvidia_node_001',
            'vendor': 'nvidia',
            'location': 'datacenter_a',
            'gpus': []
        }
        
        # NVIDIA节点5张A100卡
        for i in range(5):
            gpu = VGPUResource(
                compute=312.0,    # TFLOPS
                memory=80.0,      # GB
                bandwidth=2039.0, # GB/s
                resource_id=f"nvidia_a100_{i+1:02d}",
                vendor="nvidia",
                model="A100"
            )
            nvidia_node['gpus'].append(gpu)
        
        # 节点2：华为节点
        huawei_node = {
            'node_id': 'huawei_node_001', 
            'vendor': 'huawei',
            'location': 'datacenter_b',
            'gpus': []
        }
        
        # 华为节点5张Ascend 910B卡
        for i in range(5):
            gpu = VGPUResource(
                compute=280.0,    # TFLOPS
                memory=64.0,      # GB
                bandwidth=1600.0, # GB/s
                resource_id=f"huawei_ascend_{i+1:02d}",
                vendor="huawei",
                model="Ascend910B"
            )
            huawei_node['gpus'].append(gpu)
        
        self.nodes = {
            'nvidia_node_001': nvidia_node,
            'huawei_node_001': huawei_node
        }
        
        print(f"✅ 初始化完成：{len(self.nodes)}个节点，{sum(len(node['gpus']) for node in self.nodes.values())}张GPU卡")
    
    def _init_tasks(self):
        """初始化预定义任务"""
        print("初始化预定义任务...")
        
        # 任务1：深度学习训练任务
        task1 = {
            'task_id': 'dl_training_001',
            'task_type': 'deep_learning',
            'priority': 'high',
            'description': 'ResNet-50图像分类训练',
            'demand': VGPUResource(
                compute=200.0,    # 需要200 TFLOPS
                memory=50.0,      # 需要50GB显存
                bandwidth=800.0,  # 需要800 GB/s带宽
                resource_id='dl_training_001_demand'
            )
        }
        
        # 任务2：模型推理任务
        task2 = {
            'task_id': 'inference_001',
            'task_type': 'inference',
            'priority': 'medium',
            'description': 'BERT文本分类推理',
            'demand': VGPUResource(
                compute=120.0,
                memory=30.0,
                bandwidth=600.0,
                resource_id='inference_001_demand'
            )
        }
        
        # 任务3：大模型训练任务
        task3 = {
            'task_id': 'llm_training_001',
            'task_type': 'training',
            'priority': 'high',
            'description': 'GPT-3语言模型训练',
            'demand': VGPUResource(
                compute=250.0,
                memory=60.0,
                bandwidth=1000.0,
                resource_id='llm_training_001_demand'
            )
        }
        
        # 任务4：科学计算任务
        task4 = {
            'task_id': 'scientific_001',
            'task_type': 'deep_learning',
            'priority': 'low',
            'description': '分子动力学模拟',
            'demand': VGPUResource(
                compute=180.0,
                memory=40.0,
                bandwidth=700.0,
                resource_id='scientific_001_demand'
            )
        }
        
        # 任务5：批处理任务
        task5 = {
            'task_id': 'batch_001',
            'task_type': 'inference',
            'priority': 'low',
            'description': '图像批处理推理',
            'demand': VGPUResource(
                compute=100.0,
                memory=25.0,
                bandwidth=500.0,
                resource_id='batch_001_demand'
            )
        }
        
        self.tasks = [task1, task2, task3, task4, task5]
        print(f"✅ 初始化完成：{len(self.tasks)}个预定义任务")
    
    def _init_coefficients(self):
        """初始化折算系数"""
        print("初始化折算系数...")
        
        # NVIDIA A100 折算系数 (以A100为基准)
        nvidia_coeff = NormalizationCoefficients(
            alpha=1.0,      # 算力折算系数
            beta=1.0,       # 显存折算系数  
            gamma=1.0,      # 带宽折算系数
            vendor="nvidia",
            model="A100",
            baseline="nvidia_a100"
        )
        
        # 华为 Ascend 910B 折算系数
        huawei_coeff = NormalizationCoefficients(
            alpha=0.9,      # 算力相对A100的折算系数
            beta=0.8,       # 显存相对A100的折算系数
            gamma=0.78,     # 带宽相对A100的折算系数
            vendor="huawei",
            model="Ascend910B",
            baseline="nvidia_a100"
        )
        
        # 添加折算系数
        self.coeff_manager.add_coefficients(nvidia_coeff)
        self.coeff_manager.add_coefficients(huawei_coeff)
        
        print(f"✅ 折算系数初始化完成")
    
    def _init_allocator(self):
        """初始化资源分配器"""
        print("初始化资源分配器...")
        
        # 将所有GPU添加到分配器
        for node_id, node in self.nodes.items():
            for gpu in node['gpus']:
                self.allocator.add_available_resource(gpu)
        
        print(f"✅ 资源分配器初始化完成")
    
    def display_scenario(self):
        """显示测试场景"""
        print("\n" + "=" * 80)
        print("统一测试场景概览")
        print("=" * 80)
        
        # 显示节点信息
        for node_id, node in self.nodes.items():
            print(f"\n节点: {node_id}")
            print(f"  厂商: {node['vendor']}")
            print(f"  位置: {node['location']}")
            print(f"  GPU数量: {len(node['gpus'])}")
            
            for i, gpu in enumerate(node['gpus'], 1):
                print(f"    GPU{i}: {gpu.resource_id} - {gpu}")
        
        # 显示任务信息
        print(f"\n预定义任务 ({len(self.tasks)}个):")
        for task in self.tasks:
            print(f"  {task['task_id']}: {task['description']}")
            print(f"    类型: {task['task_type']}, 优先级: {task['priority']}")
            print(f"    需求: {task['demand']}")
    
    def test_cross_vendor_scoring(self):
        """测试跨厂商得分计算"""
        print("\n" + "=" * 80)
        print("跨厂商得分计算测试")
        print("=" * 80)
        
        # 选择第一个任务进行测试
        test_task = self.tasks[0]
        print(f"测试任务: {test_task['task_id']} - {test_task['description']}")
        print(f"任务需求: {test_task['demand']}")
        
        # 计算任务在所有GPU上的得分
        all_gpus = []
        for node in self.nodes.values():
            all_gpus.extend(node['gpus'])
        
        print(f"\n计算任务在{len(all_gpus)}张GPU上的得分:")
        
        gpu_scores = []
        for gpu in all_gpus:
            try:
                score = self.scorer.calculate_cross_vendor_score(test_task['demand'], gpu)
                gpu_scores.append((gpu, score))
                print(f"  {gpu.resource_id}: {score:.6f}")
            except Exception as e:
                print(f"  {gpu.resource_id}: 计算失败 - {e}")
        
        # 排序并显示结果
        gpu_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nGPU得分排名:")
        for i, (gpu, score) in enumerate(gpu_scores, 1):
            print(f"  第{i}名: {gpu.resource_id} - 得分: {score:.6f}")
        
        return gpu_scores
    
    def test_task_scheduling(self):
        """测试任务调度"""
        print("\n" + "=" * 80)
        print("任务调度测试")
        print("=" * 80)
        
        scheduling_results = []
        
        for task in self.tasks:
            print(f"\n调度任务: {task['task_id']}")
            print(f"  描述: {task['description']}")
            print(f"  需求: {task['demand']}")
            
            # 为任务选择最佳GPU
            best_gpu = None
            best_score = 0
            available_gpus = []
            
            # 获取所有可用GPU
            for node in self.nodes.values():
                for gpu in node['gpus']:
                    if gpu.resource_id in self.allocator.get_available_resources():
                        available_gpus.append(gpu)
            
            # 计算得分并选择最佳GPU
            for gpu in available_gpus:
                try:
                    score = self.scorer.calculate_cross_vendor_score(task['demand'], gpu)
                    print(f"    {gpu.resource_id}: 得分 = {score:.6f}")
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu
                except Exception as e:
                    print(f"    {gpu.resource_id}: 计算失败 - {e}")
            
            if best_gpu:
                # 尝试分配资源
                allocated = self.allocator.allocate_resource(best_gpu.resource_id, task['demand'])
                if allocated:
                    print(f"  ✅ 成功分配到: {best_gpu.resource_id}")
                    print(f"  📈 分配资源: {allocated}")
                    
                    scheduling_results.append({
                        'task': task,
                        'gpu': best_gpu,
                        'score': best_score,
                        'allocated': allocated,
                        'success': True
                    })
                else:
                    print(f"  ❌ 分配失败")
                    scheduling_results.append({
                        'task': task,
                        'gpu': best_gpu,
                        'score': best_score,
                        'allocated': None,
                        'success': False
                    })
            else:
                print(f"  ❌ 无可用GPU")
                scheduling_results.append({
                    'task': task,
                    'gpu': None,
                    'score': 0,
                    'allocated': None,
                    'success': False
                })
        
        return scheduling_results
    
    def test_resource_utilization(self):
        """测试资源利用率"""
        print("\n" + "=" * 80)
        print("资源利用率测试")
        print("=" * 80)
        
        # 获取资源利用率
        utilization = self.allocator.get_resource_utilization()
        
        print("各GPU资源利用率:")
        for gpu_id, util in utilization.items():
            print(f"\n{gpu_id}:")
            print(f"  算力利用率: {util['compute_utilization']*100:.1f}%")
            print(f"  显存利用率: {util['memory_utilization']*100:.1f}%")
            print(f"  带宽利用率: {util['bandwidth_utilization']*100:.1f}%")
        
        # 计算总体利用率
        total_util = {}
        for util in utilization.values():
            for key, value in util.items():
                if key not in total_util:
                    total_util[key] = []
                total_util[key].append(value)
        
        print(f"\n总体资源利用率:")
        for key, values in total_util.items():
            avg_util = sum(values) / len(values)
            print(f"  平均{key}: {avg_util*100:.1f}%")
    
    def test_node_performance(self):
        """测试节点性能对比"""
        print("\n" + "=" * 80)
        print("节点性能对比测试")
        print("=" * 80)
        
        # 计算每个节点的总资源
        for node_id, node in self.nodes.items():
            print(f"\n节点: {node_id}")
            
            # 计算节点总资源
            total_compute = sum(gpu.compute for gpu in node['gpus'])
            total_memory = sum(gpu.memory for gpu in node['gpus'])
            total_bandwidth = sum(gpu.bandwidth for gpu in node['gpus'])
            
            print(f"  总算力: {total_compute} TFLOPS")
            print(f"  总显存: {total_memory} GB")
            print(f"  总带宽: {total_bandwidth} GB/s")
            
            # 计算节点得分（使用第一个任务作为基准）
            test_task = self.tasks[0]['demand']
            node_scores = []
            
            for gpu in node['gpus']:
                try:
                    score = self.scorer.calculate_cross_vendor_score(test_task, gpu)
                    node_scores.append(score)
                except:
                    node_scores.append(0)
            
            avg_score = sum(node_scores) / len(node_scores)
            print(f"  平均得分: {avg_score:.6f}")
            print(f"  最高得分: {max(node_scores):.6f}")
            print(f"  最低得分: {min(node_scores):.6f}")
    
    def run_complete_test(self):
        """运行完整测试"""
        print("🚀 开始统一测试场景")
        
        # 1. 显示场景概览
        self.display_scenario()
        
        # 2. 测试跨厂商得分计算
        gpu_scores = self.test_cross_vendor_scoring()
        
        # 3. 测试任务调度
        scheduling_results = self.test_task_scheduling()
        
        # 4. 测试资源利用率
        self.test_resource_utilization()
        
        # 5. 测试节点性能对比
        self.test_node_performance()
        
        # 6. 测试结果总结
        print("\n" + "=" * 80)
        print("测试结果总结")
        print("=" * 80)
        
        successful_tasks = sum(1 for result in scheduling_results if result['success'])
        total_tasks = len(scheduling_results)
        
        print(f"任务调度成功率: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        
        # 按节点统计
        node_stats = {}
        for result in scheduling_results:
            if result['success'] and result['gpu']:
                node_id = result['gpu'].resource_id.split('_')[0] + '_' + result['gpu'].resource_id.split('_')[1]
                if node_id not in node_stats:
                    node_stats[node_id] = 0
                node_stats[node_id] += 1
        
        print(f"\n各节点任务分配统计:")
        for node_id, count in node_stats.items():
            print(f"  {node_id}: {count}个任务")
        
        print(f"\n✅ 统一测试场景完成")


def main():
    """主函数"""
    try:
        # 创建统一测试场景
        scenario = UnifiedTestScenario()
        
        # 运行完整测试
        scenario.run_complete_test()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
