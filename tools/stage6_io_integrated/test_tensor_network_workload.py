#!/usr/bin/env python3
"""
第六阶段测试 2: 张量网络模拟负载测试
构建简单的 MPS 链式收缩，分析各步骤的时间占比
"""

import sys
import torch
import time
import json
import os
import numpy as np

def create_mps_tensor(bond_left, physical, bond_right):
    """创建 MPS 张量"""
    return torch.randn(bond_left, physical, bond_right, requires_grad=True)

def test_mps_contraction():
    """测试 MPS 链式收缩"""
    print("=" * 80)
    print("MPS 链式收缩测试")
    print("=" * 80)
    
    # MPS 参数
    num_sites = 10
    bond_dim = 32
    physical_dim = 4
    
    print(f"\nMPS 配置:")
    print(f"  站点数: {num_sites}")
    print(f"  键维度: {bond_dim}")
    print(f"  物理维度: {physical_dim}")
    print("-" * 80)
    
    # 创建 MPS 张量链
    mps_tensors = []
    for i in range(num_sites):
        if i == 0:
            # 左边界
            tensor = create_mps_tensor(1, physical_dim, bond_dim)
        elif i == num_sites - 1:
            # 右边界
            tensor = create_mps_tensor(bond_dim, physical_dim, 1)
        else:
            # 中间
            tensor = create_mps_tensor(bond_dim, physical_dim, bond_dim)
        
        mps_tensors.append(tensor)
    
    # 创建输入状态
    input_states = [torch.randn(physical_dim) for _ in range(num_sites)]
    
    # 记录各步骤时间
    timings = {
        "contraction": [],
        "permute": [],
        "svd": [],
        "backward": []
    }
    
    num_trials = 20
    
    print(f"\n执行 {num_trials} 次试验...")
    
    for trial in range(num_trials):
        # 1. 收缩（Contraction）
        start = time.perf_counter()
        
        # 从左到右收缩
        result = torch.einsum('ij,j->i', mps_tensors[0].squeeze(0), input_states[0])
        
        for i in range(1, num_sites):
            # result: (bond_left,)
            # mps: (bond_left, physical, bond_right)
            # input: (physical,)
            result = torch.einsum('i,ipj,p->j', result, mps_tensors[i], input_states[i])
        
        contract_time = time.perf_counter() - start
        timings["contraction"].append(contract_time)
        
        # 2. 转置（Permute）- 模拟重排操作
        start = time.perf_counter()
        
        permuted_tensors = []
        for tensor in mps_tensors:
            # 交换维度
            if tensor.dim() == 3:
                perm = tensor.permute(2, 1, 0).contiguous()
                permuted_tensors.append(perm)
        
        permute_time = time.perf_counter() - start
        timings["permute"].append(permute_time)
        
        # 3. SVD 分解（用于截断）
        start = time.perf_counter()
        
        # 对一个中间张量做 SVD
        middle_tensor = mps_tensors[num_sites // 2]
        shape = middle_tensor.shape
        matrix = middle_tensor.reshape(shape[0] * shape[1], shape[2])
        
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # 截断
        k = min(bond_dim, len(S))
        U_trunc = U[:, :k]
        S_trunc = S[:k]
        Vh_trunc = Vh[:k, :]
        
        svd_time = time.perf_counter() - start
        timings["svd"].append(svd_time)
        
        # 4. 反向传播（Backward）
        start = time.perf_counter()
        
        if result.requires_grad:
            result.backward(result)  # 使用自己作为梯度
            
            # 清除梯度
            for tensor in mps_tensors:
                if tensor.grad is not None:
                    tensor.grad.zero_()
        
        backward_time = time.perf_counter() - start
        timings["backward"].append(backward_time)
    
    # 计算统计
    results = {}
    total_time = 0
    
    print("\n" + "=" * 80)
    print("时间分析")
    print("=" * 80)
    
    for operation, times in timings.items():
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        total_time += np.mean(times)
        
        results[operation] = {
            "avg_time_ms": avg_time,
            "std_time_ms": std_time
        }
        
        print(f"\n{operation.capitalize()}:")
        print(f"  平均时间: {avg_time:.2f} ± {std_time:.2f} ms")
    
    # 计算占比
    print("\n" + "=" * 80)
    print("时间占比")
    print("=" * 80)
    
    for operation, result in results.items():
        percentage = (result["avg_time_ms"] / (total_time * 1000)) * 100
        result["percentage"] = percentage
        print(f"{operation.capitalize()}: {percentage:.1f}%")
    
    return results

def test_einsum_vs_matmul():
    """对比 einsum 和 matmul 的性能"""
    print("\n" + "=" * 80)
    print("Einsum vs Matmul 性能对比")
    print("=" * 80)
    
    # 常见的张量网络操作
    bond_dim = 64
    physical_dim = 4
    
    A = torch.randn(bond_dim, physical_dim, bond_dim)
    B = torch.randn(bond_dim, physical_dim, bond_dim)
    v = torch.randn(physical_dim)
    
    results = []
    
    # 操作 1: 物理维度收缩
    print("\n操作 1: 物理维度收缩")
    print("-" * 80)
    
    # 方法 A: einsum
    times_einsum = []
    for _ in range(100):
        start = time.perf_counter()
        result = torch.einsum('ipj,jqk,p,q->ik', A, B, v, v)
        end = time.perf_counter()
        times_einsum.append(end - start)
    
    einsum_time = np.mean(times_einsum) * 1000
    
    # 方法 B: 手动 matmul
    times_matmul = []
    for _ in range(100):
        start = time.perf_counter()
        # 分步骤执行
        temp1 = torch.tensordot(A, v, dims=([1], [0]))  # (bond, bond)
        temp2 = torch.tensordot(B, v, dims=([1], [0]))  # (bond, bond)
        result = torch.matmul(temp1, temp2)
        end = time.perf_counter()
        times_matmul.append(end - start)
    
    matmul_time = np.mean(times_matmul) * 1000
    
    print(f"  Einsum: {einsum_time:.3f} ms")
    print(f"  Matmul: {matmul_time:.3f} ms")
    print(f"  比率: {einsum_time/matmul_time:.2f}x")
    
    results.append({
        "operation": "physical_contraction",
        "einsum_ms": einsum_time,
        "matmul_ms": matmul_time,
        "ratio": einsum_time / matmul_time
    })
    
    return results

def test_full_training_iteration():
    """测试完整的训练迭代"""
    print("\n" + "=" * 80)
    print("完整训练迭代测试")
    print("=" * 80)
    
    # 设置
    num_sites = 20
    bond_dim = 48
    physical_dim = 4
    batch_size = 16
    
    print(f"\n配置:")
    print(f"  站点数: {num_sites}")
    print(f"  键维度: {bond_dim}")
    print(f"  批次大小: {batch_size}")
    print("-" * 80)
    
    # 创建模型
    mps_tensors = []
    for i in range(num_sites):
        if i == 0:
            tensor = create_mps_tensor(1, physical_dim, bond_dim)
        elif i == num_sites - 1:
            tensor = create_mps_tensor(bond_dim, physical_dim, 1)
        else:
            tensor = create_mps_tensor(bond_dim, physical_dim, bond_dim)
        mps_tensors.append(tensor)
    
    # 创建优化器
    optimizer = torch.optim.Adam([t for t in mps_tensors], lr=0.01)
    
    # 模拟一个训练迭代
    timings = {
        "data_loading": 0,
        "forward": 0,
        "loss_compute": 0,
        "backward": 0,
        "optimizer_step": 0
    }
    
    num_iterations = 10
    
    print(f"\n执行 {num_iterations} 次迭代...")
    
    for iteration in range(num_iterations):
        # 1. 数据加载
        start = time.perf_counter()
        batch_inputs = [torch.randn(batch_size, physical_dim) for _ in range(num_sites)]
        batch_targets = torch.randn(batch_size)
        timings["data_loading"] += time.perf_counter() - start
        
        # 2. 前向传播
        start = time.perf_counter()
        outputs = []
        for b in range(batch_size):
            inputs = [batch_inputs[i][b] for i in range(num_sites)]
            
            # 收缩
            result = torch.einsum('ij,j->i', mps_tensors[0].squeeze(0), inputs[0])
            for i in range(1, num_sites):
                result = torch.einsum('i,ipj,p->j', result, mps_tensors[i], inputs[i])
            
            outputs.append(result.squeeze())
        
        outputs = torch.stack(outputs)
        timings["forward"] += time.perf_counter() - start
        
        # 3. 计算损失
        start = time.perf_counter()
        loss = torch.nn.functional.mse_loss(outputs, batch_targets)
        timings["loss_compute"] += time.perf_counter() - start
        
        # 4. 反向传播
        start = time.perf_counter()
        loss.backward()
        timings["backward"] += time.perf_counter() - start
        
        # 5. 优化器步骤
        start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        timings["optimizer_step"] += time.perf_counter() - start
    
    # 平均时间
    print("\n" + "=" * 80)
    print("每次迭代平均时间")
    print("=" * 80)
    
    results = {}
    total_time = sum(timings.values()) / num_iterations
    
    for operation, total_time_op in timings.items():
        avg_time = (total_time_op / num_iterations) * 1000  # ms
        percentage = (avg_time / (total_time * 1000)) * 100
        
        results[operation] = {
            "avg_time_ms": avg_time,
            "percentage": percentage
        }
        
        print(f"\n{operation.replace('_', ' ').capitalize()}:")
        print(f"  时间: {avg_time:.2f} ms ({percentage:.1f}%)")
    
    print(f"\n总时间: {total_time*1000:.2f} ms/iteration")
    results["total_time_ms"] = total_time * 1000
    
    return results

def analyze_bottlenecks(all_results):
    """分析性能瓶颈"""
    print("\n" + "=" * 80)
    print("性能瓶颈分析")
    print("=" * 80)
    
    analysis = {
        "bottleneck": "unknown",
        "recommendations": []
    }
    
    # 分析 MPS 收缩
    if "mps_contraction" in all_results:
        mps_results = all_results["mps_contraction"]
        
        # 找出最耗时的操作
        max_op = max(mps_results.items(), key=lambda x: x[1]["avg_time_ms"])
        analysis["bottleneck"] = max_op[0]
        
        print(f"\n主要瓶颈: {max_op[0]} ({max_op[1]['percentage']:.1f}%)")
        
        if max_op[0] == "backward":
            print("\n反向传播是主要瓶颈")
            analysis["recommendations"].append("考虑使用混合精度训练")
            analysis["recommendations"].append("减少梯度计算的频率（梯度累积）")
        elif max_op[0] == "contraction":
            print("\n收缩操作是主要瓶颈")
            analysis["recommendations"].append("优化 einsum 操作")
            analysis["recommendations"].append("考虑使用编译优化（torch.compile）")
        elif max_op[0] == "permute":
            print("\n转置操作是主要瓶颈")
            analysis["recommendations"].append("减少不必要的维度重排")
            analysis["recommendations"].append("使用 einsum 避免显式转置")
        elif max_op[0] == "svd":
            print("\nSVD 是主要瓶颈")
            analysis["recommendations"].append("减少 SVD 频率")
            analysis["recommendations"].append("使用近似 SVD 方法")
    
    # 分析训练迭代
    if "training_iteration" in all_results:
        train_results = all_results["training_iteration"]
        
        if train_results["forward"]["percentage"] > 60:
            print("\n前向传播占用时间过多")
            analysis["recommendations"].append("优化张量收缩路径")
        
        if train_results.get("data_loading", {}).get("percentage", 0) > 10:
            print("\n数据加载有开销")
            analysis["recommendations"].append("使用数据预加载和多进程数据加载")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage6"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "tensor_network_workload.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第六阶段测试: 张量网络模拟负载")
    print("=" * 80)
    
    all_results = {
        "test_name": "tensor_network_workload",
        "stage": 6
    }
    
    # 执行测试
    all_results["mps_contraction"] = test_mps_contraction()
    all_results["einsum_vs_matmul"] = test_einsum_vs_matmul()
    all_results["training_iteration"] = test_full_training_iteration()
    
    # 分析瓶颈
    all_results["analysis"] = analyze_bottlenecks(all_results)
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n通用优化建议:")
    print("  1. 使用 torch.compile() 加速热点函数")
    print("  2. 批处理多个样本以提高吞吐量")
    print("  3. 使用 einsum 代替多个 matmul + permute")
    print("  4. 在关键路径上使用 profiler 找出瓶颈")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
