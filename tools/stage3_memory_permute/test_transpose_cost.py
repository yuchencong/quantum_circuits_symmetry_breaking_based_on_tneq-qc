#!/usr/bin/env python3
"""
第三阶段测试 2: 张量转置代价测试
测试 permute() 和 contiguous() 操作的性能影响
"""

import sys
import torch
import time
import json
import os
import numpy as np

def benchmark_permute(shape, perm, num_trials=50):
    """测试特定维度重排的性能"""
    # 预热
    for _ in range(5):
        tensor = torch.randn(*shape)
        permuted = tensor.permute(*perm)
        _ = permuted.contiguous()
    
    # 测试 permute (不复制数据，只改变视图)
    permute_times = []
    for _ in range(num_trials):
        tensor = torch.randn(*shape)
        
        start = time.perf_counter()
        permuted = tensor.permute(*perm)
        end = time.perf_counter()
        
        permute_times.append(end - start)
    
    # 测试 contiguous (真正的数据重排)
    contiguous_times = []
    for _ in range(num_trials):
        tensor = torch.randn(*shape)
        permuted = tensor.permute(*perm)
        
        start = time.perf_counter()
        result = permuted.contiguous()
        end = time.perf_counter()
        
        contiguous_times.append(end - start)
    
    # 计算数据大小
    num_elements = np.prod(shape)
    size_mb = num_elements * 4 / (1024 * 1024)  # FP32
    
    avg_contiguous_time = np.mean(contiguous_times)
    # 带宽计算 (读+写)
    bandwidth = (2 * size_mb / 1024) / avg_contiguous_time
    
    return {
        "shape": shape,
        "permutation": perm,
        "size_mb": size_mb,
        "permute_time_us": np.mean(permute_times) * 1e6,
        "contiguous_time_ms": np.mean(contiguous_times) * 1000,
        "contiguous_std_ms": np.std(contiguous_times) * 1000,
        "bandwidth_gbps": bandwidth
    }

def test_easy_transpose():
    """测试简单转置 (交换末尾两维)"""
    print("=" * 80)
    print("简单转置测试 (交换末尾两维)")
    print("=" * 80)
    print("这种转置具有良好的局部性，性能较好")
    print("-" * 80)
    
    test_cases = [
        ((1024, 1024), (1, 0)),
        ((100, 512, 512), (0, 2, 1)),
        ((32, 256, 256), (0, 2, 1)),
        ((16, 128, 128, 8), (0, 1, 3, 2)),
    ]
    
    results = []
    for shape, perm in test_cases:
        print(f"\n形状: {shape}, 重排: {perm}")
        result = benchmark_permute(shape, perm)
        results.append(result)
        
        print(f"  数据大小: {result['size_mb']:.2f} MB")
        print(f"  Permute 时间: {result['permute_time_us']:.2f} us")
        print(f"  Contiguous 时间: {result['contiguous_time_ms']:.2f} ± {result['contiguous_std_ms']:.2f} ms")
        print(f"  带宽: {result['bandwidth_gbps']:.2f} GB/s")
    
    return results

def test_hard_transpose():
    """测试困难转置 (交换首尾维度)"""
    print("\n" + "=" * 80)
    print("困难转置测试 (交换首尾维度)")
    print("=" * 80)
    print("这种转置跨度最大，cache miss 率高")
    print("-" * 80)
    
    test_cases = [
        ((1024, 1024), (1, 0)),  # 2D 转置
        ((100, 512, 512), (2, 1, 0)),  # 首尾交换
        ((32, 256, 256), (2, 1, 0)),
        ((64, 64, 64, 64), (3, 1, 2, 0)),  # 4D 首尾交换
    ]
    
    results = []
    for shape, perm in test_cases:
        print(f"\n形状: {shape}, 重排: {perm}")
        result = benchmark_permute(shape, perm)
        results.append(result)
        
        print(f"  数据大小: {result['size_mb']:.2f} MB")
        print(f"  Permute 时间: {result['permute_time_us']:.2f} us")
        print(f"  Contiguous 时间: {result['contiguous_time_ms']:.2f} ± {result['contiguous_std_ms']:.2f} ms")
        print(f"  带宽: {result['bandwidth_gbps']:.2f} GB/s")
    
    return results

def test_tensor_network_patterns():
    """测试张量网络中常见的重排模式"""
    print("\n" + "=" * 80)
    print("张量网络常见重排模式")
    print("=" * 80)
    
    # 常见的 MPS/PEPS 张量重排
    test_cases = [
        # MPS 张量: (bond_left, physical, bond_right)
        ((64, 8, 64), (2, 1, 0), "MPS left-right swap"),
        ((128, 4, 128), (1, 0, 2), "MPS physical-left swap"),
        
        # PEPS 张量: (bond_up, bond_right, bond_down, bond_left, physical)
        ((16, 16, 16, 16, 4), (4, 0, 1, 2, 3), "PEPS physical to front"),
        ((32, 32, 32, 32, 2), (1, 2, 3, 4, 0), "PEPS rotation"),
        
        # 收缩后的中间张量
        ((8, 8, 8, 8, 8, 8), (5, 4, 3, 2, 1, 0), "6D reverse"),
    ]
    
    results = []
    for shape, perm, description in test_cases:
        print(f"\n{description}")
        print(f"  形状: {shape}, 重排: {perm}")
        result = benchmark_permute(shape, perm)
        result["description"] = description
        results.append(result)
        
        print(f"  数据大小: {result['size_mb']:.2f} MB")
        print(f"  Contiguous 时间: {result['contiguous_time_ms']:.2f} ms")
        print(f"  带宽: {result['bandwidth_gbps']:.2f} GB/s")
    
    return results

def test_transpose_vs_einsum():
    """对比显式转置 vs einsum 的性能"""
    print("\n" + "=" * 80)
    print("显式转置 vs einsum 对比")
    print("=" * 80)
    
    results = []
    
    # 测试案例: 矩阵乘法需要转置
    M, N, K = 512, 512, 512
    
    print(f"\n测试: ({M}x{K}) @ ({K}x{N})")
    print("-" * 80)
    
    # 方法 1: 显式转置
    A = torch.randn(M, K)
    B = torch.randn(N, K)  # 注意这里是 NxK，需要转置
    
    # 预热
    for _ in range(10):
        B_t = B.t().contiguous()
        _ = torch.matmul(A, B_t)
    
    times_explicit = []
    for _ in range(50):
        start = time.perf_counter()
        B_t = B.t().contiguous()
        result = torch.matmul(A, B_t)
        end = time.perf_counter()
        times_explicit.append(end - start)
    
    avg_explicit = np.mean(times_explicit) * 1000
    
    # 方法 2: einsum (隐式处理)
    for _ in range(10):
        _ = torch.einsum('mk,nk->mn', A, B)
    
    times_einsum = []
    for _ in range(50):
        start = time.perf_counter()
        result = torch.einsum('mk,nk->mn', A, B)
        end = time.perf_counter()
        times_einsum.append(end - start)
    
    avg_einsum = np.mean(times_einsum) * 1000
    
    result = {
        "shape": f"{M}x{K} @ {K}x{N}",
        "explicit_transpose_ms": avg_explicit,
        "einsum_ms": avg_einsum,
        "ratio": avg_explicit / avg_einsum
    }
    results.append(result)
    
    print(f"  显式转置: {avg_explicit:.2f} ms")
    print(f"  einsum: {avg_einsum:.2f} ms")
    print(f"  比率: {result['ratio']:.2f}x")
    
    return results

def compare_easy_vs_hard(easy_results, hard_results):
    """对比简单转置和困难转置的性能差异"""
    print("\n" + "=" * 80)
    print("简单 vs 困难转置对比")
    print("=" * 80)
    
    analysis = {
        "avg_easy_bandwidth": 0,
        "avg_hard_bandwidth": 0,
        "performance_gap": 0,
        "recommendations": []
    }
    
    if easy_results and hard_results:
        # 选择相似大小的样本比较
        easy_bw = [r["bandwidth_gbps"] for r in easy_results if 10 < r["size_mb"] < 200]
        hard_bw = [r["bandwidth_gbps"] for r in hard_results if 10 < r["size_mb"] < 200]
        
        if easy_bw and hard_bw:
            avg_easy = np.mean(easy_bw)
            avg_hard = np.mean(hard_bw)
            
            analysis["avg_easy_bandwidth"] = avg_easy
            analysis["avg_hard_bandwidth"] = avg_hard
            analysis["performance_gap"] = (avg_easy / avg_hard - 1) * 100
            
            print(f"\n简单转置平均带宽: {avg_easy:.2f} GB/s")
            print(f"困难转置平均带宽: {avg_hard:.2f} GB/s")
            print(f"性能差距: {analysis['performance_gap']:.1f}%")
            
            if analysis["performance_gap"] > 50:
                print("\n⚠ 转置模式对性能影响显著")
                analysis["recommendations"].append("在张量网络设计中优先选择局部转置")
                analysis["recommendations"].append("避免大跨度的维度重排，考虑分块处理")
            elif analysis["performance_gap"] > 20:
                print("\n✓ 转置模式有一定影响")
                analysis["recommendations"].append("关注热点路径上的转置操作")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage3"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "transpose_cost.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第三阶段测试: 张量转置代价")
    print("=" * 80)
    
    all_results = {
        "test_name": "transpose_cost",
        "stage": 3
    }
    
    # 执行测试
    all_results["easy_transpose"] = test_easy_transpose()
    all_results["hard_transpose"] = test_hard_transpose()
    all_results["tensor_network_patterns"] = test_tensor_network_patterns()
    all_results["transpose_vs_einsum"] = test_transpose_vs_einsum()
    
    # 对比分析
    all_results["comparison"] = compare_easy_vs_hard(
        all_results["easy_transpose"],
        all_results["hard_transpose"]
    )
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["comparison"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n通用建议:")
    print("  1. 尽量将常用维度放在内存连续的位置")
    print("  2. 考虑使用 einsum 让 PyTorch 自动优化转置")
    print("  3. 对于频繁转置的张量，考虑调整初始存储顺序")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
