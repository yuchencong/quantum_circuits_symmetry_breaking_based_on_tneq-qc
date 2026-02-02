#!/usr/bin/env python3
"""
第三阶段测试 3: L2 Cache 命中率与 Tiling 敏感度
找出导致 Cache 抖动的 Tensor 尺寸临界点
"""

import sys
import torch
import time
import json
import os
import numpy as np

# 富岳 A64FX: 每个 CMG 有 8MB L2 Cache (共享)

def benchmark_matmul_sizes(sizes, num_trials=30):
    """测试不同矩阵尺寸的性能，观察 cache 影响"""
    results = []
    
    for N in sizes:
        # 计算矩阵占用的内存
        matrix_size_mb = (N * N * 4) / (1024 * 1024)  # FP32
        total_size_mb = 3 * matrix_size_mb  # A, B, C 三个矩阵
        
        # 预热
        for _ in range(5):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            _ = torch.matmul(A, B)
        
        # 测试
        times = []
        for _ in range(num_trials):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            
            start = time.perf_counter()
            C = torch.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # 计算 GFLOPS
        flops = 2 * N ** 3
        gflops = flops / (avg_time * 1e9)
        
        result = {
            "size": N,
            "matrix_size_mb": matrix_size_mb,
            "total_size_mb": total_size_mb,
            "avg_time_ms": avg_time * 1000,
            "std_time_ms": std_time * 1000,
            "gflops": gflops,
            "cv_percent": (std_time / avg_time) * 100  # 变异系数
        }
        results.append(result)
    
    return results

def test_cache_working_set():
    """测试 cache 工作集大小的影响"""
    print("=" * 80)
    print("Cache 工作集测试")
    print("=" * 80)
    print("A64FX L2 Cache: 8MB per CMG")
    print("-" * 80)
    
    # 设计尺寸跨越 L2 Cache 大小
    # 8MB / 4 bytes = 2M elements = sqrt(2M) ≈ 1414 for square matrix
    # 测试从小于 L2 到远大于 L2 的范围
    
    sizes = [
        # L2 以内
        256,   # ~0.75 MB
        512,   # ~3 MB
        724,   # ~6 MB (接近 L2)
        
        # L2 临界点附近
        1024,  # ~12 MB (超出 L2)
        1448,  # ~25 MB
        
        # 远超 L2
        2048,  # ~48 MB
        2896,  # ~100 MB
        4096,  # ~192 MB
    ]
    
    results = benchmark_matmul_sizes(sizes)
    
    print("")
    print(f"{'Size':>6} | {'Total MB':>9} | {'Time (ms)':>10} | {'GFLOPS':>8} | {'CV%':>6}")
    print("-" * 60)
    
    for r in results:
        in_cache = "✓" if r["total_size_mb"] < 8 else "✗"
        print(f"{r['size']:>6} | {r['total_size_mb']:>8.1f} {in_cache} | "
              f"{r['avg_time_ms']:>10.2f} | {r['gflops']:>8.1f} | {r['cv_percent']:>5.1f}")
    
    return results

def test_blocking_strategy():
    """测试分块策略对 cache 利用的影响"""
    print("\n" + "=" * 80)
    print("矩阵分块策略测试")
    print("=" * 80)
    
    N = 2048  # 大矩阵
    block_sizes = [64, 128, 256, 512, 1024]
    
    results = []
    
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    
    # 方法 1: 不分块直接计算
    print("\n不分块计算:")
    print("-" * 80)
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        C = torch.matmul(A, B)
        end = time.perf_counter()
        times.append(end - start)
    
    no_block_time = np.mean(times)
    flops = 2 * N ** 3
    no_block_gflops = flops / (no_block_time * 1e9)
    
    print(f"  时间: {no_block_time*1000:.2f} ms")
    print(f"  性能: {no_block_gflops:.2f} GFLOPS")
    
    results.append({
        "strategy": "no_blocking",
        "block_size": N,
        "time_ms": no_block_time * 1000,
        "gflops": no_block_gflops
    })
    
    # 方法 2: 手动分块计算
    print("\n分块计算:")
    print("-" * 80)
    
    for block_size in block_sizes:
        print(f"\n  块大小: {block_size}x{block_size}")
        
        block_mb = (block_size * block_size * 4) / (1024 * 1024)
        print(f"  块大小: {block_mb:.2f} MB")
        
        # 简单的分块乘法实现
        def blocked_matmul(A, B, block_size):
            N = A.shape[0]
            C = torch.zeros(N, N)
            
            for i in range(0, N, block_size):
                for j in range(0, N, block_size):
                    for k in range(0, N, block_size):
                        i_end = min(i + block_size, N)
                        j_end = min(j + block_size, N)
                        k_end = min(k + block_size, N)
                        
                        A_block = A[i:i_end, k:k_end]
                        B_block = B[k:k_end, j:j_end]
                        C[i:i_end, j:j_end] += torch.matmul(A_block, B_block)
            
            return C
        
        # 预热
        _ = blocked_matmul(A, B, block_size)
        
        # 测试
        times = []
        for _ in range(5):  # 分块计算较慢，减少试验次数
            start = time.perf_counter()
            C_blocked = blocked_matmul(A, B, block_size)
            end = time.perf_counter()
            times.append(end - start)
        
        blocked_time = np.mean(times)
        blocked_gflops = flops / (blocked_time * 1e9)
        
        result = {
            "strategy": "blocked",
            "block_size": block_size,
            "block_mb": block_mb,
            "time_ms": blocked_time * 1000,
            "gflops": blocked_gflops,
            "vs_no_block": no_block_gflops / blocked_gflops
        }
        results.append(result)
        
        print(f"  时间: {blocked_time*1000:.2f} ms")
        print(f"  性能: {blocked_gflops:.2f} GFLOPS")
        print(f"  vs 不分块: {result['vs_no_block']:.2f}x")
    
    return results

def test_access_patterns():
    """测试不同访问模式的 cache 性能"""
    print("\n" + "=" * 80)
    print("访问模式 Cache 性能测试")
    print("=" * 80)
    
    size = 4096
    tensor = torch.randn(size, size)
    
    results = []
    
    # 模式 1: 行优先访问 (连续)
    print("\n行优先访问 (Cache 友好):")
    print("-" * 80)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        sum_val = 0.0
        for i in range(size):
            sum_val += tensor[i, :].sum().item()
        end = time.perf_counter()
        times.append(end - start)
    
    row_major_time = np.mean(times)
    print(f"  时间: {row_major_time*1000:.2f} ms")
    
    results.append({
        "pattern": "row_major",
        "time_ms": row_major_time * 1000
    })
    
    # 模式 2: 列优先访问 (跨步)
    print("\n列优先访问 (Cache 不友好):")
    print("-" * 80)
    
    times = []
    for _ in range(20):
        start = time.perf_counter()
        sum_val = 0.0
        for j in range(size):
            sum_val += tensor[:, j].sum().item()
        end = time.perf_counter()
        times.append(end - start)
    
    col_major_time = np.mean(times)
    print(f"  时间: {col_major_time*1000:.2f} ms")
    print(f"  vs 行优先: {col_major_time/row_major_time:.2f}x")
    
    results.append({
        "pattern": "column_major",
        "time_ms": col_major_time * 1000,
        "vs_row_major": col_major_time / row_major_time
    })
    
    return results

def analyze_cache_effects(working_set_results):
    """分析 cache 效应"""
    print("\n" + "=" * 80)
    print("Cache 效应分析")
    print("=" * 80)
    
    analysis = {
        "l2_cache_size_mb": 8,
        "cache_friendly_size": 0,
        "performance_cliff": False,
        "cliff_size": 0,
        "recommendations": []
    }
    
    # 寻找性能悬崖 (performance cliff)
    if len(working_set_results) > 1:
        # 找出 L2 内最好性能
        in_cache = [r for r in working_set_results if r["total_size_mb"] < 8]
        out_cache = [r for r in working_set_results if r["total_size_mb"] >= 8]
        
        if in_cache and out_cache:
            best_in_cache_gflops = max([r["gflops"] for r in in_cache])
            first_out_cache = out_cache[0]
            
            performance_drop = (1 - first_out_cache["gflops"] / best_in_cache_gflops) * 100
            
            print(f"\nL2 Cache 内最佳性能: {best_in_cache_gflops:.2f} GFLOPS")
            print(f"超出 L2 后性能: {first_out_cache['gflops']:.2f} GFLOPS")
            print(f"性能下降: {performance_drop:.1f}%")
            
            if performance_drop > 20:
                analysis["performance_cliff"] = True
                analysis["cliff_size"] = first_out_cache["size"]
                print(f"\n⚠ 检测到性能悬崖在矩阵尺寸 {first_out_cache['size']} 附近")
                analysis["recommendations"].append(
                    f"避免使用尺寸在 {first_out_cache['size']-200}~{first_out_cache['size']+200} 范围的矩阵"
                )
            
            # 推荐 cache 友好尺寸
            if in_cache:
                best_in_cache = max(in_cache, key=lambda x: x["gflops"])
                analysis["cache_friendly_size"] = best_in_cache["size"]
                print(f"\n✓ 推荐 Cache 友好尺寸: ~{best_in_cache['size']}")
                analysis["recommendations"].append(
                    f"对于小规模计算，保持工作集在 {best_in_cache['size']} 以下"
                )
        
        # 检查变异系数
        high_variance = [r for r in working_set_results if r["cv_percent"] > 5]
        if high_variance:
            print(f"\n⚠ 检测到 {len(high_variance)} 个高变异性尺寸")
            analysis["recommendations"].append("高变异性可能表示 cache 抖动，考虑调整尺寸")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage3"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cache_sensitivity.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第三阶段测试: L2 Cache 敏感度")
    print("=" * 80)
    
    all_results = {
        "test_name": "cache_sensitivity",
        "stage": 3
    }
    
    # 执行测试
    all_results["working_set"] = test_cache_working_set()
    all_results["blocking"] = test_blocking_strategy()
    all_results["access_patterns"] = test_access_patterns()
    
    # 分析结果
    all_results["analysis"] = analyze_cache_effects(all_results["working_set"])
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n通用建议:")
    print("  1. 在张量网络中使用分块稀疏张量，控制块大小在 L2 内")
    print("  2. 优先使用行优先的访问模式")
    print("  3. 避免在性能悬崖附近的尺寸")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
