#!/usr/bin/env python3
"""
第二阶段测试 2: 算子融合与开销测试
验证 CPU 是否能有效处理算子融合，减少中间内存读写
"""

import sys
import torch
import time
import json
import os
import numpy as np

def benchmark_operation(func, *args, num_warmup=2, num_trials=10):
    """通用操作基准测试（快速模式：少量预热和试验，约 5s 内完成）"""
    # 预热
    for _ in range(num_warmup):
        _ = func(*args)
    
    # 正式测试
    times = []
    for _ in range(num_trials):
        # 重新生成输入以避免缓存
        new_args = [torch.randn_like(arg) if isinstance(arg, torch.Tensor) else arg for arg in args]
        
        start = time.perf_counter()
        result = func(*new_args)
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    return {
        "avg_time_us": avg_time * 1e6,
        "std_time_us": std_time * 1e6,
        "min_time_us": min_time * 1e6
    }

def test_matmul_add_fusion():
    """测试矩阵乘加融合: C = A @ B + C"""
    print("=" * 80)
    print("矩阵乘加融合测试")
    print("=" * 80)
    
    # 缩小规模以便约 5s 内完成
    sizes = [512, 1024]
    results = []
    
    for N in sizes:
        print(f"\n测试规模: {N}x{N}")
        print("-" * 80)
        
        # 准备数据
        A_template = torch.randn(N, N)
        B_template = torch.randn(N, N)
        C_template = torch.randn(N, N)
        
        # 方法 1: 分离操作 (A @ B) + C
        def separated_ops(A, B, C):
            temp = torch.matmul(A, B)
            return temp + C
        
        result_sep = benchmark_operation(separated_ops, A_template, B_template, C_template)
        
        # 方法 2: addmm 融合算子
        def fused_addmm(A, B, C):
            return torch.addmm(C, A, B)
        
        result_fused = benchmark_operation(fused_addmm, A_template, B_template, C_template)
        
        # 计算加速比
        speedup = result_sep["avg_time_us"] / result_fused["avg_time_us"]
        
        result = {
            "size": N,
            "separated_time_us": result_sep["avg_time_us"],
            "fused_time_us": result_fused["avg_time_us"],
            "speedup": speedup
        }
        results.append(result)
        
        print(f"  分离操作: {result_sep['avg_time_us']:.2f} us")
        print(f"  融合操作: {result_fused['avg_time_us']:.2f} us")
        print(f"  加速比: {speedup:.2f}x")
    
    return results

def test_baddbmm_fusion():
    """测试批量矩阵乘加: C = beta*C + alpha*(A @ B)"""
    print("\n" + "=" * 80)
    print("批量矩阵乘加融合测试 (baddbmm)")
    print("=" * 80)
    
    batch_size = 16
    matrix_sizes = [256, 512]
    results = []
    
    for N in matrix_sizes:
        print(f"\n测试规模: batch={batch_size}, {N}x{N}")
        print("-" * 80)
        
        A_template = torch.randn(batch_size, N, N)
        B_template = torch.randn(batch_size, N, N)
        C_template = torch.randn(batch_size, N, N)
        alpha = 1.0
        beta = 1.0
        
        # 方法 1: 分离操作
        def separated_batch_ops(A, B, C):
            temp = torch.bmm(A, B)
            return beta * C + alpha * temp
        
        result_sep = benchmark_operation(separated_batch_ops, A_template, B_template, C_template)
        
        # 方法 2: baddbmm 融合
        def fused_baddbmm(A, B, C):
            return torch.baddbmm(C, A, B, beta=beta, alpha=alpha)
        
        result_fused = benchmark_operation(fused_baddbmm, A_template, B_template, C_template)
        
        speedup = result_sep["avg_time_us"] / result_fused["avg_time_us"]
        
        result = {
            "batch_size": batch_size,
            "matrix_size": N,
            "separated_time_us": result_sep["avg_time_us"],
            "fused_time_us": result_fused["avg_time_us"],
            "speedup": speedup
        }
        results.append(result)
        
        print(f"  分离操作: {result_sep['avg_time_us']:.2f} us")
        print(f"  融合操作: {result_fused['avg_time_us']:.2f} us")
        print(f"  加速比: {speedup:.2f}x")
    
    return results

def test_activation_fusion():
    """测试激活函数融合"""
    print("\n" + "=" * 80)
    print("激活函数融合测试")
    print("=" * 80)
    
    # 缩小规模以便约 5s 内完成（原 1024–16384）
    sizes = [512, 1024]
    results = []
    
    for N in sizes:
        print(f"\n测试规模: {N}x{N}")
        print("-" * 80)
        
        A_template = torch.randn(N, N)
        B_template = torch.randn(N, N)
        
        # 测试 GELU 激活
        def separated_matmul_gelu(A, B):
            temp = torch.matmul(A, B)
            return torch.nn.functional.gelu(temp)
        
        result_sep = benchmark_operation(separated_matmul_gelu, A_template, B_template)
        
        # PyTorch 没有直接的融合 matmul+gelu，但我们可以测试开销
        def inline_gelu(A, B):
            # 手动内联可能的融合
            x = torch.matmul(A, B)
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
        
        result_inline = benchmark_operation(inline_gelu, A_template, B_template)
        
        result = {
            "size": N,
            "gelu_builtin_us": result_sep["avg_time_us"],
            "gelu_inline_us": result_inline["avg_time_us"],
            "ratio": result_sep["avg_time_us"] / result_inline["avg_time_us"]
        }
        results.append(result)
        
        print(f"  内置 GELU: {result_sep['avg_time_us']:.2f} us")
        print(f"  内联 GELU: {result_inline['avg_time_us']:.2f} us")
        print(f"  比率: {result['ratio']:.2f}x")
    
    return results

def test_memory_overhead():
    """测试内存分配开销"""
    print("\n" + "=" * 80)
    print("内存分配开销测试")
    print("=" * 80)
    
    # 缩小规模与试验次数以便约 5s 内完成
    sizes = [512, 1024]
    results = []
    
    for N in sizes:
        print(f"\n测试规模: {N}x{N}")
        print("-" * 80)
        
        A = torch.randn(N, N)
        B = torch.randn(N, N)
        
        # 测试 in-place 操作 vs 新分配
        # 方法 1: 创建新张量
        def allocate_new(A, B):
            C = torch.zeros(N, N)
            C[:] = A + B
            return C
        
        result_new = benchmark_operation(allocate_new, A, B, num_trials=15)
        
        # 方法 2: 预分配后 in-place
        C_preallocated = torch.zeros(N, N)
        def inplace_op(A, B, C):
            torch.add(A, B, out=C)
            return C
        
        result_inplace = benchmark_operation(inplace_op, A, B, C_preallocated, num_trials=15)
        
        speedup = result_new["avg_time_us"] / result_inplace["avg_time_us"]
        
        result = {
            "size": N,
            "allocate_new_us": result_new["avg_time_us"],
            "inplace_us": result_inplace["avg_time_us"],
            "speedup": speedup
        }
        results.append(result)
        
        print(f"  新分配: {result_new['avg_time_us']:.2f} us")
        print(f"  In-place: {result_inplace['avg_time_us']:.2f} us")
        print(f"  加速比: {speedup:.2f}x")
    
    return results

def analyze_results(all_results):
    """分析融合效果"""
    print("\n" + "=" * 80)
    print("融合效果分析")
    print("=" * 80)
    
    analysis = {
        "avg_addmm_speedup": 0,
        "avg_baddbmm_speedup": 0,
        "avg_inplace_speedup": 0,
        "fusion_effectiveness": "unknown",
        "recommendations": []
    }
    
    # 分析 addmm 融合
    if "matmul_add" in all_results:
        speedups = [r["speedup"] for r in all_results["matmul_add"]]
        avg_speedup = np.mean(speedups)
        analysis["avg_addmm_speedup"] = avg_speedup
        print(f"\naddmm 平均加速比: {avg_speedup:.2f}x")
        
        if avg_speedup > 1.5:
            print("  ✓ 融合效果显著")
        elif avg_speedup > 1.1:
            print("  ✓ 融合有一定效果")
        else:
            print("  ⚠ 融合效果不明显")
            analysis["recommendations"].append("检查 BLAS 库是否支持融合操作")
    
    # 分析 baddbmm 融合
    if "baddbmm" in all_results:
        speedups = [r["speedup"] for r in all_results["baddbmm"]]
        avg_speedup = np.mean(speedups)
        analysis["avg_baddbmm_speedup"] = avg_speedup
        print(f"\nbaddbmm 平均加速比: {avg_speedup:.2f}x")
    
    # 分析 in-place 操作
    if "memory_overhead" in all_results:
        speedups = [r["speedup"] for r in all_results["memory_overhead"]]
        avg_speedup = np.mean(speedups)
        analysis["avg_inplace_speedup"] = avg_speedup
        print(f"\nIn-place 操作平均加速比: {avg_speedup:.2f}x")
        
        if avg_speedup > 2.0:
            print("  ✓ 内存分配开销显著")
            analysis["recommendations"].append("在张量网络训练中尽量使用 in-place 操作")
    
    # 综合评估
    if analysis["avg_addmm_speedup"] > 1.3 and analysis["avg_inplace_speedup"] > 1.5:
        analysis["fusion_effectiveness"] = "good"
    elif analysis["avg_addmm_speedup"] > 1.1:
        analysis["fusion_effectiveness"] = "moderate"
    else:
        analysis["fusion_effectiveness"] = "poor"
        analysis["recommendations"].append("考虑使用更优化的 BLAS 库或手动优化算子")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage2"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "op_fusion.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第二阶段测试: 算子融合与开销")
    print("=" * 80)
    
    all_results = {
        "test_name": "operator_fusion",
        "stage": 2
    }
    
    # 执行测试
    all_results["matmul_add"] = test_matmul_add_fusion()
    all_results["baddbmm"] = test_baddbmm_fusion()
    all_results["activation"] = test_activation_fusion()
    all_results["memory_overhead"] = test_memory_overhead()
    
    # 分析结果
    all_results["analysis"] = analyze_results(all_results)
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✓ 算子融合效果良好，无需特别优化")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
