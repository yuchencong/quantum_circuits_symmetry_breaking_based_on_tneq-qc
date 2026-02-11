#!/usr/bin/env python3
"""
第二阶段测试 1: 单 CMG 算力极限测试
绑定 1 个进程到 1 个 CMG（12 核），测试峰值性能
"""

import sys
import torch
import time
import json
import os
import numpy as np

def set_num_threads(num_threads):
    """设置线程数"""
    torch.set_num_threads(num_threads)
    print(f"设置 PyTorch 线程数: {num_threads}")
    print(f"当前线程数: {torch.get_num_threads()}")

def benchmark_gemm(M, N, K, dtype=torch.float32, num_warmup=2, num_trials=5):
    """
    基准测试矩阵乘法 (GEMM)
    C = A @ B, 其中 A 是 MxK, B 是 KxN
    （快速模式：少量预热和试验，约 5s 内完成）
    """
    # 预热
    for _ in range(num_warmup):
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, N, dtype=dtype)
        C = torch.matmul(A, B)
    
    # 正式测试
    times = []
    for _ in range(num_trials):
        A = torch.randn(M, K, dtype=dtype)
        B = torch.randn(K, N, dtype=dtype)
        
        start = time.perf_counter()
        C = torch.matmul(A, B)
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    
    # 计算 FLOPS: 2*M*N*K
    flops = 2 * M * N * K
    avg_gflops = flops / (avg_time * 1e9)
    peak_gflops = flops / (min_time * 1e9)
    
    return {
        "M": M, "N": N, "K": K,
        "dtype": str(dtype),
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "min_time_ms": min_time * 1000,
        "avg_gflops": avg_gflops,
        "peak_gflops": peak_gflops,
        "flops": flops
    }

def test_square_gemm_fp32():
    """测试方阵 FP32 GEMM"""
    print("=" * 80)
    print("FP32 方阵 GEMM 测试")
    print("=" * 80)
    
    # 缩小规模以便约 5s 内完成（原 1024–8192）
    sizes = [512, 1024, 2048]
    results = []
    
    for N in sizes:
        print(f"\n测试规模: {N}x{N} @ {N}x{N}")
        print("-" * 80)
        
        result = benchmark_gemm(N, N, N, dtype=torch.float32)
        results.append(result)
        
        print(f"  平均时间: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
        print(f"  最短时间: {result['min_time_ms']:.2f} ms")
        print(f"  平均性能: {result['avg_gflops']:.2f} GFLOPS")
        print(f"  峰值性能: {result['peak_gflops']:.2f} GFLOPS")
    
    return results

def test_square_gemm_fp16():
    """测试方阵 FP16 GEMM"""
    print("\n" + "=" * 80)
    print("FP16 方阵 GEMM 测试")
    print("=" * 80)
    
    # 缩小规模以便约 5s 内完成
    sizes = [512, 1024, 2048]
    results = []
    
    for N in sizes:
        print(f"\n测试规模: {N}x{N} @ {N}x{N}")
        print("-" * 80)
        
        try:
            result = benchmark_gemm(N, N, N, dtype=torch.float16, num_trials=5)
            results.append(result)
            
            print(f"  平均时间: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
            print(f"  最短时间: {result['min_time_ms']:.2f} ms")
            print(f"  平均性能: {result['avg_gflops']:.2f} GFLOPS")
            print(f"  峰值性能: {result['peak_gflops']:.2f} GFLOPS")
        except Exception as e:
            print(f"  错误: {e}")
            results.append({"N": N, "error": str(e)})
    
    return results

def test_rectangular_gemm():
    """测试非方阵 GEMM (常见于张量网络)"""
    print("\n" + "=" * 80)
    print("非方阵 GEMM 测试 (张量网络常见形状)")
    print("=" * 80)
    
    # 常见的张量网络收缩形状（缩小以便约 5s 内完成）
    shapes = [
        (512, 1024, 512),   # 瘦长矩阵
        (1024, 512, 1024),  # 宽短矩阵
        (256, 1024, 256),   # 极端瘦长
        (1024, 1024, 512),  # 批量小矩阵
    ]
    
    results = []
    
    for M, N, K in shapes:
        print(f"\n测试规模: {M}x{K} @ {K}x{N}")
        print("-" * 80)
        
        result = benchmark_gemm(M, N, K, dtype=torch.float32, num_trials=5)
        results.append(result)
        
        print(f"  平均时间: {result['avg_time_ms']:.2f} ms")
        print(f"  平均性能: {result['avg_gflops']:.2f} GFLOPS")
    
    return results

def analyze_results(fp32_results, fp16_results):
    """分析测试结果"""
    print("\n" + "=" * 80)
    print("性能分析")
    print("=" * 80)
    
    analysis = {
        "fp32_peak_gflops": 0,
        "fp16_peak_gflops": 0,
        "fp16_speedup": 0,
        "theoretical_peak_cmg_fp32": 768,  # 单 CMG @ 2.0GHz
        "utilization_percent": 0,
        "verdict": "unknown"
    }
    
    # 找出 FP32 最佳性能
    if fp32_results:
        fp32_peak = max([r["peak_gflops"] for r in fp32_results if "peak_gflops" in r])
        analysis["fp32_peak_gflops"] = fp32_peak
        print(f"\nFP32 峰值性能: {fp32_peak:.2f} GFLOPS")
    
    # 找出 FP16 最佳性能
    if fp16_results:
        valid_fp16 = [r for r in fp16_results if "peak_gflops" in r]
        if valid_fp16:
            fp16_peak = max([r["peak_gflops"] for r in valid_fp16])
            analysis["fp16_peak_gflops"] = fp16_peak
            print(f"FP16 峰值性能: {fp16_peak:.2f} GFLOPS")
            
            # 计算加速比
            if analysis["fp32_peak_gflops"] > 0:
                speedup = fp16_peak / analysis["fp32_peak_gflops"]
                analysis["fp16_speedup"] = speedup
                print(f"FP16 vs FP32 加速比: {speedup:.2f}x")
    
    # 计算利用率（相对于理论峰值）
    if analysis["fp32_peak_gflops"] > 0:
        utilization = (analysis["fp32_peak_gflops"] / analysis["theoretical_peak_cmg_fp32"]) * 100
        analysis["utilization_percent"] = utilization
        print(f"\n单 CMG 理论峰值: {analysis['theoretical_peak_cmg_fp32']} GFLOPS")
        print(f"实测峰值利用率: {utilization:.2f}%")
        
        # 评估
        if utilization > 70:
            analysis["verdict"] = "excellent"
            print("\n✓ 性能优秀 - 接近理论峰值")
        elif utilization > 50:
            analysis["verdict"] = "good"
            print("\n✓ 性能良好")
        elif utilization > 30:
            analysis["verdict"] = "moderate"
            print("\n⚠ 性能中等 - 仍有优化空间")
        else:
            analysis["verdict"] = "poor"
            print("\n✗ 性能较差 - 需要检查配置")
    
    return analysis

def print_environment_info():
    """打印环境信息"""
    print("\n" + "=" * 80)
    print("环境信息")
    print("=" * 80)
    
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"线程数: {torch.get_num_threads()}")
    
    # 获取 OpenMP 信息
    import os
    omp_num_threads = os.environ.get('OMP_NUM_THREADS', 'not set')
    print(f"OMP_NUM_THREADS: {omp_num_threads}")
    
    mkl_num_threads = os.environ.get('MKL_NUM_THREADS', 'not set')
    print(f"MKL_NUM_THREADS: {mkl_num_threads}")

def save_results(all_results, output_dir="../test_results/stage2"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "single_cmg_peak.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第二阶段测试: 单 CMG 算力极限")
    print("=" * 80)
    
    print_environment_info()
    
    # 设置线程数为 12 (单个 CMG 的核心数)
    # 注意：在实际运行时，应该配合 numactl 绑定到特定 CMG
    set_num_threads(12)
    
    all_results = {
        "test_name": "single_cmg_peak",
        "stage": 2,
        "num_threads": torch.get_num_threads()
    }
    
    # 执行测试
    all_results["fp32_square"] = test_square_gemm_fp32()
    all_results["fp16_square"] = test_square_gemm_fp16()
    all_results["rectangular"] = test_rectangular_gemm()
    
    # 分析结果
    all_results["analysis"] = analyze_results(
        all_results["fp32_square"],
        all_results["fp16_square"]
    )
    
    # 打印建议
    print("\n" + "=" * 80)
    print("建议")
    print("=" * 80)
    
    verdict = all_results["analysis"]["verdict"]
    if verdict == "excellent":
        print("\n✓ 性能已达到优秀水平")
        print("  - 可以继续进行下一阶段测试")
    elif verdict == "good":
        print("\n✓ 性能良好")
        print("  - 建议：尝试调整线程亲和性以进一步优化")
    else:
        print("\n⚠ 性能需要改进")
        print("  - 确保使用 numactl 绑定进程到单个 CMG")
        print("  - 检查是否链接了优化的 BLAS 库")
        print("  - 验证 SVE 指令集是否启用")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
