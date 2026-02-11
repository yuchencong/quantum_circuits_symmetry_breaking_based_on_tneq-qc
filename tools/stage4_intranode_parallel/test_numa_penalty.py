#!/usr/bin/env python3
"""
第四阶段测试 1: 跨 CMG 访问惩罚 (NUMA Penalty)
测试从不同 CMG 访问内存的延迟差异
"""

import sys
import torch
import time
import json
import os
import numpy as np
import subprocess

def get_numa_info():
    """获取 NUMA 节点信息"""
    print("=" * 80)
    print("NUMA 架构信息")
    print("=" * 80)
    
    info = {}
    
    try:
        # 尝试使用 numactl 获取信息
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
            info["numactl_output"] = result.stdout
        
        # 尝试使用 lscpu
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n" + "-" * 80)
            print("CPU 信息:")
            print(result.stdout)
            info["lscpu_output"] = result.stdout
            
    except Exception as e:
        print(f"无法获取 NUMA 信息: {e}")
        info["error"] = str(e)
    
    return info

def test_local_vs_remote_access():
    """
    测试本地 vs 远程内存访问
    注意: 需要配合 numactl 使用才能真正测试
    """
    print("\n" + "=" * 80)
    print("本地 vs 远程内存访问测试")
    print("=" * 80)
    print("提示: 需要使用 numactl 绑定进程和内存到特定 CMG")
    print("-" * 80)
    
    sizes = [1*1024, 10*1024, 100*1024, 1024*1024]  # 元素数量
    results = []
    
    for size in sizes:
        print(f"\n测试大小: {size} 元素 ({size*4/(1024*1024):.2f} MB)")
        
        # 创建数据
        data = torch.randn(size)
        
        # 预热
        for _ in range(10):
            _ = data.sum()
        
        # 测试访问延迟
        times = []
        for _ in range(100):
            start = time.perf_counter()
            result = data.sum()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        
        # 计算带宽
        size_mb = size * 4 / (1024 * 1024)
        bandwidth = size_mb / (avg_time * 1024)  # GB/s
        
        result = {
            "size": size,
            "size_mb": size_mb,
            "avg_time_us": avg_time * 1e6,
            "std_time_us": std_time * 1e6,
            "min_time_us": min_time * 1e6,
            "bandwidth_gbps": bandwidth
        }
        results.append(result)
        
        print(f"  平均时间: {avg_time*1e6:.2f} ± {std_time*1e6:.2f} us")
        print(f"  带宽: {bandwidth:.2f} GB/s")
    
    return results

def test_random_vs_sequential():
    """测试随机访问 vs 顺序访问的性能差异"""
    print("\n" + "=" * 80)
    print("随机访问 vs 顺序访问")
    print("=" * 80)
    
    size = 10 * 1024 * 1024  # 40MB
    data = torch.randn(size)
    
    results = []
    
    # 测试 1: 顺序访问
    print("\n顺序访问:")
    print("-" * 80)
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = data.sum()
        end = time.perf_counter()
        times.append(end - start)
    
    seq_time = np.mean(times)
    print(f"  时间: {seq_time*1000:.2f} ms")
    
    results.append({
        "pattern": "sequential",
        "time_ms": seq_time * 1000
    })
    
    # 测试 2: 随机访问
    print("\n随机访问 (stride 访问):")
    print("-" * 80)
    
    stride = 17  # 使用素数步长来模拟随机访问
    indices = torch.arange(0, size, stride)
    
    times = []
    for _ in range(50):
        start = time.perf_counter()
        result = data[indices].sum()
        end = time.perf_counter()
        times.append(end - start)
    
    random_time = np.mean(times)
    print(f"  时间: {random_time*1000:.2f} ms")
    print(f"  vs 顺序: {random_time/seq_time:.2f}x")
    
    results.append({
        "pattern": "random_stride",
        "stride": stride,
        "time_ms": random_time * 1000,
        "vs_sequential": random_time / seq_time
    })
    
    return results

def test_thread_affinity_impact():
    """测试线程亲和性设置的影响"""
    print("\n" + "=" * 80)
    print("线程数量与性能")
    print("=" * 80)
    print("测试不同线程数的矩阵乘法性能")
    print("-" * 80)
    
    N = 2048
    thread_counts = [1, 2, 4, 6, 8, 12]  # 12 是单 CMG 的核心数
    
    results = []
    
    for num_threads in thread_counts:
        torch.set_num_threads(num_threads)
        print(f"\n线程数: {num_threads}")
        
        # 预热
        for _ in range(5):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            _ = torch.matmul(A, B)
        
        # 测试
        times = []
        for _ in range(20):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            
            start = time.perf_counter()
            C = torch.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        flops = 2 * N ** 3
        gflops = flops / (avg_time * 1e9)
        
        result = {
            "num_threads": num_threads,
            "time_ms": avg_time * 1000,
            "gflops": gflops
        }
        results.append(result)
        
        print(f"  时间: {avg_time*1000:.2f} ms")
        print(f"  性能: {gflops:.2f} GFLOPS")
        
        # 计算扩展效率
        if len(results) > 1:
            single_thread_gflops = results[0]["gflops"]
            speedup = gflops / single_thread_gflops
            efficiency = speedup / num_threads * 100
            result["speedup"] = speedup
            result["efficiency_percent"] = efficiency
            print(f"  加速比: {speedup:.2f}x")
            print(f"  并行效率: {efficiency:.1f}%")
    
    return results

def analyze_numa_effects(all_results):
    """分析 NUMA 效应"""
    print("\n" + "=" * 80)
    print("NUMA 效应分析")
    print("=" * 80)
    
    analysis = {
        "recommendations": []
    }
    
    # 分析线程扩展性
    if "thread_scaling" in all_results:
        thread_results = all_results["thread_scaling"]
        
        if len(thread_results) >= 2:
            # 找到 12 线程（单 CMG）的结果
            thread_12 = next((r for r in thread_results if r["num_threads"] == 12), None)
            thread_1 = next((r for r in thread_results if r["num_threads"] == 1), None)
            
            if thread_12 and thread_1:
                scaling_efficiency = thread_12.get("efficiency_percent", 0)
                
                print(f"\n单 CMG (12核) 并行效率: {scaling_efficiency:.1f}%")
                
                if scaling_efficiency > 80:
                    print("✓ 并行扩展性优秀")
                    analysis["scaling_verdict"] = "excellent"
                elif scaling_efficiency > 60:
                    print("✓ 并行扩展性良好")
                    analysis["scaling_verdict"] = "good"
                    analysis["recommendations"].append("考虑优化线程绑定以提升效率")
                else:
                    print("⚠ 并行扩展性需要改进")
                    analysis["scaling_verdict"] = "poor"
                    analysis["recommendations"].append("使用 numactl 绑定线程到特定核心")
                    analysis["recommendations"].append("检查是否有线程竞争和负载不均")
    
    # 分析访问模式
    if "random_vs_sequential" in all_results:
        for r in all_results["random_vs_sequential"]:
            if r.get("pattern") == "random_stride":
                penalty = r.get("vs_sequential", 1.0)
                
                print(f"\n随机访问惩罚: {penalty:.2f}x")
                
                if penalty > 3:
                    print("⚠ 随机访问显著降低性能")
                    analysis["recommendations"].append("在张量网络中保持连续的内存访问模式")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage4"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "numa_penalty.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第四阶段测试: NUMA 惩罚测试")
    print("=" * 80)
    
    all_results = {
        "test_name": "numa_penalty",
        "stage": 4
    }
    
    # 获取 NUMA 信息
    all_results["numa_info"] = get_numa_info()
    
    # 执行测试
    all_results["local_vs_remote"] = test_local_vs_remote_access()
    all_results["random_vs_sequential"] = test_random_vs_sequential()
    all_results["thread_scaling"] = test_thread_affinity_impact()
    
    # 分析结果
    all_results["analysis"] = analyze_numa_effects(all_results)
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n通用建议:")
    print("  1. 使用 numactl 绑定进程到特定 CMG:")
    print("     numactl --cpunodebind=0 --membind=0 python script.py")
    print("  2. 对于多进程训练，每个进程绑定到不同的 CMG")
    print("  3. 避免跨 CMG 的内存访问")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
