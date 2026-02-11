#!/usr/bin/env python3
"""
第四阶段测试 2: OpenMP vs MPI 扩展性对比
对比单进程多线程 vs 多进程的性能
"""

import sys
import torch
import time
import json
import os
import numpy as np

def test_single_process_scaling():
    """测试单进程多线程扩展性"""
    print("=" * 80)
    print("单进程多线程扩展性 (OpenMP 风格)")
    print("=" * 80)
    print("测试配置: 1 进程 x N 线程")
    print("-" * 80)
    
    N = 4096  # 大矩阵，足够占用多核
    thread_configs = [1, 4, 12, 24, 48]  # 48 = 4个CMG
    
    results = []
    
    for num_threads in thread_configs:
        torch.set_num_threads(num_threads)
        print(f"\n配置: 1 进程 x {num_threads} 线程")
        
        # 预热
        for _ in range(5):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            _ = torch.matmul(A, B)
        
        # 测试
        times = []
        for _ in range(15):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            
            start = time.perf_counter()
            C = torch.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        flops = 2 * N ** 3
        gflops = flops / (avg_time * 1e9)
        
        result = {
            "num_processes": 1,
            "num_threads": num_threads,
            "total_workers": num_threads,
            "time_ms": avg_time * 1000,
            "std_ms": std_time * 1000,
            "gflops": gflops
        }
        
        # 计算扩展效率
        if len(results) > 0:
            baseline_gflops = results[0]["gflops"]
            speedup = gflops / baseline_gflops
            efficiency = (speedup / num_threads) * 100
            result["speedup"] = speedup
            result["efficiency_percent"] = efficiency
            
            print(f"  时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"  性能: {gflops:.2f} GFLOPS")
            print(f"  加速比: {speedup:.2f}x")
            print(f"  并行效率: {efficiency:.1f}%")
        else:
            print(f"  时间: {avg_time*1000:.2f} ms")
            print(f"  性能: {gflops:.2f} GFLOPS")
            result["speedup"] = 1.0
            result["efficiency_percent"] = 100.0
        
        results.append(result)
    
    return results

def test_memory_bandwidth_scaling():
    """测试内存带宽密集型任务的扩展性"""
    print("\n" + "=" * 80)
    print("内存带宽密集型任务扩展性")
    print("=" * 80)
    print("测试任务: 大向量加法 (带宽瓶颈)")
    print("-" * 80)
    
    size = 100 * 1024 * 1024  # 400MB
    thread_configs = [1, 4, 12, 24, 48]
    
    results = []
    
    for num_threads in thread_configs:
        torch.set_num_threads(num_threads)
        print(f"\n线程数: {num_threads}")
        
        # 预热
        for _ in range(5):
            a = torch.randn(size)
            b = torch.randn(size)
            _ = a + b
        
        # 测试
        times = []
        for _ in range(30):
            a = torch.randn(size)
            b = torch.randn(size)
            
            start = time.perf_counter()
            c = a + b
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        # 计算带宽 (读2次 + 写1次 = 3倍数据量)
        size_gb = (size * 4) / (1024 ** 3)
        bandwidth = (3 * size_gb) / avg_time
        
        result = {
            "num_threads": num_threads,
            "time_ms": avg_time * 1000,
            "bandwidth_gbps": bandwidth
        }
        
        if len(results) > 0:
            baseline_bw = results[0]["bandwidth_gbps"]
            speedup = bandwidth / baseline_bw
            result["speedup"] = speedup
            
            print(f"  时间: {avg_time*1000:.2f} ms")
            print(f"  带宽: {bandwidth:.2f} GB/s")
            print(f"  加速比: {speedup:.2f}x")
        else:
            print(f"  时间: {avg_time*1000:.2f} ms")
            print(f"  带宽: {bandwidth:.2f} GB/s")
        
        results.append(result)
    
    return results

def simulate_mpi_workload():
    """
    模拟 MPI 多进程工作负载
    注意: 这只是模拟，真正的 MPI 测试在第五阶段
    """
    print("\n" + "=" * 80)
    print("模拟 MPI 多进程工作负载")
    print("=" * 80)
    print("模拟: 4 进程 x 12 线程 (每个进程一个 CMG)")
    print("-" * 80)
    
    # 模拟每个进程的工作量
    N = 2048  # 每个进程处理的矩阵大小
    num_processes_sim = 4
    threads_per_process = 12
    
    torch.set_num_threads(threads_per_process)
    
    print(f"\n每个进程配置: {threads_per_process} 线程")
    print(f"矩阵大小: {N}x{N}")
    
    # 模拟并行执行（实际是串行执行4次来估算）
    process_times = []
    
    for proc_id in range(num_processes_sim):
        times = []
        for _ in range(10):
            A = torch.randn(N, N)
            B = torch.randn(N, N)
            
            start = time.perf_counter()
            C = torch.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        process_times.append(avg_time)
    
    # 假设完美并行，取最慢的进程时间
    max_time = max(process_times)
    avg_proc_time = np.mean(process_times)
    
    # 计算总吞吐量
    flops_per_process = 2 * N ** 3
    total_flops = flops_per_process * num_processes_sim
    total_gflops = total_flops / (max_time * 1e9)
    
    print(f"\n模拟结果:")
    print(f"  各进程时间: {[f'{t*1000:.2f}' for t in process_times]} ms")
    print(f"  最慢进程: {max_time*1000:.2f} ms")
    print(f"  平均时间: {avg_proc_time*1000:.2f} ms")
    print(f"  总吞吐量: {total_gflops:.2f} GFLOPS")
    print(f"  每进程吞吐量: {total_gflops/num_processes_sim:.2f} GFLOPS")
    
    result = {
        "num_processes": num_processes_sim,
        "threads_per_process": threads_per_process,
        "max_time_ms": max_time * 1000,
        "avg_time_ms": avg_proc_time * 1000,
        "total_gflops": total_gflops,
        "gflops_per_process": total_gflops / num_processes_sim
    }
    
    return result

def compare_strategies(openmp_results, mpi_simulation):
    """对比 OpenMP 和 MPI 策略"""
    print("\n" + "=" * 80)
    print("OpenMP vs MPI 策略对比")
    print("=" * 80)
    
    analysis = {
        "openmp_best_gflops": 0,
        "mpi_sim_gflops": 0,
        "recommendation": "",
        "details": []
    }
    
    # OpenMP 最佳性能 (48线程)
    openmp_48 = next((r for r in openmp_results if r["num_threads"] == 48), None)
    if openmp_48:
        analysis["openmp_best_gflops"] = openmp_48["gflops"]
        print(f"\nOpenMP (1进程 x 48线程): {openmp_48['gflops']:.2f} GFLOPS")
        print(f"  并行效率: {openmp_48['efficiency_percent']:.1f}%")
    
    # MPI 模拟性能
    if mpi_simulation:
        analysis["mpi_sim_gflops"] = mpi_simulation["total_gflops"]
        print(f"\nMPI 模拟 (4进程 x 12线程): {mpi_simulation['total_gflops']:.2f} GFLOPS")
    
    # 对比
    if analysis["openmp_best_gflops"] > 0 and analysis["mpi_sim_gflops"] > 0:
        ratio = analysis["mpi_sim_gflops"] / analysis["openmp_best_gflops"]
        analysis["mpi_vs_openmp_ratio"] = ratio
        
        print(f"\nMPI vs OpenMP 性能比: {ratio:.2f}x")
        
        if ratio > 1.2:
            analysis["recommendation"] = "MPI (推荐用于富岳)"
            print("\n✓ MPI 多进程方案性能更优")
            print("  推荐: 每个 CMG 运行一个 MPI 进程")
            analysis["details"].append("MPI 避免了跨 CMG 的线程同步开销")
            analysis["details"].append("每个进程独享一个 CMG 的内存带宽")
        elif ratio > 0.9:
            analysis["recommendation"] = "性能相当"
            print("\n✓ 两种方案性能相近")
            analysis["details"].append("可以根据应用特点选择")
        else:
            analysis["recommendation"] = "OpenMP"
            print("\n⚠ OpenMP 在此测试中表现更好")
            analysis["details"].append("但实际 MPI 可能在带宽密集型任务中表现更好")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage4"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "openmp_vs_mpi.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第四阶段测试: OpenMP vs MPI")
    print("=" * 80)
    
    all_results = {
        "test_name": "openmp_vs_mpi",
        "stage": 4
    }
    
    # 执行测试
    all_results["openmp_compute"] = test_single_process_scaling()
    all_results["openmp_bandwidth"] = test_memory_bandwidth_scaling()
    all_results["mpi_simulation"] = simulate_mpi_workload()
    
    # 对比分析
    all_results["comparison"] = compare_strategies(
        all_results["openmp_compute"],
        all_results["mpi_simulation"]
    )
    
    # 打印建议
    print("\n" + "=" * 80)
    print("总结与建议")
    print("=" * 80)
    
    comp = all_results["comparison"]
    print(f"\n推荐策略: {comp['recommendation']}")
    
    if comp.get("details"):
        print("\n原因:")
        for detail in comp["details"]:
            print(f"  - {detail}")
    
    print("\n在富岳上的最佳实践:")
    print("  1. 使用 MPI 多进程，每个进程绑定到一个 CMG")
    print("  2. 每个进程内使用 12 个 OpenMP 线程")
    print("  3. 使用 numactl 确保内存分配在本地 CMG")
    print("  4. 命令示例:")
    print("     mpirun -n 4 numactl --cpunodebind=\$RANK --membind=\$RANK python train.py")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
