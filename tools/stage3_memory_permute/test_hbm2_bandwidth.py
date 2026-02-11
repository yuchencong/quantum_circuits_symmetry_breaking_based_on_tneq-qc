#!/usr/bin/env python3
"""
第三阶段测试 1: HBM2 峰值带宽利用率测试
测试大规模 Tensor Copy 和 fill_ 操作的内存带宽
"""

import sys
import torch
import time
import json
import os
import numpy as np

def benchmark_copy(size_mb, num_trials=5):
    """测试张量复制带宽"""
    num_elements = int(size_mb * 1024 * 1024 / 4)  # FP32 = 4 bytes
    
    # 预热（缩短为 2 次，减少总耗时）
    for _ in range(2):
        src = torch.randn(num_elements)
        dst = src.clone()
    
    # 正式测试
    times = []
    for _ in range(num_trials):
        src = torch.randn(num_elements)
        
        start = time.perf_counter()
        dst = src.clone()
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    
    # 计算带宽 (GB/s)
    # 读一次 + 写一次 = 2倍数据量
    data_size_gb = size_mb / 1024
    avg_bandwidth = (2 * data_size_gb) / avg_time
    peak_bandwidth = (2 * data_size_gb) / min_time
    
    return {
        "size_mb": size_mb,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "avg_bandwidth_gbps": avg_bandwidth,
        "peak_bandwidth_gbps": peak_bandwidth
    }

def test_single_cmg_bandwidth():
    """测试单 CMG 内存带宽"""
    print("=" * 80)
    print("单 CMG 内存带宽测试")
    print("=" * 80)
    print("理论峰值: 256 GB/s (单 CMG)")
    print("-" * 80)
    
    # 测试不同大小（缩小到较小数据量，便于快速运行）
    sizes_mb = [4, 16, 64]
    results = []
    
    for size_mb in sizes_mb:
        print(f"\n测试数据大小: {size_mb} MB")
        result = benchmark_copy(size_mb, num_trials=5)
        results.append(result)
        
        print(f"  平均时间: {result['avg_time_ms']:.2f} ms")
        print(f"  平均带宽: {result['avg_bandwidth_gbps']:.2f} GB/s")
        print(f"  峰值带宽: {result['peak_bandwidth_gbps']:.2f} GB/s")
        print(f"  利用率: {result['peak_bandwidth_gbps']/256*100:.1f}%")
    
    return results

def benchmark_fill(size_mb, num_trials=5):
    """测试张量填充带宽"""
    num_elements = int(size_mb * 1024 * 1024 / 4)
    
    # 预热
    for _ in range(2):
        tensor = torch.empty(num_elements)
        tensor.fill_(1.0)
    
    # 正式测试
    times = []
    for _ in range(num_trials):
        tensor = torch.empty(num_elements)
        
        start = time.perf_counter()
        tensor.fill_(1.0)
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    
    # 只写不读
    data_size_gb = size_mb / 1024
    avg_bandwidth = data_size_gb / avg_time
    peak_bandwidth = data_size_gb / min_time
    
    return {
        "size_mb": size_mb,
        "avg_bandwidth_gbps": avg_bandwidth,
        "peak_bandwidth_gbps": peak_bandwidth
    }

def test_fill_bandwidth():
    """测试填充操作带宽"""
    print("\n" + "=" * 80)
    print("填充操作带宽测试 (纯写入)")
    print("=" * 80)
    
    sizes_mb = [16, 64]
    results = []
    
    for size_mb in sizes_mb:
        print(f"\n测试数据大小: {size_mb} MB")
        result = benchmark_fill(size_mb)
        results.append(result)
        
        print(f"  平均带宽: {result['avg_bandwidth_gbps']:.2f} GB/s")
        print(f"  峰值带宽: {result['peak_bandwidth_gbps']:.2f} GB/s")
    
    return results

def benchmark_add_inplace(size_mb, num_trials=5):
    """测试 in-place 加法带宽 (读+写)"""
    num_elements = int(size_mb * 1024 * 1024 / 4)
    
    # 预热
    for _ in range(2):
        tensor = torch.randn(num_elements)
        tensor.add_(1.0)
    
    # 正式测试
    times = []
    for _ in range(num_trials):
        tensor = torch.randn(num_elements)
        
        start = time.perf_counter()
        tensor.add_(1.0)
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    
    # 读一次 + 写一次
    data_size_gb = size_mb / 1024
    avg_bandwidth = (2 * data_size_gb) / avg_time
    
    return {
        "size_mb": size_mb,
        "avg_bandwidth_gbps": avg_bandwidth
    }

def test_readwrite_bandwidth():
    """测试读写混合带宽"""
    print("\n" + "=" * 80)
    print("读写混合带宽测试 (add_ in-place)")
    print("=" * 80)
    
    sizes_mb = [16, 64]
    results = []
    
    for size_mb in sizes_mb:
        print(f"\n测试数据大小: {size_mb} MB")
        result = benchmark_add_inplace(size_mb)
        results.append(result)
        
        print(f"  带宽: {result['avg_bandwidth_gbps']:.2f} GB/s")
    
    return results

def test_strided_access():
    """测试跨步访问的带宽影响"""
    print("\n" + "=" * 80)
    print("跨步访问带宽测试")
    print("=" * 80)
    
    # 原始为 1GB，这里缩小为 64MB 以便在本地快速运行
    size = 1024 * 1024 * 16
    strides = [1, 2, 4, 8, 16]
    results = []
    
    for stride in strides:
        print(f"\n测试跨步: {stride}")
        
        # 创建大张量
        tensor = torch.randn(size)
        
        # 预热
        for _ in range(2):
            view = tensor[::stride]
            _ = view.clone()
        
        # 测试（减少次数以缩短时间）
        times = []
        for _ in range(5):
            view = tensor[::stride]
            
            start = time.perf_counter()
            copy = view.clone()
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        effective_size_gb = (size // stride * 4) / (1024**3)
        bandwidth = (2 * effective_size_gb) / avg_time
        
        result = {
            "stride": stride,
            "effective_size_mb": effective_size_gb * 1024,
            "bandwidth_gbps": bandwidth
        }
        results.append(result)
        
        print(f"  有效大小: {effective_size_gb*1024:.1f} MB")
        print(f"  带宽: {bandwidth:.2f} GB/s")
    
    return results

def analyze_results(all_results):
    """分析带宽测试结果"""
    print("\n" + "=" * 80)
    print("带宽分析")
    print("=" * 80)
    
    analysis = {
        "peak_copy_bandwidth": 0,
        "peak_fill_bandwidth": 0,
        "single_cmg_theoretical": 256,  # GB/s
        "full_node_theoretical": 1024,  # GB/s
        "utilization_percent": 0,
        "verdict": "unknown",
        "recommendations": []
    }
    
    # 分析复制带宽
    if "copy_bandwidth" in all_results and all_results["copy_bandwidth"]:
        # 取较大数据量的带宽作为峰值（阈值从 256MB 调整为 16MB，适配快速测试）
        large_data = [r for r in all_results["copy_bandwidth"] if r["size_mb"] >= 16]
        if large_data:
            peak_bw = max([r["peak_bandwidth_gbps"] for r in large_data])
            analysis["peak_copy_bandwidth"] = peak_bw
            
            utilization = (peak_bw / analysis["single_cmg_theoretical"]) * 100
            analysis["utilization_percent"] = utilization
            
            print(f"\n复制操作峰值带宽: {peak_bw:.2f} GB/s")
            print(f"单 CMG 理论峰值: {analysis['single_cmg_theoretical']} GB/s")
            print(f"带宽利用率: {utilization:.1f}%")
            
            if utilization > 80:
                analysis["verdict"] = "excellent"
                print("\n✓ 带宽利用率优秀")
            elif utilization > 60:
                analysis["verdict"] = "good"
                print("\n✓ 带宽利用率良好")
            elif utilization > 40:
                analysis["verdict"] = "moderate"
                print("\n⚠ 带宽利用率中等")
                analysis["recommendations"].append("检查 NUMA 绑定是否正确")
            else:
                analysis["verdict"] = "poor"
                print("\n✗ 带宽利用率较低")
                analysis["recommendations"].append("验证内存访问是否绑定在正确的 CMG")
                analysis["recommendations"].append("检查是否有其他进程竞争内存带宽")
    
    # 分析填充带宽
    if "fill_bandwidth" in all_results and all_results["fill_bandwidth"]:
        large_data = [r for r in all_results["fill_bandwidth"] if r["size_mb"] >= 16]
        if large_data:
            peak_fill = max([r["peak_bandwidth_gbps"] for r in large_data])
            analysis["peak_fill_bandwidth"] = peak_fill
            print(f"\n填充操作峰值带宽: {peak_fill:.2f} GB/s")
    
    # 分析跨步访问
    if "strided_access" in all_results and len(all_results["strided_access"]) > 1:
        bw_stride1 = all_results["strided_access"][0]["bandwidth_gbps"]
        bw_stride16 = all_results["strided_access"][-1]["bandwidth_gbps"]
        degradation = (1 - bw_stride16 / bw_stride1) * 100
        
        print(f"\n跨步访问性能下降: {degradation:.1f}%")
        if degradation > 50:
            analysis["recommendations"].append("在张量网络中避免大跨步访问，使用 contiguous() 重排")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage3"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "hbm2_bandwidth.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第三阶段测试: HBM2 带宽测试")
    print("=" * 80)
    
    all_results = {
        "test_name": "hbm2_bandwidth",
        "stage": 3
    }
    
    # 执行测试
    all_results["copy_bandwidth"] = test_single_cmg_bandwidth()
    all_results["fill_bandwidth"] = test_fill_bandwidth()
    all_results["readwrite_bandwidth"] = test_readwrite_bandwidth()
    all_results["strided_access"] = test_strided_access()
    
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
        print("  ✓ 带宽利用率良好，无需特别优化")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
