#!/usr/bin/env python3
"""
第五阶段测试 1: MPI 基线性能测试
测试 MPI4Py 的基本通信性能：延迟和带宽
"""

import sys
import time
import json
import os
import numpy as np

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    print("警告: 未安装 mpi4py，将跳过 MPI 测试")

def test_mpi_pingpong():
    """测试 MPI Ping-Pong 延迟"""
    if not HAS_MPI:
        return {"error": "mpi4py not available"}
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        if rank == 0:
            print("错误: Ping-Pong 测试需要至少 2 个进程")
        return {"error": "需要至少 2 个进程"}
    
    if rank == 0:
        print("=" * 80)
        print("MPI Ping-Pong 延迟测试")
        print("=" * 80)
    
    results = []
    
    # 测试不同消息大小
    sizes_bytes = [1, 8, 64, 512, 4096, 32768, 262144]  # 1B to 256KB
    num_trials = 100
    
    for msg_size in sizes_bytes:
        if rank == 0:
            data = np.ones(msg_size, dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                comm.send(data, dest=1, tag=0)
                comm.recv(source=1, tag=0)
            
            # 测试
            times = []
            for _ in range(num_trials):
                start = time.perf_counter()
                comm.send(data, dest=1, tag=0)
                comm.recv(source=1, tag=0)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            min_time = np.min(times)
            latency_us = (avg_time / 2) * 1e6  # 单程延迟
            
            result = {
                "size_bytes": msg_size,
                "latency_us": latency_us,
                "min_latency_us": (min_time / 2) * 1e6
            }
            results.append(result)
            
            print(f"\n消息大小: {msg_size} bytes")
            print(f"  平均延迟: {latency_us:.2f} us")
            print(f"  最小延迟: {result['min_latency_us']:.2f} us")
            
        elif rank == 1:
            data = np.ones(msg_size, dtype=np.uint8)
            
            # 预热
            for _ in range(10):
                recv_data = comm.recv(source=0, tag=0)
                comm.send(data, dest=0, tag=0)
            
            # 测试
            for _ in range(num_trials):
                recv_data = comm.recv(source=0, tag=0)
                comm.send(data, dest=0, tag=0)
    
    return results

def test_mpi_bandwidth():
    """测试 MPI 带宽"""
    if not HAS_MPI:
        return {"error": "mpi4py not available"}
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        return {"error": "需要至少 2 个进程"}
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MPI 点对点带宽测试")
        print("=" * 80)
    
    results = []
    
    # 测试大消息带宽
    sizes_mb = [1, 10, 100, 500, 1000]
    num_trials = 20
    
    for size_mb in sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 / 8)  # FP64
        
        if rank == 0:
            data = np.random.randn(num_elements)
            
            # 预热
            comm.Send(data, dest=1, tag=0)
            comm.Recv(data, source=1, tag=0)
            
            # 测试
            times = []
            for _ in range(num_trials):
                start = time.perf_counter()
                comm.Send(data, dest=1, tag=0)
                comm.Recv(data, source=1, tag=0)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            # 双向传输，带宽计算
            data_size_gb = size_mb / 1024
            bandwidth = (2 * data_size_gb) / avg_time
            
            result = {
                "size_mb": size_mb,
                "time_ms": avg_time * 1000,
                "bandwidth_gbps": bandwidth
            }
            results.append(result)
            
            print(f"\n消息大小: {size_mb} MB")
            print(f"  时间: {avg_time*1000:.2f} ms")
            print(f"  带宽: {bandwidth:.2f} GB/s")
            
        elif rank == 1:
            data = np.empty(num_elements, dtype=np.float64)
            
            # 预热
            comm.Recv(data, source=0, tag=0)
            comm.Send(data, dest=0, tag=0)
            
            # 测试
            for _ in range(num_trials):
                comm.Recv(data, source=0, tag=0)
                comm.Send(data, dest=0, tag=0)
    
    return results

def test_allreduce_performance():
    """测试 AllReduce 性能"""
    if not HAS_MPI:
        return {"error": "mpi4py not available"}
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MPI AllReduce 性能测试")
        print("=" * 80)
        print(f"进程数: {size}")
    
    results = []
    
    # 测试不同数据量的 AllReduce
    sizes = [1024, 10240, 102400, 1024000, 10240000]  # 元素数量
    num_trials = 50
    
    for num_elements in sizes:
        data = np.random.randn(num_elements)
        result_data = np.empty_like(data)
        
        # 预热
        for _ in range(5):
            comm.Allreduce(data, result_data, op=MPI.SUM)
        
        # 测试
        times = []
        for _ in range(num_trials):
            comm.Barrier()
            start = time.perf_counter()
            comm.Allreduce(data, result_data, op=MPI.SUM)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        size_mb = (num_elements * 8) / (1024 * 1024)  # FP64
        
        result = {
            "num_elements": num_elements,
            "size_mb": size_mb,
            "time_ms": avg_time * 1000,
            "num_processes": size
        }
        results.append(result)
        
        if rank == 0:
            print(f"\n数据量: {size_mb:.2f} MB")
            print(f"  时间: {avg_time*1000:.2f} ms")
    
    return results

def test_alltoall_performance():
    """测试 AllToAll 性能"""
    if not HAS_MPI:
        return {"error": "mpi4py not available"}
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("MPI AllToAll 性能测试")
        print("=" * 80)
        print(f"进程数: {size}")
    
    results = []
    
    # 每个进程发送给其他每个进程的数据块大小
    block_sizes = [1024, 10240, 102400]  # 元素数量
    num_trials = 30
    
    for block_size in block_sizes:
        send_data = np.random.randn(size * block_size)
        recv_data = np.empty(size * block_size, dtype=np.float64)
        
        # 预热
        for _ in range(5):
            comm.Alltoall(send_data, recv_data)
        
        # 测试
        times = []
        for _ in range(num_trials):
            comm.Barrier()
            start = time.perf_counter()
            comm.Alltoall(send_data, recv_data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        total_size_mb = (size * block_size * 8) / (1024 * 1024)
        
        result = {
            "block_size": block_size,
            "total_size_mb": total_size_mb,
            "time_ms": avg_time * 1000,
            "num_processes": size
        }
        results.append(result)
        
        if rank == 0:
            print(f"\n每进程块大小: {block_size} 元素")
            print(f"  总数据量: {total_size_mb:.2f} MB")
            print(f"  时间: {avg_time*1000:.2f} ms")
    
    return results

def save_results(all_results, output_dir="../test_results/stage5"):
    """保存测试结果"""
    if not HAS_MPI:
        return
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "mpi_baseline.json")
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")

def main():
    if not HAS_MPI:
        print("错误: 需要安装 mpi4py")
        print("安装命令: pip install mpi4py")
        return 1
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("富岳集群 - 第五阶段测试: MPI 基线性能")
        print("=" * 80)
        print(f"MPI 进程数: {size}")
    
    all_results = {
        "test_name": "mpi_baseline",
        "stage": 5,
        "num_processes": size
    }
    
    # 执行测试
    all_results["pingpong"] = test_mpi_pingpong()
    all_results["bandwidth"] = test_mpi_bandwidth()
    all_results["allreduce"] = test_allreduce_performance()
    all_results["alltoall"] = test_alltoall_performance()
    
    # 保存结果
    save_results(all_results)
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)
        print("\n提示:")
        print("  - 对比这些结果与 Gloo 的性能")
        print("  - 富岳的 Tofu 互联预期提供极低延迟（<1us）和高带宽")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
