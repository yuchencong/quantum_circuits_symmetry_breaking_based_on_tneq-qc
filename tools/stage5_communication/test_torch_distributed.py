#!/usr/bin/env python3
"""
第五阶段测试 2: PyTorch Distributed 性能测试
测试 torch.distributed 使用 Gloo 和 MPI 后端的性能
"""

import sys
import time
import json
import os
import numpy as np
import torch

try:
    import torch.distributed as dist
    HAS_DIST = True
except ImportError:
    HAS_DIST = False

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

def init_distributed(backend='gloo'):
    """初始化分布式环境"""
    if not HAS_DIST:
        return False
    
    # 从环境变量获取 rank 和 world_size
    # 或者从 MPI 获取
    if HAS_MPI and backend == 'mpi':
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        
        # 设置环境变量
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    else:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size == 1:
        print(f"警告: 单进程模式，无法测试分布式通信")
        return False
    
    try:
        dist.init_process_group(backend=backend, 
                               init_method='env://',
                               rank=rank,
                               world_size=world_size)
        return True
    except Exception as e:
        print(f"初始化分布式失败: {e}")
        return False

def test_allreduce_torch(backend='gloo'):
    """测试 PyTorch AllReduce"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print(f"PyTorch AllReduce 测试 (backend={backend})")
        print("=" * 80)
        print(f"进程数: {world_size}")
    
    results = []
    
    # 测试不同大小的张量
    sizes = [1024, 10240, 102400, 1024000, 10240000]
    num_trials = 50
    
    for size in sizes:
        tensor = torch.randn(size)
        
        # 预热
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 测试
        times = []
        for _ in range(num_trials):
            tensor = torch.randn(size)
            dist.barrier()
            
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        size_mb = (size * 4) / (1024 * 1024)  # FP32
        
        result = {
            "size": size,
            "size_mb": size_mb,
            "time_ms": avg_time * 1000,
            "std_ms": std_time * 1000,
            "backend": backend
        }
        results.append(result)
        
        if rank == 0:
            print(f"\n大小: {size_mb:.2f} MB")
            print(f"  时间: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    
    return results

def test_broadcast_torch(backend='gloo'):
    """测试 PyTorch Broadcast"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"PyTorch Broadcast 测试 (backend={backend})")
        print("=" * 80)
    
    results = []
    sizes = [1024, 102400, 1024000, 10240000]
    num_trials = 50
    
    for size in sizes:
        if rank == 0:
            tensor = torch.randn(size)
        else:
            tensor = torch.empty(size)
        
        # 预热
        for _ in range(5):
            dist.broadcast(tensor, src=0)
        
        # 测试
        times = []
        for _ in range(num_trials):
            dist.barrier()
            
            start = time.perf_counter()
            dist.broadcast(tensor, src=0)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        size_mb = (size * 4) / (1024 * 1024)
        
        result = {
            "size": size,
            "size_mb": size_mb,
            "time_ms": avg_time * 1000,
            "backend": backend
        }
        results.append(result)
        
        if rank == 0:
            print(f"\n大小: {size_mb:.2f} MB")
            print(f"  时间: {avg_time*1000:.2f} ms")
    
    return results

def test_allgather_torch(backend='gloo'):
    """测试 PyTorch AllGather"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"PyTorch AllGather 测试 (backend={backend})")
        print("=" * 80)
    
    results = []
    sizes = [1024, 102400, 1024000]
    num_trials = 30
    
    for size in sizes:
        tensor = torch.randn(size)
        gather_list = [torch.empty(size) for _ in range(world_size)]
        
        # 预热
        for _ in range(5):
            dist.all_gather(gather_list, tensor)
        
        # 测试
        times = []
        for _ in range(num_trials):
            dist.barrier()
            
            start = time.perf_counter()
            dist.all_gather(gather_list, tensor)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = np.mean(times)
        size_mb = (size * 4) / (1024 * 1024)
        
        result = {
            "size_per_rank_mb": size_mb,
            "total_size_mb": size_mb * world_size,
            "time_ms": avg_time * 1000,
            "backend": backend
        }
        results.append(result)
        
        if rank == 0:
            print(f"\n每进程: {size_mb:.2f} MB, 总计: {size_mb*world_size:.2f} MB")
            print(f"  时间: {avg_time*1000:.2f} ms")
    
    return results

def test_compute_comm_overlap(backend='gloo'):
    """测试计算与通信重叠"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"计算-通信重叠测试 (backend={backend})")
        print("=" * 80)
    
    # 准备数据
    comm_tensor = torch.randn(1024000)  # 4MB
    compute_size = 2048
    A = torch.randn(compute_size, compute_size)
    B = torch.randn(compute_size, compute_size)
    
    num_trials = 20
    
    # 方法 1: 串行（先通信后计算）
    if rank == 0:
        print("\n串行执行（通信 -> 计算）:")
    
    times_serial = []
    for _ in range(num_trials):
        dist.barrier()
        start = time.perf_counter()
        
        # 通信
        dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM)
        
        # 计算
        C = torch.matmul(A, B)
        
        end = time.perf_counter()
        times_serial.append(end - start)
    
    avg_serial = np.mean(times_serial)
    
    # 方法 2: 异步（尝试重叠）
    if rank == 0:
        print("\n异步执行（尝试重叠）:")
    
    times_async = []
    for _ in range(num_trials):
        dist.barrier()
        start = time.perf_counter()
        
        # 异步通信
        work = dist.all_reduce(comm_tensor, op=dist.ReduceOp.SUM, async_op=True)
        
        # 同时进行计算
        C = torch.matmul(A, B)
        
        # 等待通信完成
        work.wait()
        
        end = time.perf_counter()
        times_async.append(end - start)
    
    avg_async = np.mean(times_async)
    overlap_benefit = (1 - avg_async / avg_serial) * 100
    
    if rank == 0:
        print(f"\n串行时间: {avg_serial*1000:.2f} ms")
        print(f"异步时间: {avg_async*1000:.2f} ms")
        print(f"重叠收益: {overlap_benefit:.1f}%")
        
        if overlap_benefit > 10:
            print("✓ 检测到有效的计算-通信重叠")
        elif overlap_benefit > 0:
            print("⚠ 有少量重叠，但效果有限")
        else:
            print("✗ 未检测到有效重叠（可能 CPU 资源竞争）")
    
    return {
        "serial_ms": avg_serial * 1000,
        "async_ms": avg_async * 1000,
        "overlap_benefit_percent": overlap_benefit,
        "backend": backend
    }

def save_results(all_results, backend, output_dir="../test_results/stage5"):
    """保存测试结果"""
    rank = dist.get_rank()
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"torch_distributed_{backend}.json")
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")

def main():
    if not HAS_DIST:
        print("错误: PyTorch distributed 不可用")
        return 1
    
    # 从命令行参数获取后端
    backend = sys.argv[1] if len(sys.argv) > 1 else 'gloo'
    
    if backend not in ['gloo', 'mpi', 'nccl']:
        print(f"不支持的后端: {backend}")
        print("支持的后端: gloo, mpi, nccl")
        return 1
    
    if backend == 'mpi' and not HAS_MPI:
        print("错误: MPI 后端需要安装 mpi4py")
        return 1
    
    # 初始化分布式
    if not init_distributed(backend):
        print("无法初始化分布式环境")
        return 1
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"富岳集群 - 第五阶段测试: PyTorch Distributed ({backend})")
        print("=" * 80)
        print(f"进程数: {world_size}")
    
    all_results = {
        "test_name": f"torch_distributed_{backend}",
        "stage": 5,
        "backend": backend,
        "world_size": world_size
    }
    
    # 执行测试
    all_results["allreduce"] = test_allreduce_torch(backend)
    all_results["broadcast"] = test_broadcast_torch(backend)
    all_results["allgather"] = test_allgather_torch(backend)
    all_results["overlap"] = test_compute_comm_overlap(backend)
    
    # 保存结果
    save_results(all_results, backend)
    
    # 清理
    dist.destroy_process_group()
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
