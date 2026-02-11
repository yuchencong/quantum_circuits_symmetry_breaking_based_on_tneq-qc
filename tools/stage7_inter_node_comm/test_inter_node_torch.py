#!/usr/bin/env python3
"""
第七阶段测试：机器间通信效率（PyTorch）

目标：两台机器，每台各一个进程，使用 torch.distributed 测试跨节点通信的
延迟与带宽。必须 world_size=2 时才有意义。

使用方式（两机两进程）：
  - 机器 A (rank 0)：设置 MASTER_ADDR=本机IP, RANK=0, WORLD_SIZE=2，运行本脚本
  - 机器 B (rank 1)：设置 MASTER_ADDR=机器A的IP, RANK=1, WORLD_SIZE=2，运行本脚本
  或使用 MPI：mpirun -np 2 -host node1,node2 python test_inter_node_torch.py [gloo|mpi]
"""

import sys
import time
import json
import os
import socket
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


def init_distributed(backend="gloo"):
    """初始化分布式环境（与 stage5 一致，支持 env 与 MPI）"""
    if not HAS_DIST:
        return False

    if HAS_MPI and backend == "mpi":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size != 2:
        if rank == 0 or not HAS_DIST:
            print(
                "stage7 机器间通信测试要求 exactly 2 个进程（两台机器各 1 个）。"
                "当前 world_size=%s。请使用 mpirun -np 2 或两机分别启动。"
                % world_size,
                file=sys.stderr,
            )
        return False

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        return True
    except Exception as e:
        if rank == 0:
            print("初始化分布式失败: %s" % e, file=sys.stderr)
        return False


def _warmup(backend):
    """简单预热一次 AllReduce"""
    t = torch.randn(1024)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)


def test_allreduce_inter_node(backend, num_trials=30):
    """AllReduce 延迟与带宽（两进程）"""
    rank = dist.get_rank()
    # 从 4KB 到约 64MB，覆盖典型梯度通信规模
    sizes = [
        1024,
        10240,
        102400,
        1024000,
        4 * 1024000,
        16 * 1024000,
        # 32 * 1024000,
    ]
    results = []
    _warmup(backend)

    for size in sizes:
        tensor = torch.randn(size)
        times = []
        for _ in range(num_trials):
            dist.barrier()
            start = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_s = np.mean(times)
        std_s = np.std(times)
        bytes_per_rank = size * 4  # float32
        total_bytes = bytes_per_rank * 2  # 两进程 AllReduce 近似传输量
        bandwidth_mbps = (total_bytes / (1024 * 1024)) / avg_s if avg_s > 0 else 0
        results.append({
            "size": size,
            "size_mb": bytes_per_rank / (1024 * 1024),
            "time_ms": avg_s * 1000,
            "std_ms": std_s * 1000,
            "bandwidth_mbps": bandwidth_mbps,
        })
        if rank == 0:
            print(
                "  AllReduce size=%.2f MB: %.2f ± %.2f ms, %.2f MB/s"
                % (bytes_per_rank / (1024 * 1024), avg_s * 1000, std_s * 1000, bandwidth_mbps)
            )
    return results


def test_broadcast_inter_node(backend, num_trials=30):
    """Broadcast 延迟与带宽（rank0 -> rank1）"""
    rank = dist.get_rank()
    sizes = [1024, 102400, 1024000, 4 * 1024000, 16 * 1024000]
    results = []
    _warmup(backend)

    for size in sizes:
        if rank == 0:
            tensor = torch.randn(size)
        else:
            tensor = torch.empty(size)
        times = []
        for _ in range(num_trials):
            dist.barrier()
            start = time.perf_counter()
            dist.broadcast(tensor, src=0)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg_s = np.mean(times)
        bytes_mb = (size * 4) / (1024 * 1024)
        bandwidth_mbps = bytes_mb / avg_s if avg_s > 0 else 0
        results.append({
            "size": size,
            "size_mb": bytes_mb,
            "time_ms": avg_s * 1000,
            "bandwidth_mbps": bandwidth_mbps,
        })
        if rank == 0:
            print("  Broadcast size=%.2f MB: %.2f ms, %.2f MB/s" % (bytes_mb, avg_s * 1000, bandwidth_mbps))
    return results


def test_point_to_point_latency(backend, num_trials=100):
    """小消息点对点延迟（ping-pong，仅 world_size=2）"""
    rank = dist.get_rank()
    small_sizes = [1, 4, 16, 64, 256, 1024, 4096]
    results = []
    for size in small_sizes:
        tensor = torch.randn(size)
        if rank == 0:
            times = []
            for _ in range(num_trials):
                dist.barrier()
                start = time.perf_counter()
                dist.send(tensor, dst=1)
                dist.recv(tensor, src=1)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            avg_s = np.mean(times)
            one_way_us = (avg_s * 0.5) * 1e6
            results.append({
                "size": size,
                "round_trip_ms": avg_s * 1000,
                "one_way_latency_us": one_way_us,
            })
            print("  P2P size=%d: RTT=%.2f ms, one_way~%.0f us" % (size, avg_s * 1000, one_way_us))
        else:
            for _ in range(num_trials):
                dist.barrier()
                dist.recv(tensor, src=0)
                dist.send(tensor, dst=0)
    return results


def save_results(all_results, output_dir="../test_results/stage7"):
    rank = dist.get_rank()
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "inter_node_torch.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n结果已保存: %s" % out_path)


def main():
    if not HAS_DIST:
        print("错误: PyTorch distributed 不可用", file=sys.stderr)
        return 1

    backend = (sys.argv[1] if len(sys.argv) > 1 else "gloo").lower()
    if backend not in ("gloo", "mpi"):
        print("用法: python test_inter_node_torch.py [gloo|mpi]", file=sys.stderr)
        return 1
    if backend == "mpi" and not HAS_MPI:
        print("错误: MPI 后端需要 mpi4py", file=sys.stderr)
        return 1

    if not init_distributed(backend):
        return 1

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    if rank == 0:
        print("=" * 80)
        print("第七阶段：机器间通信效率（PyTorch, backend=%s）" % backend)
        print("=" * 80)
        print("world_size=%d, 本机 hostname=%s" % (world_size, hostname))
        print("")

    all_results = {
        "stage": 7,
        "test_name": "inter_node_torch",
        "backend": backend,
        "world_size": world_size,
        "hostname_rank0": hostname if rank == 0 else None,
    }
    # rank1 的 hostname 在 barrier 后由 rank0 收不到，这里只记录 rank0；可选后续用 all_gather 收集
    dist.barrier()
    if rank == 0:
        print("AllReduce (两机):")
    all_results["allreduce"] = test_allreduce_inter_node(backend)
    if rank == 0:
        print("\nBroadcast (rank0 -> rank1):")
    all_results["broadcast"] = test_broadcast_inter_node(backend)
    if rank == 0:
        print("\nP2P 延迟 (ping-pong):")
    all_results["p2p_latency"] = test_point_to_point_latency(backend)

    save_results(all_results)
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 80)
        print("stage7 机器间通信测试完成")
        print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
