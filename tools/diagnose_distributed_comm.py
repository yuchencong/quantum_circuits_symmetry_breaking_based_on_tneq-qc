#!/usr/bin/env python3
"""
分布式通信诊断工具

用于诊断分布式训练中的通信问题，包括：
1. 基本连接测试
2. 带宽测试
3. 延迟测试
4. 死锁检测
5. 通信模式测试
"""

import torch
import torch.distributed as dist
import time
import sys
from typing import List, Tuple


def setup_distributed():
    """初始化分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"[Rank {rank}] Process initialized: {world_size} total processes")
    return rank, world_size


def test_basic_connectivity(rank: int, world_size: int) -> bool:
    """测试基本连接"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 1: Basic Connectivity")
    print(f"{'='*60}")
    
    try:
        # 测试 barrier
        print(f"[Rank {rank}] Testing barrier...")
        start = time.time()
        dist.barrier()
        elapsed = time.time() - start
        print(f"[Rank {rank}] ✓ Barrier successful ({elapsed*1000:.2f}ms)")
        
        # 测试 allreduce
        print(f"[Rank {rank}] Testing allreduce...")
        tensor = torch.tensor([rank], dtype=torch.float32)
        start = time.time()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elapsed = time.time() - start
        
        expected = sum(range(world_size))
        if abs(tensor.item() - expected) < 1e-6:
            print(f"[Rank {rank}] ✓ Allreduce successful ({elapsed*1000:.2f}ms)")
            return True
        else:
            print(f"[Rank {rank}] ✗ Allreduce failed: expected {expected}, got {tensor.item()}")
            return False
            
    except Exception as e:
        print(f"[Rank {rank}] ✗ Basic connectivity test failed: {e}")
        return False


def test_point_to_point(rank: int, world_size: int) -> bool:
    """测试点对点通信"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 2: Point-to-Point Communication")
    print(f"{'='*60}")
    
    try:
        # 每个 rank 与下一个 rank 通信（环形）
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1 + world_size) % world_size
        
        send_tensor = torch.tensor([rank], dtype=torch.float32)
        recv_tensor = torch.zeros(1, dtype=torch.float32)
        
        print(f"[Rank {rank}] Sending to rank {next_rank}, receiving from rank {prev_rank}")
        
        start = time.time()
        
        # 使用 sendrecv 避免死锁
        if rank % 2 == 0:
            dist.send(send_tensor, next_rank)
            dist.recv(recv_tensor, prev_rank)
        else:
            dist.recv(recv_tensor, prev_rank)
            dist.send(send_tensor, next_rank)
        
        elapsed = time.time() - start
        
        if abs(recv_tensor.item() - prev_rank) < 1e-6:
            print(f"[Rank {rank}] ✓ P2P successful ({elapsed*1000:.2f}ms)")
            return True
        else:
            print(f"[Rank {rank}] ✗ P2P failed: expected {prev_rank}, got {recv_tensor.item()}")
            return False
            
    except Exception as e:
        print(f"[Rank {rank}] ✗ P2P test failed: {e}")
        return False


def test_bandwidth(rank: int, world_size: int, sizes: List[int] = None) -> None:
    """测试通信带宽"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 3: Bandwidth Test")
    print(f"{'='*60}")
    
    if sizes is None:
        sizes = [1024, 10240, 102400, 1024000]  # 1KB, 10KB, 100KB, 1MB
    
    for size in sizes:
        try:
            tensor = torch.randn(size, dtype=torch.float32)
            
            # Warmup
            dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
            
            # Measure
            dist.barrier()
            start = time.time()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            elapsed = time.time() - start
            
            # Calculate bandwidth
            data_size_mb = size * 4 / (1024 * 1024)  # float32 = 4 bytes
            bandwidth = data_size_mb / elapsed if elapsed > 0 else 0
            
            if rank == 0:
                print(f"  Size: {size:>8} elements ({data_size_mb:.2f} MB) | "
                      f"Time: {elapsed*1000:>8.2f}ms | "
                      f"Bandwidth: {bandwidth:>8.2f} MB/s")
                
        except Exception as e:
            print(f"[Rank {rank}] ✗ Bandwidth test failed for size {size}: {e}")


def test_latency(rank: int, world_size: int, num_iterations: int = 100) -> None:
    """测试通信延迟"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 4: Latency Test ({num_iterations} iterations)")
    print(f"{'='*60}")
    
    try:
        tensor = torch.tensor([rank], dtype=torch.float32)
        latencies = []
        
        for i in range(num_iterations):
            dist.barrier()
            start = time.time()
            dist.all_reduce(tensor.clone(), op=dist.ReduceOp.SUM)
            elapsed = time.time() - start
            latencies.append(elapsed * 1000)  # Convert to ms
        
        # Statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        if rank == 0:
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  Min latency: {min_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            
            # Check for anomalies
            if max_latency > avg_latency * 3:
                print(f"  ⚠ WARNING: High latency variance detected!")
                print(f"    Max latency is {max_latency/avg_latency:.1f}x average")
                
    except Exception as e:
        print(f"[Rank {rank}] ✗ Latency test failed: {e}")


def test_deadlock_scenarios(rank: int, world_size: int) -> None:
    """测试常见的死锁场景"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 5: Deadlock Scenarios")
    print(f"{'='*60}")
    
    # Scenario 1: Mismatched operations
    print(f"[Rank {rank}] Scenario 1: Testing synchronized operations...")
    try:
        dist.barrier()
        tensor = torch.tensor([rank], dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] ✓ Synchronized operations successful")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Synchronized operations failed: {e}")
    
    # Scenario 2: Multiple consecutive barriers
    print(f"[Rank {rank}] Scenario 2: Testing multiple barriers...")
    try:
        for i in range(5):
            dist.barrier()
        print(f"[Rank {rank}] ✓ Multiple barriers successful")
    except Exception as e:
        print(f"[Rank {rank}] ✗ Multiple barriers failed: {e}")
    
    # Scenario 3: Alternating send/recv pattern (like in TP matmul)
    if world_size >= 2:
        print(f"[Rank {rank}] Scenario 3: Testing alternating send/recv...")
        try:
            partner = 1 if rank == 0 else 0 if rank == 1 else -1
            
            if partner >= 0:
                send_tensor = torch.tensor([rank], dtype=torch.float32)
                recv_tensor = torch.zeros(1, dtype=torch.float32)
                
                # Lower rank sends first
                if rank < partner:
                    dist.send(send_tensor, partner)
                    dist.recv(recv_tensor, partner)
                else:
                    dist.recv(recv_tensor, partner)
                    dist.send(send_tensor, partner)
                
                print(f"[Rank {rank}] ✓ Alternating send/recv successful")
            else:
                print(f"[Rank {rank}] - Skipped (not in pair)")
                
        except Exception as e:
            print(f"[Rank {rank}] ✗ Alternating send/recv failed: {e}")


def test_communication_pattern(rank: int, world_size: int) -> None:
    """测试实际训练中的通信模式"""
    print(f"\n{'='*60}")
    print(f"[Rank {rank}] Test 6: Training Communication Pattern")
    print(f"{'='*60}")
    
    try:
        # Simulate multi-stage reduction
        num_stages = 0
        temp_size = world_size
        while temp_size > 1:
            temp_size = (temp_size + 1) // 2
            num_stages += 1
        
        print(f"[Rank {rank}] Simulating {num_stages} reduction stages...")
        
        for stage in range(num_stages):
            group_size = 2 ** (stage + 1)
            my_group = rank // group_size
            
            print(f"[Rank {rank}] Stage {stage + 1}: group_size={group_size}, my_group={my_group}")
            
            # Barrier before stage
            dist.barrier()
            
            # Simulate allgather within group
            tensor = torch.tensor([rank], dtype=torch.float32)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            # Barrier after stage
            dist.barrier()
            
            print(f"[Rank {rank}] Stage {stage + 1} completed")
        
        print(f"[Rank {rank}] ✓ Training pattern simulation successful")
        
    except Exception as e:
        print(f"[Rank {rank}] ✗ Training pattern simulation failed: {e}")


def main():
    """运行所有诊断测试"""
    print("\n" + "="*60)
    print("Distributed Communication Diagnostic Tool")
    print("="*60)
    
    # Setup
    rank, world_size = setup_distributed()
    
    print(f"\n[Rank {rank}] Configuration:")
    print(f"  Backend: {dist.get_backend()}")
    print(f"  World size: {world_size}")
    print(f"  Rank: {rank}")
    
    # Run tests
    all_passed = True
    
    # Test 1: Basic connectivity
    if not test_basic_connectivity(rank, world_size):
        all_passed = False
        print(f"\n[Rank {rank}] ⚠ WARNING: Basic connectivity test failed!")
        print(f"[Rank {rank}] Stopping further tests.")
        return
    
    # Test 2: Point-to-point
    if not test_point_to_point(rank, world_size):
        all_passed = False
    
    # Test 3: Bandwidth
    test_bandwidth(rank, world_size)
    
    # Test 4: Latency
    test_latency(rank, world_size)
    
    # Test 5: Deadlock scenarios
    test_deadlock_scenarios(rank, world_size)
    
    # Test 6: Communication pattern
    test_communication_pattern(rank, world_size)
    
    # Summary
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print("Diagnostic Summary")
        print(f"{'='*60}")
        if all_passed:
            print("✓ All critical tests passed!")
        else:
            print("⚠ Some tests failed. Check logs above.")
        print(f"{'='*60}\n")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Diagnostic] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Diagnostic] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
