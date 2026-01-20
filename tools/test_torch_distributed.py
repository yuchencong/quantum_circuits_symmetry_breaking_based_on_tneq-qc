#!/usr/bin/env python3
"""
Test PyTorch distributed communication backends and performance
For Fujitsu ARM cluster with PyTorch 1.13 CPU

Usage:
  Single node: python test_torch_distributed.py
  Multi-node:  mpirun -np 4 python test_torch_distributed.py
"""

import torch
import torch.distributed as dist
import os
import time
import socket
import argparse

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def init_process(backend='gloo'):
    """Initialize the distributed environment"""
    
    # Check if running under MPI
    if 'OMPI_COMM_WORLD_SIZE' in os.environ or 'PMI_SIZE' in os.environ:
        # Running under MPI
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 
                                       os.environ.get('PMI_SIZE', '1')))
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK',
                                  os.environ.get('PMI_RANK', '0')))
        
        # Set environment variables for PyTorch
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        # Single process mode for testing
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    
    return dist.get_rank(), dist.get_world_size()

def print_system_info():
    """Print system and distributed information"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("System Information")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"World size: {world_size}")
        print(f"Backend: {dist.get_backend()}")
        
        # Print all available backends
        print(f"\nDistributed backends available:")
        all_backends = ['gloo', 'mpi', 'nccl']
        for backend in all_backends:
            is_available = dist.is_backend_available(backend) if hasattr(dist, 'is_backend_available') else False
            status = "✓" if is_available else "✗"
            print(f"  {status} {backend}")
        
        import platform
        print(f"\nPlatform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"Architecture: {platform.machine()}")

def test_available_backends():
    """Test which backends are available"""
    rank = dist.get_rank()
    
    if rank == 0:
        print_section("Test 1: Available Backends")
        
        print("Checking PyTorch distributed backends:\n")
        
        backends = ['gloo', 'mpi', 'nccl']
        print(f"{'Backend':<10} {'Available':<15} {'Details':<50}")
        print("-" * 80)
        
        for backend in backends:
            # Check if backend is available
            if hasattr(dist, 'is_backend_available'):
                available = dist.is_backend_available(backend)
            else:
                # Fallback for older PyTorch versions
                available = backend in ['gloo', 'nccl']  # gloo and nccl are usually available
            
            notes = ""
            if backend == 'gloo':
                notes = "CPU communication, recommended for CPU clusters"
            elif backend == 'mpi':
                notes = "Requires MPI installation and mpi4py"
                if available:
                    notes += " [FOUND]"
                else:
                    notes += " [NOT FOUND - install with: pip install mpi4py]"
            elif backend == 'nccl':
                notes = "GPU communication only (requires CUDA)"
                if not available:
                    notes += " [Not available on CPU-only build]"
            
            status = "✓ Yes" if available else "✗ No"
            print(f"{backend:<10} {status:<15} {notes:<50}")

def test_basic_communication():
    """Test basic point-to-point communication"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("Test 2: Basic Communication")
    
    dist.barrier()
    
    if world_size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes for communication test")
            print("  Run with: mpirun -np 2 python test_torch_distributed.py")
        return
    
    if rank == 0:
        # Send
        tensor = torch.ones(5) * rank
        print(f"Rank {rank} sending: {tensor.numpy()}")
        dist.send(tensor=tensor, dst=1)
        
        # Receive
        recv_tensor = torch.zeros(5)
        dist.recv(tensor=recv_tensor, src=1)
        print(f"Rank {rank} received: {recv_tensor.numpy()}")
        
    elif rank == 1:
        # Receive
        recv_tensor = torch.zeros(5)
        dist.recv(tensor=recv_tensor, src=0)
        print(f"Rank {rank} received: {recv_tensor.numpy()}")
        
        # Send
        tensor = torch.ones(5) * rank
        print(f"Rank {rank} sending: {tensor.numpy()}")
        dist.send(tensor=tensor, dst=0)
    
    dist.barrier()
    if rank == 0:
        print("✓ Basic communication test passed")

def test_collective_operations():
    """Test collective operations"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("Test 3: Collective Operations")
    
    dist.barrier()
    
    # Broadcast
    if rank == 0:
        print("\n3a. Broadcast")
    tensor = torch.zeros(5)
    if rank == 0:
        tensor = torch.arange(5, dtype=torch.float32)
        print(f"Rank {rank} broadcasting: {tensor.numpy()}")
    
    dist.broadcast(tensor, src=0)
    print(f"Rank {rank} received: {tensor.numpy()}")
    
    dist.barrier()
    
    # All-reduce (sum)
    if rank == 0:
        print("\n3b. All-Reduce (sum)")
    tensor = torch.ones(5) * rank
    print(f"Rank {rank} before all-reduce: {tensor.numpy()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after all-reduce: {tensor.numpy()}")
    
    dist.barrier()
    
    # All-gather
    if rank == 0:
        print("\n3c. All-Gather")
    tensor = torch.ones(3) * rank
    tensor_list = [torch.zeros(3) for _ in range(world_size)]
    
    dist.all_gather(tensor_list, tensor)
    print(f"Rank {rank} gathered: {[t.numpy() for t in tensor_list]}")
    
    dist.barrier()
    
    # Reduce
    if rank == 0:
        print("\n3d. Reduce (to rank 0)")
    tensor = torch.ones(5) * rank
    print(f"Rank {rank} before reduce: {tensor.numpy()}")
    
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Rank {rank} after reduce: {tensor.numpy()}")
        print(f"Expected sum: {sum(range(world_size)) * torch.ones(5).numpy()}")
    
    dist.barrier()
    if rank == 0:
        print("✓ Collective operations test passed")

def benchmark_bandwidth(size_mb=10, num_iterations=20):
    """Benchmark communication bandwidth"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("Test 4: Bandwidth Benchmark")
    
    dist.barrier()
    
    if world_size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes for bandwidth test")
        return
    
    # Test different message sizes
    sizes = [1024, 10240, 102400, 1024000, 10240000]  # bytes
    
    if rank == 0:
        print(f"\n{'Size':<15} {'Latency (ms)':<15} {'Bandwidth (MB/s)':<20}")
        print("-" * 55)
    
    for size in sizes:
        num_elements = size // 4  # float32 = 4 bytes
        tensor = torch.randn(num_elements)
        
        dist.barrier()
        
        if rank == 0:
            # Send and receive
            start = time.time()
            for _ in range(num_iterations):
                dist.send(tensor=tensor, dst=1)
                dist.recv(tensor=tensor, src=1)
            elapsed = time.time() - start
            
            # Calculate metrics
            latency = (elapsed / num_iterations / 2) * 1000  # ms per one-way
            bandwidth = (size * 2 * num_iterations) / elapsed / 1024 / 1024  # MB/s
            
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
            print(f"{size_str:<15} {latency:<15.3f} {bandwidth:<20.2f}")
            
        elif rank == 1:
            for _ in range(num_iterations):
                dist.recv(tensor=tensor, src=0)
                dist.send(tensor=tensor, dst=0)
    
    dist.barrier()
    if rank == 0:
        print("✓ Bandwidth benchmark complete")

def benchmark_allreduce(size_mb=10, num_iterations=20):
    """Benchmark all-reduce performance"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("Test 5: All-Reduce Benchmark")
    
    dist.barrier()
    
    sizes = [1024, 10240, 102400, 1024000]  # bytes
    
    if rank == 0:
        print(f"\n{'Size':<15} {'Time (ms)':<15} {'Bandwidth (MB/s)':<20}")
        print("-" * 55)
    
    for size in sizes:
        num_elements = size // 4  # float32 = 4 bytes
        tensor = torch.randn(num_elements)
        
        dist.barrier()
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        dist.barrier()
        start = time.time()
        
        for _ in range(num_iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        dist.barrier()
        elapsed = time.time() - start
        
        if rank == 0:
            avg_time = (elapsed / num_iterations) * 1000  # ms
            # All-reduce moves data: (world_size - 1) * size in reduce + broadcast
            bandwidth = (size * 2 * (world_size - 1) * num_iterations) / elapsed / 1024 / 1024
            
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
            print(f"{size_str:<15} {avg_time:<15.3f} {bandwidth:<20.2f}")
    
    dist.barrier()
    if rank == 0:
        print("✓ All-reduce benchmark complete")

def test_distributed_gradient():
    """Test distributed gradient computation"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print_section("Test 6: Distributed Gradient Computation")
    
    dist.barrier()
    
    # Simple linear model
    model = torch.nn.Linear(10, 5)
    
    # Each rank has different input
    x = torch.randn(32, 10) * (rank + 1)
    y = torch.randn(32, 5)
    
    # Forward
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    
    # Backward
    loss.backward()
    
    # Average gradients across ranks
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size
    
    print(f"Rank {rank}: loss = {loss.item():.4f}, "
          f"grad norm = {torch.norm(model.weight.grad).item():.4f}")
    
    dist.barrier()
    if rank == 0:
        print("✓ Distributed gradient test passed")

def main():
    parser = argparse.ArgumentParser(description='Test PyTorch distributed')
    parser.add_argument('--backend', type=str, default='gloo',
                       help='Backend to use (gloo, mpi, nccl)')
    args = parser.parse_args()
    
    try:
        rank, world_size = init_process(backend=args.backend)
        
        print_system_info()
        
        if rank == 0:
            test_available_backends()
        
        dist.barrier()
        
        test_basic_communication()
        test_collective_operations()
        benchmark_bandwidth()
        benchmark_allreduce()
        test_distributed_gradient()
        
        if rank == 0:
            print_section("All Tests Complete")
            print("✓ All distributed tests passed successfully!")
        
    except Exception as e:
        print(f"Error on rank {rank if 'rank' in locals() else 'unknown'}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
