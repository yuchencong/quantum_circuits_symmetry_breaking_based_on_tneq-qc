#!/usr/bin/env python3
"""
Test einsum performance: opt_einsum vs torch.einsum
For Fujitsu ARM cluster with PyTorch 1.13 CPU
"""

import time
import torch
import numpy as np
try:
    import opt_einsum
    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False
    print("Warning: opt_einsum not available")

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def benchmark_einsum(equation, *tensors, num_runs=100, warmup=10):
    """Benchmark einsum operation"""
    
    # Warmup
    for _ in range(warmup):
        _ = torch.einsum(equation, *tensors)
    
    # Benchmark torch.einsum
    start = time.time()
    for _ in range(num_runs):
        result_torch = torch.einsum(equation, *tensors)
    torch_time = (time.time() - start) / num_runs
    
    # Benchmark opt_einsum if available
    opt_time = None
    if HAS_OPT_EINSUM:
        # Warmup
        for _ in range(warmup):
            _ = opt_einsum.contract(equation, *tensors)
        
        start = time.time()
        for _ in range(num_runs):
            result_opt = opt_einsum.contract(equation, *tensors)
        opt_time = (time.time() - start) / num_runs
        
        # Verify results match
        max_diff = torch.max(torch.abs(result_torch - result_opt)).item()
        assert max_diff < 1e-5, f"Results don't match! Max diff: {max_diff}"
    
    return torch_time, opt_time

def test_simple_matmul():
    """Test simple matrix multiplication: ij,jk->ik"""
    print_section("Test 1: Simple Matrix Multiplication (ij,jk->ik)")
    
    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    
    print(f"{'Size':<15} {'torch.einsum':<15} {'opt_einsum':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        A = torch.randn(size[0], size[1])
        B = torch.randn(size[1], size[0])
        
        torch_time, opt_time = benchmark_einsum('ij,jk->ik', A, B)
        
        speedup = torch_time / opt_time if opt_time else None
        opt_str = f"{opt_time*1000:.3f}ms" if opt_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        
        print(f"{str(size):<15} {torch_time*1000:.3f}ms      {opt_str:<15} {speedup_str:<10}")

def test_batch_matmul():
    """Test batch matrix multiplication: bij,bjk->bik"""
    print_section("Test 2: Batch Matrix Multiplication (bij,bjk->bik)")
    
    configs = [
        (32, 64, 64),   # (batch, m, n)
        (64, 128, 128),
        (128, 256, 256),
    ]
    
    print(f"{'Batch×Size':<20} {'torch.einsum':<15} {'opt_einsum':<15} {'Speedup':<10}")
    print("-" * 65)
    
    for batch, m, n in configs:
        A = torch.randn(batch, m, n)
        B = torch.randn(batch, n, m)
        
        torch_time, opt_time = benchmark_einsum('bij,bjk->bik', A, B)
        
        speedup = torch_time / opt_time if opt_time else None
        opt_str = f"{opt_time*1000:.3f}ms" if opt_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        
        print(f"{batch}×{m}×{n:<15} {torch_time*1000:.3f}ms      {opt_str:<15} {speedup_str:<10}")

def test_tensor_contraction():
    """Test complex tensor contraction"""
    print_section("Test 3: Complex Tensor Contraction")
    
    # Test case: abcd,cdef,efgh->abgh
    configs = [
        (4, 8, 8, 4),    # Small
        (8, 16, 16, 8),  # Medium
        (16, 32, 32, 16), # Large
    ]
    
    print(f"{'Dimensions':<25} {'torch.einsum':<15} {'opt_einsum':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for dim in configs:
        a, b, c, d = dim
        T1 = torch.randn(a, b, c, d)
        T2 = torch.randn(c, d, c, d)
        T3 = torch.randn(c, d, a, b)
        
        equation = 'abcd,cdef,efgh->abgh'
        torch_time, opt_time = benchmark_einsum(equation, T1, T2, T3, num_runs=50)
        
        speedup = torch_time / opt_time if opt_time else None
        opt_str = f"{opt_time*1000:.3f}ms" if opt_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        
        print(f"{str(dim):<25} {torch_time*1000:.3f}ms      {opt_str:<15} {speedup_str:<10}")

def test_trace_and_diagonal():
    """Test trace and diagonal operations"""
    print_section("Test 4: Trace and Diagonal Operations")
    
    sizes = [64, 128, 256, 512]
    
    print("\n4a. Matrix Trace (ii->)")
    print(f"{'Size':<15} {'torch.einsum':<15} {'opt_einsum':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        A = torch.randn(size, size)
        
        torch_time, opt_time = benchmark_einsum('ii->', A, num_runs=500)
        
        speedup = torch_time / opt_time if opt_time else None
        opt_str = f"{opt_time*1000:.3f}ms" if opt_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        
        print(f"{size}×{size:<10} {torch_time*1000:.3f}ms      {opt_str:<15} {speedup_str:<10}")

def test_outer_product():
    """Test outer product"""
    print_section("Test 5: Outer Product (i,j->ij)")
    
    sizes = [64, 128, 256, 512]
    
    print(f"{'Size':<15} {'torch.einsum':<15} {'opt_einsum':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        a = torch.randn(size)
        b = torch.randn(size)
        
        torch_time, opt_time = benchmark_einsum('i,j->ij', a, b, num_runs=200)
        
        speedup = torch_time / opt_time if opt_time else None
        opt_str = f"{opt_time*1000:.3f}ms" if opt_time else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        
        print(f"{size}×{size:<10} {torch_time*1000:.3f}ms      {opt_str:<15} {speedup_str:<10}")

def test_einsum_path_optimization():
    """Test einsum path optimization (opt_einsum feature)"""
    if not HAS_OPT_EINSUM:
        return
    
    print_section("Test 6: Einsum Path Optimization")
    
    # Complex multi-tensor contraction
    A = torch.randn(10, 12)
    B = torch.randn(12, 14)
    C = torch.randn(14, 10)
    
    equation = 'ij,jk,ki->'
    
    print("\nEquation:", equation)
    print("Tensor shapes:", [t.shape for t in [A, B, C]])
    
    # Get optimal path
    path_info = opt_einsum.contract_path(equation, A, B, C)
    print("\nOptimal contraction path:")
    print(path_info[1])
    
    # Benchmark with and without path
    print("\nPerformance comparison:")
    
    # Without path (default)
    start = time.time()
    for _ in range(100):
        result1 = opt_einsum.contract(equation, A, B, C)
    time_no_path = (time.time() - start) / 100
    
    # With pre-computed path
    path = path_info[0]
    start = time.time()
    for _ in range(100):
        result2 = opt_einsum.contract(equation, A, B, C, optimize=path)
    time_with_path = (time.time() - start) / 100
    
    print(f"Without path: {time_no_path*1000:.3f}ms")
    print(f"With path:    {time_with_path*1000:.3f}ms")
    print(f"Speedup:      {time_no_path/time_with_path:.2f}x")

def print_system_info():
    """Print system information"""
    print_section("System Information")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    if HAS_OPT_EINSUM:
        print(f"opt_einsum version: {opt_einsum.__version__}")
    else:
        print("opt_einsum: Not installed")
    
    print(f"\nCPU threads: {torch.get_num_threads()}")
    print(f"MKL available: {torch.backends.mkl.is_available()}")
    print(f"OpenMP available: {torch.backends.openmp.is_available()}")
    
    # ARM-specific info
    import platform
    print(f"\nPlatform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")

def main():
    print_system_info()
    
    test_simple_matmul()
    test_batch_matmul()
    test_tensor_contraction()
    test_trace_and_diagonal()
    test_outer_product()
    
    if HAS_OPT_EINSUM:
        test_einsum_path_optimization()
    
    print_section("Test Complete")
    print("\nSummary:")
    if HAS_OPT_EINSUM:
        print("✓ opt_einsum is available and tested")
        print("✓ All tests passed successfully")
    else:
        print("✗ opt_einsum is not available")
        print("  Install with: pip install opt_einsum")

if __name__ == "__main__":
    main()
