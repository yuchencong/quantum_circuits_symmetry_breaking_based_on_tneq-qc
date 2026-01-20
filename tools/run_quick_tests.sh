#!/bin/bash
# Quick test script for local/interactive testing
# Usage: ./run_quick_tests.sh

set -e  # Exit on error

echo "=========================================="
echo "  Quick Cluster Tests"
echo "=========================================="
echo ""

# Check Python
echo "Checking Python..."
python3 --version
echo ""

# Check PyTorch
echo "Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo ""

# Check PyTorch Distributed Backends
echo "Checking PyTorch Distributed Backends..."
python3 -c "
import torch.distributed as dist
backends = ['gloo', 'mpi', 'nccl']
print('Available backends:')
for backend in backends:
    try:
        available = dist.is_backend_available(backend) if hasattr(dist, 'is_backend_available') else (backend in ['gloo'])
        status = '✓' if available else '✗'
        print(f'  {status} {backend}')
    except:
        print(f'  ? {backend} (unknown)')
"
echo ""

# Check MPI
echo "Checking MPI..."
which mpirun && mpirun --version || echo "MPI not found"
echo ""

# Test 1: Einsum (quick)
echo "=========================================="
echo "  Test 1: Einsum Performance (quick)"
echo "=========================================="
python3 test_einsum_performance.py
echo ""

# Test 2: Autograd
echo "=========================================="
echo "  Test 2: Autograd"
echo "=========================================="
python3 test_autograd.py
echo ""

# Test 3: MPI4py (if available)
# echo "=========================================="
# echo "  Test 3: MPI4py (2 processes)"
# echo "=========================================="
# if command -v mpirun &> /dev/null; then
#     mpirun -np 2 python3 test_mpi4py.py
# else
#     echo "MPI not available, skipping"
# fi
# echo ""

# Test 4: PyTorch Distributed (if available)
echo "=========================================="
echo "  Test 4: PyTorch Distributed"
echo "=========================================="
if command -v mpirun &> /dev/null; then
    mpirun -np 2 python3 test_torch_distributed.py --backend gloo
else
    echo "MPI not available, running single process mode"
    python3 test_torch_distributed.py --backend gloo
fi
echo ""

# echo "=========================================="
# echo "  All Quick Tests Complete!"
# echo "=========================================="echo ""

# A64FX Architecture Information
echo "=========================================="
echo "  A64FX Architecture Information"
echo "=========================================="
echo ""

# CPU Model and Basic Info
echo "--- CPU Information ---"
if [ -f /proc/cpuinfo ]; then
    echo "CPU Model:"
    grep -m1 "CPU implementer\|CPU architecture\|CPU variant\|CPU part" /proc/cpuinfo || \
    grep -m1 "model name" /proc/cpuinfo || echo "  Not available"
    echo ""
    
    echo "Total CPU cores:"
    grep -c "^processor" /proc/cpuinfo || echo "  Not available"
    echo ""
fi

# Check for A64FX specific features
echo "--- A64FX Features ---"
if command -v lscpu &> /dev/null; then
    echo "Architecture details:"
    lscpu | grep -E "Architecture|CPU\(s\)|Thread|Core|Socket|NUMA|Model name|Vendor ID" || echo "  lscpu not available"
    echo ""
    
    echo "SIMD Extensions:"
    lscpu | grep -i "flags\|features" || grep -i "Features" /proc/cpuinfo | head -1 || echo "  Not available"
    echo ""
fi

# NUMA and CMG Information
echo "--- NUMA/CMG Topology ---"
if command -v numactl &> /dev/null; then
    echo "NUMA nodes (likely corresponding to CMGs on A64FX):"
    numactl --hardware 2>/dev/null || echo "  numactl not available or no NUMA"
    echo ""
else
    echo "numactl not available"
    if [ -d /sys/devices/system/node ]; then
        echo "NUMA nodes from sysfs:"
        ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l | xargs echo "  Number of nodes:"
        echo ""
    fi
fi

# Memory Information
echo "--- Memory Information ---"
if [ -f /proc/meminfo ]; then
    echo "Total Memory:"
    grep "MemTotal" /proc/meminfo || echo "  Not available"
    
    echo "Memory per NUMA node:"
    if command -v numactl &> /dev/null; then
        numactl --hardware 2>/dev/null | grep "size:" || echo "  Not available"
    fi
    echo ""
fi

# Cache Information
echo "--- Cache Hierarchy ---"
if command -v lscpu &> /dev/null; then
    lscpu | grep -i "cache" || echo "  Cache info not available"
    echo ""
fi

# SVE Support (A64FX specific)
echo "--- SVE (Scalable Vector Extension) ---"
if [ -f /proc/cpuinfo ]; then
    if grep -q "sve" /proc/cpuinfo; then
        echo "  ✓ SVE is supported"
        grep "Features" /proc/cpuinfo | head -1 | grep -o "sve[^ ]*" || echo "  SVE features detected"
    else
        echo "  ✗ SVE not detected (or not in cpuinfo)"
    fi
    echo ""
fi

# CPU Frequency
echo "--- CPU Frequency ---"
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
    echo "Current CPU frequency:"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq | awk '{printf "  %.2f GHz\n", $1/1000000}'
elif [ -f /proc/cpuinfo ]; then
    echo "CPU MHz:"
    grep "cpu MHz" /proc/cpuinfo | head -1 || echo "  Not available"
fi
echo ""

# Python hardware detection
echo "--- Python Hardware Detection ---"
python3 -c "
import os
import platform

print('Platform:', platform.machine())
print('System:', platform.system())
print('CPU count:', os.cpu_count())

# Try to detect A64FX specific info
try:
    import subprocess
    result = subprocess.run(['lscpu'], capture_output=True, text=True)
    if 'A64FX' in result.stdout or 'ARM' in result.stdout:
        print('Detected: Likely A64FX or ARM architecture')
except:
    pass

# Check for NUMA
try:
    numa_nodes = len([d for d in os.listdir('/sys/devices/system/node') if d.startswith('node')])
    print(f'NUMA nodes: {numa_nodes} (likely CMGs on A64FX)')
except:
    print('NUMA nodes: Unable to detect')
"
echo ""

echo "=========================================="
echo "  Architecture Detection Complete"
echo "=========================================="