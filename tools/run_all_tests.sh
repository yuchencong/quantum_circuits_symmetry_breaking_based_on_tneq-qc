#!/bin/bash
#PBS -N cluster_tests
#PBS -l select=2:ncpus=48:mpiprocs=48
#PBS -l walltime=02:00:00
#PBS -j oe

# Batch job script for running cluster tests on Fujitsu ARM cluster
# Submit with: qsub run_all_tests.sh

echo "========================================================================"
echo "  Cluster Testing Suite - Fujitsu ARM Cluster"
echo "  Start Time: $(date)"
echo "========================================================================"

cd $PBS_O_WORKDIR
source ~/venv_tneq/bin/activate

# Create results directory
RESULTS_DIR="test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Test 1: Einsum Performance
echo "========================================================================"
echo "  Test 1: Einsum Performance (opt_einsum vs torch.einsum)"
echo "========================================================================"
python test_einsum_performance.py | tee $RESULTS_DIR/einsum_performance.txt
echo ""

# Test 2: Autograd Functionality
echo "========================================================================"
echo "  Test 2: PyTorch Autograd Functionality"
echo "========================================================================"
python test_autograd.py | tee $RESULTS_DIR/autograd_tests.txt
echo ""

# Test 3: MPI4py Tests (with different process counts)
echo "========================================================================"
echo "  Test 3: MPI4py Communication Tests"
echo "========================================================================"

echo "--- 3a. MPI4py with 2 processes ---"
mpirun -np 2 python test_mpi4py.py | tee $RESULTS_DIR/mpi4py_np2.txt
echo ""

echo "--- 3b. MPI4py with 4 processes ---"
mpirun -np 4 python test_mpi4py.py | tee $RESULTS_DIR/mpi4py_np4.txt
echo ""

echo "--- 3c. MPI4py with 8 processes ---"
mpirun -np 8 python test_mpi4py.py | tee $RESULTS_DIR/mpi4py_np8.txt
echo ""

# Test 4: PyTorch Distributed (with different backends)
echo "========================================================================"
echo "  Test 4: PyTorch Distributed Communication"
echo "========================================================================"

echo "--- 4a. PyTorch Distributed with gloo backend (2 processes) ---"
mpirun -np 2 python test_torch_distributed.py --backend gloo | tee $RESULTS_DIR/torch_dist_gloo_np2.txt
echo ""

echo "--- 4b. PyTorch Distributed with gloo backend (4 processes) ---"
mpirun -np 4 python test_torch_distributed.py --backend gloo | tee $RESULTS_DIR/torch_dist_gloo_np4.txt
echo ""

echo "--- 4c. PyTorch Distributed with mpi backend (4 processes) ---"
mpirun -np 4 python test_torch_distributed.py --backend mpi | tee $RESULTS_DIR/torch_dist_mpi_np4.txt 2>&1
echo ""

# Generate summary report
echo "========================================================================"
echo "  Generating Summary Report"
echo "========================================================================"

cat > $RESULTS_DIR/SUMMARY.txt << EOF
Cluster Testing Summary
=======================

Test Run: $(date)
Hostname: $(hostname)
Working Directory: $(pwd)

Test Results:
-------------

1. Einsum Performance Tests
   File: einsum_performance.txt
   Status: $(grep -q "Test Complete" $RESULTS_DIR/einsum_performance.txt && echo "PASSED" || echo "FAILED")

2. Autograd Tests
   File: autograd_tests.txt
   Status: $(grep -q "All Tests Complete" $RESULTS_DIR/autograd_tests.txt && echo "PASSED" || echo "FAILED")

3. MPI4py Tests
   - 2 processes: $(grep -q "All Tests Complete" $RESULTS_DIR/mpi4py_np2.txt && echo "PASSED" || echo "FAILED")
   - 4 processes: $(grep -q "All Tests Complete" $RESULTS_DIR/mpi4py_np4.txt && echo "PASSED" || echo "FAILED")
   - 8 processes: $(grep -q "All Tests Complete" $RESULTS_DIR/mpi4py_np8.txt && echo "PASSED" || echo "FAILED")

4. PyTorch Distributed Tests
   - gloo 2 procs: $(grep -q "All Tests Complete" $RESULTS_DIR/torch_dist_gloo_np2.txt && echo "PASSED" || echo "FAILED")
   - gloo 4 procs: $(grep -q "All Tests Complete" $RESULTS_DIR/torch_dist_gloo_np4.txt && echo "PASSED" || echo "FAILED")
   - mpi 4 procs: $(grep -q "All Tests Complete" $RESULTS_DIR/torch_dist_mpi_np4.txt && echo "PASSED" || echo "CHECK LOG")

Performance Highlights:
-----------------------

Einsum Speedup (opt_einsum vs torch.einsum):
$(grep -A 10 "Simple Matrix Multiplication" $RESULTS_DIR/einsum_performance.txt | grep -E "x$" | tail -1 || echo "N/A")

MPI Latency (ping-pong):
$(grep "Average latency" $RESULTS_DIR/mpi4py_np2.txt | tail -1 || echo "N/A")

MPI Bandwidth (large messages):
$(grep "MB/s" $RESULTS_DIR/mpi4py_np2.txt | grep -E "[0-9]+\.[0-9]+ MB" | tail -1 || echo "N/A")

All result files are in: $RESULTS_DIR/

EOF

cat $RESULTS_DIR/SUMMARY.txt

echo ""
echo "========================================================================"
echo "  All Tests Complete!"
echo "  End Time: $(date)"
echo "========================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "Summary: $RESULTS_DIR/SUMMARY.txt"
