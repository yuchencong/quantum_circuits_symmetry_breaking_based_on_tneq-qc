#!/usr/bin/env python3
"""
Test mpi4py functionality and performance
For Fujitsu ARM cluster

Usage:
  mpirun -np 4 python test_mpi4py.py
"""

import sys
import time
import numpy as np

try:
    from mpi4py import MPI
    HAS_MPI4PY = True
except ImportError:
    print("Error: mpi4py not installed")
    print("Install with: pip install mpi4py")
    HAS_MPI4PY = False
    sys.exit(1)

def print_section(title, comm):
    """Print section header on rank 0"""
    if comm.Get_rank() == 0:
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
        sys.stdout.flush()  # Force flush output buffer
    comm.Barrier()

def print_system_info(comm):
    """Print MPI system information"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Import modules outside the conditional blocks
    import platform
    import socket
    
    if rank == 0:
        print_section("System Information", comm)
        print(f"MPI4py version: {MPI.Get_version()}")
        print(f"MPI library version: {MPI.Get_library_version()}")
        print(f"Number of processes: {size}")
        
        print(f"\nPlatform: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
        print(f"Architecture: {platform.machine()}")
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Print info from each rank
    for i in range(size):
        if rank == i:
            print(f"Rank {rank}: hostname = {socket.gethostname()}")
            sys.stdout.flush()
            comm.Barrier()

def test_basic_communication(comm):
    """Test basic point-to-point communication"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print('run test_basic_communication on rank', rank)
    sys.stdout.flush()

    # print_section("Test 1: Basic Point-to-Point Communication", comm)
    
    print('start test_basic_communication on rank', rank)
    sys.stdout.flush()

    if size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes for communication test")
            sys.stdout.flush()
        return
    
    if rank == 0:
        data = {'number': 42, 'text': 'Hello from rank 0', 'array': np.arange(5)}
        print(f"Rank {rank} sending: {data}")
        sys.stdout.flush()
        comm.send(data, dest=1, tag=11)
        
        # Receive reply
        data = comm.recv(source=1, tag=22)
        print(f"Rank {rank} received: {data}")
        sys.stdout.flush()
        
    elif rank == 1:
        # Receive from rank 0
        data = comm.recv(source=0, tag=11)
        print(f"Rank {rank} received: {data}")
        sys.stdout.flush()
        
        # Send reply
        reply = {'number': 99, 'text': 'Reply from rank 1', 'array': np.arange(10, 15)}
        print(f"Rank {rank} sending: {reply}")
        sys.stdout.flush()
        comm.send(reply, dest=0, tag=22)
    
    comm.Barrier()
    if rank == 0:
        print("✓ Basic communication test passed")
        sys.stdout.flush()

def test_numpy_arrays(comm):
    """Test numpy array communication"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 2: NumPy Array Communication", comm)
    
    if size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes")
        return
    
    # Test with uppercase (buffer protocol)
    if rank == 0:
        print("\n2a. Using Send/Recv (buffer protocol)")
        
        # Send
        data = np.arange(10, dtype=np.float64)
        print(f"Rank {rank} sending array: {data}")
        comm.Send(data, dest=1, tag=11)
        
        # Receive
        recv_data = np.empty(10, dtype=np.float64)
        comm.Recv(recv_data, source=1, tag=22)
        print(f"Rank {rank} received array: {recv_data}")
        
    elif rank == 1:
        # Receive
        recv_data = np.empty(10, dtype=np.float64)
        comm.Recv(recv_data, source=0, tag=11)
        print(f"Rank {rank} received array: {recv_data}")
        
        # Send
        data = np.arange(10, 20, dtype=np.float64)
        print(f"Rank {rank} sending array: {data}")
        comm.Send(data, dest=0, tag=22)
    
    comm.Barrier()
    if rank == 0:
        print("✓ NumPy array communication test passed")

def test_collective_operations(comm):
    """Test collective operations"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 3: Collective Operations", comm)
    
    # Barrier
    if rank == 0:
        print("\n3a. Barrier synchronization")
    comm.Barrier()
    print(f"Rank {rank} passed barrier")
    comm.Barrier()
    
    # Broadcast
    if rank == 0:
        print("\n3b. Broadcast")
        data = {'key': 'value', 'number': 42}
    else:
        data = None
    
    data = comm.bcast(data, root=0)
    print(f"Rank {rank} received broadcast: {data}")
    comm.Barrier()
    
    # Scatter
    if rank == 0:
        print("\n3c. Scatter")
        send_data = [f"data_{i}" for i in range(size)]
        print(f"Rank {rank} scattering: {send_data}")
    else:
        send_data = None
    
    recv_data = comm.scatter(send_data, root=0)
    print(f"Rank {rank} received: {recv_data}")
    comm.Barrier()
    
    # Gather
    if rank == 0:
        print("\n3d. Gather")
    
    send_data = rank * 10
    recv_data = comm.gather(send_data, root=0)
    
    if rank == 0:
        print(f"Rank {rank} gathered: {recv_data}")
    comm.Barrier()
    
    # Reduce (sum)
    if rank == 0:
        print("\n3e. Reduce (sum)")
    
    send_data = np.array([rank] * 5, dtype=np.float64)
    recv_data = np.zeros(5, dtype=np.float64)
    
    comm.Reduce(send_data, recv_data, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"Sum of all ranks: {recv_data}")
        expected = np.array([sum(range(size))] * 5)
        print(f"Expected: {expected}")
        assert np.allclose(recv_data, expected)
    comm.Barrier()
    
    # Allreduce
    if rank == 0:
        print("\n3f. Allreduce (sum)")
    
    send_data = np.array([rank] * 5, dtype=np.float64)
    recv_data = np.zeros(5, dtype=np.float64)
    
    comm.Allreduce(send_data, recv_data, op=MPI.SUM)
    print(f"Rank {rank} allreduce result: {recv_data}")
    
    comm.Barrier()
    if rank == 0:
        print("✓ Collective operations test passed")

def benchmark_latency(comm, num_iterations=1000):
    """Benchmark message latency"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 4: Latency Benchmark", comm)
    
    if size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes")
        return
    
    # Ping-pong test
    if rank == 0:
        print(f"\nPing-pong test ({num_iterations} iterations)")
        
        # Small message
        data = np.zeros(1, dtype=np.float64)
        
        comm.Barrier()
        start = time.time()
        
        for _ in range(num_iterations):
            comm.Send(data, dest=1, tag=0)
            comm.Recv(data, source=1, tag=0)
        
        elapsed = time.time() - start
        latency = (elapsed / num_iterations / 2) * 1e6  # microseconds
        
        print(f"Average latency: {latency:.2f} µs")
        
    elif rank == 1:
        data = np.zeros(1, dtype=np.float64)
        
        comm.Barrier()
        
        for _ in range(num_iterations):
            comm.Recv(data, source=0, tag=0)
            comm.Send(data, dest=0, tag=0)
    
    comm.Barrier()
    if rank == 0:
        print("✓ Latency benchmark complete")

def benchmark_bandwidth(comm, num_iterations=100):
    """Benchmark communication bandwidth"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 5: Bandwidth Benchmark", comm)
    
    if size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes")
        return
    
    # Test different message sizes
    sizes = [1024, 10240, 102400, 1024000, 10240000]  # bytes
    
    if rank == 0:
        print(f"\n{'Size':<15} {'Time (ms)':<15} {'Bandwidth (MB/s)':<20}")
        print("-" * 55)
    
    for size in sizes:
        num_elements = size // 8  # float64 = 8 bytes
        data = np.zeros(num_elements, dtype=np.float64)
        
        comm.Barrier()
        
        if rank == 0:
            # Send and receive
            start = time.time()
            for _ in range(num_iterations):
                comm.Send(data, dest=1, tag=0)
                comm.Recv(data, source=1, tag=0)
            elapsed = time.time() - start
            
            # Calculate bandwidth (bidirectional)
            bandwidth = (size * 2 * num_iterations) / elapsed / 1024 / 1024  # MB/s
            avg_time = (elapsed / num_iterations) * 1000  # ms
            
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
            print(f"{size_str:<15} {avg_time:<15.3f} {bandwidth:<20.2f}")
            
        elif rank == 1:
            for _ in range(num_iterations):
                comm.Recv(data, source=0, tag=0)
                comm.Send(data, dest=0, tag=0)
    
    comm.Barrier()
    if rank == 0:
        print("✓ Bandwidth benchmark complete")

def benchmark_allreduce(comm, num_iterations=100):
    """Benchmark allreduce performance"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 6: Allreduce Benchmark", comm)
    
    sizes = [1024, 10240, 102400, 1024000]  # bytes
    
    if rank == 0:
        print(f"\n{'Size':<15} {'Time (ms)':<15} {'Throughput (MB/s)':<20}")
        print("-" * 55)
    
    for msg_size in sizes:
        num_elements = msg_size // 8  # float64 = 8 bytes
        send_data = np.ones(num_elements, dtype=np.float64) * rank
        recv_data = np.zeros(num_elements, dtype=np.float64)
        
        # Warmup
        for _ in range(10):
            comm.Allreduce(send_data, recv_data, op=MPI.SUM)
        
        comm.Barrier()
        start = time.time()
        
        for _ in range(num_iterations):
            comm.Allreduce(send_data, recv_data, op=MPI.SUM)
        
        comm.Barrier()
        elapsed = time.time() - start
        
        if rank == 0:
            avg_time = (elapsed / num_iterations) * 1000  # ms
            throughput = (msg_size * num_iterations) / elapsed / 1024 / 1024  # MB/s
            
            size_str = f"{msg_size/1024:.1f} KB" if msg_size < 1024*1024 else f"{msg_size/1024/1024:.1f} MB"
            print(f"{size_str:<15} {avg_time:<15.3f} {throughput:<20.2f}")
    
    comm.Barrier()
    if rank == 0:
        print("✓ Allreduce benchmark complete")

def test_nonblocking_communication(comm):
    """Test non-blocking communication"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print_section("Test 7: Non-blocking Communication", comm)
    
    if size < 2:
        if rank == 0:
            print("⚠ Need at least 2 processes")
        return
    
    if rank == 0:
        # Initiate non-blocking send
        data = np.arange(10, dtype=np.float64)
        print(f"Rank {rank} initiating non-blocking send: {data}")
        req = comm.Isend(data, dest=1, tag=0)
        
        # Do some computation while sending
        result = np.sum(np.arange(1000000))
        print(f"Rank {rank} did computation while sending: sum = {result}")
        
        # Wait for send to complete
        req.Wait()
        print(f"Rank {rank} send completed")
        
        # Non-blocking receive
        recv_data = np.empty(10, dtype=np.float64)
        req = comm.Irecv(recv_data, source=1, tag=0)
        
        # Do some computation while receiving
        result = np.sum(np.arange(1000000))
        print(f"Rank {rank} did computation while receiving: sum = {result}")
        
        # Wait for receive to complete
        req.Wait()
        print(f"Rank {rank} received: {recv_data}")
        
    elif rank == 1:
        # Non-blocking receive
        recv_data = np.empty(10, dtype=np.float64)
        req = comm.Irecv(recv_data, source=0, tag=0)
        req.Wait()
        print(f"Rank {rank} received: {recv_data}")
        
        # Non-blocking send
        data = np.arange(10, 20, dtype=np.float64)
        print(f"Rank {rank} sending: {data}")
        req = comm.Isend(data, dest=0, tag=0)
        req.Wait()
    
    comm.Barrier()
    if rank == 0:
        print("✓ Non-blocking communication test passed")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    try:
        print_system_info(comm)
        test_basic_communication(comm)
        test_numpy_arrays(comm)
        test_collective_operations(comm)
        benchmark_latency(comm)
        benchmark_bandwidth(comm)
        benchmark_allreduce(comm)
        test_nonblocking_communication(comm)
        
        if rank == 0:
            print_section("All Tests Complete", comm)
            print("✓ All mpi4py tests passed successfully!")
    
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        comm.Abort(1)

if __name__ == "__main__":
    if not HAS_MPI4PY:
        sys.exit(1)
    main()
