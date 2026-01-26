"""
Example: Distributed TNEQ Training with Autograd

Demonstrates distributed training with:
- Hierarchical tensor contraction across multiple processes
- Gradient-aware allreduce operations
- SGDG optimizer on Stiefel manifold

Usage:
    # Run with shell script:
    ./run.sh

    # Or directly with torchrun:
    torchrun --nproc_per_node=4 example_distributed_training.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from tneq_qc.distributed import DistributedTrainer, DistributedConfig
from tneq_qc.distributed.optim import DistributedSGDG, LRScheduler

from tneq_qc.core.qctn import QCTNHelper
from tqdm import tqdm

def generate_2qubits_graph(n):
    graph = ""

    import opt_einsum
    char_list = [opt_einsum.get_symbol(i) for i in range(n)]

    # dim_char = '3'
    dim_char = '16'

    line = ("-"+dim_char+"-").join(char_list[:n])
    line = f"-{dim_char}-" + line + f"-{dim_char}-"
    graph += line + "\n" + line + "\n"
    return graph



def main():
    """Main function for distributed training example."""
    
    print("Initializing distributed training example...")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initialize PyTorch distributed (if running with torchrun)
    if 'RANK' in os.environ:
        dist.init_process_group(backend='gloo')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    # Print node information
    num_nodes = int(os.environ.get('NNODES', 1))
    node_rank = int(os.environ.get('NODE_RANK', 0))
    
    print(f"Distributed setup: node rank {node_rank} / {num_nodes}, "
          f"processes rank {rank} / {world_size}")


    if rank == 0:
        print("=" * 60)
        print("Distributed TNEQ Training Example")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Backend: PyTorch CPU (gloo)")
        print()
    
    # graph_type = "std"
    # graph_type = "tree"
    graph_type = "wall"

    # graph = QCTNHelper.generate_example_graph(n=2)
    # graph = QCTNHelper.generate_example_graph(n=3)
    # graph = QCTNHelper.generate_example_graph(n=5)
    graph = QCTNHelper.generate_example_graph(n=17, graph_type=graph_type, dim_char='3')
    # graph = QCTNHelper.generate_example_graph(n=17, dim_char='16')
    # graph = generate_2qubits_graph(n=16)
    print(f"[Rank {rank}] QCTN graph:\n{graph}")

    # Create distributed configuration
    config = DistributedConfig(
        # Backend configuration
        backend_type='pytorch',
        device='cpu',
        strategy_mode='balanced',
        
        # QCTN configuration
        qctn_graph=graph,  # 3-core tensor network
        # num_qubits=1,
        
        # Communication configuration
        comm_backend='torch' if world_size > 1 else 'auto',
        use_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,
        
        # Partitioning configuration
        partition_strategy='layer',
        # partitions = [
        #     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'],
        #     ['m', 'n', 'o', 'p'],
        #     ['q', 'r', 's', 't'],
        #     ['u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F'],
        # ],
        partitions = [
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'o'],

            ['n', 'p'],
            ['r', 't'],

            ['q', 's', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F'],
        ],
        
        # Training configuration
        max_steps=1000,
        log_interval=10,
        learning_rate=0.01,
        optimizer='sgdg',
        momentum=0.9,
        stiefel=True,
    )


    """
    -3-a-3-----b-----3-
    -3-a-3-c-3-b-3-d-3-
    -3-e-3-c-3-f-3-d-3-
    -3-e-3-g-3-f-3-h-3-
    -3-i-3-g-3-j-3-h-3-
    -3-i-3-k-3-j-3-l-3-
    -3-m-3-k-3-n-3-l-3-
    -3-m-3-o-3-n-3-p-3-
    -3-q-3-o-3-r-3-p-3-
    -3-q-3-s-3-r-3-t-3-
    -3-u-3-s-3-v-3-t-3-
    -3-u-3-w-3-v-3-x-3-
    -3-y-3-w-3-z-3-x-3-
    -3-y-3-A-3-z-3-B-3-
    -3-C-3-A-3-D-3-B-3-
    -3-C-3-E-3-D-3-F-3-
    -3-----E-3-----F-3-



    {'from_core': 'P0', 'to_core': 'P1', 'from_partition': 0, 'to_partition': 1, 'from_core_idx': 10, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 6, 'from_core_raw': 'k', 'to_core_raw': 'n'}, 
    {'from_core': 'P1', 'to_core': 'P0', 'from_partition': 1, 'to_partition': 0, 'from_core_idx': 0, 'to_core_idx': 11, 'edge_rank': 3, 'qubit_idx': 6, 'from_core_raw': 'n', 'to_core_raw': 'l'}, 
    {'from_core': 'P0', 'to_core': 'P1', 'from_partition': 0, 'to_partition': 1, 'from_core_idx': 13, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 7, 'from_core_raw': 'o', 'to_core_raw': 'n'}, 
    {'from_core': 'P0', 'to_core': 'P2', 'from_partition': 0, 'to_partition': 2, 'from_core_idx': 13, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'o', 'to_core_raw': 'r'}, 
    
    {'from_core': 'P2', 'to_core': 'P1', 'from_partition': 2, 'to_partition': 1, 'from_core_idx': 0, 'to_core_idx': 1, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'r', 'to_core_raw': 'p'}, 
    
    {'from_core': 'P3', 'to_core': 'P0', 'from_partition': 3, 'to_partition': 0, 'from_core_idx': 0, 'to_core_idx': 13, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'q', 'to_core_raw': 'o'}, 
    {'from_core': 'P3', 'to_core': 'P2', 'from_partition': 3, 'to_partition': 2, 'from_core_idx': 1, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 9, 'from_core_raw': 's', 'to_core_raw': 'r'}, 
    {'from_core': 'P3', 'to_core': 'P2', 'from_partition': 3, 'to_partition': 2, 'from_core_idx': 3, 'to_core_idx': 1, 'edge_rank': 3, 'qubit_idx': 10, 'from_core_raw': 'v', 'to_core_raw': 't'}



    {'from_core': 'P0', 'to_core': 'P2', 'from_partition': 0, 'to_partition': 2, 'from_core_idx': 10, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 6, 'from_core_raw': 'k', 'to_core_raw': 'n'}, 
    {'from_core': 'P2', 'to_core': 'P0', 'from_partition': 2, 'to_partition': 0, 'from_core_idx': 0, 'to_core_idx': 11, 'edge_rank': 3, 'qubit_idx': 6, 'from_core_raw': 'n', 'to_core_raw': 'l'}, 
    {'from_core': 'P0', 'to_core': 'P2', 'from_partition': 0, 'to_partition': 2, 'from_core_idx': 13, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 7, 'from_core_raw': 'o', 'to_core_raw': 'n'}, 
    {'from_core': 'P0', 'to_core': 'P1', 'from_partition': 0, 'to_partition': 1, 'from_core_idx': 13, 'to_core_idx': 1, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'o', 'to_core_raw': 'r'}, 
    {'from_core': 'P3', 'to_core': 'P0', 'from_partition': 3, 'to_partition': 0, 'from_core_idx': 0, 'to_core_idx': 13, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'q', 'to_core_raw': 'o'}, 
    {'from_core': 'P1', 'to_core': 'P2', 'from_partition': 1, 'to_partition': 2, 'from_core_idx': 1, 'to_core_idx': 1, 'edge_rank': 3, 'qubit_idx': 8, 'from_core_raw': 'r', 'to_core_raw': 'p'}, 
    {'from_core': 'P3', 'to_core': 'P1', 'from_partition': 3, 'to_partition': 1, 'from_core_idx': 1, 'to_core_idx': 1, 'edge_rank': 3, 'qubit_idx': 9, 'from_core_raw': 's', 'to_core_raw': 'r'}, 
    {'from_core': 'P3', 'to_core': 'P1', 'from_partition': 3, 'to_partition': 1, 'from_core_idx': 3, 'to_core_idx': 0, 'edge_rank': 3, 'qubit_idx': 10, 'from_core_raw': 'v', 'to_core_raw': 't'}
    """
    
    if rank == 0:
        print("Configuration:")
        print(f"  QCTN graph: {config.qctn_graph}")
        print(f"  Max steps: {config.max_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Optimizer: {config.optimizer}")
        print()
    
    # Create trainer (this initializes comm, engine, and QCTN)
    trainer = DistributedTrainer(config)
    
    # Print initialization status
    print(f"[Rank {trainer.comm.rank}] Trainer initialized successfully!")
    print(f"[Rank {trainer.comm.rank}] QCTN cores: {len(trainer.qctn.cores)}")
    print(f"[Rank {trainer.comm.rank}] Engine backend: {trainer.engine.backend}")
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print()
        print("=" * 60)
        print("Initialization completed successfully!")
        print("=" * 60)
        print()
        print("Running contract example...")
    
    # ==================== Contract Example ====================
    # Generate sample data and perform one contraction
    
    # Create sample input data
    batch_size = 1024
    num_qubits = trainer.qctn.nqubits
    K = 3  # Hermite polynomial order
    
    # Generate random input x of shape [B, D] where D = num_qubits
    # x = torch.randn(batch_size, num_qubits)
    
    # Generate measurement matrices (Mx) and circuit states
    # Mx_list, phi_x = trainer.engine.generate_data(x, K=K)
    
    # Create circuit states (one per qubit)
    # For simplicity, use identity-like states
    def generate_circuit_states_list(num_qubits, K, device='cuda'):
        circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]
        for i in range(len(circuit_states_list)):
            circuit_states_list[i][-1] = 1.0
        return circuit_states_list
    circuit_states_list = generate_circuit_states_list(num_qubits, K, device='cpu')
    
    # print(f"[Rank {rank}] Input shape: {x.shape}")
    # print(f"[Rank {rank}] Mx_list length: {len(Mx_list)}, each shape: {Mx_list[0].shape}")
    # print(f"[Rank {rank}] circuit_states_list : {[(c.shape) for c in circuit_states_list]}")
    
    # Perform contraction
    # result = trainer.engine.contract_with_compiled_strategy(
    #     trainer.qctn,
    #     circuit_states_list=circuit_states_list,
    #     measure_input_list=Mx_list,
    #     measure_is_matrix=True
    # )
    
    # result = trainer.engine.contract_distributed(
    #     circuit_states_list=circuit_states_list,
    #     measure_input_list=Mx_list,
    #     measure_is_matrix=True
    # )

    # print(f"[Rank {rank}] Contract result shape: {result.shape}")
    # print(f"[Rank {rank}] Contract result (first 5 values): {result.flatten()[:5]}")
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print()
        print("=" * 60)
        print("Contract completed successfully!")
        print("=" * 60)
        print()
        print("Starting distributed training with autograd...")
    
    # ==================== Distributed Training Example ====================
    # Use the new autograd-based distributed training
    
    # Generate training data
    N_batches = 100  # Number of training batches
    train_data_list = []
    
    for i in range(N_batches):
        # Generate new input data for each batch
        # x_train = torch.randn(batch_size, num_qubits)
        x_train = torch.empty(batch_size, num_qubits).normal_(mean=0.0, std=1.0)
        Mx_train, _ = trainer.engine.generate_data(x_train, K=K, ret_type='TNTensor')
        # Mx_train, _ = trainer.engine.generate_data(x_train, K=K)
        train_data_list.append({'measure_input_list': Mx_train})

        if i == 0:
            print(f"[Rank {rank}] Mx_list length {len(Mx_train)} each shape: {Mx_train[0].shape}")
    print(f"[Rank {rank}] circuit_states_list : {[(c.shape) for c in circuit_states_list]}")
    if rank == 0:
        print(f"Generated {N_batches} training batches")
    
    # Option 1: Use trainer.train_distributed() - high-level API
    # if rank == 0:
    #     print()
    #     print("=" * 60)
    #     print("Training with trainer.train_distributed()...")
    #     print("=" * 60)
    
    # stats = trainer.train_distributed(
    #     data_list=train_data_list,
    #     circuit_states_list=circuit_states_list,
    #     num_epochs=10,
    #     log_interval=2
    # )
    
    # if rank == 0:
    #     print(f"Training stats: {stats}")
    
    # Option 2: Manual training loop - low-level API
    if rank == 0:
        print()
        print("=" * 60)
        print("Training with manual loop (low-level API)...")
        print("=" * 60)
    
    # Create optimizer manually
    optimizer = DistributedSGDG(
        lr=0.01 / world_size,

        momentum=0.9,
        stiefel=True
    )
    
    # Manual training loop
    for step in tqdm(range(1000), desc=f"Rank {rank} Training Progress"):
        data = train_data_list[step % len(train_data_list)]
        
        loss = trainer.engine.train_step(
            circuit_states_list=circuit_states_list,
            measure_input_list=data['measure_input_list'],
            optimizer=optimizer,
            measure_is_matrix=True
        )
        
        if rank == 0:
            print(f"  Step {step}: loss = {loss:.6f}")
    
    # Synchronize all processes
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print()
        print("=" * 60)
        print("Distributed training completed successfully!")
        print("=" * 60)
    
    # ==================== Save Distributed Model ====================
    if rank == 0:
        print()
        print("=" * 60)
        print("Saving distributed model...")
        print("=" * 60)
    
    # Save trained weights from all processes to a single file
    # save_path = "assets/qctn_cores_5qubits_dist.safetensors"
    # save_path = "assets/qctn_cores_17qubits_dist_00.safetensors"
    save_path = f"assets/qctn_cores_{num_qubits}qubits{graph_type}_dist_00.safetensors"
    trainer.engine.save_cores_distributed(
        file_path=save_path,
        metadata={
            "model": "distributed_qctn",
            "num_qubits": str(num_qubits),
            "batch_size": str(batch_size),
        }
    )
    
    if rank == 0:
        print(f"Model saved to {save_path}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
