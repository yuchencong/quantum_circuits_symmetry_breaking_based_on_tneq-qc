"""
Distributed Trainer

Main entry point for distributed TNEQ training. Combines all distributed
components into a simple, high-level API.
"""

from __future__ import annotations
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

from ..comm.mpi_backend import MPIBackend, MockMPIBackend, get_backend
from ..parallel.data_parallel import DataParallelTrainer, TrainingConfig, TrainingStats
from ..engine.distributed_engine import DistributedEngineSiamese

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN


class DistributedTrainer:
    """
    High-level Distributed Trainer.
    
    Provides a simple API for distributed TNEQ training, handling:
    - MPI initialization
    - QCTN model creation and synchronization
    - Data generation and partitioning
    - Distributed training loop
    - Checkpoint management
    
    Example:
        >>> config = {
        ...     'backend_type': 'pytorch',
        ...     'qctn_graph': '-3-A-3-B-3-',
        ...     'max_steps': 1000,
        ... }
        >>> trainer = DistributedTrainer(config)
        >>> data_list, circuit_states = trainer.prepare_data(N=100, B=128, K=3)
        >>> stats = trainer.train(data_list, circuit_states)
    
    Usage with mpiexec:
        $ mpiexec -n 4 python -m tneq_qc.distributed.trainer.distributed_trainer --config config.yaml
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize distributed trainer.
        
        Args:
            config: Configuration dictionary containing:
                - backend_type: 'pytorch' or 'jax'
                - device: 'cpu' or 'cuda'
                - qctn_graph: QCTN graph string
                - strategy_mode: 'fast', 'balanced', or 'full'
                - mx_K: Maximum Hermite polynomial order
                - max_steps, learning_rate, etc.: Training parameters
        """
        self.config = config
        
        # Initialize MPI
        self.mpi = get_backend(use_mpi=True)
        self.ctx = self.mpi.get_context()
        
        # Initialize distributed engine
        self.engine = DistributedEngineSiamese(
            backend=config.get('backend_type', 'pytorch'),
            strategy_mode=config.get('strategy_mode', 'balanced'),
            mx_K=config.get('mx_K', 100),
            mpi_backend=self.mpi
        )
        
        # Initialize QCTN
        self._init_qctn()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        if self.mpi.is_main_process():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"DistributedTrainer initialized: {self.ctx}")
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.mpi.is_main_process():
            print(f"[DistributedTrainer] {msg}")
    
    def _init_qctn(self):
        """Initialize QCTN model."""
        from ...core.qctn import QCTN, QCTNHelper
        
        qctn_graph = self.config.get('qctn_graph')
        
        if qctn_graph is None:
            # Use default example graph
            qctn_graph = QCTNHelper.generate_example_graph()
            self._log(f"Using default QCTN graph")
        
        self.qctn = QCTN(qctn_graph, backend=self.engine.backend)
        self._log(f"QCTN initialized: {self.qctn.nqubits} qubits, {len(self.qctn.cores)} cores")
        
        # Synchronize initial weights across all workers
        self._sync_model_weights()
    
    def _sync_model_weights(self):
        """Synchronize model weights from main process to all workers."""
        from tneq_qc.core.tn_tensor import TNTensor
        
        if self.ctx.world_size == 1:
            return
        
        backend = self.engine.backend
        
        for core_name in self.qctn.cores:
            weight = self.qctn.cores_weights[core_name]
            
            # Handle TNTensor objects
            if isinstance(weight, TNTensor):
                # Broadcast the underlying tensor
                synced_tensor = self.mpi.broadcast(weight.tensor, src=0)
                # Broadcast the scale factor
                if hasattr(weight.scale, 'detach'):
                    synced_scale = self.mpi.broadcast(weight.scale, src=0)
                else:
                    # Scale is a scalar, broadcast as tensor then extract
                    scale_tensor = backend.convert_to_tensor([weight.scale])
                    synced_scale = self.mpi.broadcast(scale_tensor, src=0)
                    synced_scale = float(backend.tensor_to_numpy(synced_scale)[0])
                # Reconstruct TNTensor
                self.qctn.cores_weights[core_name] = TNTensor(synced_tensor, synced_scale)
            else:
                # Regular tensor
                synced_weight = self.mpi.broadcast(weight, src=0)
                self.qctn.cores_weights[core_name] = synced_weight
        
        self._log("Model weights synchronized across workers")
    
    # ==================== Data Preparation ====================
    
    def prepare_data(self, N: int, B: int, K: int) -> Tuple[List[Dict], List]:
        """
        Prepare training data.
        
        Generates N batches of measurement matrices using Hermite polynomials.
        
        Args:
            N: Number of data batches
            B: Batch size (samples per batch)
            K: Hermite polynomial order
            
        Returns:
            (data_list, circuit_states_list)
        """
        import numpy as np
        
        backend = self.engine.backend
        D = self.qctn.nqubits
        
        data_list = []
        
        # Only main process generates data, then broadcast
        if self.mpi.is_main_process():
            self._log(f"Generating data: N={N}, B={B}, D={D}, K={K}")
            
            for i in range(N):
                # Generate random data using numpy, then convert to backend tensor
                x_np = np.random.randn(B, D).astype(np.float32)
                x = backend.convert_to_tensor(x_np)
                Mx_list, _ = self.engine.generate_data(x, K=K)
                data_list.append({"measure_input_list": Mx_list})
        
        # Broadcast data list size
        n_batches = len(data_list) if self.mpi.is_main_process() else 0
        n_batches = self.mpi.broadcast_object(n_batches, src=0)
        
        # Broadcast each batch
        if not self.mpi.is_main_process():
            data_list = [None] * n_batches
        
        for i in range(n_batches):
            data_list[i] = self.mpi.broadcast_object(data_list[i], src=0)
        
        # Generate circuit states
        circuit_states_list = [backend.zeros(K) for _ in range(D)]
        for s in circuit_states_list:
            s[-1] = 1.0
        
        self._log(f"Data prepared: {len(data_list)} batches")
        
        return data_list, circuit_states_list
    
    # ==================== Training ====================
    
    def train(self, 
              data_list: List[Dict], 
              circuit_states_list: List['torch.Tensor'],
              training_config: Optional[TrainingConfig] = None) -> TrainingStats:
        """
        Execute distributed training.
        
        Args:
            data_list: Training data list
            circuit_states_list: Circuit states
            training_config: Training configuration (uses config dict if None)
            
        Returns:
            Training statistics
        """
        # Build training config from dict if not provided
        if training_config is None:
            training_config = TrainingConfig(
                max_steps=self.config.get('max_steps', 1000),
                log_interval=self.config.get('log_interval', 10),
                checkpoint_interval=self.config.get('checkpoint_interval', 100),
                learning_rate=self.config.get('learning_rate', 1e-2),
                lr_schedule=self.config.get('lr_schedule'),
                optimizer_method=self.config.get('optimizer', 'sgdg'),
                momentum=self.config.get('momentum', 0.9),
                stiefel=self.config.get('stiefel', True),
                tol=self.config.get('tol'),
            )
        
        # Create data parallel trainer
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,  # Use base engine
            qctn=self.qctn,
            config=training_config,
            mpi_backend=self.mpi
        )
        
        # Execute training
        self._log("Starting distributed training...")
        stats = trainer.train(data_list, circuit_states_list)
        
        # Save final model
        self._save_final_model()
        
        return stats
    
    # ==================== Checkpointing ====================
    
    def _save_final_model(self):
        """Save final model (main process only)."""
        if not self.mpi.is_main_process():
            return
        
        try:
            model_path = self.checkpoint_dir / "final_model.safetensors"
            self.qctn.save_cores(str(model_path), metadata={
                'config': json.dumps(self.config)
            })
            self._log(f"Final model saved: {model_path}")
        except Exception as e:
            self._log(f"Warning: Could not save model: {e}", level="warn")
    
    def save_checkpoint(self, step: int, stats: Optional[TrainingStats] = None):
        """
        Save training checkpoint.
        
        Args:
            step: Current training step
            stats: Optional training stats
        """
        if not self.mpi.is_main_process():
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.safetensors"
        
        metadata = {
            'step': str(step),
            'config': json.dumps(self.config),
        }
        
        if stats:
            metadata['final_loss'] = str(stats.final_loss)
        
        try:
            self.qctn.save_cores(str(checkpoint_path), metadata=metadata)
            self._log(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self._log(f"Warning: Could not save checkpoint: {e}", level="warn")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        from ...core.qctn import QCTN
        
        self.qctn = QCTN.from_pretrained(
            self.qctn.graph, 
            checkpoint_path, 
            backend=self.engine.backend
        )
        
        # Sync weights after loading
        self._sync_model_weights()
        
        self._log(f"Loaded checkpoint: {checkpoint_path}")
    
    # ==================== Evaluation ====================
    
    def evaluate(self, data_list: List[Dict], 
                 circuit_states_list: List['torch.Tensor']) -> float:
        """
        Evaluate model on given data.
        
        Args:
            data_list: Evaluation data
            circuit_states_list: Circuit states
            
        Returns:
            Average loss
        """
        # Create temporary trainer for evaluation
        config = TrainingConfig(max_steps=0)
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,
            qctn=self.qctn,
            config=config,
            mpi_backend=self.mpi
        )
        
        return trainer.evaluate(data_list, circuit_states_list)


def main():
    """Command-line entry point for distributed training."""
    import argparse
    
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
    
    parser = argparse.ArgumentParser(description='TNEQ Distributed Training')
    parser.add_argument('--config', type=str, help='Config file path (YAML or JSON)')
    parser.add_argument('--backend', type=str, default='pytorch', help='Backend type')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-batches', type=int, default=100, help='Number of data batches')
    parser.add_argument('--hermite-order', type=int, default=3, help='Hermite polynomial order')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml'] and has_yaml:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
    else:
        config = {
            'backend_type': args.backend,
            'max_steps': args.max_steps,
            'learning_rate': args.learning_rate,
        }
    
    # Create trainer
    trainer = DistributedTrainer(config)
    
    # Prepare data
    data_list, circuit_states_list = trainer.prepare_data(
        N=args.data_batches,
        B=args.batch_size,
        K=args.hermite_order
    )
    
    # Train
    stats = trainer.train(data_list, circuit_states_list)
    
    if trainer.mpi.is_main_process():
        print(f"\nTraining completed: {stats.to_dict()}")


if __name__ == "__main__":
    main()
