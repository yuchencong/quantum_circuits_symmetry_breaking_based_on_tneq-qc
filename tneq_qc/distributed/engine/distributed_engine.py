"""
Distributed Engine Siamese

Extends EngineSiamese with distributed computing capabilities:
- Distributed tensor contraction
- Distributed gradient computation
- Tensor parallel support (reserved for future)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

from ..comm.mpi_backend import MPIBackend, MockMPIBackend, ReduceOp, get_backend

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN
    from ...backends.backend_interface import ComputeBackend


class DistributedEngineSiamese:
    """
    Distributed EngineSiamese.
    
    Wraps the standard EngineSiamese with distributed computing support,
    enabling data parallel and (future) tensor parallel training.
    
    Key features:
    1. Distributed tensor contraction with gradient sync
    2. Data partitioning helpers
    3. Tensor parallel support (reserved)
    
    Example:
        >>> engine = DistributedEngineSiamese(backend='pytorch')
        >>> loss, grads = engine.contract_with_gradient_distributed(
        ...     qctn, circuit_states_list, measure_input_list
        ... )
    """
    
    def __init__(self, 
                 backend: Optional[Union[str, 'ComputeBackend']] = None, 
                 strategy_mode: str = 'balanced', 
                 mx_K: int = 100,
                 mpi_backend: Optional[Union[MPIBackend, MockMPIBackend]] = None,
                 enable_tensor_parallel: bool = False):
        """
        Initialize distributed engine.
        
        Args:
            backend: Compute backend ('pytorch', 'jax', or ComputeBackend instance)
            strategy_mode: Contraction strategy mode ('fast', 'balanced', 'full')
            mx_K: Maximum Hermite polynomial order
            mpi_backend: MPI backend (auto-created if None)
            enable_tensor_parallel: Enable tensor parallel (reserved for future)
        """
        # Import and create base engine
        from ...core.engine_siamese import EngineSiamese
        
        self._base_engine = EngineSiamese(
            backend=backend,
            strategy_mode=strategy_mode,
            mx_K=mx_K
        )
        
        # Initialize MPI
        self.mpi = mpi_backend or get_backend(use_mpi=True)
        self.ctx = self.mpi.get_context()
        
        # Tensor parallel configuration (reserved)
        self.enable_tensor_parallel = enable_tensor_parallel
        self.tensor_parallel_config: Optional[Dict[str, Any]] = None
        
        self._log(f"DistributedEngineSiamese initialized: {self.ctx}")
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.mpi.is_main_process():
            print(f"[DistributedEngine] {msg}")
    
    # ==================== Proxy to Base Engine ====================
    
    @property
    def backend(self):
        """Access the underlying compute backend."""
        return self._base_engine.backend
    
    @property
    def contractor(self):
        """Access the einsum strategy contractor."""
        return self._base_engine.contractor
    
    @property
    def strategy_compiler(self):
        """Access the strategy compiler."""
        return self._base_engine.strategy_compiler
    
    def generate_data(self, x, K=None):
        """
        Generate measurement matrices from input data.
        
        Proxies to base engine's generate_data method.
        
        Args:
            x: Input tensor of shape (B, D)
            K: Hermite polynomial order (uses engine default if None)
            
        Returns:
            (Mx_list, extra_info)
        """
        return self._base_engine.generate_data(x, K=K)
    
    def contract_with_compiled_strategy(self, qctn, circuit_states_list, measure_input_list=None, **kwargs):
        """
        Standard tensor contraction.
        
        Proxies to base engine.
        """
        return self._base_engine.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
            **kwargs
        )
    
    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list=None, 
                                                      measure_input_list=None, **kwargs):
        """
        Tensor contraction with gradient computation.
        
        Proxies to base engine.
        """
        return self._base_engine.contract_with_compiled_strategy_for_gradient(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
            **kwargs
        )
    
    # ==================== Distributed Operations ====================
    
    def contract_distributed(self,
                             qctn: 'QCTN',
                             circuit_states_list: List,
                             measure_input_list: List,
                             sync_result: bool = True) -> 'torch.Tensor':
        """
        Distributed tensor contraction.
        
        Each worker computes on its local batch. Results are optionally
        gathered via AllGather.
        
        Args:
            qctn: QCTN model
            circuit_states_list: Circuit states list
            measure_input_list: Measurement matrices (local batch)
            sync_result: Whether to AllGather results
            
        Returns:
            Contraction result
        """
        import torch
        
        # Local computation
        local_result = self.contract_with_compiled_strategy(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list
        )
        
        if not sync_result:
            return local_result
        
        # AllGather to collect all results
        if hasattr(local_result, 'tensor'):
            # TNTensor wrapper
            all_tensors = self.mpi.allgather(local_result.tensor)
        else:
            all_tensors = self.mpi.allgather(local_result)
        
        return torch.cat(all_tensors, dim=0)
    
    def contract_with_gradient_distributed(self,
                                           qctn: 'QCTN',
                                           circuit_states_list: List,
                                           measure_input_list: List,
                                           sync_gradients: bool = True) -> Tuple[float, List['torch.Tensor']]:
        """
        Distributed gradient computation.
        
        Each worker computes local gradients on its data batch.
        Gradients are synchronized via AllReduce (average).
        
        Args:
            qctn: QCTN model
            circuit_states_list: Circuit states list
            measure_input_list: Measurement matrices (local batch)
            sync_gradients: Whether to AllReduce gradients
            
        Returns:
            (global_loss, global_gradients)
        """
        import torch
        
        # Local computation
        local_loss, local_grads = self.contract_with_compiled_strategy_for_gradient(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list
        )
        
        if not sync_gradients:
            return float(local_loss), local_grads
        
        # Sync loss
        loss_value = float(local_loss)
        global_loss = self.mpi.allreduce_scalar(loss_value, op=ReduceOp.AVG)
        
        # Sync gradients
        global_grads = self.mpi.allreduce_tensors(local_grads, op=ReduceOp.AVG)
        
        return global_loss, global_grads
    
    # ==================== Tensor Parallel (Reserved) ====================
    
    def setup_tensor_parallel(self, config: Dict[str, Any]):
        """
        Configure tensor parallel (reserved for future).
        
        Args:
            config: Tensor parallel configuration
                - partition_dim: Partition dimension ('batch' or 'bond')
                - min_size_for_partition: Minimum tensor size to trigger partition
        """
        self.tensor_parallel_config = config
        self.enable_tensor_parallel = True
        self._log(f"Tensor parallel configured: {config}")
    
    def contract_tensor_parallel(self,
                                  qctn: 'QCTN',
                                  circuit_states_list: List,
                                  measure_input_list: List) -> 'torch.Tensor':
        """
        Tensor parallel contraction (reserved for future).
        
        For large intermediate tensors during contraction, this would
        partition and distribute computation across workers.
        
        Args:
            qctn: QCTN model
            circuit_states_list: Circuit states list
            measure_input_list: Measurement matrices
            
        Returns:
            Contraction result
            
        Raises:
            NotImplementedError: Tensor parallel not yet implemented
        """
        if not self.enable_tensor_parallel:
            raise RuntimeError("Tensor parallel not enabled. Call setup_tensor_parallel first.")
        
        # TODO: Implement tensor parallel contraction
        # 1. Monitor intermediate tensor sizes during contraction
        # 2. When tensor exceeds threshold, partition along specified dimension
        # 3. Each worker computes partial result
        # 4. AllReduce to combine results
        
        raise NotImplementedError("Tensor parallel contraction not yet implemented")
    
    # ==================== Batch Partitioning ====================
    
    def partition_measure_matrices(self, measure_input_list: List['torch.Tensor']) -> List['torch.Tensor']:
        """
        Partition measurement matrices across workers.
        
        Splits the batch dimension evenly across workers.
        
        Args:
            measure_input_list: List of Mx tensors, each (B, K, K)
            
        Returns:
            List of partitioned Mx tensors
        """
        if not measure_input_list:
            return measure_input_list
        
        B = measure_input_list[0].shape[0]
        per_worker = B // self.ctx.world_size
        remainder = B % self.ctx.world_size
        
        # Calculate this worker's slice
        if self.ctx.rank < remainder:
            start = self.ctx.rank * (per_worker + 1)
            end = start + per_worker + 1
        else:
            start = remainder * (per_worker + 1) + (self.ctx.rank - remainder) * per_worker
            end = start + per_worker
        
        return [mx[start:end] for mx in measure_input_list]
    
    def gather_results(self, local_results: 'torch.Tensor') -> 'torch.Tensor':
        """
        Gather results from all workers.
        
        Args:
            local_results: This worker's results
            
        Returns:
            Concatenated results from all workers
        """
        import torch
        
        all_results = self.mpi.allgather(local_results)
        return torch.cat(all_results, dim=0)
