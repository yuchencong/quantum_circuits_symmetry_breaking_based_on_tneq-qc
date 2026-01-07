"""
TNEQ Distributed Computing Module

Provides distributed training capabilities for TNEQ-QC tensor network models.

Key Components:
- comm: MPI communication backend
- parallel: Data parallel training strategy
- engine: Distributed engine
- trainer: High-level distributed trainer

Existing (Genetic Algorithm):
- mpi_overlord: Overlord for genetic algorithm distributed search
- mpi_agent: Agent workers for genetic algorithm
- mpi_core: Core MPI utilities

Example:
    >>> from tneq_qc.distributed import DistributedTrainer
    >>> trainer = DistributedTrainer({'backend_type': 'pytorch'})
    >>> stats = trainer.train(data_list, circuit_states_list)
"""

# New distributed training modules
from .comm import MPIBackend, ReduceOp, DistributedContext, MockMPIBackend
from .parallel import DataParallelTrainer, TrainingConfig
from .engine import DistributedEngineSiamese
from .trainer import DistributedTrainer

# Legacy genetic algorithm modules (existing)
from .mpi_core import TAGS, SURVIVAL, REASONS, AGENT_STATUS, INDIVIDUAL_STATUS

__all__ = [
    # Communication
    'MPIBackend',
    'MockMPIBackend',
    'ReduceOp',
    'DistributedContext',
    
    # Parallel strategies
    'DataParallelTrainer',
    'TrainingConfig',
    
    # Engine
    'DistributedEngineSiamese',
    
    # High-level trainer
    'DistributedTrainer',
    
    # Legacy (genetic algorithm)
    'TAGS',
    'SURVIVAL',
    'REASONS',
    'AGENT_STATUS',
    'INDIVIDUAL_STATUS',
]
