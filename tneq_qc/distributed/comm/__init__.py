"""
Communication module for distributed training.

Provides MPI-based communication primitives for tensor synchronization.
"""

from .mpi_backend import MPIBackend, MockMPIBackend, ReduceOp, DistributedContext, AsyncHandle, get_backend

__all__ = ['MPIBackend', 'MockMPIBackend', 'ReduceOp', 'DistributedContext', 'AsyncHandle', 'get_backend']
