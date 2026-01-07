"""
Parallel training strategies module.
"""

from .data_parallel import DataParallelTrainer, TrainingConfig

__all__ = ['DataParallelTrainer', 'TrainingConfig']
