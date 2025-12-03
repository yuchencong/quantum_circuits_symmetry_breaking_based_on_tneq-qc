"""
Base classes for contraction strategies.

This module provides the abstract base class for all contraction strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any


class ContractionStrategy(ABC):
    """Abstract base class for contraction strategies"""
    
    @abstractmethod
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        Check if the network structure is compatible with this strategy.
        
        Args:
            qctn: QCTN object
            shapes_info: dict, containing circuit_states_shapes, measure_shapes etc.
        
        Returns:
            bool: Whether it is compatible
        """
        pass
    
    @abstractmethod
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        Generate computation function.
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information
            backend: Backend
        
        Returns:
            Callable: compute_fn(cores_dict, circuit_states, measure_matrices)
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        Estimate computation cost (FLOPs).
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information
        
        Returns:
            float: Estimated FLOPs
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name"""
        pass
