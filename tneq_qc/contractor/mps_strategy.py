"""
MPS chain contraction strategy.

This module provides the MPSChainStrategy class for Matrix Product State chain contraction.
"""

from __future__ import annotations
from typing import Dict, Any, Callable

from .base import ContractionStrategy


class MPSChainStrategy(ContractionStrategy):
    """Optimization strategy for MPS chain structure (Balanced/Full mode)"""
    
    def check_compatibility(self, qctn, shapes_info: Dict[str, Any]) -> bool:
        """
        Check if it is a chain structure.
        
        Currently simplified implementation: returns True directly.
        More strict topology checks can be added in the future.
        """
        # TODO: Implement stricter chain structure check
        # 1. Check if topology is a chain
        # 2. Check connection method of each core
        # 3. Check if input/output dimensions meet expectations
        
        return True
    
    def get_compute_function(self, qctn, shapes_info: Dict[str, Any], backend) -> Callable:
        """
        Return computation function for MPS chain contraction.
        
        This is similar to contract_with_std_graph implementation.
        """
        def compute_fn(cores_dict, circuit_states, measure_matrices):
            """
            MPS chain contraction.
            
            Args:
                cores_dict: {core_name: core_tensor} dictionary
                circuit_states: List of circuit input states
                measure_matrices: List of measurement matrices
            
            Returns:
                Contraction result
            """
            import torch
            
            new_core_dict = {}
            
            # Step 1: Contract cores with circuit_states
            for idx, core_name in enumerate(qctn.cores):
                core_tensor = cores_dict[core_name]
                
                if idx == 0:
                    # First core contracts two states
                    state1 = circuit_states[0]
                    state2 = circuit_states[1]
                    contracted = torch.einsum('i,j,ij...->...', state1, state2, core_tensor)
                else:
                    # Other cores contract one state
                    state = circuit_states[idx + 1]
                    contracted = torch.einsum('i,ji...->j...', state, core_tensor)
                
                new_core_dict[core_name] = contracted
            
            print('new_core_dict', [(core_name, new_core_dict[core_name].shape) for core_name in new_core_dict])

            # Step 2: Chain contraction with measurements
            n = len(new_core_dict)
            
            if n == 1:
                core_tensor = new_core_dict[qctn.cores[0]]
                measure_matrix_1 = measure_matrices[0]
                measure_matrix_2 = measure_matrices[1]
                contracted = torch.einsum('ka,zkl,zab,lb->z', 
                                         core_tensor, measure_matrix_1, 
                                         measure_matrix_2, core_tensor)
                return contracted

            for idx in range(n):
                
                if idx == 0:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix = measure_matrices[idx]
                    contracted = torch.einsum('ka,zkl,lb->zab', 
                                             core_tensor, measure_matrix, core_tensor)
                    
                elif idx < n - 1:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix = measure_matrices[idx]
                    contracted = torch.einsum('zab,akc,zkl,bld->zcd', 
                                             contracted, core_tensor, 
                                             measure_matrix, core_tensor)
                else:
                    core_tensor = new_core_dict[qctn.cores[idx]]
                    measure_matrix_1 = measure_matrices[idx]
                    measure_matrix_2 = measure_matrices[idx + 1]
                    contracted = torch.einsum('zab,akc,zkl,zcd,bld->z', 
                                             contracted, core_tensor, 
                                             measure_matrix_1, measure_matrix_2, core_tensor)
                
                print('contract step', idx, contracted.shape)

            return contracted
        
        return compute_fn
    
    def estimate_cost(self, qctn, shapes_info: Dict[str, Any]) -> float:
        """
        Estimate cost of MPS chain contraction.
        
        Currently simplified implementation: returns a small fixed value.
        Future implementation can estimate precisely based on dimensions of each einsum step.
        """
        # TODO: Implement precise FLOPs estimation
        # Calculate based on dimensions of each einsum step
        
        circuit_states_shapes = shapes_info.get('circuit_states_shapes', [])
        measure_shapes = shapes_info.get('measure_shapes', [])
        
        # Simplified estimation: assume MPS strategy is usually better than einsum
        total_flops = 1e6  # Return a small fixed value
        
        return total_flops
    
    @property
    def name(self) -> str:
        return "mps_chain"
