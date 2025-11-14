"""
Unified executor that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use TensorContractor to generate
expressions and then execute them using the specified backend.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import numpy as np

from .contractor import TensorContractor
from ..backends.backend_factory import BackendFactory, ComputeBackend


class ContractExecutor:
    """
    Executor that combines tensor contraction expression generation with backend execution.
    
    This class separates concerns:
    - TensorContractor: Generates einsum expressions using opt_einsum
    - ComputeBackend: Executes expressions using JAX, PyTorch, etc.
    """

    def __init__(self, backend: Optional[Union[str, ComputeBackend]] = None):
        """
        Initialize the executor with a specific backend.
        
        Args:
            backend (str or ComputeBackend, optional): Backend to use. 
                Can be 'jax', 'pytorch', or a ComputeBackend instance.
                If None, uses the default backend.
        """
        if backend is None:
            self.backend = BackendFactory.get_default_backend()
        elif isinstance(backend, str):
            self.backend = BackendFactory.create_backend(backend)
        else:
            self.backend = backend

        self.contractor = TensorContractor()

    def contract_core_only(self, qctn):
        """
        Contract QCTN cores only (no inputs).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_core_only_expression(qctn)
        
        # Create optimized expression
        if not hasattr(qctn, '_contract_expr_core_only'):
            qctn._contract_expr_core_only = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(qctn._contract_expr_core_only)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_inputs(self, qctn, inputs):
        """
        Contract QCTN with input tensor.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (array): Input tensor.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        inputs = self.backend.convert_to_tensor(inputs)
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_inputs_expression(qctn, inputs.shape)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_inputs_{inputs.shape}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = [inputs] + [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_vector_inputs(self, qctn, inputs: List):
        """
        Contract QCTN with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (list): List of input tensors.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        inputs = [self.backend.convert_to_tensor(inp) for inp in inputs]
        inputs_shapes = [inp.shape for inp in inputs]
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_vector_inputs_expression(qctn, inputs_shapes)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_vector_inputs_{tuple(inputs_shapes)}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = inputs + [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_qctn(self, qctn, target_qctn):
        """
        Contract QCTN with another QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target QCTN.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_qctn_expression(qctn, target_qctn)
        
        # Create optimized expression if not cached
        if not hasattr(qctn, '_contract_expr_with_qctn'):
            qctn._contract_expr_with_qctn = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores] + \
                  [self.backend.convert_to_tensor(target_qctn.cores_weights[c]) for c in target_qctn.cores]
        
        # Execute
        jit_fn = self.backend.jit_compile(qctn._contract_expr_with_qctn)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_self(self, qctn, circuit_array_input=None):
        """
        Contract QCTN with itself (hermitian conjugate).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_array_input (array, optional): Optional input array.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        input_shape = None
        if circuit_array_input is not None:
            circuit_array_input = self.backend.convert_to_tensor(circuit_array_input)
            input_shape = circuit_array_input.shape
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(qctn, input_shape)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{input_shape}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = []
        if circuit_array_input is not None:
            tensors.append(circuit_array_input)
        
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        if circuit_array_input is not None:
            tensors.append(circuit_array_input)
        
        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_self_for_gradient(self, qctn, circuit_array_input=None) -> Tuple:
        """
        Contract QCTN with itself and compute gradients.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_array_input (array, optional): Optional input array.
        
        Returns:
            tuple: (loss, gradients)
        """
        input_shape = None
        if circuit_array_input is not None:
            circuit_array_input = self.backend.convert_to_tensor(circuit_array_input)
            input_shape = circuit_array_input.shape
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(qctn, input_shape)
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{input_shape}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Define loss function
        def mse_loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # Compute MSE loss
            diff = result - 1.0
            return (diff * diff).mean()
        
        # Determine which arguments to compute gradients for
        num_cores = len(qctn.cores)
        if circuit_array_input is not None:
            # Skip first and last tensor (inputs), only compute gradients for cores
            argnums = list(range(1, 1 + num_cores))
        else:
            # All tensors are cores
            argnums = list(range(num_cores))
        
        # Create value_and_grad function
        cache_key_grad = f'_grad_fn_self_{input_shape}'
        if not hasattr(qctn, cache_key_grad):
            value_and_grad_fn = self.backend.compute_value_and_grad(mse_loss_fn, argnums=argnums)
            value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            setattr(qctn, cache_key_grad, value_and_grad_fn)
        else:
            value_and_grad_fn = getattr(qctn, cache_key_grad)
        
        # Prepare tensors
        tensors = []
        if circuit_array_input is not None:
            tensors.append(circuit_array_input)
        
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        if circuit_array_input is not None:
            tensors.append(circuit_array_input)
        
        # Compute value and gradients
        loss, grads = value_and_grad_fn(*tensors)
        
        return loss, grads

    def contract_with_qctn_for_gradient(self, qctn, target_qctn) -> Tuple:
        """
        Contract QCTN with target QCTN and compute gradients.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            target_qctn (QCTN): The target QCTN.
        
        Returns:
            tuple: (loss, gradients)
        """
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_qctn_expression(qctn, target_qctn)
        
        # Create optimized expression if not cached
        if not hasattr(qctn, '_contract_expr_with_qctn'):
            qctn._contract_expr_with_qctn = self.contractor.create_contract_expression(
                einsum_eq, tensor_shapes
            )
        
        expr = qctn._contract_expr_with_qctn
        
        # Define loss function
        def mse_loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # Compute MSE loss
            diff = result - 1.0
            return (diff * diff).mean()
        
        # Only compute gradients for qctn cores, not target_qctn
        argnums = list(range(len(qctn.cores)))
        
        # Create value_and_grad function
        if not hasattr(qctn, '_grad_fn_with_qctn'):
            value_and_grad_fn = self.backend.compute_value_and_grad(mse_loss_fn, argnums=argnums)
            value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            qctn._grad_fn_with_qctn = value_and_grad_fn
        else:
            value_and_grad_fn = qctn._grad_fn_with_qctn
        
        # Prepare tensors
        tensors = [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores] + \
                  [self.backend.convert_to_tensor(target_qctn.cores_weights[c]) for c in target_qctn.cores]
        
        # Compute value and gradients
        loss, grads = value_and_grad_fn(*tensors)
        
        return loss, grads
