"""
Unified executor that combines contractor (expression generation) with backend (execution).

This module provides high-level functions that use TensorContractor to generate
expressions and then execute them using the specified backend.
"""

from __future__ import annotations
from typing import Optional, Union, List, Tuple
import numpy as np
import torch

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
            self.backend = BackendFactory.create_backend(backend, device="cuda")
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

    def contract_with_self(self, qctn, circuit_states=None, measure_input=None, measure_is_matrix=False):
        """
        Contract QCTN with itself (hermitian conjugate).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states. 
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_input (array or list, optional): Measurement input.
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            Backend tensor: Result of the contraction.
        """
        # Convert circuit_states
        states_shape = None
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
        
        # Convert measure_input
        measure_shape = None
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(
            qctn, states_shape, measure_shape, measure_is_matrix
        )
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Prepare tensors
        tensors = []
        
        # Add circuit_states
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # Add cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]

        # Add measure_input
        if measure_input is not None:
            if isinstance(measure_input, list):
                tensors.extend(measure_input)
            else:
                tensors.append(measure_input)
        
        # Add inverse cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        # Add circuit_states again (conjugate side)
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # TODO: use general interface, this will cause error with jax
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda()

        # Execute
        jit_fn = self.backend.jit_compile(expr)
        return self.backend.execute_expression(jit_fn, *tensors)

    def contract_with_self_for_gradient(self, qctn, circuit_states=None, measure_input=None, measure_is_matrix=False) -> Tuple:
        """
        Contract QCTN with itself and compute gradients.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_states (array or list, optional): Circuit input states. 
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_input (array or list, optional): Measurement input.
                Can be numpy array, torch tensor, jax ndarray, or a list of such tensors.
            measure_is_matrix (bool): If True, measure_input is the outer product matrix Mx;
                If False, measure_input is the vector phi_x.
        
        Returns:
            tuple: (loss, gradients)
        """
        # Convert circuit_states
        states_shape = None
        num_states_tensors = 0
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                circuit_states = [self.backend.convert_to_tensor(s) for s in circuit_states]
                states_shape = tuple([s.shape for s in circuit_states])
                num_states_tensors = len(circuit_states)
            else:
                circuit_states = self.backend.convert_to_tensor(circuit_states)
                states_shape = circuit_states.shape
                num_states_tensors = 1
        
        # Convert measure_input
        measure_shape = None
        num_measure_tensors = 0
        if measure_input is not None:
            if isinstance(measure_input, list):
                measure_input = [self.backend.convert_to_tensor(m) for m in measure_input]
                measure_shape = tuple([m.shape for m in measure_input])
                num_measure_tensors = len(measure_input)
            else:
                measure_input = self.backend.convert_to_tensor(measure_input)
                measure_shape = measure_input.shape
                num_measure_tensors = 1
        
        # Generate expression
        einsum_eq, tensor_shapes = self.contractor.build_with_self_expression(
            qctn, states_shape, measure_shape, measure_is_matrix
        )
        
        # Create optimized expression if not cached
        cache_key = f'_contract_expr_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key):
            expr = self.contractor.create_contract_expression(einsum_eq, tensor_shapes)
            setattr(qctn, cache_key, expr)
        else:
            expr = getattr(qctn, cache_key)
        
        # Define loss function
        def loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # Compute MSE loss
            # diff = result - 1.0
            # return (diff * diff).mean()

            # compute cross entropy loss
            target = torch.ones_like(result)
            log_result = torch.log(result + 1e-10)
            return -torch.mean(target * log_result)

        # Determine which arguments to compute gradients for
        # Skip circuit_states and measure_input, only compute gradients for cores
        num_cores = len(qctn.cores)
        total_args = num_states_tensors * 2 + num_cores * 2 + num_measure_tensors
        # argnums = list(range(0, total_args))

        cores_index = list(range(num_states_tensors, num_states_tensors + num_cores))
        inv_cores_index = list(range(num_states_tensors + num_cores + num_measure_tensors,
                                   num_states_tensors + num_cores + num_measure_tensors + num_cores))
        argnums = cores_index + inv_cores_index
        
        # Create value_and_grad function
        cache_key_grad = f'_grad_fn_self_{states_shape}_{measure_shape}_{measure_is_matrix}'
        if not hasattr(qctn, cache_key_grad):
            value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
            value_and_grad_fn = self.backend.jit_compile(value_and_grad_fn)
            setattr(qctn, cache_key_grad, value_and_grad_fn)
        else:
            value_and_grad_fn = getattr(qctn, cache_key_grad)
        
        # Prepare tensors
        tensors = []
        
        # Add circuit_states
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # Add cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores]

        # Add measure_input
        if measure_input is not None:
            if isinstance(measure_input, list):
                tensors.extend(measure_input)
            else:
                tensors.append(measure_input)

        # Add inverse cores
        tensors += [self.backend.convert_to_tensor(qctn.cores_weights[c]) for c in qctn.cores[::-1]]
        
        # Add circuit_states again (conjugate side)
        if circuit_states is not None:
            if isinstance(circuit_states, list):
                tensors.extend(circuit_states)
            else:
                tensors.append(circuit_states)
        
        # TODO: use general interface, this will cause error with jax
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda()

        # print('input tensors', len(tensors))

        # Compute value and gradients
        loss, grads = value_and_grad_fn(*tensors)


        grads = [grads[i] + grads[-j] for i, j in zip(range(0, len(cores_index)), range(1, len(cores_index)+1))]
        
        # print(f'grads for cores: {len(grads)}')

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
        def loss_fn(*tensors):
            jit_fn = self.backend.jit_compile(expr)
            result = self.backend.execute_expression(jit_fn, *tensors)
            # # Compute MSE loss
            # diff = result - 1.0
            # return (diff * diff).mean()

            # compute cross entropy loss
            target = torch.ones_like(result)
            log_result = torch.log(result + 1e-10)
            return -torch.mean(target * log_result)
        
        # Only compute gradients for qctn cores, not target_qctn
        argnums = list(range(len(qctn.cores)))
        
        # Create value_and_grad function
        if not hasattr(qctn, '_grad_fn_with_qctn'):
            value_and_grad_fn = self.backend.compute_value_and_grad(loss_fn, argnums=argnums)
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
