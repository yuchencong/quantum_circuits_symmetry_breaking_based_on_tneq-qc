"""
Contractor module for generating tensor contraction expressions using opt_einsum.

This module separates the contraction logic from backend execution:
- Contractor: Generates einsum expressions and manages contraction strategies
- Backend: Executes the expressions using JAX, PyTorch, or other frameworks
"""

from __future__ import annotations
import itertools
import opt_einsum
from typing import Tuple, List, Optional, Union
import numpy as np


class TensorContractor:
    """
    TensorContractor class for generating optimized tensor contraction expressions.
    
    This class uses opt_einsum to generate contraction expressions but does not
    execute them. The execution is delegated to backend implementations.
    """

    @staticmethod
    def build_core_only_expression(qctn) -> Tuple[str, List]:
        """
        Build einsum expression for contracting cores only (no inputs).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_inputs_expression(qctn, inputs_shape) -> Tuple[str, List]:
        """
        Build einsum expression for contracting with single input tensor.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shape (tuple): Shape of the input tensor.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand},{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [inputs_shape] + [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_vector_inputs_expression(qctn, inputs_shapes: List) -> Tuple[str, List]:
        """
        Build einsum expression for contracting with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs_shapes (list): List of input tensor shapes.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                inputs_equation_lefthand += ','
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1

            einsum_equation_lefthand += ','

        einsum_equation_lefthand = f'{inputs_equation_lefthand}{einsum_equation_lefthand[:-1]}'
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = inputs_shapes + [qctn.cores_weights[core_name].shape for core_name in cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_qctn_expression(qctn, target_qctn) -> Tuple[str, List]:
        """
        Build einsum expression for contracting two QCTNs together.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        target_input_ranks, target_adjacency_matrix, target_output_ranks = target_qctn.circuit
        target_cores_name = target_qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        target_adjacency_matrix_for_interaction = target_adjacency_matrix.copy()
        for element in QCTNHelper.jax_triu_ndindex(len(target_cores_name)):
            i, j = element
            if target_adjacency_matrix_for_interaction[i, j]:
                connection_num = len(target_adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                target_adjacency_matrix[i, j] = connection_symbols
                target_adjacency_matrix[j, i] = connection_symbols

        input_symbols_stack = []
        output_symbols_stack = []

        for idx, _ in enumerate(cores_name):
            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                einsum_equation_lefthand += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += ','

        for idx, _ in enumerate(target_cores_name):
            for _ in target_input_ranks[idx]:
                target_einsum_equation_lefthand += input_symbols_stack.pop(0)

            target_einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(target_adjacency_matrix[idx])))

            for _ in target_output_ranks[idx]:
                target_einsum_equation_lefthand += output_symbols_stack.pop(0)

            target_einsum_equation_lefthand += ','

        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name] + \
                        [target_qctn.cores_weights[core_name].shape for core_name in target_cores_name]

        return einsum_equation, tensor_shapes

    @staticmethod
    def build_with_self_expression(qctn, circuit_array_input_shape=None) -> Tuple[str, List]:
        """
        Build einsum expression for contracting QCTN with itself (hermitian conjugate).
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_array_input_shape (tuple, optional): Shape of circuit array input.
        
        Returns:
            tuple: (einsum_equation, tensor_shapes)
        """
        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores

        symbol_id = 0
        einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from .tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols

        input_symbols_stack = []
        output_symbols_stack = []

        equation_list = []
        new_symbol_mapping = {}

        for idx, _ in enumerate(cores_name):
            core_equation = ""

            for _ in input_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                input_symbols_stack.append(symbol)
                symbol_id += 1

            core_equation += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            for _ in output_ranks[idx]:
                symbol = opt_einsum.get_symbol(symbol_id)
                core_equation += symbol
                output_symbols_stack.append(symbol)
                symbol_id += 1

            einsum_equation_lefthand += core_equation
            equation_list.append(core_equation)

        real_output_symbols_stack = []

        inv_equation_list = []
        for core_equation in equation_list[::-1]:
            new_equation = ""
            for char in core_equation:
                if char in output_symbols_stack:
                    new_equation += char
                else:
                    if char in new_symbol_mapping:
                        symbol = new_symbol_mapping[char]
                    else:
                        symbol = opt_einsum.get_symbol(symbol_id)
                        symbol_id += 1
                        new_symbol_mapping[char] = symbol

                        if char in input_symbols_stack:
                            real_output_symbols_stack.append(symbol)

                    new_equation += symbol

            inv_equation_list.append(new_equation)

        equation_list += inv_equation_list

        einsum_equation_lefthand = ",".join(equation_list)

        if circuit_array_input_shape is not None:
            einsum_equation_output = ''.join(real_output_symbols_stack)
            einsum_equation_lefthand = f"{''.join(input_symbols_stack)},{einsum_equation_lefthand},{einsum_equation_output}"

        einsum_equation = f'{einsum_equation_lefthand}->'

        tensor_shapes = [qctn.cores_weights[core_name].shape for core_name in cores_name] + \
                        [qctn.cores_weights[core_name].shape for core_name in cores_name[::-1]]

        if circuit_array_input_shape is not None:
            tensor_shapes = [circuit_array_input_shape] + tensor_shapes + [circuit_array_input_shape]

        return einsum_equation, tensor_shapes

    @staticmethod
    def create_contract_expression(einsum_equation: str, tensor_shapes: List, optimize='auto'):
        """
        Create optimized contraction expression using opt_einsum.
        
        Args:
            einsum_equation (str): The einsum equation string.
            tensor_shapes (list): List of tensor shapes.
            optimize (str or bool): Optimization strategy for opt_einsum.
        
        Returns:
            opt_einsum.ContractExpression: Optimized contraction expression.
        """
        from ..config import Configuration
        print('einsum_equation', einsum_equation)
        print('tensor_shapes', tensor_shapes)

        return opt_einsum.contract_expression(
            einsum_equation, 
            *tensor_shapes, 
            optimize=optimize if optimize != 'auto' else Configuration.opt_einsum_optimize
        )
