from __future__ import annotations
from ..config import Configuration
import itertools
import opt_einsum
import torch
import torch.nn.functional as F

local_debug = True

def local_print(*args, **kwargs):
    """Print function that only prints when local_debug is True."""
    if local_debug:
        print(*args, **kwargs)

class ContractorPyTorch:
    """
    ContractorPyTorch class for optimized tensor contraction using PyTorch.
    
    This class provides methods to contract tensors using PyTorch's einsum,
    with automatic differentiation support for gradient-based optimization.
    """

    @staticmethod
    def contract_core_only(qctn):
        """
        Contract the given QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            torch.Tensor: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..core.tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
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

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]  # Remove the last comma
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN: {qctn.circuit}')
        local_print(f'Einsum Equation: {einsum_equation}')

        # Convert weights to PyTorch tensors if needed
        torch_weights = []
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_weights.append(weight)

        retracted_QCTN = torch.einsum(einsum_equation, *torch_weights)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_inputs(qctn, inputs):
        """
        Contract the given QCTN with specified inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (torch.Tensor): The inputs for the contraction operation.
        
        Returns:
            torch.Tensor: The result of the tensor contraction.
        """

        local_print(f'ContractorPyTorch.contract_with_inputs called with inputs shape: {inputs.shape}')

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..core.tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry
                
        
        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            local_print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
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

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN: {qctn.circuit}')
        local_print(f'Einsum Equation: {einsum_equation}')

        # Convert to PyTorch tensors if needed
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()
        
        torch_weights = [inputs]
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_weights.append(weight)

        retracted_QCTN = torch.einsum(einsum_equation, *torch_weights)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_vector_inputs(qctn, inputs):
        """
        Contract the given QCTN with vector inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (list[torch.Tensor]): The vector inputs for the contraction operation.

        Returns:
            torch.Tensor: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()

        from ..core.tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        inputs_equation_lefthand = ''
        for idx, _ in enumerate(cores_name):
            local_print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
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

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        local_print(f'QCTN: {qctn.circuit}')
        local_print(f'Einsum Equation: {einsum_equation}')

        # Convert to PyTorch tensors
        torch_inputs = []
        for v in inputs:
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(v).float()
            torch_inputs.append(v)
        
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_inputs.append(weight)

        retracted_QCTN = torch.einsum(einsum_equation, *torch_inputs)

        return retracted_QCTN
    
    @staticmethod
    def contract_with_QCTN(qctn, target_qctn):
        """
        Contract the given QCTN with a target QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            target_qctn (QCTN): The target quantum circuit tensor network for contraction.
        
        Returns:
            torch.Tensor: The result of the tensor contraction.
        """

        local_print("ContractorPyTorch.contract_with_QCTN called")

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        target_input_ranks, target_adjacency_matrix, target_output_ranks = target_qctn.circuit
        target_cores_name = target_qctn.cores
        target_cores_weights = target_qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''
        target_einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()
        local_print(f'adjacency_matrix for interaction: \n{adjacency_matrix_for_interaction}')
        
        from ..core.tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        local_print(f"adjacency_matrix after symbol assignment: \n{adjacency_matrix}")

        target_adjacency_matrix_for_interaction = target_adjacency_matrix.copy()
        for element in QCTNHelper.jax_triu_ndindex(len(target_cores_name)):
            i, j = element
            if target_adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(target_adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                target_adjacency_matrix[i, j] = connection_symbols
                target_adjacency_matrix[j, i] = connection_symbols

        local_print(f"target_adjacency_matrix after symbol assignment: \n{target_adjacency_matrix}")

        input_symbols_stack = []
        output_symbols_stack = []

        # self.einsum_equation_lefthand
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
        
        local_print(f'input_symbols_stack after source QCTN processing: {input_symbols_stack}')
        local_print(f'output_symbols_stack after source QCTN processing: {output_symbols_stack}')

        # target.einsum_equation_lefthand
        for idx, _ in enumerate(target_cores_name):
            for _ in target_input_ranks[idx]:
                target_einsum_equation_lefthand += input_symbols_stack.pop(0)

            target_einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(target_adjacency_matrix[idx])))

            for _ in target_output_ranks[idx]:
                target_einsum_equation_lefthand += output_symbols_stack.pop(0)
            
            target_einsum_equation_lefthand += ','

        einsum_equation = f'{einsum_equation_lefthand}{target_einsum_equation_lefthand[:-1]}->'

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')

        for core_name in target_cores_name:
            local_print(f'Target Core: {core_name}, Shape: {target_cores_weights[core_name].shape}')
        
        local_print(f'QCTN: {qctn.circuit}')
        local_print(f'Target QCTN: {target_qctn.circuit}')
        local_print(f'Einsum Equation: {einsum_equation}')

        # Convert to PyTorch tensors
        torch_tensors = []
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_tensors.append(weight)
        
        for core_name in target_cores_name:
            weight = target_cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_tensors.append(weight)
        
        retracted_QCTN = torch.einsum(einsum_equation, *torch_tensors)

        local_print(f"retracted_QCTN: {retracted_QCTN}")

        return retracted_QCTN
    
    @staticmethod
    def contract_with_self(qctn, circuit_array_input=None):
        """
        Contract the given QCTN with itself.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_array_input (torch.Tensor, optional): Input array for the circuit.
        
        Returns:
            torch.Tensor: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        symbol_id = 0
        einsum_equation_lefthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()
        local_print(f'adjacency_matrix for interaction: \n{adjacency_matrix_for_interaction}')
        
        from ..core.tenmul_qc import QCTNHelper
        for element in QCTNHelper.jax_triu_ndindex(len(cores_name)):
            i, j = element
            if adjacency_matrix_for_interaction[i, j]:
                # If there is a connection between core i and core j
                connection_num = len(adjacency_matrix[i, j])
                connection_symbols = [opt_einsum.get_symbol(symbol_id + k) for k in range(connection_num)]
                symbol_id += connection_num
                adjacency_matrix[i, j] = connection_symbols
                adjacency_matrix[j, i] = connection_symbols  # Ensure symmetry

        local_print(f"adjacency_matrix after symbol assignment: \n{adjacency_matrix}")

        input_symbols_stack = []
        output_symbols_stack = []

        equation_list = []
        new_symbol_mapping = {}

        # self.einsum_equation_lefthand
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

        local_print(f'einsum_equation_lefthand: {einsum_equation_lefthand}')
        local_print(f'input_symbols_stack after source QCTN processing: {input_symbols_stack}')
        local_print(f'output_symbols_stack after source QCTN processing: {output_symbols_stack}')

        if circuit_array_input is not None:
            einsum_equation_output = ''.join(real_output_symbols_stack)
            einsum_equation_lefthand = f"{''.join(input_symbols_stack)},{einsum_equation_lefthand},{einsum_equation_output}"

        einsum_equation = f'{einsum_equation_lefthand}->'

        for core_name in cores_name:
            local_print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')

        local_print(f'QCTN: {qctn.circuit}')
        local_print(f'Einsum Equation: {einsum_equation}')

        # Convert to PyTorch tensors
        torch_tensors = []
        
        if circuit_array_input is not None:
            if not isinstance(circuit_array_input, torch.Tensor):
                circuit_array_input = torch.from_numpy(circuit_array_input).float()
            torch_tensors.append(circuit_array_input)

        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_tensors.append(weight)
        
        for core_name in cores_name[::-1]:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            torch_tensors.append(weight)
        
        if circuit_array_input is not None:
            torch_tensors.append(circuit_array_input)

        retracted_QCTN = torch.einsum(einsum_equation, *torch_tensors)

        local_print(f"retracted_QCTN: {retracted_QCTN}")

        return retracted_QCTN

    @staticmethod
    def contract_with_self_for_gradient(qctn, circuit_array_input=None):
        """
        Contract QCTN with itself and compute gradients using PyTorch autograd.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            circuit_array_input (torch.Tensor, optional): Input array for the circuit.
        
        Returns:
            tuple: (loss, gradients) where gradients is a list of gradient tensors.
        """
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        # Convert to PyTorch tensors with gradient tracking
        torch_weights = {}
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            weight.requires_grad_(True)
            torch_weights[core_name] = weight

        # Store in qctn for contraction
        original_weights = qctn.cores_weights
        qctn.cores_weights = torch_weights

        # Perform contraction
        retracted_QCTN = ContractorPyTorch.contract_with_self(qctn, circuit_array_input)
        
        # Compute MSE loss
        loss = torch.mean((retracted_QCTN - 1.0) ** 2)
        
        # Compute gradients
        loss.backward()
        
        grad_cores = [torch_weights[core_name].grad for core_name in cores_name]
        
        # Restore original weights
        qctn.cores_weights = original_weights
        
        local_print(f'Loss: {loss.item()}')
        
        return loss.item(), grad_cores

    @staticmethod
    def contract_with_QCTN_for_gradient(qctn, target_qctn):
        """
        Contract QCTN with target QCTN and compute gradients using PyTorch autograd.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to optimize.
            target_qctn (QCTN): The target quantum circuit tensor network.
        
        Returns:
            tuple: (loss, gradients) where gradients is a list of gradient tensors.
        """
        cores_name = qctn.cores
        cores_weights = qctn.cores_weights

        # Convert to PyTorch tensors with gradient tracking
        torch_weights = {}
        for core_name in cores_name:
            weight = cores_weights[core_name]
            if not isinstance(weight, torch.Tensor):
                weight = torch.from_numpy(weight).float()
            weight.requires_grad_(True)
            torch_weights[core_name] = weight

        # Store in qctn for contraction
        original_weights = qctn.cores_weights
        qctn.cores_weights = torch_weights

        # Perform contraction
        retracted_QCTN = ContractorPyTorch.contract_with_QCTN(qctn, target_qctn)
        
        # Compute MSE loss
        loss = torch.mean((retracted_QCTN - 1.0) ** 2)
        
        # Compute gradients
        loss.backward()
        
        grad_cores = [torch_weights[core_name].grad for core_name in cores_name]
        
        # Restore original weights
        qctn.cores_weights = original_weights
        
        local_print(f'Loss: {loss.item()}')
        
        return loss.item(), grad_cores
