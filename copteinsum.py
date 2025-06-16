from config import Configuration
import itertools
import opt_einsum
import jax
from tenmul_qc import QCTN, QCTNHelper

class ContractorOptEinsum:
    """
    ContractorOptEinsum class for optimized tensor contraction using opt_einsum.
    
    This class provides methods to contract tensors using the opt_einsum library,
    which is optimized for performance and memory efficiency.
    """

    @staticmethod
    def contract_core_only(qctn: 'QCTN'):
        """
        Contract the given QCTN.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """

        input_ranks, adjacency_matrix, output_ranks = qctn.circuit
        cores_name = qctn.cores
        cores_weights = qctn.cores_weigts

        symbol_id = 0
        einsum_equation_lefthand = ''
        einsum_equation_righthand = ''

        adjacency_matrix_for_interaction = adjacency_matrix.copy()
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
            print(f"Processing core: {cores_name[idx]} with input ranks {input_ranks[idx]}, adjacency matrix {adjacency_matrix[idx, :]}, output ranks {output_ranks[idx]}")
            for _ in input_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            print('ckpt 1')
            print(f"Current einsum equation left-hand side: {einsum_equation_lefthand}")
            print(f"Current einsum equation right-hand side: {einsum_equation_righthand}")
            einsum_equation_lefthand += "".join(list(itertools.chain.from_iterable(adjacency_matrix[idx])))

            print('ckpt 2')
            print(f"Current einsum equation left-hand side: {einsum_equation_lefthand}")
            print(f"Current einsum equation right-hand side: {einsum_equation_righthand}")
            for _ in output_ranks[idx]:
                einsum_equation_lefthand += opt_einsum.get_symbol(symbol_id)
                einsum_equation_righthand += opt_einsum.get_symbol(symbol_id)
                symbol_id += 1
            
            einsum_equation_lefthand += ','
            print('ckpt 3')
            print(f"Current einsum equation left-hand side: {einsum_equation_lefthand}")
            print(f"Current einsum equation right-hand side: {einsum_equation_righthand}")

        einsum_equation_lefthand = einsum_equation_lefthand[:-1]  # Remove the last comma
        einsum_equation = f'{einsum_equation_lefthand}->{einsum_equation_righthand}'

        tensor_shapes = [cores_weights[core_name].shape for core_name in cores_name]

        for core_name in cores_name:
            print(f'Core: {core_name}, Shape: {cores_weights[core_name].shape}')
        
        print(f'QCTN: {qctn.circuit}')
        print(f'Einsum Equation: {einsum_equation}')
        print(f'Tensor Shapes: {tensor_shapes}')

        einsum_expr = opt_einsum.contract_expression(einsum_equation, *tensor_shapes, optimize=Configuration.opt_einsum_optimize)
        jit_retraction = jax.jit(einsum_expr)
        retracted_QCTN = jit_retraction(*[cores_weights[core_name] for core_name in cores_name])

        return retracted_QCTN
    

    @staticmethod
    def contract(tensors, equation, optimize='greedy'):
        """
        Contract tensors using opt_einsum.
        
        Args:
            tensors (list): List of tensors to be contracted.
            equation (str): The einsum equation specifying the contraction.
            optimize (str): Optimization strategy for contraction. Default is 'greedy'.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """
        return opt_einsum.contract(equation, *tensors, optimize=optimize)
