import time
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.config import Configuration
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.core.cqctn import ContractorQCTN
from tneq_qc.backends.copteinsum import ContractorOptEinsum
# from tneq_qc.core.engine import Engine
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.optim.optimizer import Optimizer
import numpy as np
import torch
import math
from typing import Any
from tqdm import tqdm

def generate_circuit_states_list(num_qubits, K, device='cuda'):
    """
    Generate circuit states list with status [0, 0, ..., 1] for each qubit
    Parameters:
    - num_qubits: Number of qubits
    - K: Dimension of each qubit state
    Returns:
    - circuit_states_list: List of tensors representing the circuit states for each qubit
    """
    circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]

    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0

    return circuit_states_list

if __name__ == "__main__":
    backend_type = 'pytorch'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    backend = BackendFactory.create_backend(backend_type, device='cpu')

    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=100)
    # engine = EngineSiamese(backend=backend, strategy_mode="fast", mx_K=100)

    suffix = "_exp01"
    # graph_type = "tree"
    # graph_type = "wall"
    graph_type = "std"
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, graph_type="std", dim_char='3')
    # print(f"std qctn_graph: \n{qctn_graph}")

    # qctn_graph = QCTNHelper.generate_example_graph(n=17, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=5, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=5, graph_type=graph_type, dim_char='3')
    qctn_graph = QCTNHelper.generate_example_graph(n=257, graph_type=graph_type, dim_char='3')
    print(f"{graph_type} qctn_graph: \n{qctn_graph}")
    
    
    qctn = QCTN(qctn_graph, backend=engine.backend)

    # qctn = QCTN.from_pretrained(qctn_graph, "assets/qctn_cores_3qubitswall_dist_00.safetensors", backend=engine.backend)
    # pretrained_qctn = QCTN.from_pretrained(qctn_graph, "assets/qctn_cores_3qubitswall_dist_00.safetensors", backend=engine.backend)
    # right_qctn = QCTN(qctn_graph, backend=engine.backend)
    N = 100
    B = 1024
    # B = 1
    D = qctn.nqubits
    K = 3
    num_step = 1000

    data_list = []
    
    device = backend.backend_info.device
    
    for i in range(N):
        x = torch.empty((B, D), device=device).normal_(mean=0.0, std=1.0)

        Mx_list, out = engine.generate_data(x, K=K, ret_type='TNTensor')
        # Mx_list, out = engine.generate_data(x, K=K)

        data_list += [({"measure_input_list": Mx_list}, out)]

    # data_list structure is [(dict, out), ...]
    # We only need the dict part for optimizer
    data_list_for_optim = [x[0] for x in data_list]

    # for i in range(len(data_list_for_optim)):
    #     x = data_list_for_optim[i]["measure_input_list"]
    #     data_list_for_optim[i]["measure_input_list"] = [torch.eye(tensor.shape[-1], device=tensor.device) for tensor in x]

    # print(data_list_for_optim[0]["measure_input_list"][0].shape)
    
    circuit_states_list = generate_circuit_states_list(num_qubits=D, K=K, device=backend.backend_info.device)

    for c_name in qctn.cores:
        core_tensor = qctn.cores_weights[c_name]
        if isinstance(core_tensor, torch.Tensor):
            core_tensor.requires_grad_(True)
        else:
            core_tensor.tensor.requires_grad_(True)

        # elif isinstance(core_tensor, TNTensor):
        #     core_tensor.tensor.requires_grad_(True)
        requires_grad = core_tensor.tensor.requires_grad
        print(f"core {c_name} requires_grad: {requires_grad}")
    

    # for c_name in right_qctn.cores:
    #     core_tensor = right_qctn.cores_weights[c_name]
    #     core_tensor.requires_grad_(True)

    #     requires_grad = core_tensor.requires_grad
    #     print(f"pretrained core {c_name} requires_grad: {requires_grad}")
    
    lr_schedule = [
        (0, 1e-2),
        (200, 5e-3),
        (600, 2e-3),
        (800, 1e-3),
    ]

    optimizer = Optimizer(
        method='sgdg', 
        max_iter=num_step, 
        # tol=1e-6, 
        tol=0.0, 
        # learning_rate=1e-2, 
        learning_rate=1e-3, 
        beta1=0.9, 
        beta2=0.95, 
        epsilon=1e-8,
        engine=engine,
        # lr_schedule=lr_schedule,

        momentum=0.9,            # 动量因子
        stiefel=True,            # 启用 Stiefel 流形优化
    )
    
    torch.cuda.empty_cache()
    
    tic = time.time()

    optimizer.optimize(qctn, 
                       data_list=data_list_for_optim, 
                    #    circuit_states=circuit_states_list,
                       circuit_states_list=circuit_states_list,
                    #    circuit_states_list=None,
                    #    right_qctn=right_qctn,
                       )

    toc = time.time()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Optimization Time: {toc - tic:.2f} seconds")

    # save cores
    cores_file = f"./assets/qctn_cores_{qctn.nqubits}qubits{graph_type}{suffix}.safetensors"
    qctn.save_cores(cores_file, metadata={"graph": graph_type})
    
    print(f"Saved trained QCTN cores to {cores_file}")

    exit()

    # Choose the first data of each batch for testing
    test_loss_list = []
    for i in range(N):
        data_slice = [x[0:1] for x in data_list[i]["measure_input_list"]]
        result = engine.contract_with_std_graph(qctn, 
                                                circuit_states_list=circuit_states_list,
                                                measure_input_list=data_slice, 
                                                )
        print(f"Test {i}, Result: {result.item()}")
        test_loss_list.append(result.item())

    print(f"Average Result: {sum(test_loss_list) / len(test_loss_list)}")
    print(f"Max Result: {max(test_loss_list)}")
    print(f"Min Result: {min(test_loss_list)}")

    
    # load pretrained qctn
    pretrained_qctn = QCTN.from_pretrained(qctn_graph, cores_file, backend=engine.backend)
    with torch.no_grad():
        pretrained_result = engine.contract_with_std_graph(
            pretrained_qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=data_list[0]["measure_input_list"],
        )
    print(f"Pretrained Result (std graph): {pretrained_result} {pretrained_result.shape}")

    # get cross entropy loss for all pretrained results (shape = 512)
    loss = - torch.log(pretrained_result)
    print(f"Cross Entropy Loss: {loss} {torch.mean(loss)}")
