from symmetry_breaking_quantum import (
    incidence_to_graph,
    build_brick_wall_IM,
    target_tensor_init,
    validate_target_tensor,
    symmetry_breaking,
)
from tneq_qc.backends.backend_factory import BackendFactory
import numpy as np
import torch
import os


is_data_loaded = False
backend=BackendFactory.create_backend("pytorch", device="cpu")
# n_qubits = 8  # number of qubits
# n_cells = 5 # number of brick-wall unit cells
n_qubits = 4  # number of qubits
n_cells = 3 # number of brick-wall unit cells
n_cores = (n_qubits - 1) * n_cells  # number of cores
data_size = 20


IM = build_brick_wall_IM(n_qubits, n_cells, rank=2)
print("Incidence Matrix:\n", IM)
# import pdb; pdb.set_trace()

target_mask_list = [2,3,5,8]
# target_mask_list = [0,4,6,8]    # experiment for 4 qubits
# target_mask_list = [2,3,5,8,9,12,13,14,15,17,18,20,21,23,25,26,29,31,32,33]     # experiment for 8 qubits


# IM = np.array([[2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
#             [2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2],
#             [0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2],
#             [0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0]])
# n_qubits = IM.shape[0]  # number of qubits
# n_cores = IM.shape[1]

if  is_data_loaded:
    # load target tensor from file
    target_tensor = torch.load(os.path.join(os.path.dirname(__file__), "data", f"nqubit_{n_qubits}_ncore_{n_cores}.pt"))
    print("Target tensor loaded from file.")
    n_qubits = target_tensor.dim()/2
    # need to record n_cores when generating target tensor
    # n_cores = 

else:
    # generate target tensor
    for i in range(data_size):
        print("This is generation",i)
        while True:
            target_tensor = target_tensor_init(IM, n_cores, backend=backend, target_mask_list=target_mask_list)
            flag = validate_target_tensor(target_tensor, IM, backend=backend, n_qubits=n_qubits, n_cores=n_cores,idx=i)
            if flag:
                print("Target tensor generated and validated.")
                break
            print("Regenerating target tensor...")


# Start symmetry breaking

pruned_list, prune_count = symmetry_breaking(IM=IM, target_tensor=target_tensor, backend=backend, n_qubits=n_qubits, n_cores=n_cores)

original_graph_for_display = incidence_to_graph(IM, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Original QCTN graph:\n" + original_graph_for_display)
target_graph_for_display = incidence_to_graph(IM, mask_list=target_mask_list, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Target QCTN graph:\n" + target_graph_for_display)
pruned_graph_for_display = incidence_to_graph(IM, mask_list=pruned_list, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Pruned QCTN graph:\n" + pruned_graph_for_display)

print(f"Total cores pruned: {len(pruned_list)} out of {n_cores}, {len(pruned_list)/n_cores}% tried {prune_count} times.")