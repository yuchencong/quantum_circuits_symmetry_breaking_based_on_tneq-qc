import numpy as np
import torch
import opt_einsum as oe

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.contractor.einsum_strategy import EinsumStrategy

import random



# functions 
def incidence_to_graph(
    incidence: np.ndarray,
    core_symbols=None,
    mask_list=None,
    *,
    for_display: bool = False,
    keep_zeros: bool = False,
    mask_char: str = "█",
    pad_dim = None,
) -> str:
    """
    Convert incidence matrix (rows=qubits, cols=cores) to QCTN graph string.

    for_display=True  : insert '-' placeholders for zeros (alignment only).
    for_display=False : emit valid QCTN graph (zeros are skipped).
    keep_zeros        : only meaningful when for_display=True.
    mask_char         : display-only replacement for masked core symbols.
    """
    if incidence.ndim != 2:
        raise ValueError("incidence must be 2D (n_qubits x n_cores)")
    if (incidence < 0).any():
        raise ValueError("incidence entries must be >= 0")

    n_qubits, n_cores = incidence.shape

    if core_symbols is None:
        core_symbols = [oe.get_symbol(i) for i in range(n_cores)]
    if len(core_symbols) != n_cores:
        raise ValueError("core_symbols length must match n_cores")

    if mask_list is None:
        mask_list = []
    mask_set = set(mask_list)

    for idx in mask_set:
        if idx < 0 or idx >= n_cores:
            raise IndexError(f"mask_index={idx} out of range: 0 ~ {n_cores-1}")

    def symbol(c: int) -> str:
        if for_display and c in mask_set:
            return mask_char
        return core_symbols[c]

    # ---- valid QCTN graph (no placeholders) ----
    if not for_display:
        lines = []
        for q in range(n_qubits):
            entries = [(symbol(c), int(incidence[q, c]))
                       for c in range(n_cores) if incidence[q, c] > 0]
            if not entries:
                raise ValueError(f"Row {q} has no cores; graph line would be invalid.")

            line = f"-{entries[0][1]}-{entries[0][0]}"
            for core, dim in entries[1:]:
                line += f"-{dim}-" + core
            line += f"-{entries[-1][1]}-"
            lines.append(line)
        return "\n".join(lines)

    # ---- display-only graph with aligned placeholders ----
    if keep_zeros:
        # compute per-column width for alignment
        col_width = []
        for c in range(n_cores):
            vals = incidence[:, c]
            vals = vals[vals > 0]
            if len(vals) > 0:
                dim = int(vals.max())
            elif pad_dim is not None:
                dim = int(pad_dim)
            else:
                dim = 1
            slot = f"-{dim}-{symbol(c)}"
            col_width.append(len(slot))

        lines = []
        for q in range(n_qubits):
            line = ""
            for c in range(n_cores):
                dim = int(incidence[q, c])
                if dim > 0:
                    slot = f"-{dim}-{symbol(c)}"
                    # pad slot to column width
                    if len(slot) < col_width[c]:
                        slot = slot + "-" * (col_width[c] - len(slot))
                    line += slot
                else:
                    line += "-" * col_width[c]
            line += "-"
            lines.append(line)
        return "\n".join(lines)

    # display but no zero placeholders
    return incidence_to_graph(
        incidence, core_symbols=core_symbols, mask_list=mask_list,
        for_display=False
    )



# 0) Prepare brick-wall structure graph and backend
IM = np.array([[2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0],
               [2, 0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2],
               [0, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 2],
               [0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0]])
n_qubits = IM.shape[0]  # number of qubits
n_cores = IM.shape[1]
backend = BackendFactory.create_backend("pytorch", device="cpu")

# 1) Target tensor (brick wall with random mask)
target_mask_list = [2,3,5]
mask_IM = IM.copy()
for mask_idx in target_mask_list:
    # print(f"Processing mask_index={mask_idx} ...")
    if mask_idx >= n_cores:   # 注意用 >=，因为 index 最大是 n_cores-1
        raise IndexError(
            f"mask_index={mask_idx} out of range: 0 ~ {mask_idx-1}"
        )
    mask_IM[:, mask_idx] = 0 

target_graph = incidence_to_graph(mask_IM)  # For tensor generation
# print("Target QCTN graph:\n" + target_graph)
target_qctn = QCTN(target_graph,backend=backend)
einsum_eq, tensor_shapes = EinsumStrategy.build_core_only_expression(target_qctn)
expr = oe.contract_expression(einsum_eq, *tensor_shapes, optimize="auto")
params = [target_qctn.cores_weights[c] for c in target_qctn.cores]
target_tensor = expr(*params)
target_tensor = target_tensor.detach()
# print("Target tensor shape:", target_tensor.shape, target_tensor)

# 2) Prepare for symmetric breaking
pruned_list = []
pruned_IM = IM.copy()
prune_list = list(range(n_cores))
# 3) Strat symmetric breaking
for i in range(max_iterations:=50):
    if len(pruned_list) == len(prune_list):
        print("All cores have been pruned.")
        break
    random.shuffle(prune_list)
    for idx in prune_list:
        print(f"Trying to prune core index={idx} ...")
        if idx in pruned_list:
            continue
        candidate = pruned_list + [idx]
        cand_IM = IM.copy()
        cand_IM[:, candidate] = 0

        prune_graph_for_display = incidence_to_graph(IM, mask_list=candidate, for_display=True, keep_zeros=True, mask_char="█") # For display only
        print("Now QCTN graph:\n" + prune_graph_for_display)
        for pruned_idx in pruned_list:
            pruned_IM[:, pruned_idx] = 0
        graph = incidence_to_graph(pruned_IM)
        train_qctn = QCTN(graph, backend=backend)
        einsum_eq, tensor_shapes = EinsumStrategy.build_core_only_expression(train_qctn)
        expr = oe.contract_expression(einsum_eq, *tensor_shapes, optimize="auto")
        params = [torch.nn.Parameter(train_qctn.cores_weights[c]) for c in train_qctn.cores]

        opt = torch.optim.Adam(params, lr=1e-2)
        out = expr(*params)
        # try to fit target tensor
        pruned_flag = False
        for i in range(500):
            opt.zero_grad()
            out = expr(*params)
            loss = ((out - target_tensor) ** 2).mean()
            loss.backward()
            opt.step()
            # if i % 10 == 0:
            #     print(i, float(loss))
            fidelity = (2**n_qubits -2 ** (2*n_qubits-1) *loss)/2**n_qubits
            # print(f"Fidelity after step {i}: {fidelity}")
            if 1-fidelity < 1e-4:
                pruned_flag = True
                print(f"Successfully pruned core index={idx} , fidelity={fidelity}")
                pruned_list = candidate
                break
        # if not pruned_flag:
        #     pruned_list.pop()
                
# backend = BackendFactory.create_backend("pytorch", device="cuda")
# qctn = QCTN(graph, backend=backend)
# einsum_eq, tensor_shapes = EinsumStrategy.build_core_only_expression(qctn)
# expr = oe.contract_expression(einsum_eq, *tensor_shapes, optimize="auto")

# 4) Prepare torch params
# for c in qctn.cores:
#     print(c, type(qctn.cores_weights[c]))

# params = [torch.nn.Parameter(qctn.cores_weights[c].tensor) for c in qctn.cores]
# params = [torch.nn.Parameter(qctn.cores_weights[c]) for c in qctn.cores]
# device = backend.backend_info.device
# target = torch.from_numpy(T).float().to(device)


# 5) Fit cores to target



original_graph_for_display = incidence_to_graph(IM, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Original QCTN graph:\n" + original_graph_for_display)
target_graph_for_display = incidence_to_graph(IM, mask_list=target_mask_list, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Target QCTN graph:\n" + target_graph_for_display)
pruned_graph_for_display = incidence_to_graph(IM, mask_list=pruned_list, for_display=True, keep_zeros=True, mask_char="█") # For display only
print("Pruned QCTN graph:\n" + pruned_graph_for_display)
