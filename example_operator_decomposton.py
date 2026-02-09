import numpy as np
import torch
import opt_einsum as oe

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.contractor.einsum_strategy import EinsumStrategy



# functions 
def incidence_to_graph(incidence: np.ndarray, core_symbols=None) -> str:
    """
    Convert incidence matrix (rows=qubits, cols=cores) to QCTN graph string.
    0 entries are rendered as a dash-block with the SAME width as "-dim-core"
    so columns stay aligned.
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

    # width of one cell: "-<max_dim>-<symbol>"
    max_dim = int(incidence.max()) if incidence.size else 0
    cell_len = len(f"-{max_dim}-{core_symbols[0]}")  # core symbol is 1 char typically
    empty_cell = "-" * cell_len

    lines = []
    for q in range(n_qubits):
        line = ""
        for c in range(n_cores):
            dim = int(incidence[q, c])
            if dim > 0:
                line += f"-{dim}-{core_symbols[c]}"
            else:
                line += empty_cell
        line += "-"  # ending dash
        lines.append(line)

    return "\n".join(lines)

'''====== Example: Fit QCTN cores to target tensor via Adam optimization ======'''


# 0) Prepare brick-wall structure graph
IM = np.array([[2, 0, 0, 2, 0, 0,],
               [2, 0, 2, 2, 0, 2,],
               [0, 2, 2, 0, 2, 2,],
               [0, 2, 0, 0, 2, 0,]])
n_qubits = IM.shape[0]  # number of qubits
n_cores = IM.shape[1]

# 1) Target tensor (brick wall with random mask)
musk_list = [2,3,5]
mask_IM = IM.copy()
for musk_idx in musk_list:
    # print(f"Processing musk_index={musk_idx} ...")
    if musk_idx >= n_cores:   # 注意用 >=，因为 index 最大是 n_cores-1
        raise IndexError(
            f"musk_index={musk_idx} out of range: 0 ~ {musk_idx-1}"
        )
    mask_IM[:, musk_idx] = 0  # 把该 core 对应的列清零
musked_graph = incidence_to_graph(mask_IM)
print("Masked QCTN graph:\n" + musked_graph)
import pdb; pdb.set_trace()

# 2) Simple 2‑core, 2‑qubit graph with bond dim = 2
# Output shape is (2,2,2,2)
# graph = "-2-A-2-B-2-\n-2-A-2-B-2-"

n_qubits = IM.shape[0]  # number of qubits
n_cores = IM.shape[1]
# print(f"Number of qubits: {n_qubits}, number of cores: {n_cores}")
graph = incidence_to_graph(IM)
print("QCTN graph:\n" + graph)

# backend = BackendFactory.create_backend("pytorch", device="cuda")
backend = BackendFactory.create_backend("pytorch", device="cpu")
qctn = QCTN(graph, backend=backend)

# 3) Build a core‑only contraction expression (opt_einsum supports torch tensors)
einsum_eq, tensor_shapes = EinsumStrategy.build_core_only_expression(qctn)
expr = oe.contract_expression(einsum_eq, *tensor_shapes, optimize="auto")

# 4) Prepare torch params
# for c in qctn.cores:
#     print(c, type(qctn.cores_weights[c]))

# params = [torch.nn.Parameter(qctn.cores_weights[c].tensor) for c in qctn.cores]
params = [torch.nn.Parameter(qctn.cores_weights[c]) for c in qctn.cores]
device = backend.backend_info.device
target = torch.from_numpy(T).float().to(device)

opt = torch.optim.Adam(params, lr=1e-2)

# 5) Fit cores to target
for i in range(2000):
    opt.zero_grad()
    out = expr(*params)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    opt.step()

    if i % 20 == 0:
        print(i, float(loss))


