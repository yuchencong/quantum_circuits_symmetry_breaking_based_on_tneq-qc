"""
Example script demonstrating QCTN split / merge operations and simple visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper


def _adjacency_to_array(adj):
    """
    Convert QCTN.adjacency_matrix (list-of-ranks per entry) to a numeric array.

    Each entry (i, j) stores the sum of bond dimensions between cores i and j.
    """
    n = adj.shape[0]
    arr = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if isinstance(adj[i, j], (list, tuple)):
                arr[i, j] = float(sum(adj[i, j]))
            elif adj[i, j] is None:
                arr[i, j] = 0.0
            else:
                # Fallback: try to interpret as scalar
                try:
                    arr[i, j] = float(adj[i, j])
                except Exception:
                    arr[i, j] = 0.0
    return arr


def main():
    # ------------------------------------------------------------------
    # 1. 构造一个简单的 MPS 图并创建 QCTN
    # ------------------------------------------------------------------
    backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")

    num_qubits = 8
    graph_type = "mps"
    dim_char = "2"

    graph = QCTNHelper.generate_example_graph(
        n=num_qubits,
        graph_type=graph_type,
        dim_char=dim_char,
    )
    print("Original QCTN graph:")
    print(graph)

    qctn = QCTN(graph, backend=backend)
    print(f"Original QCTN: nqubits = {qctn.nqubits}, ncores = {qctn.ncores}, graph: \n{qctn.tn_graph.to_string()}")

    # ------------------------------------------------------------------
    # 2. 对 QCTN 做 split 操作
    # ------------------------------------------------------------------
    left_qctn, right_qctn = qctn.split()
    print(f"Left  QCTN: nqubits = {left_qctn.nqubits}, ncores = {left_qctn.ncores}, graph = {left_qctn}")
    print(f"Right QCTN: nqubits = {right_qctn.nqubits}, ncores = {right_qctn.ncores}, graph = {right_qctn}")

    # ------------------------------------------------------------------
    # 3. 对两个子 QCTN 做 merge 操作
    # ------------------------------------------------------------------
    merged_qctn = QCTN.merge(left_qctn, right_qctn)
    print(
        f"Merged QCTN: nqubits = {merged_qctn.nqubits}, "
        f"ncores = {merged_qctn.ncores}"
    )

    # ------------------------------------------------------------------
    # 4. 可视化：原始 / 子网络 / 合并后 的邻接矩阵
    # ------------------------------------------------------------------
    adj_orig = _adjacency_to_array(qctn.adjacency_matrix)
    adj_left = _adjacency_to_array(left_qctn.adjacency_matrix)
    adj_right = _adjacency_to_array(right_qctn.adjacency_matrix)
    adj_merged = _adjacency_to_array(merged_qctn.adjacency_matrix)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax00, ax01, ax10, ax11 = axes.flatten()

    im0 = ax00.imshow(adj_orig, cmap="viridis")
    ax00.set_title("Original adjacency")
    fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)

    im1 = ax01.imshow(adj_left, cmap="viridis")
    ax01.set_title("Left part adjacency")
    fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)

    im2 = ax10.imshow(adj_right, cmap="viridis")
    ax10.set_title("Right part adjacency")
    fig.colorbar(im2, ax=ax10, fraction=0.046, pad=0.04)

    im3 = ax11.imshow(adj_merged, cmap="viridis")
    ax11.set_title("Merged adjacency")
    fig.colorbar(im3, ax=ax11, fraction=0.046, pad=0.04)

    for ax in (ax00, ax01, ax10, ax11):
        ax.set_xlabel("core index")
        ax.set_ylabel("core index")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

