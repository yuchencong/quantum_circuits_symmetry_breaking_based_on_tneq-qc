# Quantum Circuit Tensor Network (QCTN)

This module provides classes and utilities for representing and simulating quantum circuits as tensor networks using JAX.

## Features

- **QCTN**: Main class for parsing a quantum circuit graph, building the corresponding tensor network, and performing contractions.

## Example Usage

```shell
python train.py
```

## Graph Format

A quantum circuit is described as a multi-line string, where each line represents a qubit and connections to tensor cores (labeled by uppercase letters or CJK characters):

```text
-3-A-------------------3-
-3-A--3--B-------------3-
-3-------B--3--C-------3-
-3-------------C--3--D-3-
-3-------------------D-3-
```

- Letters (A, B, C, ...) denote tensor cores.
- Numbers denote the rank (dimension) of each connection.

## API Overview

- `QCTNHelper.generate_example_graph()`: Returns a sample graph string.
- `QCTNHelper.generate_random_example_graph(nqubits, ncores)`: Generates a random graph.
- `QCTN(graph)`: Parses the graph and builds the tensor network.
- `QCTN.contract(attach=None, engine=...)`: Contracts the network, optionally with inputs or another QCTN.

## Requirements

- [JAX](https://github.com/google/jax)
- [NumPy](https://numpy.org/)

## Notes

- The contraction logic relies on an external engine (`ContractorOptEinsum`).
- Some advanced contraction methods are placeholders and need implementation.

## Roadmap

### 1. Tensor Network Architecture
- [x] Basic tensor network architecture
- [x] Mx module support (multi-dimensional tensor operations)
- [x] Batch Training support

### 2. Genetic Algorithm
- [x] Graph representation method (TNGraph with ASCII art)
- [x] Basic mutation operations (modify bond, insert tensor, remove tensor)
- [ ] Composite mutation operations (multiple mutations in sequence)
- [ ] Crossover operation (breeding from two parents)

### 3. Backend Support
- [x] JAX backend
- [ ] cuTensorNet backend (NVIDIA GPU acceleration)
- [x] PyTorch backend
- [ ] Remove other backend and only support PyTorch
- [x] opt_einsum contract
- [ ] Custom hand-crafted contraction method (optimized for specific patterns)

### 4. Distributed Computing Support
- [x] MPI-based distribution (master-worker architecture)
- [ ] ... (other distributed frameworks)
- [x] Multi-GPU support (single node)
- [ ] Multi-node support (cluster computing)

### 5. Observations
- [ ] 经过测试，可支持至少27个qubits，在batch_size<=8的情况下训练，显存占用37G。batch_size > 8 OOM（超过50G）
- [ ] 显存占用不呈线性增长。batch size < 8 时始终37G，batch_size > 8 突变
- [ ] 训练有明显的梯度消失
- [ ] 由于TN结果不能保证输出在0-1范围，交叉熵loss会出现优化到负值的情况
- [ ] MSE loss 梯度更小，消失更明显，且在对梯度进行缩放后也难以收敛。
- [ ] adam训练时如果对梯度缩放容易nan
