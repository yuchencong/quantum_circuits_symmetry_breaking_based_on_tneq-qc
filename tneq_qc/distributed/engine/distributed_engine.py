"""
Distributed Engine Siamese

Extends EngineSiamese with distributed computing capabilities:
- QCTN graph partitioning across workers
- Hierarchical tensor contraction (log(n)+1 stages)
- Tensor parallel matrix multiplication
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

import numpy as np

from ..comm import CommBase, get_comm_backend, ReduceOp

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN
    from ...backends.backend_interface import ComputeBackend


@dataclass
class PartitionConfig:
    """Configuration for QCTN graph partitioning."""
    
    # Partitioning strategy: 'layer' (by layers/stages) or 'core' (by cores)
    strategy: str = 'layer'
    
    # Number of partitions (usually equals world_size)
    num_partitions: int = 1
    
    # Minimum cores per partition
    min_cores_per_partition: int = 1
    
    # Whether to balance partition sizes
    balance_partitions: bool = True


@dataclass
class ContractStage:
    """Information about a contraction stage."""
    
    stage_idx: int
    stage_type: str  # 'local' or 'reduce'
    
    # For 'local' stage: which cores to contract
    local_cores: List[str] = field(default_factory=list)
    
    # For 'reduce' stage: group info
    group_ranks: List[int] = field(default_factory=list)  # ranks in this group
    group_size: int = 1
    is_group_leader: bool = False  # True if this rank should send result
    partner_rank: int = -1  # rank to receive from or send to
    
    def __str__(self) -> str:
        lines = [f"  Stage {self.stage_idx} ({self.stage_type}):"]
        if self.stage_type == 'local':
            cores_str = ', '.join(self.local_cores[:5])
            if len(self.local_cores) > 5:
                cores_str += f", ... (+{len(self.local_cores) - 5} more)"
            lines.append(f"    Local cores: [{cores_str}]")
        else:
            lines.append(f"    Group size: {self.group_size}")
            if self.group_ranks:
                lines.append(f"    Group ranks: {self.group_ranks}")
            if self.partner_rank >= 0:
                lines.append(f"    Partner rank: {self.partner_rank}")
            lines.append(f"    Is group leader: {self.is_group_leader}")
        return '\n'.join(lines)


@dataclass
class DistributedContractPlan:
    """Execution plan for distributed contraction."""
    
    # Total number of stages
    num_stages: int
    
    # List of stages to execute
    stages: List[ContractStage] = field(default_factory=list)
    
    # Partition info for this rank
    local_partition_idx: int = 0
    local_cores: List[str] = field(default_factory=list)
    
    # Inter-node contraction graph (computed by master)
    inter_node_graph: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "DistributedContractPlan",
            "=" * 60,
            f"Partition Index: {self.local_partition_idx}",
            f"Total Stages: {self.num_stages}",
        ]
        
        # Local cores summary
        cores_str = ', '.join(self.local_cores[:5])
        if len(self.local_cores) > 5:
            cores_str += f", ... (+{len(self.local_cores) - 5} more)"
        lines.append(f"Local Cores ({len(self.local_cores)}): [{cores_str}]")
        
        # Stages
        lines.append("-" * 40)
        lines.append("Stages:")
        for stage in self.stages:
            lines.append(str(stage))
        
        # Inter-node graph summary
        if self.inter_node_graph:
            lines.append("-" * 40)
            lines.append("Inter-Node Graph:")
            lines.append(f"  Num partitions: {self.inter_node_graph.get('num_partitions', 'N/A')}")
            lines.append(f"  Partition sizes: {self.inter_node_graph.get('partition_sizes', [])}")
            
            cross_edges = self.inter_node_graph.get('cross_edges', [])
            lines.append(f"  Cross edges: {len(cross_edges)}")
            if cross_edges and len(cross_edges) <= 5:
                for edge in cross_edges:
                    lines.append(f"    {edge['from_core']} -> {edge['to_core']} (rank={edge['edge_rank']} qubit={edge['qubit_idx']})")
            elif cross_edges:
                for edge in cross_edges[:3]:
                    lines.append(f"    {edge['from_core']} -> {edge['to_core']} (rank={edge['edge_rank']} qubit={edge['qubit_idx']})")
                lines.append(f"    ... (+{len(cross_edges) - 3} more edges)")
            
            # Partition adjacency tables summary
            partition_tables = self.inter_node_graph.get('partition_adjacency_tables', [])
            if partition_tables:
                lines.append(f"  Partition adjacency tables: {len(partition_tables)} partitions")
                for p_idx, table in enumerate(partition_tables):
                    # Count out_edge_list
                    out_internal_edges = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if not e.get('is_cross_partition', False) and e.get('neighbor_name')
                    )
                    out_cross_edges = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if e.get('is_cross_partition', False)
                    )
                    # Count in_edge_list
                    in_internal_edges = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if not e.get('is_cross_partition', False) and e.get('neighbor_name')
                    )
                    in_cross_edges = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if e.get('is_cross_partition', False)
                    )
                    in_external = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if not e.get('neighbor_name')
                    )
                    out_external = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if not e.get('neighbor_name')
                    )
                    lines.append(f"    P{p_idx}: {len(table)} cores | "
                                 f"in: {in_internal_edges} internal, {in_cross_edges} cross, {in_external} ext | "
                                 f"out: {out_internal_edges} internal, {out_cross_edges} cross, {out_external} ext")
        
        lines.append("=" * 60)
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return self.__str__()


class DistributedEngineSiamese:
    """
    Distributed EngineSiamese.
    
    Provides distributed tensor network contraction with hierarchical reduction:
    
    1. **Initialization (init)**: 
       - Master process partitions the QCTN graph into subgraphs
       - Each subgraph is assigned to a worker
       - Inter-node contraction graph is computed
    
    2. **Contraction (contract)**:
       - Stage 0: Each worker contracts its local subgraph using StrategyCompiler
       - Stage 1 to log2(n): Hierarchical reduction via tensor parallel
         - Workers are grouped (e.g., [0,1], [2,3], then [0,1,2,3])
         - Each group performs matrix multiplication using TP
    
    Example (4 workers):
        - Stage 0: All workers contract their local subgraphs in parallel
        - Stage 1: Groups [0,1] and [2,3] do TP matrix multiply
        - Stage 2: Group [0,1,2,3] does final TP matrix multiply
    
    Usage:
        >>> engine = DistributedEngineSiamese(backend='pytorch')
        >>> engine.init_distributed(qctn)  # Partition and distribute
        >>> result = engine.contract_distributed(circuit_states, measure_input)
    """
    
    def __init__(self, 
                 backend: Optional[Union[str, 'ComputeBackend']] = None, 
                 strategy_mode: str = 'balanced', 
                 mx_K: int = 100,
                 comm: Optional[CommBase] = None,
                 partition_config: Optional[PartitionConfig] = None):
        """
        Initialize distributed engine.
        
        Args:
            backend: Compute backend ('pytorch', 'jax', or ComputeBackend instance)
            strategy_mode: Contraction strategy mode ('fast', 'balanced', 'full')
            mx_K: Maximum Hermite polynomial order
            comm: Communication backend (auto-created if None)
            partition_config: Configuration for graph partitioning
        """
        # Import and create base engine
        from ...core.engine_siamese import EngineSiamese
        
        self._base_engine = EngineSiamese(
            backend=backend,
            strategy_mode=strategy_mode,
            mx_K=mx_K
        )
        
        # Initialize communication
        self.comm = comm or get_comm_backend(backend.backend_info.backend_type)

        self.rank = self.comm.rank
        self.world_size = self.comm.world_size
        
        # Partition configuration
        self.partition_config = partition_config or PartitionConfig(
            num_partitions=self.world_size
        )
        
        # Distributed state (set after init_distributed)
        self._is_initialized = False
        self._qctn: Optional['QCTN'] = None
        self._local_qctn: Optional['QCTN'] = None  # Local partition QCTN
        self._contract_plan: Optional[DistributedContractPlan] = None
        
        self._log(f"DistributedEngineSiamese initialized: rank={self.rank}/{self.world_size}")
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.rank == 0:
            print(f"[DistributedEngine] {msg}")
    
    # ==================== Proxy to Base Engine ====================
    
    @property
    def backend(self):
        """Access the underlying compute backend."""
        return self._base_engine.backend
    
    @property
    def contractor(self):
        """Access the einsum strategy contractor."""
        return self._base_engine.contractor
    
    @property
    def strategy_compiler(self):
        """Access the strategy compiler."""
        return self._base_engine.strategy_compiler
    
    def generate_data(self, x, K=None):
        """
        Generate measurement matrices from input data.
        
        Proxies to base engine's generate_data method.
        
        Args:
            x: Input tensor of shape (B, D)
            K: Hermite polynomial order (uses engine default if None)
            
        Returns:
            (Mx_list, extra_info)
        """
        return self._base_engine.generate_data(x, K=K)
    
    # ==================== Distributed Initialization ====================
    
    def init_distributed(self, qctn: 'QCTN') -> DistributedContractPlan:
        """
        Initialize distributed contraction for a QCTN.
        
        This method:
        1. Master process partitions the QCTN graph into subgraphs
        2. Distributes subgraph assignments to all workers
        3. Computes the inter-node contraction plan
        
        Args:
            qctn: The full QCTN to distribute
            
        Returns:
            DistributedContractPlan: Execution plan for this worker
        """
        self._qctn = qctn
        
        # Step 1: All processes compute the same partition (deterministic)
        # No broadcast needed - each process runs the same partitioning logic
        partitions = self._partition_qctn(qctn)
        contract_plan = self._compute_contract_plan(qctn, partitions)
        
        # print(f"[Rank {self.rank}] Distributed contraction plan computed.")
        # print(f"[Rank {self.rank}] Partitions: {partitions}")
        # print(f"[Rank {self.rank}] Contract Plan: \n{contract_plan}")

        # Step 2: Extract this worker's local partition based on rank
        local_cores = partitions[self.rank] if self.rank < len(partitions) else []
        
        # Update plan with local info
        contract_plan.local_partition_idx = self.rank
        contract_plan.local_cores = local_cores
        self._contract_plan = contract_plan
        
        # Step 3: Create local QCTN (subgraph) and load corresponding weights
        if local_cores:
            self._local_qctn = self._create_local_qctn(qctn, local_cores)
        else:
            self._local_qctn = None


        # self._print_local_qctn(self._local_qctn)
        
        
        self._is_initialized = True
        
        self._log(f"Distributed init complete: {len(local_cores)} local cores, "
                  f"{contract_plan.num_stages} stages")
        
        return contract_plan
    
    def _partition_qctn(self, qctn: 'QCTN') -> List[List[str]]:
        """
        Partition QCTN cores across workers.
        
        Uses a simple layer-based partitioning strategy:
        - Cores are ordered by their topological order
        - Cores are distributed evenly across partitions
        
        Args:
            qctn: QCTN to partition
            
        Returns:
            List of core name lists, one per partition
        """
        num_partitions = self.partition_config.num_partitions
        cores = qctn.cores  # Already sorted
        ncores = len(cores)
        
        if ncores < num_partitions:
            # More partitions than cores: some partitions will be empty
            partitions = [[cores[i]] if i < ncores else [] for i in range(num_partitions)]
        else:
            # Distribute cores evenly
            base_size = ncores // num_partitions
            remainder = ncores % num_partitions
            
            partitions = []
            idx = 0
            for i in range(num_partitions):
                size = base_size + (1 if i < remainder else 0)
                partitions.append(cores[idx:idx + size])
                idx += size
        
        self._log(f"Partitioned {ncores} cores into {num_partitions} partitions: "
                  f"{[len(p) for p in partitions]}")
        
        return partitions
    
    def _compute_contract_plan(self, qctn: 'QCTN', 
                                partitions: List[List[str]]) -> DistributedContractPlan:
        """
        Compute the hierarchical contraction plan.
        
        For n workers, we have log2(n) + 1 stages:
        - Stage 0: Local contraction within each partition
        - Stages 1 to log2(n): Hierarchical reduction
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            
        Returns:
            DistributedContractPlan
        """
        n = self.world_size
        num_reduce_stages = int(math.ceil(math.log2(n))) if n > 1 else 0
        num_stages = 1 + num_reduce_stages  # 1 for local + reduction stages
        
        stages = []
        
        # Stage 0: Local contraction
        stage0 = ContractStage(
            stage_idx=0,
            stage_type='local',
            local_cores=partitions[0] if partitions else [],  # Will be updated per-rank
        )
        stages.append(stage0)
        
        # Reduction stages
        for s in range(1, num_stages):
            # Group size doubles each stage: 2, 4, 8, ...
            group_size = 2 ** s
            
            # Determine which group this rank belongs to
            # Not needed here as this is master-side; each rank computes its own
            
            stage = ContractStage(
                stage_idx=s,
                stage_type='reduce',
                group_size=group_size,
            )
            stages.append(stage)
        
        # Compute inter-node graph edges (which partitions need to contract together)
        inter_node_graph = self._compute_inter_node_graph(qctn, partitions)
        
        plan = DistributedContractPlan(
            num_stages=num_stages,
            stages=stages,
            inter_node_graph=inter_node_graph,
        )
        
        return plan
    
    def _compute_inter_node_graph(self, qctn: 'QCTN', 
                                   partitions: List[List[str]]) -> Dict[str, Any]:
        """
        Compute the inter-node contraction graph.
        
        This determines which edges connect different partitions and how
        they should be contracted in the reduction stages.
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            
        Returns:
            Dict containing:
            - raw_cross_edges: original cross-partition edges with core names
            - cross_edges: cross-partition edges with partition+index naming
            - partition_adjacency_tables: adjacency table for each partition
            - num_partitions: number of partitions
            - partition_sizes: size of each partition
        """
        # Map each core to its partition and index within partition
        core_to_partition = {}
        core_to_partition_idx = {}
        for p_idx, cores in enumerate(partitions):
            for local_idx, core in enumerate(cores):
                core_to_partition[core] = p_idx
                core_to_partition_idx[core] = local_idx
        
        # Build partition adjacency tables
        partition_adjacency_tables = self._build_partition_adjacency_tables(
            qctn, partitions, core_to_partition, core_to_partition_idx
        )
        
        # Find edges that cross partition boundaries
        raw_cross_edges = []
        cross_edges = []
        
        for core_info in qctn.adjacency_table:
            core_name = core_info['core_name']
            core_partition = core_to_partition.get(core_name, -1)
            core_local_idx = core_to_partition_idx.get(core_name, -1)
            
            for edge in core_info['out_edge_list']:
                neighbor_name = edge['neighbor_name']
                if neighbor_name and neighbor_name in core_to_partition:
                    neighbor_partition = core_to_partition[neighbor_name]
                    neighbor_local_idx = core_to_partition_idx[neighbor_name]
                    
                    if core_partition != neighbor_partition:
                        # Raw cross edge with original core names
                        raw_cross_edges.append({
                            'from_core': core_name,
                            'to_core': neighbor_name,
                            'from_partition': core_partition,
                            'to_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                        })
                        
                        # Cross edge with partition naming (P0, P1, etc.)
                        cross_edges.append({
                            'from_core': f"P{core_partition}",
                            'to_core': f"P{neighbor_partition}",
                            'from_partition': core_partition,
                            'to_partition': neighbor_partition,
                            'from_core_idx': core_local_idx,
                            'to_core_idx': neighbor_local_idx,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            # Keep original names for reference
                            'from_core_raw': core_name,
                            'to_core_raw': neighbor_name,
                        })
        
        return {
            'raw_cross_edges': raw_cross_edges,
            'cross_edges': cross_edges,
            'partition_adjacency_tables': partition_adjacency_tables,
            'num_partitions': len(partitions),
            'partition_sizes': [len(p) for p in partitions],
        }
    
    def _build_partition_adjacency_tables(
        self, 
        qctn: 'QCTN', 
        partitions: List[List[str]],
        core_to_partition: Dict[str, int],
        core_to_partition_idx: Dict[str, int]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build adjacency table for each partition.
        
        Each partition's adjacency table contains entries for its local cores,
        with edges pointing to either:
        - Other local cores (internal edges)
        - External cores from other partitions (cross edges, marked with partition info)
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            core_to_partition: Mapping from core name to partition index
            core_to_partition_idx: Mapping from core name to index within partition
            
        Returns:
            List of adjacency tables, one per partition
        """
        partition_adjacency_tables = []
        
        for p_idx, partition_cores in enumerate(partitions):
            partition_table = []
            partition_core_set = set(partition_cores)
            
            for local_idx, core_name in enumerate(partition_cores):
                # Find original core info
                original_info = None
                for info in qctn.adjacency_table:
                    if info['core_name'] == core_name:
                        original_info = info
                        break
                
                if original_info is None:
                    continue
                
                # Build local adjacency entry using original core names
                local_entry = {
                    'core_idx': local_idx,
                    'core_name': core_name,  # Use original core name
                    'in_edge_list': [],
                    'out_edge_list': [],
                    'input_shape': original_info['input_shape'].copy(),
                    'output_shape': original_info['output_shape'].copy(),
                    'input_dim': original_info['input_dim'],
                    'output_dim': original_info['output_dim'],
                }
                
                # Process in_edge_list
                for edge in original_info['in_edge_list']:
                    neighbor_name = edge['neighbor_name']
                    if not neighbor_name:
                        # Input from circuit (external input)
                        local_entry['in_edge_list'].append({
                            'neighbor_idx': -1,
                            'neighbor_name': "",
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    elif neighbor_name in partition_core_set:
                        # Internal edge within partition
                        neighbor_local_idx = core_to_partition_idx[neighbor_name]
                        local_entry['in_edge_list'].append({
                            'neighbor_idx': neighbor_local_idx,
                            'neighbor_name': neighbor_name,  # Use original name
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    else:
                        # Cross-partition edge
                        neighbor_partition = core_to_partition.get(neighbor_name, -1)
                        neighbor_local_idx = core_to_partition_idx.get(neighbor_name, -1)
                        local_entry['in_edge_list'].append({
                            # 'neighbor_idx': neighbor_local_idx,
                            'neighbor_idx': -1,
                            'neighbor_name': neighbor_name,  # Use original name
                            'neighbor_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': True,
                        })
                
                # Process out_edge_list
                for edge in original_info['out_edge_list']:
                    neighbor_name = edge['neighbor_name']
                    if not neighbor_name:
                        # Output to circuit (external output)
                        local_entry['out_edge_list'].append({
                            'neighbor_idx': -1,
                            'neighbor_name': "",
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    elif neighbor_name in partition_core_set:
                        # Internal edge within partition
                        neighbor_local_idx = core_to_partition_idx[neighbor_name]
                        local_entry['out_edge_list'].append({
                            'neighbor_idx': neighbor_local_idx,
                            'neighbor_name': neighbor_name,  # Use original name
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    else:
                        # Cross-partition edge
                        neighbor_partition = core_to_partition.get(neighbor_name, -1)
                        neighbor_local_idx = core_to_partition_idx.get(neighbor_name, -1)
                        local_entry['out_edge_list'].append({
                            # 'neighbor_idx': neighbor_local_idx,
                            'neighbor_idx': -1,
                            'neighbor_name': neighbor_name,  # Use original name
                            'neighbor_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': True,
                        })
                
                partition_table.append(local_entry)
            
            partition_adjacency_tables.append(partition_table)
        
        return partition_adjacency_tables
    
    def _create_local_qctn(self, qctn: 'QCTN', local_cores: List[str]) -> 'QCTN':
        """
        Create a local QCTN subgraph containing only the specified cores.
        
        Each process retrieves its partition's adjacency table from the contract plan
        and extracts only the local cores' weights from the full QCTN.
        Creates a new QCTN instance with the local partition's data.
        
        Args:
            qctn: Full QCTN (with all cores initialized)
            local_cores: List of core names for this partition
            
        Returns:
            QCTN instance with:
            - adjacency_table: Local partition's adjacency table (using original core names)
            - cores_weights: Dict mapping core names to their weights
            - cores: List of core names (original naming)
        """
        from ...core.qctn import QCTN
        
        # Get partition adjacency table from contract plan
        partition_tables = self._contract_plan.inter_node_graph.get('partition_adjacency_tables', [])
        if self.rank < len(partition_tables):
            local_adjacency_table = partition_tables[self.rank]
        else:
            local_adjacency_table = []
        
        # Extract core names and weights from adjacency table
        # Now core_name is the original name
        local_weights = {}
        local_cores_list = []  # Original naming
        qubit_indices_set = set()  # Collect all qubit indices involved in this partition
        
        for entry in local_adjacency_table:
            core_name = entry.get('core_name', '')
            
            if core_name and core_name in qctn.cores_weights:
                local_weights[core_name] = qctn.cores_weights[core_name]
                local_cores_list.append(core_name)
            
            # Collect qubit indices from in_edge_list and out_edge_list
            for e in entry.get('in_edge_list', []):
                qubit_idx = e.get('qubit_idx', -1)
                if qubit_idx >= 0:
                    qubit_indices_set.add(qubit_idx)
            for e in entry.get('out_edge_list', []):
                qubit_idx = e.get('qubit_idx', -1)
                if qubit_idx >= 0:
                    qubit_indices_set.add(qubit_idx)
        
        # Sort qubit indices to maintain order
        qubit_indices = sorted(qubit_indices_set)
        
        # Create a new QCTN-like object with local partition data
        # We use object.__new__ to avoid calling __init__ which parses a graph string
        local_qctn = object.__new__(QCTN)
        
        # Set essential attributes
        local_qctn.graph = None  # No graph string for partitioned QCTN
        local_qctn.nqubits = qctn.nqubits  # Keep original nqubits for reference
        local_qctn.cores = local_cores_list  # Original core names
        local_qctn.ncores = len(local_cores_list)
        local_qctn.adjacency_table = local_adjacency_table
        local_qctn.cores_weights = local_weights
        local_qctn.backend = qctn.backend
        local_qctn.einsum_expr = None
        local_qctn._loaded_metadata = None
        
        # Store partition-specific info as extra attribute
        local_qctn.partition_idx = self.rank
        local_qctn.qubit_indices = qubit_indices  # All qubit indices involved in this partition
        
        self._log(f"Rank {self.rank}: Local QCTN created with {len(local_cores_list)} cores: "
                  f"{local_cores_list}, qubits: {qubit_indices}")
        
        return local_qctn
    
    def _print_local_qctn(self, local_qctn: 'QCTN'):
        """
        Print formatted local QCTN information.
        
        Args:
            local_qctn: Local QCTN instance
        """
        lines = [
            "=" * 60,
            f"Local QCTN - Partition {local_qctn.partition_idx} (Rank {self.rank})",
            "=" * 60,
        ]
        
        # Core names (using original names)
        lines.append(f"Cores ({local_qctn.ncores}):")
        for i, core_name in enumerate(local_qctn.cores):
            weight = local_qctn.cores_weights.get(core_name)
            if weight is not None:
                shape_str = f"shape={tuple(weight.tensor.shape)}"
            else:
                shape_str = "no weight"
            lines.append(f"  [{i}] {core_name} - {shape_str}")
        
        # Adjacency table
        lines.append("-" * 40)
        lines.append("Adjacency Table:")
        for entry in local_qctn.adjacency_table:
            core_name = entry.get('core_name', '')
            lines.append(f"  {core_name}:")
            
            # In edges
            in_edges = entry.get('in_edge_list', [])
            if in_edges:
                lines.append(f"    In edges ({len(in_edges)}):")
                for e in in_edges:
                    neighbor = e.get('neighbor_name', '')
                    is_cross = e.get('is_cross_partition', False)
                    edge_rank = e.get('edge_rank', 0)
                    qubit_idx = e.get('qubit_idx', -1)
                    cross_mark = " [CROSS]" if is_cross else ""
                    if neighbor:
                        lines.append(f"      <- {neighbor} (rank={edge_rank}, qubit={qubit_idx}){cross_mark}")
                    else:
                        lines.append(f"      <- [INPUT] (rank={edge_rank}, qubit={qubit_idx})")
            
            # Out edges
            out_edges = entry.get('out_edge_list', [])
            if out_edges:
                lines.append(f"    Out edges ({len(out_edges)}):")
                for e in out_edges:
                    neighbor = e.get('neighbor_name', '')
                    is_cross = e.get('is_cross_partition', False)
                    edge_rank = e.get('edge_rank', 0)
                    qubit_idx = e.get('qubit_idx', -1)
                    cross_mark = " [CROSS]" if is_cross else ""
                    if neighbor:
                        lines.append(f"      -> {neighbor} (rank={edge_rank}, qubit={qubit_idx}){cross_mark}")
                    else:
                        lines.append(f"      -> [OUTPUT] (rank={edge_rank}, qubit={qubit_idx})")
        
        lines.append("=" * 60)
        
        # Print all lines (each rank prints its own)
        print('\n'.join(lines))
    
    # ==================== Distributed Contraction ====================
    
    def contract_distributed(self,
                             circuit_states_list: List,
                             measure_input_list: List,
                             measure_is_matrix: bool = True) -> 'torch.Tensor':
        """
        Execute distributed tensor contraction.
        
        Follows the hierarchical contraction plan:
        1. Stage 0: Local contraction using StrategyCompiler
        2. Stages 1-log2(n): Hierarchical tensor parallel reduction
        
        Args:
            circuit_states_list: Circuit states for each qubit
            measure_input_list: Measurement matrices (Mx) for each qubit
            measure_is_matrix: Whether measure_input is matrix form
            
        Returns:
            Final contracted result tensor
        """
        if not self._is_initialized:
            raise RuntimeError("Must call init_distributed() before contract_distributed()")
        
        import torch
        
        plan = self._contract_plan
        
        # Stage 0: Local contraction
        # Extract local circuit_states and measure_input based on QCTN's adjacency table
        
        # Get INPUT qubit indices (in_edge_list where neighbor_name is empty)
        input_qubit_indices = set()
        for entry in self._local_qctn.adjacency_table:
            for e in entry.get('in_edge_list', []):
                if not e.get('neighbor_name'):  # INPUT edge
                    qubit_idx = e.get('qubit_idx', -1)
                    if qubit_idx >= 0:
                        input_qubit_indices.add(qubit_idx)
        
        # Get OUTPUT qubit indices (out_edge_list where neighbor_name is empty)
        output_qubit_indices = set()
        for entry in self._local_qctn.adjacency_table:
            for e in entry.get('out_edge_list', []):
                if not e.get('neighbor_name'):  # OUTPUT edge
                    qubit_idx = e.get('qubit_idx', -1)
                    if qubit_idx >= 0:
                        output_qubit_indices.add(qubit_idx)
        
        # Sort indices to maintain order
        input_qubit_indices = sorted(input_qubit_indices)
        output_qubit_indices = sorted(output_qubit_indices)
        
        # Extract local circuit_states_list based on INPUT qubits
        local_circuit_states_list = {i: circuit_states_list[i] for i in input_qubit_indices}
        
        # Extract local measure_input_list based on OUTPUT qubits
        local_measure_input_list = {i: measure_input_list[i] for i in output_qubit_indices}
        
        print(f"[Rank {self.rank}] Local contraction: "
              f"input_qubits={input_qubit_indices}, output_qubits={output_qubit_indices}")

        local_result = self._contract_local(local_circuit_states_list, local_measure_input_list, 
                                            measure_is_matrix)
        
        if self.world_size == 1:
            return local_result
        
        print(f"[Rank {self.rank}] Local contraction complete. Result shape: {local_result.shape}")

        # Reduction stages
        current_result = local_result
        
        for stage_idx in range(1, plan.num_stages):
            current_result = self._contract_reduce_stage(
                stage_idx, current_result
            )
        
        return current_result
    
    def _contract_local(self,
                        circuit_states_list: List,
                        measure_input_list: List,
                        measure_is_matrix: bool = True) -> 'torch.Tensor':
        """
        Execute local contraction (Stage 0).
        
        Uses the base engine's strategy compiler for optimal contraction.
        
        Args:
            circuit_states_list: Circuit states
            measure_input_list: Measurement matrices
            measure_is_matrix: Whether measure_input is matrix form
            
        Returns:
            Local contraction result
        """
        qctn = self._local_qctn
        
        # Use base engine's compiled strategy
        result = self._base_engine.contract_with_compiled_strategy(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
            measure_is_matrix=measure_is_matrix
        )
        
        return result
    
    def _contract_reduce_stage(self, stage_idx: int, 
                                local_result: 'torch.Tensor') -> 'torch.Tensor':
        """
        Execute a reduction stage using tensor parallel matrix multiplication.
        
        In stage k (1-indexed):
        - Group size = 2^k
        - Each group of 2^k ranks performs pairwise TP matrix multiplication
        - First stage: n matrices -> n/2 matrices (pairs: [0,1], [2,3], ...)
        - Second stage: 4 ranks per group, left 2 ranks have one matrix, right 2 have another
        
        Args:
            stage_idx: Which reduction stage (1-indexed)
            local_result: This worker's current result tensor
            
        Returns:
            Reduced result after this stage
        """
        import torch
        
        # Compute group membership
        group_size = 2 ** stage_idx
        my_group = self.rank // group_size
        my_position_in_group = self.rank % group_size
        group_start = my_group * group_size
        group_end = group_start + group_size
        group_ranks = list(range(group_start, min(group_end, self.world_size)))
        
        # Determine which "sub-matrix" this rank belongs to (left or right half of group)
        half_group_size = group_size // 2
        is_left_half = my_position_in_group < half_group_size
        
        # Get cross edges between adjacent partitions for this stage
        cross_edges = self._get_cross_edges_for_stage(stage_idx, my_group)
        
        # Extract the qubit indices that are contracted in this stage
        contract_qubit_indices = set()
        for edge in cross_edges:
            contract_qubit_indices.add(edge['qubit_idx'])
        contract_qubit_indices = sorted(contract_qubit_indices)
        
        print(f"[Rank {self.rank}] Stage {stage_idx}: group={my_group}, "
              f"is_left={is_left_half}, contract_qubits={contract_qubit_indices}")
        
        # Perform TP matrix multiplication
        result = self._tensor_parallel_matmul(
            local_result,
            stage_idx,
            group_ranks,
            my_position_in_group,
            is_left_half,
            contract_qubit_indices
        )
        
        return result
    
    def _get_cross_edges_for_stage(self, stage_idx: int, my_group: int) -> List[Dict]:
        """
        Get cross-partition edges relevant for this reduction stage.
        
        In stage 1: edges between partition 0-1, 2-3, etc.
        In stage 2: edges between partitions 0,1 and 2,3, etc.
        
        Args:
            stage_idx: Which reduction stage
            my_group: Which group this rank belongs to
            
        Returns:
            List of cross edges for this stage
        """
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        
        # In stage k, group size is 2^k
        # Left half partitions: [group_start, group_start + 2^(k-1) - 1]
        # Right half partitions: [group_start + 2^(k-1), group_start + 2^k - 1]
        group_size = 2 ** stage_idx
        half_size = group_size // 2
        group_start = my_group * group_size
        left_partitions = set(range(group_start, group_start + half_size))
        right_partitions = set(range(group_start + half_size, group_start + group_size))
        
        # Find edges going from left partitions to right partitions (or vice versa)
        relevant_edges = []
        for edge in all_cross_edges:
            from_p = edge['from_partition']
            to_p = edge['to_partition']
            # Edge crosses between left and right halves of this group
            if (from_p in left_partitions and to_p in right_partitions) or \
               (from_p in right_partitions and to_p in left_partitions):
                relevant_edges.append(edge)
        
        return relevant_edges
    
    def _tensor_parallel_matmul(self, 
                                 local_tensor: 'torch.Tensor',
                                 stage_idx: int,
                                 group_ranks: List[int],
                                 my_position: int,
                                 is_left_half: bool,
                                 contract_qubit_indices: List[int]) -> 'torch.Tensor':
        """
        Perform tensor parallel matrix multiplication for distributed contraction.
        
        The tensor has structure: [batch, in_dims..., out_dims...]
        where in_dims and out_dims correspond to qubit indices.
        
        For contraction:
        1. Identify which dims correspond to contract_qubit_indices
        2. Transpose to put non-contract dims first, contract dims last
        3. Reshape to [batch, N, K] where K is product of contract dims
        4. Partner matrix is [batch, M, K]
        5. Shard N across group, each computes partial [batch, N/group_size, K] @ [batch, M, K]^T
        6. AllGather to get full [batch, N, M] result
        
        Args:
            local_tensor: Local result tensor
            stage_idx: Current reduction stage
            group_ranks: Ranks in current group
            my_position: Position within group
            is_left_half: Whether this rank is in left half of group
            contract_qubit_indices: Qubit indices being contracted in this stage
            
        Returns:
            Result tensor after TP matmul
        """
        import torch
        
        group_size = len(group_ranks)
        half_group_size = group_size // 2
        
        # Print cross-edge information for this partition from _contract_plan
        my_partition = self.rank  # Assuming 1:1 mapping between rank and partition
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        
        if self.rank == 0:
            print(f"\n[Rank {self.rank}] === Cross Edges for Partition {my_partition} (Stage {stage_idx}) ===")
            print(f"  Input tensor shape: {local_tensor.shape}")
            print(f"  Contract qubit indices for this stage: {contract_qubit_indices}")
            
            # Find edges where this partition is the source (OUT edges)
            out_edges = [e for e in all_cross_edges if e['from_partition'] == my_partition]
            print(f"  OUT edges (from this partition):")
            if out_edges:
                for edge in out_edges:
                    is_contract = edge['qubit_idx'] in contract_qubit_indices
                    marker = " [CONTRACT]" if is_contract else ""
                    print(f"    -> P{edge['to_partition']}: qubit={edge['qubit_idx']}, "
                        f"edge_rank={edge['edge_rank']}, "
                        f"from_core={edge.get('from_partition', 'N/A')}, "
                        f"to_core={edge.get('to_partition', 'N/A')}{marker}")
            else:
                print(f"    (none)")
            
            # Find edges where this partition is the destination (IN edges)
            in_edges = [e for e in all_cross_edges if e['to_partition'] == my_partition]
            print(f"  IN edges (to this partition):")
            if in_edges:
                for edge in in_edges:
                    is_contract = edge['qubit_idx'] in contract_qubit_indices
                    marker = " [CONTRACT]" if is_contract else ""
                    print(f"    <- P{edge['from_partition']}: qubit={edge['qubit_idx']}, "
                        f"edge_rank={edge['edge_rank']}, "
                        f"from_core={edge.get('from_partition', 'N/A')}, "
                        f"to_core={edge.get('to_partition', 'N/A')}{marker}")
            else:
                print(f"    (none)")
            print(f"[Rank {self.rank}] =====================================\n")
        
        # Get qubit dimension info from local_qctn
        # The tensor dims after batch are ordered based on cross-partition edges
        
        batch_size = local_tensor.shape[0]
        remaining_dims = list(local_tensor.shape[1:])
        n_dims = len(remaining_dims)
        
        # =====================================================================
        # Determine contract_dim_indices and non_contract_dim_indices
        # based on cross_edges from _contract_plan
        # =====================================================================
        
        # For stage k, group_size = 2^k
        # My group contains partitions: [group_start, group_start + group_size - 1]
        # Left half: [group_start, group_start + half_group_size - 1]
        # Right half: [group_start + half_group_size, group_start + group_size - 1]
        my_partition = self.rank
        my_group_idx = my_partition // group_size
        group_start = my_group_idx * group_size
        
        # Compute my_partitions (partitions in same half as me) and partner_partitions
        if is_left_half:
            my_partitions = set(range(group_start, group_start + half_group_size))
            partner_partitions = set(range(group_start + half_group_size, group_start + group_size))
        else:
            my_partitions = set(range(group_start + half_group_size, group_start + group_size))
            partner_partitions = set(range(group_start, group_start + half_group_size))
        
        print(f"[Rank {self.rank}] Stage {stage_idx}: my_partitions={sorted(my_partitions)}, "
              f"partner_partitions={sorted(partner_partitions)}")
        
        # Get all cross edges from contract plan
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        
        # Find edges relevant to current partition
        # OUT edges: from this partition to others
        # IN edges: from others to this partition
        my_out_edges = [e for e in all_cross_edges if e['from_partition'] in my_partitions]
        my_in_edges = [e for e in all_cross_edges if e['to_partition'] in my_partitions]
        
        # # Build list of external edges in dimension order
        # # After local contraction, remaining dims correspond to:
        # #   - IN edges that are cross-partition (in order of qubit_idx)
        # #   - OUT edges that are cross-partition (in order of qubit_idx)
        # # Also include INPUT/OUTPUT edges (no neighbor)
        
        # # Get partition adjacency table for this partition
        # partition_tables = self._contract_plan.inter_node_graph.get('partition_adjacency_tables', [])
        # my_adj_table = partition_tables[my_partition] if my_partition < len(partition_tables) else []
        

        # print(f"[Rank {self.rank}] my_in_edges {my_in_edges} my_out_edges {my_out_edges}")
        
        # # Debug: print partition adjacency table
        # print(f"[Rank {self.rank}] my_adj_table has {len(my_adj_table)} entries:")
        # for entry in my_adj_table:
        #     core_name = entry.get('core_name', 'unknown')
        #     in_edges = entry.get('in_edge_list', [])
        #     out_edges = entry.get('out_edge_list', [])
        #     print(f"  Core '{core_name}':")
        #     for e in in_edges:
        #         print(f"    IN: neighbor={e.get('neighbor_name')}, qubit={e.get('qubit_idx')}, "
        #               f"is_cross={e.get('is_cross_partition')}, neighbor_part={e.get('neighbor_partition', 'N/A')}")
        #     for e in out_edges:
        #         print(f"    OUT: neighbor={e.get('neighbor_name')}, qubit={e.get('qubit_idx')}, "
        #               f"is_cross={e.get('is_cross_partition')}, neighbor_part={e.get('neighbor_partition', 'N/A')}")
        
        # # Collect all external edges (ordered by qubit_idx for consistency)
        # # An "external edge" is either:
        # #   1. Cross-partition edge (neighbor in different partition)
        # #   2. INPUT/OUTPUT edge (no neighbor)
        # external_edges = []
        
        # for entry in my_adj_table:
        #     # Process IN edges
        #     for e in entry.get('in_edge_list', []):
        #         # print(f"[Rank {self.rank}] Found external IN edge: qubit={e.get('qubit_idx')}, "
        #         #         f"neighbor={e.get('neighbor_name')}, is_cross={e.get('is_cross_partition')}, "
        #         #         f"neighbor_part={e.get('neighbor_partition', 'N/A')}")
        #         # if e.get('is_cross_partition', False) or not e.get('neighbor_name'):
        #         if e.get('is_cross_partition', False):
        #             external_edges.append({
        #                 'type': 'in',
        #                 'qubit_idx': e.get('qubit_idx', -1),
        #                 'edge_rank': e.get('edge_rank', 0),
        #                 'neighbor_partition': e.get('neighbor_partition', -1),
        #                 'is_cross_partition': e.get('is_cross_partition', False),
        #             })
        #     # Process OUT edges
        #     for e in entry.get('out_edge_list', []):
        #         # print(f"[Rank {self.rank}] Found external OUT edge: qubit={e.get('qubit_idx')}, "
        #         #         f"neighbor={e.get('neighbor_name')}, is_cross={e.get('is_cross_partition')}, "
        #         #         f"neighbor_part={e.get('neighbor_partition', 'N/A')}")
        #         # if e.get('is_cross_partition', False) or not e.get('neighbor_name'):
        #         if e.get('is_cross_partition', False):
        #             external_edges.append({
        #                 'type': 'out',
        #                 'qubit_idx': e.get('qubit_idx', -1),
        #                 'edge_rank': e.get('edge_rank', 0),
        #                 'neighbor_partition': e.get('neighbor_partition', -1),
        #                 'is_cross_partition': e.get('is_cross_partition', False),
        #             })
        
        # # Sort by qubit_idx to match tensor dimension order
        # external_edges.sort(key=lambda x: (x['qubit_idx'], 0 if x['type'] == 'in' else 1))
        
        # print(f"[Rank {self.rank}] External edges ({len(external_edges)} total, {n_dims} tensor dims):")
        # for i, e in enumerate(external_edges):
        #     print(f"  dim[{i}]: {e['type'].upper()} qubit={e['qubit_idx']}, "
        #           f"rank={e['edge_rank']}, neighbor_part={e['neighbor_partition']}, "
        #           f"is_cross={e['is_cross_partition']}")
        
        # Now determine contract_dim_indices and non_contract_dim_indices
        # Contract dims: edges connecting my_partitions to partner_partitions
        contract_dim_indices = []
        non_contract_dim_indices = []
        
        # for dim_idx, edge in enumerate(external_edges):
        #     if dim_idx >= n_dims:
        #         break  # Safety check
            
        #     neighbor_part = edge.get('neighbor_partition', -1)
        #     is_cross = edge.get('is_cross_partition', False)
            
        #     # An edge is a contract edge if:
        #     # - It's a cross-partition edge AND
        #     # - The neighbor partition is in partner_partitions
        #     if is_cross and neighbor_part in partner_partitions:
        #         contract_dim_indices.append(dim_idx)
        #     else:
        #         non_contract_dim_indices.append(dim_idx)
        
        # sort my_in_edges and my_out_edge with qubit_idx to match tensor dim order
        my_in_edges.sort(key=lambda x: x['qubit_idx'])
        my_out_edges.sort(key=lambda x: x['qubit_idx'])

        # print(f"[Rank {self.rank}] Sorted my_in_edges by qubit_idx:")
        # for i, e in enumerate(my_in_edges):
        #     print(f"[Rank {self.rank}]  dim[{i}]: IN qubit={e['qubit_idx']}, to_part={e['to_partition']}, from_part={e['from_partition']}")
        # print(f"[Rank {self.rank}] Sorted my_out_edges by qubit_idx:")
        # for i, e in enumerate(my_out_edges):
        #     print(f"[Rank {self.rank}]  dim[{i}]: OUT qubit={e['qubit_idx']}, to_part={e['to_partition']}, from_part={e['from_partition']}")

        for dim_idx, in_edge in enumerate(my_in_edges):
            neighbor_part = in_edge.get('from_partition', -1)
            if neighbor_part in partner_partitions:
                contract_dim_indices.append(dim_idx)
                contract_dim_indices.append(n_dims - dim_idx - 1)
            else:
                non_contract_dim_indices.append(dim_idx)
                non_contract_dim_indices.append(n_dims - dim_idx - 1)
        offset = len(my_in_edges)
        for dim_idx, out_edge in enumerate(my_out_edges):
            neighbor_part = out_edge.get('to_partition', -1)
            if neighbor_part in partner_partitions:
                contract_dim_indices.append(offset + dim_idx)
                contract_dim_indices.append(n_dims - (offset + dim_idx) - 1)
            else:
                non_contract_dim_indices.append(offset + dim_idx)
                non_contract_dim_indices.append(n_dims - (offset + dim_idx) - 1)
        
        print(f"[Rank {self.rank}] Contract dim indices: {contract_dim_indices}")
        print(f"[Rank {self.rank}] Non-contract dim indices: {non_contract_dim_indices}")
        
        # If no contract dims identified, fallback
        if not contract_dim_indices:
            print(f"[Rank {self.rank}] Warning: no contract dims found, using identity")
            return local_tensor

        # Transpose: [batch, non_contract_dims..., contract_dims...]
        perm = [0] + [d + 1 for d in non_contract_dim_indices] + [d + 1 for d in contract_dim_indices]
        transposed = local_tensor.permute(perm)
        
        # Compute shapes
        non_contract_shape = [remaining_dims[d] for d in non_contract_dim_indices]
        contract_shape = [remaining_dims[d] for d in contract_dim_indices]
        
        # If no non-contract dims, N = 1; if no contract dims, K = 1
        N = 1
        for s in non_contract_shape:
            N *= s
        K = 1
        for s in contract_shape:
            K *= s
        
        # Reshape to [batch, N, K] - N=1 if no non-contract dims, K=1 if no contract dims
        reshaped = transposed.reshape(batch_size, N, K)
        
        # Synchronize reshaped tensor shapes across all ranks
        # Each rank broadcasts its [batch_size, N, K] shape
        # Result: all_shapes is a [world_size, 3] tensor
        my_shape_tensor = torch.tensor([batch_size, N, K], dtype=torch.long, device=local_tensor.device)
        all_shapes_list = self.comm.allgather(my_shape_tensor)
        all_shapes = torch.stack(all_shapes_list, dim=0)  # [world_size, 3]
        
        print(f"[Rank {self.rank}] Synchronized reshaped shapes across all ranks: {all_shapes}")
        
        print(f"[Rank {self.rank}] TP matmul: tensor shape {local_tensor.shape} -> "
              f"reshaped {reshaped.shape}, N={N}, K={K}, is_left={is_left_half}")
        
        # =====================================================================
        # TP Matrix Multiplication with K-dimension sharding
        # =====================================================================
        
        # Identify left and right partitions for this stage
        left_partition_ranks = group_ranks[:half_group_size]
        right_partition_ranks = group_ranks[half_group_size:]
        
        # Get shapes from first node of each partition (they hold the authoritative shape)
        left_first_rank = left_partition_ranks[0]
        right_first_rank = right_partition_ranks[0]
        
        left_shape = all_shapes[left_first_rank]  # [batch, N_left, K_left]
        right_shape = all_shapes[right_first_rank]  # [batch, N_right, K_right]
        
        # For matmul: left [B, N, K] @ right [B, M, K]^T = [B, N, M]
        # K dimension must match and will be sharded across all nodes
        K_total = left_shape[2].item()  # K dimension to shard
        N_left = left_shape[1].item()
        M_right = right_shape[1].item()  # This is M (right's N dimension)
        
        print(f"[Rank {self.rank}] TP setup: left_partition={left_partition_ranks}, "
              f"right_partition={right_partition_ranks}, K_total={K_total}, "
              f"N_left={N_left}, M_right={M_right}")
        
        # Compute K sharding across all nodes in the group
        # Total nodes = group_size, shard K evenly with remainder to first nodes
        total_nodes = group_size
        base_k = K_total // total_nodes
        remainder_k = K_total % total_nodes
        
        # Compute start/end indices for each rank's K shard
        k_shards = []
        k_start = 0
        for i in range(total_nodes):
            k_size = base_k + (1 if i < remainder_k else 0)
            k_shards.append((k_start, k_start + k_size))
            k_start += k_size
        
        my_global_position = my_position  # Position within group
        my_k_start, my_k_end = k_shards[my_global_position]
        my_k_size = my_k_end - my_k_start
        
        print(f"[Rank {self.rank}] K sharding: position={my_global_position}, "
              f"k_range=[{my_k_start}:{my_k_end}], k_size={my_k_size}")
        print(f"[Rank {self.rank}] All K shards: {k_shards}")
        
        # Determine partner rank for data exchange
        # Left node i pairs with right node i
        # e.g., rank 0 <-> rank 2, rank 1 <-> rank 3
        if is_left_half:
            partner_rank = right_partition_ranks[my_position]  # my_position is position in left half
            partner_position = my_position + half_group_size  # partner's position in group
        else:
            partner_rank = left_partition_ranks[my_position - half_group_size]
            partner_position = my_position - half_group_size
        
        # Get partner's K shard range
        partner_k_start, partner_k_end = k_shards[partner_position]
        
        print(f"[Rank {self.rank}] Partner for data exchange: rank {partner_rank}, "
              f"partner_position={partner_position}, partner_k_range=[{partner_k_start}:{partner_k_end}]")
        
        # Extract my K shard from local tensor (for my own computation)
        my_k_shard = reshaped[:, :, my_k_start:my_k_end]  # [B, N, K_shard] or [B, M, K_shard]
        
        # Extract the K shard to send to partner (using partner's K range)
        k_shard_to_partner = reshaped[:, :, partner_k_start:partner_k_end]  # Partner needs this shard
        
        print(f"[Rank {self.rank}] My K shard shape: {my_k_shard.shape}, "
              f"K shard to send to partner: {k_shard_to_partner.shape}")
        
        # Exchange K shards with partner
        # I send my tensor sliced at partner's K range, partner sends their tensor sliced at my K range
        partner_k_shard = self._exchange_tensor_with_partner(k_shard_to_partner, partner_rank)
        
        print(f"[Rank {self.rank}] Partner K shard received shape: {partner_k_shard.shape}")
        
        # Perform partial matrix multiplication
        # Left: [B, N, K_shard] @ [B, K_shard, M] -> [B, N, M]
        # Right: [B, M, K_shard] @ [B, K_shard, N] -> [B, M, N]
        if is_left_half:
            # I'm in left partition, I have left[B, N, K_shard], partner has right[B, M, K_shard]
            # Compute: left @ right^T = [B, N, K_shard] @ [B, K_shard, M] = [B, N, M]
            partner_T = partner_k_shard.transpose(1, 2)  # [B, K_shard, M]
            partial_result = torch.bmm(my_k_shard, partner_T)  # [B, N, M]
        else:
            # I'm in right partition, I have right[B, M, K_shard], partner has left[B, N, K_shard]
            # Compute: left @ right^T = [B, N, K_shard] @ [B, K_shard, M] = [B, N, M]
            # partner has [B, N, K_shard], I have [B, M, K_shard]
            my_T = my_k_shard.transpose(1, 2)  # [B, K_shard, M]
            partial_result = torch.bmm(partner_k_shard, my_T)  # [B, N, M]
        
        print(f"[Rank {self.rank}] Partial result shape: {partial_result.shape}")
        
        # AllReduce within group to sum up partial results
        # Each node computed a partial [B, N, M] from its K shard
        # Sum them up to get full [B, N, M]
        full_result = self.comm.allreduce(partial_result, op=ReduceOp.SUM)
        
        print(f"[Rank {self.rank}] Full result after allreduce: {full_result.shape}")
        
        # Reshape back to multi-dimensional tensor
        # New shape: [batch, non_contract_dims..., M]
        new_shape = [batch_size] + non_contract_shape + [M_right]
        result = full_result.reshape(new_shape)
        
        print(f"[Rank {self.rank}] Final result shape: {result.shape}")
        
        return result
    
    def _gather_shards_in_subgroup(self, my_shard: 'torch.Tensor', sub_group_ranks: List[int]) -> List['torch.Tensor']:
        """
        Gather shards from all ranks in a sub-group using P2P communication.
        
        Each rank sends its shard to all other ranks in the sub-group and receives
        shards from all other ranks.
        
        Args:
            my_shard: This rank's shard tensor
            sub_group_ranks: List of ranks in the sub-group
            
        Returns:
            List of shards from all ranks in sub-group (in rank order)
        """
        import torch
        
        shards = [None] * len(sub_group_ranks)
        my_position = sub_group_ranks.index(self.rank)
        shards[my_position] = my_shard
        
        # Exchange with all other ranks in sub-group
        for i, other_rank in enumerate(sub_group_ranks):
            if other_rank == self.rank:
                continue
            
            # Exchange shard with other_rank using P2P
            other_shard = self._exchange_tensor_with_partner(my_shard, other_rank)
            shards[i] = other_shard
        
        return shards
    
    def _exchange_tensor_with_partner(self, tensor: 'torch.Tensor', partner_rank: int) -> 'torch.Tensor':
        """
        Exchange tensor with partner rank using point-to-point send/recv.
        
        Args:
            tensor: Tensor to send
            partner_rank: Rank of partner
            
        Returns:
            Tensor received from partner
        """
        import torch
        
        # Step 1: Exchange shapes first so we know how to allocate receive buffer
        # Send our shape to partner, receive partner's shape
        my_shape = torch.tensor(tensor.shape, dtype=torch.long, device=tensor.device)
        n_dims = torch.tensor([len(tensor.shape)], dtype=torch.long, device=tensor.device)
        
        # Exchange number of dimensions first
        partner_n_dims = torch.zeros(1, dtype=torch.long, device=tensor.device)
        
        # Use sendrecv pattern: lower rank sends first to avoid deadlock
        tag_ndims = 100
        tag_shape = 101
        tag_data = 102
        
        if self.rank < partner_rank:
            # I send first, then receive
            self.comm.send(n_dims, partner_rank, tag=tag_ndims)
            self.comm.recv(partner_rank, tag=tag_ndims, tensor=partner_n_dims)
        else:
            # I receive first, then send
            self.comm.recv(partner_rank, tag=tag_ndims, tensor=partner_n_dims)
            self.comm.send(n_dims, partner_rank, tag=tag_ndims)
        
        # Exchange shapes
        partner_shape_tensor = torch.zeros(partner_n_dims.item(), dtype=torch.long, device=tensor.device)
        
        if self.rank < partner_rank:
            self.comm.send(my_shape, partner_rank, tag=tag_shape)
            self.comm.recv(partner_rank, tag=tag_shape, tensor=partner_shape_tensor)
        else:
            self.comm.recv(partner_rank, tag=tag_shape, tensor=partner_shape_tensor)
            self.comm.send(my_shape, partner_rank, tag=tag_shape)
        
        partner_shape = tuple(partner_shape_tensor.tolist())
        
        # Step 2: Allocate receive buffer and exchange data
        recv_buffer = torch.zeros(partner_shape, dtype=tensor.dtype, device=tensor.device)
        
        if self.rank < partner_rank:
            self.comm.send(tensor.contiguous(), partner_rank, tag=tag_data)
            self.comm.recv(partner_rank, tag=tag_data, tensor=recv_buffer)
        else:
            self.comm.recv(partner_rank, tag=tag_data, tensor=recv_buffer)
            self.comm.send(tensor.contiguous(), partner_rank, tag=tag_data)
        
        return recv_buffer
    
    # ==================== Backward Compatibility ====================
    
    def contract_with_compiled_strategy(self, qctn, circuit_states_list, 
                                         measure_input_list=None, **kwargs):
        """
        Standard tensor contraction (non-distributed).
        
        Proxies to base engine.
        """
        return self._base_engine.contract_with_compiled_strategy(
            qctn, 
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
            **kwargs
        )
    
    def contract_with_compiled_strategy_for_gradient(self, qctn, circuit_states_list=None, 
                                                      measure_input_list=None, **kwargs):
        """
        Tensor contraction with gradient computation.
        
        Proxies to base engine.
        """
        return self._base_engine.contract_with_compiled_strategy_for_gradient(
            qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list,
            **kwargs
        )
    
    # Backward compatibility aliases
    @property
    def mpi(self):
        """Alias for comm (backward compatibility)."""
        return self.comm
    
    @property
    def ctx(self):
        """Get distributed context."""
        return self.comm.get_context()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
