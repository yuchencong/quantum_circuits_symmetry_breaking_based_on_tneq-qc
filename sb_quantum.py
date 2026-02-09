import torch
import random
import math
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_graph import TNGraph
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.optim import stiefel_optimizer_complex
from typing import List, Tuple
import opt_einsum as oe


class CircuitPruner:
    """
    Prune quantum circuits by removing cores that maintain fidelity above threshold.
    """
    
    def __init__(self, engine: EngineSiamese, fidelity_threshold: float = 0.95):
        """
        Args:
            engine: EngineSiamese instance for circuit contraction
            fidelity_threshold: Minimum fidelity to maintain (0-1)
        """
        self.engine = engine
        self.fidelity_threshold = fidelity_threshold
        self.backend = engine.backend
    
    def generate_target_data(self, qctn: QCTN, mask_ratio: float = 0.3) -> Tuple[QCTN, List[str]]:
        """
        Generate target data by randomly masking cores from the template circuit.
        
        Args:
            qctn: Template QCTN
            mask_ratio: Ratio of cores to mask (0.0 - 1.0)
            
        Returns:
            (target_qctn, masked_cores): Target circuit with masked cores and list of masked core names
        """
        # Create a copy of the QCTN
        target_qctn = QCTN(qctn.graph, backend=self.backend)
        
        # Deep copy all weights - IMPORTANT: must clone tensors to avoid sharing references
        for core in target_qctn.cores:
            if core in qctn.cores_weights:
                # Clone the tensor to create a separate copy (cores_weights now stores tensors directly)
                target_qctn.cores_weights[core] = qctn.cores_weights[core].clone().detach()
        
        # Randomly select cores to mask
        num_cores_to_mask = int(len(qctn.cores) * mask_ratio)
        masked_cores = random.sample(qctn.cores, num_cores_to_mask)
        
        print(f"[Data Generation] Masking {num_cores_to_mask} cores out of {len(qctn.cores)} ({mask_ratio*100:.1f}%)")
        print(f"[Data Generation] Masked cores: {masked_cores}")
        
        # Replace masked cores with identity tensors
        for core_name in masked_cores:
            core_tensor = qctn.cores_weights[core_name]
            core_shape = core_tensor.shape
            
            # Create identity tensor
            if len(core_shape) == 2:
                identity = torch.eye(core_shape[0], core_shape[1], device=self.backend.backend_info.device)
            elif len(core_shape) == 3:
                min_dim = min(core_shape)
                identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
                for i in range(min_dim):
                    identity[i, i, i] = 1.0
            elif len(core_shape) == 4:
                min_dim = min(core_shape)
                identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
                for i in range(min_dim):
                    identity[i, i, i, i] = 1.0
            else:
                identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
                min_dim = min(core_shape)
                for i in range(min_dim):
                    idx = tuple([i] * len(core_shape))
                    identity[idx] = 1.0
            
            target_qctn.cores_weights[core_name] = identity
        
        return target_qctn, masked_cores
    
    def visualize_masked_circuit(self, qctn: QCTN, masked_cores: List[str]) -> str:
        """
        Visualize the circuit with masked cores shown as 'X'.
        
        Args:
            qctn: QCTN circuit
            masked_cores: List of masked core names
            
        Returns:
            String representation of the circuit with masked cores marked
        """
        # Create a modified graph string where masked cores are replaced with 'X'
        graph_str = qctn.graph
        for core_name in masked_cores:
            # Replace the core character with 'X'
            graph_str = graph_str.replace(core_name, '█')
        
        return graph_str
    
    def compute_fidelity(self, qctn1: QCTN, qctn2: QCTN) -> float:
        # Generate identity measure_input_list based on number of qubits
        # Each qubit gets a 2x2 identity matrix (for qubit dimension 2)
        num_qubits = len(qctn1.qubit_indices)  # Number of qubits
        device = self.backend.backend_info.device
        measure_input_list = [torch.eye(2, device=device) for _ in range(num_qubits)]
        
        contraction_result = self.engine.contract_with_compiled_strategy(
            qctn1,
            circuit_states_list=None,
            measure_input_list=measure_input_list,
            right_qctn=qctn2,
        )
        contraction_result = oe.contract("abcdabcd ->", contraction_result)
        fidelity = contraction_result / 2**num_qubits
        # import pdb; pdb.set_trace()
        
        return fidelity
    
    def try_remove_core(self, qctn: QCTN, core_name: str) -> Tuple[QCTN, bool]:
        """
        Try to "remove" a core by replacing it with an identity tensor.
        This makes the core transparent without changing graph structure or dimensions.
        
        Args:
            qctn: Original QCTN
            core_name: Name of core to remove
            
        Returns:
            (new_qctn, success): New QCTN and whether removal was successful
        """
        print(f"    [DEBUG] Trying to remove core: {core_name}")
        if core_name not in qctn.cores:
            print(f"    [DEBUG] Core '{core_name}' not in qctn.cores")
            return None, False
        
        # Create a copy of the QCTN with the same graph
        new_qctn = QCTN(qctn.graph, backend=self.backend)
        
        # Deep copy all weights - IMPORTANT: must clone tensors to avoid sharing references
        for core in new_qctn.cores:
            if core in qctn.cores_weights:
                # Clone the tensor to create a separate copy
                cloned_tensor = qctn.cores_weights[core].clone().detach()
                new_qctn.cores_weights[core] = cloned_tensor
        
        # Replace the target core with an identity tensor
        core_tensor = qctn.cores_weights[core_name]
        core_shape = core_tensor.shape
        print(f"    [DEBUG] Core '{core_name}' shape: {core_shape}")
        
        # Create identity tensor with the same shape
        # For a tensor with shape (d1, d2, d3, ...), create an identity-like tensor
        # that acts as identity for the contraction
        if len(core_shape) == 2:
            # Matrix case: use identity matrix
            identity = torch.eye(core_shape[0], core_shape[1], device=self.backend.backend_info.device)
        elif len(core_shape) == 3:
            # 3D tensor: create diagonal tensor
            min_dim = min(core_shape)
            identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
            for i in range(min_dim):
                identity[i, i, i] = 1.0
        elif len(core_shape) == 4:
            # 4D tensor: create diagonal tensor
            min_dim = min(core_shape)
            identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
            for i in range(min_dim):
                identity[i, i, i, i] = 1.0
        else:
            # General case: create diagonal tensor along first dimensions
            identity = torch.zeros(core_shape, device=self.backend.backend_info.device)
            min_dim = min(core_shape)
            for i in range(min_dim):
                idx = tuple([i] * len(core_shape))
                identity[idx] = 1.0
        
        # Replace the core's weight with identity
        new_qctn.cores_weights[core_name] = identity
        
        print(f"    [DEBUG] Replaced core '{core_name}' with identity tensor")
        
        return new_qctn, True
    
    def optimize_cores(self, qctn: QCTN, target_qctn: QCTN, 
                      removed_cores_set: set = None,
                      learning_rate: float = 0.01, num_steps: int = 10) -> QCTN:
        """
        Optimize the cores of qctn to match target_qctn using gradient descent.
        
        IMPORTANT: Cores that have been removed (in removed_cores_set) are kept as identity.
        Only the REMAINING cores (not removed) are optimized to compensate.
        
        Args:
            qctn: QCTN to optimize
            target_qctn: Target QCTN to match
            removed_cores_set: Set of core names that have been removed (kept as identity, NOT optimized).
            learning_rate: Learning rate for optimization
            num_steps: Number of gradient descent steps
            
        Returns:
            Optimized QCTN
        """
        if removed_cores_set is None:
            removed_cores_set = set()
            
        from tneq_qc.backends.copteinsum import ContractorOptEinsum
        
        optimized_qctn = QCTN(qctn.graph, backend=self.backend)
        
        # Copy tensors - only make NON-removed cores trainable
        trainable_params = []
        for core_name in qctn.cores:
            tensor = qctn.cores_weights[core_name].clone().detach().float()
            
            # Only make NON-removed cores trainable (removed cores stay as identity)
            if core_name not in removed_cores_set:
                tensor.requires_grad = True
                trainable_params.append(tensor)
            else:
                tensor.requires_grad = False  # Removed cores stay fixed as identity
            
            optimized_qctn.cores_weights[core_name] = tensor
        
        # If no trainable params, return as is
        if not trainable_params:
            print(f"      [Optimization] No trainable cores, skipping optimization")
            return optimized_qctn
            
        # Create optimizer only for trainable params
        optimizer = stiefel_optimizer_complex.SGDG(trainable_params, lr=learning_rate)
        
        # Create a detached copy of target tensors for comparison (don't modify the original target_qctn)
        target_tensors = {}
        for c in target_qctn.cores:
            target_tensors[c] = target_qctn.cores_weights[c].detach()
        
        # Optimize
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 计算fidelity: 所有core的cosine similarity
            total_inner_product = 0.0
            total_norm1_sq = 0.0
            total_norm2_sq = 0.0
            
            for core_name in optimized_qctn.cores:
                if core_name in target_tensors:
                    t1 = optimized_qctn.cores_weights[core_name]
                    t2 = target_tensors[core_name]  # Use detached copy
                    
                    inner_prod = torch.sum(t1 * t2)
                    norm1_sq = torch.sum(t1 * t1)
                    norm2_sq = torch.sum(t2 * t2)
                    
                    total_inner_product = total_inner_product + inner_prod
                    total_norm1_sq = total_norm1_sq + norm1_sq
                    total_norm2_sq = total_norm2_sq + norm2_sq
            
            # Cosine similarity
            fidelity = self.compute_fidelity(optimized_qctn, target_qctn)
            # fidelity = torch.clamp(fidelity, 0.0, 1.0)
            
            loss = 1.0 - fidelity
            
            loss.backward()
            optimizer.step()
            
            if step % max(1, num_steps // 5) == 0:
                print(f"      [Optimization] Step {step + 1}/{num_steps}, Loss: {loss.item():.6f}, Fidelity: {fidelity.item():.6f}")
        
        # Detach and finalize
        for core_name in optimized_qctn.cores:
            # Get the optimized tensor and detach it
            optimized_tensor = optimized_qctn.cores_weights[core_name].detach()
            optimized_qctn.cores_weights[core_name] = optimized_tensor
        
        return optimized_qctn
    
    def prune(self, template_qctn: QCTN, target_qctn: QCTN,
              max_iterations: int = 100) -> Tuple[QCTN, List[str]]:
        """
        Iteratively prune the circuit to match target data.
        
        Uses greedy strategy WITH optimization:
        - Try removing each core and optimize remaining cores to learn target
        - Remove the core that gives highest fidelity after optimization, if above threshold
        - Repeat until no more cores can be removed
        
        Args:
            template_qctn: Template QCTN to start pruning from
            target_qctn: Target data (masked template) to match
            max_iterations: Maximum pruning iterations
            
        Returns:
            (pruned_qctn, removed_cores): Final pruned QCTN and list of removed cores
        """
        # CRITICAL: Create a deep copy of template to avoid modifying original
        current_qctn = QCTN(template_qctn.graph, backend=self.backend)
        for core in current_qctn.cores:
            if core in template_qctn.cores_weights:
                cloned_tensor = template_qctn.cores_weights[core].clone().detach()
                current_qctn.cores_weights[core] = cloned_tensor
        
        removed_cores = []
        removed_cores_set = set()  # Track removed cores to avoid duplicate removal
        
        for iteration in range(max_iterations):
            print(f"\n=== Pruning Iteration {iteration + 1} ===")
            print(f"Current number of active cores: {len(current_qctn.cores) - len(removed_cores_set)}")
            print(f"Already removed: {removed_cores}")
            
            # Compute baseline fidelity (current circuit vs target)
            baseline_fidelity = self.compute_fidelity(current_qctn, target_qctn)
            print(f"Baseline fidelity (vs target): {baseline_fidelity:.6f}")
            
            # Early stopping: if current fidelity is very high, we've already matched target well
            # Disabled for now to see full pruning behavior
            # if baseline_fidelity >= 0.999:
            #     print(f"\n✓ Fidelity is already very high ({baseline_fidelity:.6f} >= 0.999), stopping pruning.")
            #     break
            
            # Get list of cores to try (not yet removed)
            cores_to_try = [c for c in current_qctn.cores if c not in removed_cores_set]
            print(f"Trying {len(cores_to_try)} cores...")
            
            # Find the best core to remove (one that gives highest fidelity after optimization)
            best_core = None
            best_fidelity = -1.0
            best_pruned_qctn = None
            
            for core_name in cores_to_try:
                print(f"  Trying to remove core: {core_name}")
                
                pruned_qctn, success = self.try_remove_core(current_qctn, core_name)
                
                if not success or pruned_qctn is None:
                    print(f"    -> Failed to remove")
                    continue
                
                # Compute fidelity before optimization
                fidelity_before = self.compute_fidelity(pruned_qctn, target_qctn)
                print(f"    -> Fidelity before optimization: {fidelity_before:.6f}")
                
                # Create a tentative removed set including this core
                tentative_removed = removed_cores_set | {core_name}
                
                # Optimize the REMAINING (non-removed) cores to compensate
                print(f"    -> Optimizing remaining cores (excluding {tentative_removed})...")
                pruned_qctn_optimized = self.optimize_cores(pruned_qctn, target_qctn, 
                                                           removed_cores_set=tentative_removed,
                                                           learning_rate=0.01, num_steps=500)
                
                # Compute fidelity after optimization
                fidelity_after = self.compute_fidelity(pruned_qctn_optimized, target_qctn)
                print(f"    -> Fidelity after optimization: {fidelity_after:.6f}")
                
                # Track best candidate
                if fidelity_after > best_fidelity:
                    best_fidelity = fidelity_after
                    best_core = core_name
                    best_pruned_qctn = pruned_qctn_optimized
            
            # Check if best removal meets threshold
            if best_core is not None and best_fidelity >= self.fidelity_threshold:
                print(f"  ✓ Removing core '{best_core}' (fidelity {best_fidelity:.6f} >= {self.fidelity_threshold})")
                current_qctn = best_pruned_qctn
                removed_cores.append(best_core)
                removed_cores_set.add(best_core)
            else:
                print(f"\n✗ No more cores can be removed while maintaining fidelity >= {self.fidelity_threshold}")
                if best_core:
                    print(f"   Best candidate was '{best_core}' with fidelity {best_fidelity:.6f}")
                break
        
        print(f"\n=== Pruning Complete ===")
        print(f"Total removed cores: {len(removed_cores)}")
        print(f"Removed cores: {removed_cores}")
        print(f"Remaining active cores: {len(current_qctn.cores) - len(removed_cores_set)}")
        
        return current_qctn, removed_cores


# Usage example
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize backend and engine
    backend = BackendFactory.create_backend('pytorch', device='cpu')
    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=100)
    
    # Generate template circuit (mother board)
    from tneq_qc.core.qctn import QCTNHelper
    n_qubits = 4
    qctn_graph = QCTNHelper.generate_example_graph(n=n_qubits, graph_type="wall", dim_char='2')
    
    print(f"[Template] Generated {n_qubits}-qubit wall circuit")
    print(f"[Template] Graph:\n{qctn_graph}")
    
    template_qctn = QCTN(qctn_graph, backend=backend)
    print(f"[Template] Total cores: {len(template_qctn.cores)}")
    print(f"[Template] Cores: {template_qctn.cores}")
    
    # Create pruner
    mask_ratio = 0.5  # Hyperparameter: mask 60% of cores
    fidelity_threshold = 0.999  # High threshold - only remove if remaining cores can fully compensate
    
    pruner = CircuitPruner(engine, fidelity_threshold=fidelity_threshold)
    
    # Generate target data by masking template
    print(f"\n{'='*60}")
    print("GENERATING TARGET DATA")
    print(f"{'='*60}")
    target_qctn, masked_cores = pruner.generate_target_data(template_qctn, mask_ratio=mask_ratio)
    
    # Visualize masked circuit
    masked_graph = pruner.visualize_masked_circuit(template_qctn, masked_cores)
    print(f"\n[Target] Circuit with masked cores (marked as '█'):")
    print(masked_graph)
    
    
    # Start pruning from template to match target
    print(f"\n{'='*60}")
    print("PRUNING TEMPLATE TO MATCH TARGET")
    print(f"{'='*60}")
    pruned_qctn, removed_cores = pruner.prune(
        template_qctn,
        target_qctn,
        max_iterations=50
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Template cores: {len(template_qctn.cores)}")
    print(f"Target masked cores: {len(masked_cores)} - {masked_cores}")
    print(f"Pruned removed cores: {len(removed_cores)} - {removed_cores}")
    print(f"Remaining active cores: {len(template_qctn.cores) - len(removed_cores)}")
    
    # Check overlap
    masked_set = set(masked_cores)
    removed_set = set(removed_cores)
    overlap = masked_set.intersection(removed_set)
    print(f"\nOverlap (correctly identified): {len(overlap)} cores - {overlap}")
    print(f"False positives (wrongly removed): {len(removed_set - masked_set)} cores - {removed_set - masked_set}")
    print(f"False negatives (missed): {len(masked_set - removed_set)} cores - {masked_set - removed_set}")
    
    # Visualize pruned circuit
    pruned_graph = pruner.visualize_masked_circuit(template_qctn, removed_cores)
    print(f"\n[Pruned] Circuit with removed cores (marked as '█'):")
    print(pruned_graph)
    
    # Save pruned circuit
    import os
    os.makedirs("./result", exist_ok=True)
    pruned_qctn.save_cores(f"./result/pruned_{n_qubits}qubits_mask{int(mask_ratio*100)}.safetensors")
    print(f"\n[Saved] Pruned circuit to ./result/pruned_{n_qubits}qubits_mask{int(mask_ratio*100)}.safetensors")