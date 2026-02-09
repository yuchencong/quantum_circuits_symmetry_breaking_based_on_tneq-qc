import torch
import random
import numpy as np
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.backends.backend_factory import BackendFactory
# from tneq_qc.optim import stiefel_optimizer_complex
from typing import List, Tuple, Set
import opt_einsum as oe
import copy

class CircuitPrunerParentsInit:
    """
    Implements the 'ParentsInit' pruning logic (Iterative Backward Elimination)
    using PyTorch and Tensor Network Engine.
    """
    
    def __init__(self, engine: EngineSiamese, fidelity_threshold: float = 0.999):
        self.engine = engine
        self.fidelity_threshold = fidelity_threshold
        self.backend = engine.backend
        self.device = engine.backend.backend_info.device

        # Statistics containers
        self.cnt = 0
        self.sum_rse = 0.0 # RSE ~ 1 - Fidelity
        self.delete_counts = []
        self.deleted_flags = []
        self.max_delete = 0.0

    def compute_fidelity(self, qctn1: QCTN, qctn2: QCTN) -> torch.Tensor:
        """Computes Fidelity = |<psi1|psi2>|^2 / (dim^2) roughly, normalized."""
        num_qubits = len(qctn1.qubit_indices)
        measure_input_list = [torch.eye(2, device=self.device) for _ in range(num_qubits)]
        
        contraction_result = self.engine.contract_with_compiled_strategy(
            qctn1,
            circuit_states_list=None,
            measure_input_list=measure_input_list,
            right_qctn=qctn2,
        )
        val = oe.contract("abcdabcd ->", contraction_result)
        fidelity = val.abs() / (2**num_qubits)
        return fidelity

    def sequential_optimization(self, current_qctn: QCTN, target_qctn: QCTN, 
                              active_cores_set: Set[str],
                              learning_rate: float = 0.05, num_steps: int = 50) -> QCTN:
        """
        Corresponds to C++: SequentialOptimization(myU, C, args);
        Optimizes the parameters of active cores in current_qctn to match target_qctn.
        Uses Gradient Descent on Loss = 1 - Fidelity.
        """
        # 1. Create a copy for optimization
        optimized_qctn = QCTN(current_qctn.graph, backend=self.backend)
        
        trainable_params = []
        
        # 2. Setup weights and gradients
        for core_name in current_qctn.cores:
            # Clone data
            tensor_data = current_qctn.cores_weights[core_name].clone().detach()
            
            # Only active (non-deleted) cores are trainable
            if core_name in active_cores_set:
                tensor_data.requires_grad = True
                trainable_params.append(tensor_data)
            else:
                tensor_data.requires_grad = False # Deleted/Identity cores are fixed
                
            optimized_qctn.cores_weights[core_name] = tensor_data

        if not trainable_params:
            return optimized_qctn

        # 3. Optimizer
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
        
        # Target tensors (detached)
        # optimization requires reference to target, but we shouldn't compute grad for target
        # In this implementation, we contract optimized_qctn vs target_qctn directly.
        
        # 4. Gradient Descent Loop
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Compute Fidelity
            fid = self.compute_fidelity(optimized_qctn, target_qctn)
            
            # Loss Function
            loss = 1.0 - fid
            
            loss.backward()
            optimizer.step()
            
            # Optional: decay LR or print logs
            # if step % 20 == 0:
            #     print(f"        Step {step} Loss: {loss.item():.6f}")

        # 5. Finalize (detach tensors)
        final_qctn = QCTN(optimized_qctn.graph, backend=self.backend)
        for core_name in optimized_qctn.cores:
            final_qctn.cores_weights[core_name] = optimized_qctn.cores_weights[core_name].detach()
            
        return final_qctn

    def _replace_core_with_identity(self, qctn: QCTN, core_name: str) -> QCTN:
        """Helper to physically replace a tensor with Identity in a QCTN copy."""
        new_qctn = QCTN(qctn.graph, backend=self.backend)
        # Copy existing
        for c in qctn.cores:
            if c in qctn.cores_weights:
                new_qctn.cores_weights[c] = qctn.cores_weights[c].clone().detach()
        
        # Replace target
        shape = new_qctn.cores_weights[core_name].shape
        if len(shape) == 4:
            dim = min(shape)
            eye = torch.zeros(shape, device=self.device)
            for i in range(dim): eye[i,i,i,i] = 1.0
        elif len(shape) == 2:
            eye = torch.eye(shape[0], shape[1], device=self.device)
        else:
             # Generic diagonal
            eye = torch.zeros(shape, device=self.device)
            min_dim = min(shape)
            for i in range(min_dim):
                idx = tuple([i] * len(shape))
                eye[idx] = 1.0
                
        new_qctn.cores_weights[core_name] = eye
        return new_qctn

    def run_parents_init(self, template_qctn: QCTN, target_provider_func, data_size: int = 20):
        """
        Main logic mimicking the C++ ParentsInit function.
        
        Args:
            template_qctn: The initial full structure (vector<Qposition> p equivalent).
            target_provider_func: Function(epoch_idx) -> target_qctn (Simulating loading U).
            data_size: Number of experiments.
        """
        print("Please enter the dataset size")
        print(f"Dataset size is {data_size}")
        
        # Pre-identify all core names as our "positions"
        all_cores_list = list(template_qctn.cores) 
        num_gates = len(all_cores_list)

        for twentyepoch in range(data_size):
            # --- 1. Initialize sequence p (Restoring full circuit) ---
            # Corresponds to: vector<Qposition> p ... p.push_back...
            current_qctn = QCTN(template_qctn.graph, backend=self.backend)
            for c in template_qctn.cores:
                current_qctn.cores_weights[c] = template_qctn.cores_weights[c].clone().detach()
            
            # Status tracking
            is_deleted = {name: False for name in all_cores_list}
            active_cores_set = set(all_cores_list)
            
            print("-" * 17)
            print(f"Experiment {twentyepoch + 1}: ")
            
            epoch_delete = False
            rse = 0.0 # Represents current best error
            
            # --- 2. Read U ---
            myU = target_provider_func(twentyepoch) # Simulating loading U
            
            count_opt_calls = 0
            
            # Initialize indices for shuffling
            random_idx = list(range(num_gates))
            
            # --- 3. Start SB (Iterative Pruning Loop) ---
            # Corresponds to: for(int epoch=0; epoch < 9; epoch++)
            # In C++, loop limit was fixed 9. Here we use num_gates or a fixed number.
            for epoch in range(num_gates):
                random.shuffle(random_idx)
                rse_changed = False
                
                for i in range(num_gates):
                    core_idx = random_idx[i]
                    core_name = all_cores_list[core_idx]
                    
                    # Skip if deleted
                    if is_deleted[core_name]:
                        continue
                        
                    # --- Build temporary C (Trial Deletion) ---
                    # We create a temporary circuit where 'core_name' is set to Identity
                    # And all other previously deleted cores are also Identity
                    # The C++ code: C *= Identity ... if deleted.
                    # Here: We take current_qctn (which has cumulative deletions) and mask ONE more.
                    
                    temp_qctn = self._replace_core_with_identity(current_qctn, core_name)
                    
                    # Define active cores for this trial (current active - the one we are testing)
                    trial_active_set = active_cores_set - {core_name}
                    
                    # --- SequentialOptimization ---
                    # Fit the remaining active parameters to U
                    optimized_temp_qctn = self.sequential_optimization(
                        temp_qctn, myU, trial_active_set, num_steps=30
                    )
                    count_opt_calls += 1
                    
                    # --- Calculate RSE/Fidelity ---
                    fidelity_val = self.compute_fidelity(optimized_temp_qctn, myU).item()
                    rseU = 1.0 - fidelity_val # Using 1-Fidelity as RSE equivalent
                    
                    # --- Threshold Check ---
                    # C++: if(abs(rseU) < 1e-3)
                    # Note: Threshold might need adjustment depending on problem difficulty
                    threshold_check = (1.0 - self.fidelity_threshold) # e.g. 1e-3
                    
                    if rseU < threshold_check:
                        # Mark deleted
                        is_deleted[core_name] = True
                        active_cores_set.remove(core_name)
                        
                        # Update current best circuit to the optimized one
                        current_qctn = optimized_temp_qctn
                        
                        if rseU < abs(rse) or not rse_changed:
                            rse = rseU
                            rse_changed = True
                        
                        # C++ Logic: break immediately after a successful deletion to re-shuffle
                        # "Greedy" approach
                        print(f"  > Pruned core {core_name} (Loss: {rseU:.6f})")
                        break 
                
                # If rse does not change in a full pass, terminate this experiment
                if not rse_changed:
                    # Count deleted
                    deleted_count = sum(1 for v in is_deleted.values() if v)
                    
                    print(f"Deleted gates in this SB: ", end="")
                    for name, deleted in is_deleted.items():
                        if deleted: print(f"{name} ", end="")
                    print(f", Number of deleted gates: {deleted_count}")
                    
                    self.delete_counts.append(deleted_count)
                    
                    if deleted_count > 0:
                        self.cnt += 1
                        print(f"Successful deletion this time, total successful deletions: {self.cnt}")
                        print(f"Current deletion ratio: {deleted_count/num_gates*100:.2f}%")
                        self.sum_rse += abs(rse)
                        self.max_delete += deleted_count/num_gates
                        self.deleted_flags.append(True)
                    else:
                        self.deleted_flags.append(False)
                        
                    print()
                    print(f"Fidelity is: {1-abs(rse):.6f}")
                    print(f"Number of SequentialOptimization calls: {count_opt_calls}")
                    break
                    
        print("-" * 17)
        print("Experiment ended")
        
        # --- Statistics ---
        if self.cnt > 0:
            mean_delete = self.max_delete / self.cnt
            var_delete = 0.0
            for i in range(len(self.deleted_flags)):
                if self.deleted_flags[i]:
                    val = (self.delete_counts[i]/num_gates) - mean_delete
                    var_delete += val * val
            var_delete /= self.cnt
            
            print(f"Saved memory: {mean_delete*100:.2f}% +-{np.sqrt(var_delete)*100:.2f}%")
            print(f"Success rate: {float(self.cnt)/data_size*100:.2f}%")
            print(f"Mean fidelity: {1 - self.sum_rse/self.cnt:.6f}")
        else:
            print("No successful deletions.")


# --- Helper to create dummy data for the example ---
def create_template_and_target_generator(backend):
    from tneq_qc.core.qctn import QCTNHelper
    # 1. Create Template (Full Circuit)
    n_qubits = 4
    # Wall structure typically has ~8-12 gates for 4 qubits depth 2-3
    qctn_graph = QCTNHelper.generate_example_graph(n=n_qubits, graph_type="wall", dim_char='2')
    template_qctn = QCTN(qctn_graph, backend=backend)
    
    # Randomly initialize template weights
    for c in template_qctn.cores:
        template_qctn.cores_weights[c] = torch.randn_like(template_qctn.cores_weights[c])
        # Make unitary-ish (optional)
    
    # 2. Define Target Generator
    # In C++, it reads files. Here we generate a target by randomly masking the template once per epoch
    # or returning a fixed target. Let's assume we want to prune the template to match a "Masked Ground Truth".
    
    def target_provider(epoch_idx):
        # Create a ground truth U by masking 30% of the original template
        # This simulates that the "True U" is simpler than the full template
        target = QCTN(template_qctn.graph, backend=backend)
        random.seed(epoch_idx) # Deterministic per epoch
        
        cores = list(template_qctn.cores)
        mask_indices = random.sample(range(len(cores)), int(len(cores) * 0.3)) # Mask 30%
        
        for i, c in enumerate(cores):
            w = template_qctn.cores_weights[c].clone().detach()
            if i in mask_indices:
                # Set to Identity
                shape = w.shape
                if len(shape) == 4:
                    w = torch.zeros_like(w)
                    for k in range(min(shape)): w[k,k,k,k] = 1.0
                # ... simplified for brevity
            target.cores_weights[c] = w
            
        return target

    return template_qctn, target_provider

# --- Main Execution Block ---
if __name__ == "__main__":
    # Settings
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 1. Setup Backend
    backend = BackendFactory.create_backend('pytorch', device='cpu') # or 'cuda'
    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=100)
    
    # 2. Prepare Data
    template, target_func = create_template_and_target_generator(backend)
    
    print(f"Template Cores: {len(template.cores)}")
    
    # 3. Instantiate and Run Pruner
    # Threshold 0.99 means error < 1e-2. Adjust to 0.999 for 1e-3 strictness.
    pruner = CircuitPrunerParentsInit(engine, fidelity_threshold=0.99) 
    
    pruner.run_parents_init(template, target_func, data_size=5)