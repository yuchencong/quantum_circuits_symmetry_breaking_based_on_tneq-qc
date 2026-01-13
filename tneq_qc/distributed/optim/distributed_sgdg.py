"""
Distributed SGDG Optimizer

Stiefel Gradient Descent (SGDG) optimizer for distributed training.
Maintains orthogonality constraints using Cayley transform.

Reference:
- SGD-G: Stiefel Gradient Descent for Decorrelated Weight Matrix
"""

import random
from typing import Dict, List, Optional, Any

import torch


class DistributedSGDG:
    """
    Distributed Stiefel Gradient Descent (SGDG) Optimizer.
    
    This optimizer updates weights on the Stiefel manifold using Cayley transform,
    which preserves orthogonality constraints. Each rank independently optimizes
    its local partition weights.
    
    Key features:
    - Cayley transform for manifold-preserving updates
    - Momentum support
    - Adaptive step size based on matrix norm
    - Periodic QR re-orthogonalization for numerical stability
    
    Args:
        lr: Learning rate
        momentum: Momentum factor (default: 0.0)
        stiefel: Whether to use Stiefel manifold optimization (default: True)
        epsilon: Small constant for numerical stability (default: 1e-8)
        qr_retraction_prob: Probability of QR retraction per step (default: 0.01)
    """
    
    def __init__(self, 
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 stiefel: bool = True,
                 epsilon: float = 1e-8,
                 qr_retraction_prob: float = 0.01):
        self.lr = lr
        self.momentum = momentum
        self.stiefel = stiefel
        self.epsilon = epsilon
        self.qr_retraction_prob = qr_retraction_prob
        
        # Momentum buffers: {param_name: velocity_tensor}
        self.momentum_buffer: Dict[str, torch.Tensor] = {}
        
        # Step counter
        self.step_count = 0
    
    def step(self, local_qctn, grads: List[torch.Tensor]):
        """
        Perform a single optimization step.
        
        Args:
            local_qctn: LocalQCTN containing the weights to update
            grads: List of gradients corresponding to each core weight
        """
        self.step_count += 1
        
        for i, name in enumerate(local_qctn.cores):
            param = local_qctn.cores_weights[name]
            grad = grads[i]
            
            if param is None or grad is None:
                continue
            
            # Handle TNTensor
            from ...core.tn_tensor import TNTensor
            if isinstance(param, TNTensor):
                tensor = param.tensor
                scale = param.scale
            else:
                tensor = param
                scale = 1.0
            
            if self.stiefel:
                # Stiefel manifold optimization
                updated_tensor = self._stiefel_update(tensor, grad, name)
            else:
                # Standard SGD
                updated_tensor = tensor - self.lr * grad
            
            # CRITICAL: Detach the updated tensor from the computation graph
            # This prevents "backward through the graph a second time" error
            # when training multiple iterations
            updated_tensor = updated_tensor.detach().requires_grad_(True)
            
            # Update the weight
            if isinstance(param, TNTensor):
                local_qctn.cores_weights[name] = TNTensor(updated_tensor, scale)
            else:
                local_qctn.cores_weights[name] = updated_tensor
    
    def _stiefel_update(self, param: torch.Tensor, grad: torch.Tensor, 
                        name: str) -> torch.Tensor:
        """
        Perform Stiefel manifold update using Cayley transform.
        
        The update preserves the orthogonality constraint X^T X = I.
        
        Args:
            param: Current parameter tensor
            grad: Gradient tensor
            name: Parameter name (for momentum buffer)
            
        Returns:
            Updated parameter tensor
        """
        original_shape = param.shape
        
        # Reshape to matrix form [rows, cols] for Stiefel optimization
        # For 4D tensor [n_in, n_out, rank_in, rank_out], reshape to [n_in*n_out, rank_in*rank_out]
        if param.dim() == 4:
            p_reshaped = param.view(param.size(0) * param.size(1), -1)
            g_reshaped = grad.view(grad.size(0) * grad.size(1), -1)
        elif param.dim() == 2:
            p_reshaped = param
            g_reshaped = grad
        else:
            # Fall back to standard SGD for other dimensions
            return param - self.lr * grad
        
        # Unit normalization (project to Stiefel manifold)
        unity = self._unit(p_reshaped)
        
        # Check Stiefel condition: rows <= cols
        if unity.size(0) > unity.size(1):
            # Cannot satisfy Stiefel constraint, fall back to SGD
            return param - self.lr * grad
        
        # Periodic QR retraction for numerical stability
        if random.random() < self.qr_retraction_prob:
            unity = self._qr_retraction(unity)
        
        # Initialize or retrieve momentum buffer
        if name not in self.momentum_buffer:
            self.momentum_buffer[name] = torch.zeros(
                g_reshaped.t().size(),
                device=param.device,
                dtype=param.dtype
            )
        
        V = self.momentum_buffer[name]
        V = self.momentum * V - g_reshaped.t()
        
        # Compute skew-symmetric matrix W
        MX = torch.mm(V, unity)
        XMX = torch.mm(unity, MX)
        XXMX = torch.mm(unity.t(), XMX)
        W_hat = MX - 0.5 * XXMX
        W = W_hat - W_hat.t()  # Ensure skew-symmetry
        
        # Adaptive step size based on matrix norm
        t = 0.5 * 2 / (self._matrix_norm_one(W) + self.epsilon)
        alpha = min(t, self.lr)
        
        # Cayley transform: Y = (I - alpha/2 * W)^(-1) @ (I + alpha/2 * W) @ X
        p_new = self._compute_Y(alpha, W, unity.t()).t()
        
        # Update momentum buffer
        V_new = torch.mm(W, unity.t())
        self.momentum_buffer[name] = V_new
        
        return p_new.view(original_shape)
    
    def _unit(self, W: torch.Tensor) -> torch.Tensor:
        """
        Project matrix to have orthonormal rows using QR decomposition.
        
        Args:
            W: Input matrix [rows, cols]
            
        Returns:
            Matrix with orthonormal rows
        """
        Q, R = torch.linalg.qr(W.t())
        return Q.t()
    
    def _qr_retraction(self, X: torch.Tensor) -> torch.Tensor:
        """
        QR retraction to ensure we stay on Stiefel manifold.
        
        Args:
            X: Matrix to retract [rows, cols]
            
        Returns:
            Retracted matrix on Stiefel manifold
        """
        Q, R = torch.linalg.qr(X.t())
        return Q.t()
    
    def _compute_Y(self, alpha: float, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: Y = (I - alpha/2 * W)^(-1) @ (I + alpha/2 * W) @ X
        
        This transform preserves orthogonality: if X^T X = I, then Y^T Y = I.
        
        Args:
            alpha: Step size
            W: Skew-symmetric matrix
            X: Current point (transposed form)
            
        Returns:
            Transformed point
        """
        I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
        left_matrix = I - (alpha / 2) * W
        right_matrix = I + (alpha / 2) * W
        left_inv = torch.inverse(left_matrix)
        return left_inv @ right_matrix @ X
    
    def _matrix_norm_one(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix 1-norm (maximum absolute column sum).
        
        Args:
            W: Input matrix
            
        Returns:
            1-norm value
        """
        return torch.abs(W).sum(dim=0).max()
    
    def zero_grad(self, local_qctn):
        """
        Zero out gradients on all parameters.
        
        Args:
            local_qctn: LocalQCTN containing the weights
        """
        from ...core.tn_tensor import TNTensor
        
        for name in local_qctn.cores:
            weight = local_qctn.cores_weights[name]
            
            if isinstance(weight, TNTensor):
                tensor = weight.tensor
            else:
                tensor = weight
            
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                tensor.grad.zero_()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return optimizer state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'stiefel': self.stiefel,
            'step_count': self.step_count,
            'momentum_buffer': {
                k: v.cpu() for k, v in self.momentum_buffer.items()
            }
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], device: str = 'cpu'):
        """
        Load optimizer state from dictionary.
        
        Args:
            state_dict: State dictionary
            device: Device to load tensors to
        """
        self.lr = state_dict.get('lr', self.lr)
        self.momentum = state_dict.get('momentum', self.momentum)
        self.stiefel = state_dict.get('stiefel', self.stiefel)
        self.step_count = state_dict.get('step_count', 0)
        
        if 'momentum_buffer' in state_dict:
            self.momentum_buffer = {
                k: v.to(device) 
                for k, v in state_dict['momentum_buffer'].items()
            }


class LRScheduler:
    """
    Learning rate scheduler for DistributedSGDG.
    
    Supports step-based learning rate schedules.
    
    Args:
        optimizer: DistributedSGDG optimizer
        lr_schedule: List of (step, lr) tuples defining the schedule
    """
    
    def __init__(self, optimizer: DistributedSGDG, 
                 lr_schedule: List[tuple]):
        self.optimizer = optimizer
        self.lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        self.current_step = 0
    
    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1
        
        for step, lr in reversed(self.lr_schedule):
            if self.current_step >= step:
                self.optimizer.lr = lr
                return
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr
