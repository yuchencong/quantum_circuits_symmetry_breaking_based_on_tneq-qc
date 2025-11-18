"""
Backend factory for creating computational backends (JAX, PyTorch, etc.).

This module provides a factory pattern for creating and managing different
computational backends for tensor operations.
"""

from __future__ import annotations
from typing import Optional, Type, Any, Union
from abc import ABC, abstractmethod
import numpy as np


class BackendInfo:
    """
    Simple information class for backend configuration.
    
    This class only records backend information, without providing
    tensor allocation or conversion methods. The actual tensor operations
    should be implemented in the caller based on this information.
    """
    
    def __init__(self, backend_type: str, device: Optional[str] = None, **kwargs):
        """
        Initialize backend information.
        
        Args:
            backend_type (str): Type of backend ('jax', 'pytorch', etc.).
            device (Optional[str]): Device specification (e.g., 'cpu', 'cuda', 'cuda:0', 'gpu').
            **kwargs: Additional backend-specific configuration.
        """
        self.backend_type = backend_type.lower()
        self.device = device
        self.config = kwargs
    
    def __repr__(self):
        return f"BackendInfo(backend_type='{self.backend_type}', device='{self.device}', config={self.config})"
    
    def __str__(self):
        return self.__repr__()

class ComputeBackend(ABC):
    """
    Abstract base class for computational backends.
    
    All backends must implement these methods for tensor operations,
    gradient computation, and JIT compilation.
    """
    
    def __init__(self):
        """Initialize backend with BackendInfo."""
        self.backend_info: Optional[BackendInfo] = None

    @abstractmethod
    def execute_expression(self, expression, *tensors):
        """
        Execute a contraction expression with given tensors.
        
        Args:
            expression: Optimized contraction expression (opt_einsum.ContractExpression).
            *tensors: Variable number of input tensors.
        
        Returns:
            Tensor result of the contraction.
        """
        pass

    @abstractmethod
    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Create a function that computes both value and gradient.
        
        Args:
            loss_fn: Loss function to differentiate.
            argnums: Argument indices to compute gradients for.
        
        Returns:
            Function that returns (value, gradients).
        """
        pass

    @abstractmethod
    def jit_compile(self, func):
        """
        JIT compile a function for faster execution.
        
        Args:
            func: Function to compile.
        
        Returns:
            JIT-compiled function.
        """
        pass

    @abstractmethod
    def convert_to_tensor(self, array):
        """
        Convert array-like object to backend tensor.
        
        Args:
            array: Array-like object (numpy array, list, etc.).
        
        Returns:
            Backend-specific tensor.
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of the backend."""
        pass
    
    def get_backend_info(self) -> BackendInfo:
        """
        Get the BackendInfo instance for this backend.
        
        Returns:
            BackendInfo instance.
        """
        if self.backend_info is None:
            # Create default BackendInfo
            self.backend_info = BackendInfo(self.get_backend_name())
        return self.backend_info
    
    def set_backend_info(self, backend_info: BackendInfo):
        """
        Set the BackendInfo instance for this backend.
        
        Args:
            backend_info (BackendInfo): BackendInfo instance to use.
        """
        if backend_info.backend_type != self.get_backend_name():
            raise ValueError(
                f"BackendInfo type '{backend_info.backend_type}' does not match "
                f"backend '{self.get_backend_name()}'"
            )
        self.backend_info = backend_info


class JAXBackend(ComputeBackend):
    """JAX computational backend."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize JAX backend.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'gpu', etc.).
                If None, automatically detects available devices.
        """
        super().__init__()
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            
            # Auto-detect device if not specified
            if device is None:
                device = 'gpu' if jax.devices('gpu') else 'cpu'
            
            # Create BackendInfo
            self.backend_info = BackendInfo('jax', device=device)
        except ImportError:
            raise ImportError("JAX is not installed. Please install it with: pip install jax jaxlib")

    def execute_expression(self, expression, *tensors):
        """Execute contraction expression using JAX."""
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """Compute value and gradient using JAX autodiff."""
        return self.jax.value_and_grad(loss_fn, argnums=argnums)

    def jit_compile(self, func):
        """JIT compile function using JAX."""
        return self.jax.jit(func)

    def convert_to_tensor(self, array):
        """Convert to JAX array."""
        if isinstance(array, self.jnp.ndarray):
            return array
        
        # Convert based on backend_info
        tensor = self.jnp.array(array)
        
        # Device placement if GPU
        if self.backend_info.device and 'gpu' in self.backend_info.device.lower():
            devices = self.jax.devices('gpu')
            if devices:
                tensor = self.jax.device_put(tensor, devices[0])
        
        return tensor

    def get_backend_name(self) -> str:
        return "jax"


class PyTorchBackend(ComputeBackend):
    """PyTorch computational backend."""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch backend.
        
        Args:
            device (Optional[str]): Device specification ('cpu', 'cuda', 'cuda:0', etc.).
                If None, automatically selects 'cuda' if available, otherwise 'cpu'.
        """
        super().__init__()
        try:
            import torch
            self.torch = torch
            
            # Auto-detect device if not specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create BackendInfo
            self.backend_info = BackendInfo('pytorch', device=device)
        except ImportError:
            raise ImportError("PyTorch is not installed. Please install it with: pip install torch")

    def execute_expression(self, expression, *tensors):
        """
        Execute contraction expression using PyTorch.
        
        Note: opt_einsum's ContractExpression works with PyTorch tensors.
        """
        return expression(*tensors)

    def compute_value_and_grad(self, loss_fn, argnums):
        """
        Compute value and gradient using PyTorch autograd.
        
        Returns a function that computes loss and gradients for specified arguments.
        """
        def value_and_grad_fn(*args):
            # Convert argnums to list if it's a range
            if isinstance(argnums, range):
                arg_indices = list(argnums)
            elif isinstance(argnums, (list, tuple)):
                arg_indices = argnums
            else:
                arg_indices = [argnums]

            # Prepare tensors and enable gradients for specified arguments
            tensor_args = []
            for i, arg in enumerate(args):
                if not isinstance(arg, self.torch.Tensor):
                    arg = self.torch.from_numpy(arg).float()
                
                if i in arg_indices:
                    # Enable gradient tracking for parameters to optimize
                    arg.requires_grad_(True)
                else:
                    # Disable gradient tracking for other tensors
                    arg.requires_grad_(False)
                
                tensor_args.append(arg)

            # Compute loss
            loss = loss_fn(*tensor_args)

            # Compute gradients using torch.autograd.grad
            # This works correctly with both leaf and non-leaf tensors
            grad_tensors = [tensor_args[i] for i in arg_indices]
            gradients = self.torch.autograd.grad(
                outputs=loss,
                inputs=grad_tensors,
                create_graph=False,  # Don't build computation graph for gradients
                retain_graph=False   # Release computation graph after computing gradients
            )

            # Return loss as tensor value (not .item()) to preserve type
            # Detach it from computation graph to free memory
            return loss.detach(), gradients

        return value_and_grad_fn

    def jit_compile(self, func):
        """
        JIT compile function using PyTorch.
        
        Note: PyTorch's torch.jit.script has limitations.
        For now, we return the function as-is and rely on PyTorch's built-in optimizations.
        """
        # Could use torch.jit.trace or torch.compile in PyTorch 2.0+
        return func

    def convert_to_tensor(self, array):
        """Convert to PyTorch tensor."""
        if isinstance(array, self.torch.Tensor):
            # Move to correct device if needed
            if str(array.device) != self.backend_info.device:
                return array.to(self.backend_info.device)
            return array
        
        # Convert array to tensor based on backend_info
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        tensor = self.torch.from_numpy(array).float()
        return tensor.to(self.backend_info.device)

    def get_backend_name(self) -> str:
        return "pytorch"


class BackendFactory:
    """
    Factory class for creating computational backends.
    
    Usage:
        backend = BackendFactory.create_backend('jax')
        backend = BackendFactory.create_backend('pytorch', device='cuda')
    """

    _backends = {
        'jax': JAXBackend,
        'pytorch': PyTorchBackend,
    }

    _default_backend: Optional[str] = None
    _backend_instance: Optional[ComputeBackend] = None

    @classmethod
    def create_backend(cls, backend_name: str, device: Optional[str] = None, **kwargs) -> ComputeBackend:
        """
        Create a backend instance.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
            device (Optional[str]): Device specification.
            **kwargs: Additional backend-specific configuration.
        
        Returns:
            ComputeBackend: Backend instance.
        
        Raises:
            ValueError: If backend name is not supported.
        """
        backend_name = backend_name.lower()
        
        if backend_name not in cls._backends:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available backends: {list(cls._backends.keys())}"
            )

        return cls._backends[backend_name](device=device, **kwargs)

    @classmethod
    def set_default_backend(cls, backend_name: str, device: Optional[str] = None, **kwargs):
        """
        Set the default backend for the application.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
            device (Optional[str]): Device specification.
            **kwargs: Additional backend-specific configuration.
        """
        cls._default_backend = backend_name.lower()
        cls._backend_instance = cls.create_backend(backend_name, device=device, **kwargs)

    @classmethod
    def get_default_backend(cls) -> ComputeBackend:
        """
        Get the default backend instance.
        
        Returns:
            ComputeBackend: Default backend instance (creates JAX backend if not set).
        """
        if cls._backend_instance is None:
            cls.set_default_backend('jax', 'gpu')
        return cls._backend_instance

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[ComputeBackend]):
        """
        Register a custom backend.
        
        Args:
            name (str): Name for the backend.
            backend_class (Type[ComputeBackend]): Backend class implementing ComputeBackend.
        """
        cls._backends[name.lower()] = backend_class
