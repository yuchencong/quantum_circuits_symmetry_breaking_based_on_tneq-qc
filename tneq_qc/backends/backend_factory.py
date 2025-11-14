"""
Backend factory for creating computational backends (JAX, PyTorch, etc.).

This module provides a factory pattern for creating and managing different
computational backends for tensor operations.
"""

from __future__ import annotations
from typing import Optional, Type
from abc import ABC, abstractmethod
import numpy as np

class ComputeBackend(ABC):
    """
    Abstract base class for computational backends.
    
    All backends must implement these methods for tensor operations,
    gradient computation, and JIT compilation.
    """

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


class JAXBackend(ComputeBackend):
    """JAX computational backend."""

    def __init__(self):
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
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
        return self.jnp.array(array)

    def get_backend_name(self) -> str:
        return "jax"


class PyTorchBackend(ComputeBackend):
    """PyTorch computational backend."""

    def __init__(self):
        try:
            import torch
            self.torch = torch
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
                if i in arg_indices:
                    if not isinstance(arg, self.torch.Tensor):
                        arg = self.torch.from_numpy(arg).float()
                    arg.requires_grad_(True)
                else:
                    if not isinstance(arg, self.torch.Tensor):
                        arg = self.torch.from_numpy(arg).float()
                tensor_args.append(arg)

            # Compute loss
            loss = loss_fn(*tensor_args)

            # Compute gradients
            loss.backward()

            # Extract gradients for specified arguments
            gradients = tuple(tensor_args[i].grad for i in arg_indices)

            return loss.item(), gradients

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
            return array
        # if not a numpy array, convert jax array to numpy first
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        return self.torch.from_numpy(array).float()

    def get_backend_name(self) -> str:
        return "pytorch"


class BackendFactory:
    """
    Factory class for creating computational backends.
    
    Usage:
        backend = BackendFactory.create_backend('jax')
        backend = BackendFactory.create_backend('pytorch')
    """

    _backends = {
        'jax': JAXBackend,
        'pytorch': PyTorchBackend,
    }

    _default_backend: Optional[str] = None
    _backend_instance: Optional[ComputeBackend] = None

    @classmethod
    def create_backend(cls, backend_name: str) -> ComputeBackend:
        """
        Create a backend instance.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
        
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

        return cls._backends[backend_name]()

    @classmethod
    def set_default_backend(cls, backend_name: str):
        """
        Set the default backend for the application.
        
        Args:
            backend_name (str): Name of the backend ('jax' or 'pytorch').
        """
        cls._default_backend = backend_name.lower()
        cls._backend_instance = cls.create_backend(backend_name)

    @classmethod
    def get_default_backend(cls) -> ComputeBackend:
        """
        Get the default backend instance.
        
        Returns:
            ComputeBackend: Default backend instance (creates JAX backend if not set).
        """
        if cls._backend_instance is None:
            cls.set_default_backend('jax')
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
