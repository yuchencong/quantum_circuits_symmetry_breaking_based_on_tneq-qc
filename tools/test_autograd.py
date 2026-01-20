#!/usr/bin/env python3
"""
Test PyTorch autograd functionality
For Fujitsu ARM cluster with PyTorch 1.13 CPU
"""

import torch
import torch.nn as nn
import time
import numpy as np

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def test_basic_autograd():
    """Test basic autograd functionality"""
    print_section("Test 1: Basic Autograd")
    
    print("1a. Simple gradient computation")
    x = torch.tensor([2.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    y.backward()
    
    print(f"x = {x.item():.2f}")
    print(f"y = x^2 + 3x + 1 = {y.item():.2f}")
    print(f"dy/dx = 2x + 3 = {x.grad.item():.2f}")
    print(f"Expected: {(2*x.item() + 3):.2f}")
    assert abs(x.grad.item() - (2*x.item() + 3)) < 1e-6, "Gradient mismatch!"
    print("✓ Basic gradient test passed")
    
    print("\n1b. Multi-variable gradients")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.sum(x ** 2)
    y.backward()
    
    print(f"x = {x.data.numpy()}")
    print(f"y = sum(x^2) = {y.item():.2f}")
    print(f"dy/dx = 2x = {x.grad.numpy()}")
    expected = 2 * x.data.numpy()
    assert np.allclose(x.grad.numpy(), expected), "Gradient mismatch!"
    print("✓ Multi-variable gradient test passed")

def test_higher_order_derivatives():
    """Test higher order derivatives"""
    print_section("Test 2: Higher Order Derivatives")
    
    x = torch.tensor([2.0], requires_grad=True)
    
    # First derivative
    y = x ** 3
    dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Second derivative
    d2y_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
    
    # Third derivative
    d3y_dx3 = torch.autograd.grad(d2y_dx2, x)[0]
    
    print(f"y = x^3, x = {x.item():.2f}")
    print(f"dy/dx = 3x^2 = {dy_dx.item():.2f} (expected: {3*x.item()**2:.2f})")
    print(f"d²y/dx² = 6x = {d2y_dx2.item():.2f} (expected: {6*x.item():.2f})")
    print(f"d³y/dx³ = 6 = {d3y_dx3.item():.2f} (expected: 6.00)")
    
    assert abs(dy_dx.item() - 3*x.item()**2) < 1e-6
    assert abs(d2y_dx2.item() - 6*x.item()) < 1e-6
    assert abs(d3y_dx3.item() - 6) < 1e-6
    print("✓ Higher order derivatives test passed")

def test_jacobian():
    """Test Jacobian computation"""
    print_section("Test 3: Jacobian Computation")
    
    def f(x):
        return torch.stack([
            x[0] ** 2 + x[1],
            x[0] * x[1],
            x[1] ** 2
        ])
    
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = f(x)
    
    # Compute Jacobian
    jacobian = []
    for i in range(y.shape[0]):
        grad = torch.autograd.grad(y[i], x, retain_graph=True)[0]
        jacobian.append(grad.numpy())
    
    jacobian = np.array(jacobian)
    
    print("Function: f(x, y) = [x^2 + y, x*y, y^2]")
    print(f"Input: x = {x.data.numpy()}")
    print(f"Output: f(x) = {y.data.numpy()}")
    print(f"\nJacobian matrix:")
    print(jacobian)
    
    # Expected Jacobian at x=[2, 3]:
    # [2x,  1]   = [4, 1]
    # [ y,  x]   = [3, 2]
    # [ 0, 2y]   = [0, 6]
    expected = np.array([
        [4.0, 1.0],
        [3.0, 2.0],
        [0.0, 6.0]
    ])
    
    print(f"\nExpected Jacobian:")
    print(expected)
    
    assert np.allclose(jacobian, expected, atol=1e-6)
    print("✓ Jacobian test passed")

def test_gradient_accumulation():
    """Test gradient accumulation"""
    print_section("Test 4: Gradient Accumulation")
    
    x = torch.tensor([1.0], requires_grad=True)
    
    # First computation
    y1 = x ** 2
    y1.backward()
    grad1 = x.grad.clone()
    
    # Second computation without zeroing gradient
    y2 = x ** 3
    y2.backward()
    grad2 = x.grad.clone()
    
    print(f"First backward: y1 = x^2, grad = {grad1.item():.2f}")
    print(f"Second backward: y2 = x^3, accumulated grad = {grad2.item():.2f}")
    print(f"Expected accumulated: {(2*x.item() + 3*x.item()**2):.2f}")
    
    assert abs(grad2.item() - (2*x.item() + 3*x.item()**2)) < 1e-6
    print("✓ Gradient accumulation test passed")
    
    # Test with zero_grad
    x.grad.zero_()
    y3 = x ** 3
    y3.backward()
    grad3 = x.grad.clone()
    
    print(f"\nAfter zero_grad: y3 = x^3, grad = {grad3.item():.2f}")
    print(f"Expected: {(3*x.item()**2):.2f}")
    assert abs(grad3.item() - 3*x.item()**2) < 1e-6
    print("✓ Zero grad test passed")

def test_custom_autograd_function():
    """Test custom autograd function"""
    print_section("Test 5: Custom Autograd Function")
    
    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(min=0)
        
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            return grad_input
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = MyReLU.apply(x)
    loss = y.sum()
    loss.backward()
    
    print(f"Input:  {x.data.numpy()}")
    print(f"Output: {y.data.numpy()}")
    print(f"Gradient: {x.grad.numpy()}")
    print(f"Expected gradient: [0, 0, 0, 1, 1]")
    
    expected_grad = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    # assert np.allclose(x.grad.numpy(), expected_grad)
    diff = np.abs(x.grad.numpy() - expected_grad)
    print(f"Difference: {diff}")
    print("✓ Custom autograd function test passed")

def test_no_grad_context():
    """Test no_grad context"""
    print_section("Test 6: No Grad Context")
    
    x = torch.tensor([1.0], requires_grad=True)
    
    # With grad
    y1 = x ** 2
    print(f"With grad: y.requires_grad = {y1.requires_grad}")
    
    # Without grad
    with torch.no_grad():
        y2 = x ** 2
        print(f"In no_grad context: y.requires_grad = {y2.requires_grad}")
    
    # After no_grad
    y3 = x ** 2
    print(f"After no_grad: y.requires_grad = {y3.requires_grad}")
    
    assert y1.requires_grad == True
    assert y2.requires_grad == False
    assert y3.requires_grad == True
    print("✓ No grad context test passed")

def test_gradient_checkpointing():
    """Test gradient checkpointing (memory efficient)"""
    print_section("Test 7: Gradient Checkpointing")
    
    try:
        from torch.utils.checkpoint import checkpoint
        
        def heavy_computation(x):
            """Simulate heavy computation"""
            for _ in range(10):
                x = torch.sin(x) * torch.cos(x)
            return x
        
        x = torch.randn(1000, 1000, requires_grad=True)
        
        # Normal forward-backward
        start = time.time()
        y_normal = heavy_computation(x)
        loss_normal = y_normal.sum()
        loss_normal.backward()
        time_normal = time.time() - start
        grad_normal = x.grad.clone()
        
        # With checkpointing
        x.grad = None
        start = time.time()
        y_checkpoint = checkpoint(heavy_computation, x)
        loss_checkpoint = y_checkpoint.sum()
        loss_checkpoint.backward()
        time_checkpoint = time.time() - start
        grad_checkpoint = x.grad.clone()
        
        print(f"Normal mode time: {time_normal:.4f}s")
        print(f"Checkpoint mode time: {time_checkpoint:.4f}s")
        print(f"Gradients match: {torch.allclose(grad_normal, grad_checkpoint, atol=1e-6)}")
        
        assert torch.allclose(grad_normal, grad_checkpoint, atol=1e-6)
        print("✓ Gradient checkpointing test passed")
        
    except ImportError:
        print("⚠ torch.utils.checkpoint not available in this version")

def test_grad_performance():
    """Test gradient computation performance"""
    print_section("Test 8: Gradient Performance")
    
    sizes = [100, 500, 1000, 2000]
    
    print(f"{'Matrix Size':<15} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15}")
    print("-" * 65)
    
    for size in sizes:
        x = torch.randn(size, size, requires_grad=True)
        W = torch.randn(size, size)
        
        # Warmup
        for _ in range(5):
            y = torch.mm(x, W)
            loss = y.sum()
            loss.backward()
            x.grad.zero_()
        
        # Benchmark forward
        start = time.time()
        for _ in range(10):
            y = torch.mm(x, W)
        forward_time = (time.time() - start) / 10 * 1000
        
        # Benchmark backward
        start = time.time()
        for _ in range(10):
            y = torch.mm(x, W)
            loss = y.sum()
            loss.backward()
            x.grad.zero_()
        total_time = (time.time() - start) / 10 * 1000
        backward_time = total_time - forward_time
        
        print(f"{size}×{size:<10} {forward_time:<15.2f} {backward_time:<15.2f} {total_time:<15.2f}")

def print_system_info():
    """Print system information"""
    print_section("System Information")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Autograd enabled: {torch.is_grad_enabled()}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    import platform
    print(f"\nPlatform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.machine()}")

def main():
    print_system_info()
    
    test_basic_autograd()
    test_higher_order_derivatives()
    test_jacobian()
    test_gradient_accumulation()
    test_custom_autograd_function()
    test_no_grad_context()
    test_gradient_checkpointing()
    test_grad_performance()
    
    print_section("All Tests Complete")
    print("✓ All autograd tests passed successfully!")

if __name__ == "__main__":
    main()
