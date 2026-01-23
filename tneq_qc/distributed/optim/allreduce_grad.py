"""
AllReduce with Gradient Support

Provides a PyTorch autograd-compatible allreduce operation for distributed training.
This allows gradients to flow through allreduce operations during backpropagation.
"""

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp as TorchReduceOp


class AllReduceGrad(torch.autograd.Function):
    """
    AllReduce operation that supports gradient computation.
    
    Forward: performs allreduce on the input tensor
    Backward: allreduce the gradient (for SUM, gradient is also summed)
    
    This is necessary because torch.distributed.all_reduce is not autograd-aware.
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, op: TorchReduceOp = TorchReduceOp.SUM, group=None) -> torch.Tensor:
        """
        AllReduce forward pass.
        
        Args:
            tensor: Input tensor to allreduce
            op: Reduce operation (SUM, AVG, etc.)
            group: Process group (default: WORLD)
            
        Returns:
            Allreduced tensor
        """
        ctx.op = op
        ctx.group = group
        result = tensor.clone()
        dist.all_reduce(result, op=op, group=group)
        return result
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        AllReduce backward pass.
        
        For SUM operation: gradient also needs to be allreduced
        This ensures all ranks have consistent gradients.
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient to propagate downstream, None for op and group parameters
        """
        # For SUM: each rank contributed to the sum, so gradient flows back equally
        # We allreduce the gradient so all ranks have the same gradient
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=ctx.op, group=ctx.group)
        return grad_input, None, None


def allreduce_with_grad(tensor: torch.Tensor, op: TorchReduceOp = TorchReduceOp.SUM, group=None) -> torch.Tensor:
    """
    Perform allreduce with gradient support.
    
    This is a wrapper around AllReduceGrad.apply for convenience.
    
    Args:
        tensor: Input tensor to allreduce
        op: Reduce operation (default: SUM)
        group: Process group (default: WORLD)
        
    Returns:
        Allreduced tensor (maintains gradient computation)
        
    Example:
        >>> # In distributed forward pass
        >>> partial_result = compute_partial()  # requires_grad=True
        >>> full_result = allreduce_with_grad(partial_result)
        >>> loss = compute_loss(full_result)
        >>> loss.backward()  # gradients flow through allreduce
    """
    return AllReduceGrad.apply(tensor, op)


class AllGatherGrad(torch.autograd.Function):
    """
    AllGather operation that supports gradient computation.
    
    Forward: gathers tensors from all ranks
    Backward: scatters gradient back to respective ranks
    """
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """
        AllGather forward pass.
        
        Args:
            tensor: Local tensor to gather
            world_size: Number of ranks
            
        Returns:
            Concatenated tensor from all ranks along dim 0
        """
        ctx.world_size = world_size
        ctx.local_size = tensor.shape[0]
        
        # Prepare output list
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        # Concatenate along first dimension
        return torch.cat(gathered, dim=0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        AllGather backward pass.
        
        Splits gradient and returns the portion corresponding to this rank.
        """
        rank = dist.get_rank()
        local_size = ctx.local_size
        
        # Extract this rank's portion of the gradient
        start_idx = rank * local_size
        end_idx = start_idx + local_size
        grad_input = grad_output[start_idx:end_idx].clone()
        
        return grad_input, None


def allgather_with_grad(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Perform allgather with gradient support.
    
    Args:
        tensor: Local tensor to gather
        world_size: Number of ranks
        
    Returns:
        Gathered tensor (maintains gradient computation)
    """
    return AllGatherGrad.apply(tensor, world_size)


class SendRecvGrad(torch.autograd.Function):
    """
    Point-to-point send/recv with gradient support.
    
    For tensor exchange between two ranks.
    """
    
    @staticmethod
    def forward(ctx, tensor_to_send: torch.Tensor, 
                recv_buffer: torch.Tensor,
                partner_rank: int,
                my_rank: int) -> torch.Tensor:
        """
        Exchange tensor with partner rank.
        
        Args:
            tensor_to_send: Tensor to send to partner
            recv_buffer: Pre-allocated buffer for receiving
            partner_rank: Rank to exchange with
            my_rank: This process's rank
            
        Returns:
            Received tensor
        """
        ctx.partner_rank = partner_rank
        ctx.my_rank = my_rank
        ctx.send_shape = tensor_to_send.shape
        
        # Lower rank sends first to avoid deadlock
        if my_rank < partner_rank:
            dist.send(tensor_to_send.contiguous(), partner_rank)
            dist.recv(recv_buffer, partner_rank)
        else:
            dist.recv(recv_buffer, partner_rank)
            dist.send(tensor_to_send.contiguous(), partner_rank)
        
        return recv_buffer.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: exchange gradients in reverse direction.
        """
        partner_rank = ctx.partner_rank
        my_rank = ctx.my_rank
        
        # Create buffer for receiving gradient
        grad_input = torch.zeros(ctx.send_shape, dtype=grad_output.dtype, 
                                  device=grad_output.device)
        
        # Exchange gradients (reverse direction)
        if my_rank < partner_rank:
            dist.send(grad_output.contiguous(), partner_rank)
            dist.recv(grad_input, partner_rank)
        else:
            dist.recv(grad_input, partner_rank)
            dist.send(grad_output.contiguous(), partner_rank)
        
        return grad_input, None, None, None
