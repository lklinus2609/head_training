import os

import torch
import torch.distributed as dist


def setup_ddp():
    """Initialize DDP process group and set CUDA device."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy the DDP process group."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and return mean across ranks."""
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor
