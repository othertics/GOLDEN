"""Configuration and training utilities for the classification-only DeepSet model."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class TrainConfig:
    """Training configuration for binary-mask 3-class classification."""

    task: str = 'classification3'

    # Model (stage-1 MLP classifier)
    n_genes: int = 5000
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [1024, 256, 64])
    cls_hidden_dim: int = 64
    prop_hidden_dim: int = 128
    input_mode: str = 'binary'  # 'binary' | 'expression'
    use_class_head: bool = True
    presence_dropout: float = 0.0
    classifier_dropout: float = 0.1
    layer_norm_mode: str = 'all'  # 'none' | 'all' | 'first' | 'second'

    # Optimization
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    batch_size: Optional[int] = 256

    # Labels / class balancing
    class_lo_thr: float = 0.1
    class_hi_thr: float = 0.9
    class_label_smoothing: float = 0.0
    class_weight_maternal: float = 1.0
    class_weight_fetal: float = 1.0
    class_weight_mix: float = 1.0
    loss_type: str = 'ce'   # 'ce' | 'focal'
    focal_gamma: float = 2.0
    cls_loss_weight: float = 1.0
    prop_loss_weight: float = 1.0
    prop_consistency_weight: float = 0.2
    aac_purity_weight: float = 0.1
    prop_loss_type: str = 'huber'  # 'huber' | 'mse'
    aac_column: str = 'auto'

    # Monitoring
    monitor: str = 'cls_acc'
    early_stop_patience: int = 15
    min_epochs: int = 0
    min_delta: float = 0.0
    eval_every: int = 1

    # Data
    matrix_type: str = 'cp10k_log1p'

    # Runtime
    distributed: bool = False
    seed: Optional[int] = None


def set_global_seed(seed: Optional[int], rank: int = 0) -> Optional[int]:
    if seed is None:
        return None

    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    return effective_seed


def get_gpu_memory_stats(device: torch.device) -> str:
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return 'CPU mode'


def init_distributed() -> Tuple[int, int, torch.device]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device(f'cuda:{local_rank}')

        if rank == 0:
            print(f"[Distributed] Initialized with {world_size} processes")

        return rank, world_size, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return 0, 1, device


def ddp_barrier(distributed: bool):
    if distributed and torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def bcast_float(val: float, src: int, device: torch.device) -> float:
    if not torch.distributed.is_initialized():
        return val
    t = torch.tensor(val, dtype=torch.float32, device=device)
    torch.distributed.broadcast(t, src=src)
    return float(t.item())


def bcast_int(val: int, src: int, device: torch.device) -> int:
    if not torch.distributed.is_initialized():
        return val
    t = torch.tensor(val, dtype=torch.long, device=device)
    torch.distributed.broadcast(t, src=src)
    return int(t.item())


def bcast_flag(val: bool, src: int, device: torch.device) -> bool:
    if not torch.distributed.is_initialized():
        return val
    t = torch.tensor(int(val), dtype=torch.long, device=device)
    torch.distributed.broadcast(t, src=src)
    return bool(t.item())