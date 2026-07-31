"""Binary-mask multi-task classification package."""

from .architecture import (
    SpotStateClassificationHead,
    FetalProportionRegressionHead,
    GeneFeatureEncoder,
    MultiTaskArchitecture,
)
from .data_loader import load_slide_matrix_csr, load_spot_and_gene_metadata, save_training_run_config
from .losses import SPOT_STATE_NAMES, build_spot_state_labels, compute_multitask_loss, compute_multitask_metrics
from .trainer import Trainer
from .utils import (
    TrainConfig,
    bcast_flag,
    bcast_float,
    bcast_int,
    ddp_barrier,
    get_gpu_memory_stats,
    init_distributed,
    is_main_process,
    set_global_seed,
    unwrap,
)

__all__ = [
    'SpotStateClassificationHead',
    'SPOT_STATE_NAMES',
    'FetalProportionRegressionHead',
    'GeneFeatureEncoder',
    'MultiTaskArchitecture',
    'Trainer',
    'TrainConfig',
    'build_spot_state_labels',
    'compute_multitask_loss',
    'compute_multitask_metrics',
    'load_slide_matrix_csr',
    'load_spot_and_gene_metadata',
    'save_training_run_config',
    'bcast_flag',
    'bcast_float',
    'bcast_int',
    'ddp_barrier',
    'get_gpu_memory_stats',
    'init_distributed',
    'is_main_process',
    'set_global_seed',
    'unwrap',
]
