"""
Data loading utilities for spatial transcriptomics
"""
import json
import sys
import platform
import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse as sp


def read_metadata_table_auto(path: Path) -> pd.DataFrame:
    """Auto-detect and read parquet/csv/tsv files"""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    else:
        for suf, sep in [(".parquet", None), (".csv", ","), (".tsv", "\t")]:
            alt = path.with_suffix(suf)
            if alt.exists():
                return pd.read_parquet(alt) if suf == ".parquet" else pd.read_csv(alt, sep=sep)
        raise FileNotFoundError(f"Could not read table: {path}")


def load_spot_and_gene_metadata(out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load spots and genes metadata"""
    spots = read_metadata_table_auto(out_root / "spots")
    genes = read_metadata_table_auto(out_root / "genes")
    return spots, genes


def load_slide_matrix_csr(out_root: Path, slide_id: str, matrix_type: str = 'raw') -> sp.csr_matrix:
    """
    Load count matrix without normalization

    Args:
        out_root: Directory containing count matrices
        slide_id: Slide identifier
        matrix_type: Type of matrix to load ('binary', 'raw', 'cp10k_log1p', 'cp10k_int', 'spot_max_bin', 'spot_rank_norm')

    Returns:
        CSR sparse matrix (as stored in file)
    """
    if matrix_type == 'binary':
        p = out_root / f"counts_binary_{slide_id}.npz"
    elif matrix_type == 'raw':
        p = out_root / f"counts_raw_{slide_id}.npz"
    elif matrix_type == 'cp10k_int':
        p = out_root / f"counts_cp10k_int_{slide_id}.npz"
    elif matrix_type == 'cp10k_log1p':
        p = out_root / f"counts_cp10k_log1p_{slide_id}.npz"
    elif matrix_type == 'spot_max_bin':
        p = out_root / f"counts_spot_max_bin_{slide_id}.npz"
    elif matrix_type == 'spot_rank_norm':
        p = out_root / f"counts_spot_rank_norm_{slide_id}.npz"
    else:
        raise ValueError(f"Unknown matrix_type: {matrix_type}. Choose from 'binary', 'raw', 'cp10k_int', 'cp10k_log1p', 'spot_max_bin', 'spot_rank_norm'")

    if not p.exists():
        raise FileNotFoundError(f"Count matrix not found: {p}")
    return sp.load_npz(p)


def save_training_run_config(args, cfg, exp_dir: Path):
    """Save training configuration"""
    exp_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "cfg": vars(cfg),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    }
    if torch.cuda.is_available():
        run["env"]["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    (exp_dir / "run_config.json").write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str))


def csr_rows_to_torch_sparse_batch(M: sp.csr_matrix, rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert CSR matrix rows to PyTorch sparse tensor
    MEMORY EFFICIENT: On-the-fly conversion without caching entire matrix
    """
    sub = M[rows, :].tocsr().tocoo()
    idx = torch.from_numpy(np.vstack([sub.row, sub.col])).long()
    vals = torch.from_numpy(sub.data).float()
    sparse_tensor = torch.sparse_coo_tensor(idx, vals, size=sub.shape, device=device)
    return sparse_tensor.coalesce()
