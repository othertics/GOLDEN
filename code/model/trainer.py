"""Training and evaluation logic for binary-mask DeepSet classification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from .architecture import MultiTaskArchitecture
from .data_loader import csr_rows_to_torch_sparse_batch, load_slide_matrix_csr, load_spot_and_gene_metadata
from .losses import SPOT_STATE_NAMES, compute_multitask_loss, compute_multitask_metrics
from .utils import (
    TrainConfig,
    bcast_flag,
    bcast_float,
    bcast_int,
    ddp_barrier,
    get_gpu_memory_stats,
    is_main_process,
    unwrap,
)


class Trainer:
    """Trainer for maternal / fetal / mix classification."""

    AGGREGATE_METRIC_KEYS = (
        'loss_total', 'cls_loss', 'cls_acc', 'cls_macro_acc',
        'cls_acc_maternal', 'cls_acc_fetal', 'cls_acc_mix',
        'prop_loss', 'prop_mae', 'prop_rmse', 'prop_corr',
    )

    def __init__(
        self,
        out_root: Path,
        slides_train: List[str],
        slides_val: List[str],
        slides_test: List[str],
        cfg: TrainConfig,
        device: torch.device,
        exp_dir: Path,
        rank: int,
        world_size: int,
        load_checkpoint: Optional[Path] = None,
        matrix_type: str = 'cp10k_log1p',
        hvg_json: Optional[Path] = None,
    ):
        self.out_root = out_root
        self.slides_train = [str(s) for s in slides_train]
        self.slides_val = [str(s) for s in slides_val]
        self.slides_test = [str(s) for s in slides_test]
        self.cfg = cfg
        self.device = device
        self.exp_dir = exp_dir
        self.rank = rank
        self.world_size = world_size
        self.load_checkpoint = load_checkpoint
        self.matrix_type = matrix_type
        self.hvg_json = hvg_json

        self.spots = None
        self.genes = None
        self.n_genes = None
        self.hvg_indices = None
        self.per_slide = {}

        self.model = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None

        self.history = {
            'epoch': [],
            'lr': [],
            'train_loss_total': [],
            'train_cls_loss': [],
            'train_cls_acc': [],
            'train_cls_macro_acc': [],
            'train_prop_loss': [],
            'train_prop_mae': [],
            'train_prop_corr': [],
            'val_loss_total': [],
            'val_cls_loss': [],
            'val_cls_acc': [],
            'val_cls_macro_acc': [],
            'val_cls_acc_maternal': [],
            'val_cls_acc_fetal': [],
            'val_cls_acc_mix': [],
            'val_prop_loss': [],
            'val_prop_mae': [],
            'val_prop_corr': [],
        }
        self.best_epoch = 0
        self.best_score = float('-inf') if self._higher_is_better(self.cfg.monitor) else float('inf')

    def _higher_is_better(self, metric_name: str) -> bool:
        name = str(metric_name).lower()
        return not any(tok in name for tok in ('loss', 'mae', 'rmse'))

    def _empty_raw_summary(self) -> dict:
        raw = {
            'loss_sum': 0.0,
            'loss_total_sum': 0.0,
            'prop_loss_sum': 0.0,
            'prop_consistency_loss_sum': 0.0,
            'aac_purity_loss_sum': 0.0,
            'n_labeled': 0,
            'n_cls_labeled': 0,
            'n_correct': 0,
            'n_prop': 0,
            'prop_abs_err_sum': 0.0,
            'prop_sq_err_sum': 0.0,
            'prop_sum_true': 0.0,
            'prop_sum_pred': 0.0,
            'prop_sum_true2': 0.0,
            'prop_sum_pred2': 0.0,
            'prop_sum_true_pred': 0.0,
        }
        for name in SPOT_STATE_NAMES:
            raw[f'n_class_{name}'] = 0
            raw[f'n_correct_{name}'] = 0
        return raw

    def _accumulate_raw_summary(self, raw: dict, metrics: dict):
        n_labeled = int(metrics.get('n_labeled', 0))
        n_cls_labeled = int(metrics.get('n_cls_labeled', 0))
        raw['n_labeled'] += n_labeled
        raw['n_cls_labeled'] += n_cls_labeled
        raw['n_correct'] += int(metrics.get('n_correct', 0))

        cls_loss = metrics.get('cls_loss', float('nan'))
        if n_cls_labeled > 0 and np.isfinite(cls_loss):
            raw['loss_sum'] += float(cls_loss) * n_cls_labeled

        loss_total = metrics.get('loss_total', float('nan'))
        if n_labeled > 0 and np.isfinite(loss_total):
            raw['loss_total_sum'] += float(loss_total) * n_labeled

        prop_loss = metrics.get('prop_loss', float('nan'))
        if n_labeled > 0 and np.isfinite(prop_loss):
            raw['prop_loss_sum'] += float(prop_loss) * n_labeled

        cons_loss = metrics.get('prop_consistency_loss', float('nan'))
        if n_labeled > 0 and np.isfinite(cons_loss):
            raw['prop_consistency_loss_sum'] += float(cons_loss) * n_labeled

        purity_loss = metrics.get('aac_purity_loss', float('nan'))
        if n_labeled > 0 and np.isfinite(purity_loss):
            raw['aac_purity_loss_sum'] += float(purity_loss) * n_labeled

        raw['n_prop'] += int(metrics.get('n_prop', 0))
        raw['prop_abs_err_sum'] += float(metrics.get('_prop_abs_err_sum', 0.0))
        raw['prop_sq_err_sum'] += float(metrics.get('_prop_sq_err_sum', 0.0))
        raw['prop_sum_true'] += float(metrics.get('_prop_sum_true', 0.0))
        raw['prop_sum_pred'] += float(metrics.get('_prop_sum_pred', 0.0))
        raw['prop_sum_true2'] += float(metrics.get('_prop_sum_true2', 0.0))
        raw['prop_sum_pred2'] += float(metrics.get('_prop_sum_pred2', 0.0))
        raw['prop_sum_true_pred'] += float(metrics.get('_prop_sum_true_pred', 0.0))

        for name in SPOT_STATE_NAMES:
            raw[f'n_class_{name}'] += int(metrics.get(f'n_class_{name}', 0))
            raw[f'n_correct_{name}'] += int(metrics.get(f'n_correct_{name}', 0))

    def _merge_raw_summaries(self, summaries: List[dict]) -> dict:
        merged = self._empty_raw_summary()
        for summary in summaries:
            if summary is None:
                continue
            merged['loss_sum'] += float(summary.get('loss_sum', 0.0))
            merged['loss_total_sum'] += float(summary.get('loss_total_sum', 0.0))
            merged['prop_loss_sum'] += float(summary.get('prop_loss_sum', 0.0))
            merged['prop_consistency_loss_sum'] += float(summary.get('prop_consistency_loss_sum', 0.0))
            merged['aac_purity_loss_sum'] += float(summary.get('aac_purity_loss_sum', 0.0))
            merged['n_labeled'] += int(summary.get('n_labeled', 0))
            merged['n_cls_labeled'] += int(summary.get('n_cls_labeled', 0))
            merged['n_correct'] += int(summary.get('n_correct', 0))
            merged['n_prop'] += int(summary.get('n_prop', 0))
            merged['prop_abs_err_sum'] += float(summary.get('prop_abs_err_sum', 0.0))
            merged['prop_sq_err_sum'] += float(summary.get('prop_sq_err_sum', 0.0))
            merged['prop_sum_true'] += float(summary.get('prop_sum_true', 0.0))
            merged['prop_sum_pred'] += float(summary.get('prop_sum_pred', 0.0))
            merged['prop_sum_true2'] += float(summary.get('prop_sum_true2', 0.0))
            merged['prop_sum_pred2'] += float(summary.get('prop_sum_pred2', 0.0))
            merged['prop_sum_true_pred'] += float(summary.get('prop_sum_true_pred', 0.0))
            for name in SPOT_STATE_NAMES:
                merged[f'n_class_{name}'] += int(summary.get(f'n_class_{name}', 0))
                merged[f'n_correct_{name}'] += int(summary.get(f'n_correct_{name}', 0))
        return merged

    def _finalize_raw_summary(self, raw: dict) -> dict:
        n_labeled = int(raw['n_labeled'])
        n_cls_labeled = int(raw['n_cls_labeled'])
        summary = {
            'loss_total': float(raw['loss_total_sum'] / n_labeled) if n_labeled > 0 else float('nan'),
            'cls_loss': float(raw['loss_sum'] / n_cls_labeled) if n_cls_labeled > 0 else float('nan'),
            'cls_acc': float(raw['n_correct'] / n_cls_labeled) if n_cls_labeled > 0 else float('nan'),
            'cls_macro_acc': float('nan'),
            'prop_loss': float(raw['prop_loss_sum'] / n_labeled) if n_labeled > 0 else float('nan'),
            'prop_consistency_loss': float(raw['prop_consistency_loss_sum'] / n_labeled) if n_labeled > 0 else float('nan'),
            'aac_purity_loss': float(raw['aac_purity_loss_sum'] / n_labeled) if n_labeled > 0 else float('nan'),
            'prop_mae': float('nan'),
            'prop_rmse': float('nan'),
            'prop_corr': float('nan'),
            'n_labeled': n_labeled,
        }

        n_prop = int(raw['n_prop'])
        if n_prop > 0:
            summary['prop_mae'] = float(raw['prop_abs_err_sum'] / n_prop)
            summary['prop_rmse'] = float(np.sqrt(raw['prop_sq_err_sum'] / n_prop))
            mean_true = raw['prop_sum_true'] / n_prop
            mean_pred = raw['prop_sum_pred'] / n_prop
            cov = raw['prop_sum_true_pred'] - n_prop * mean_true * mean_pred
            var_true = raw['prop_sum_true2'] - n_prop * mean_true * mean_true
            var_pred = raw['prop_sum_pred2'] - n_prop * mean_pred * mean_pred
            denom = float(np.sqrt(max(var_true, 0.0) * max(var_pred, 0.0)))
            if denom > 1e-12:
                summary['prop_corr'] = float(cov / denom)

        recall_values = []
        for name in SPOT_STATE_NAMES:
            n_class = int(raw[f'n_class_{name}'])
            n_correct_class = int(raw[f'n_correct_{name}'])
            summary[f'n_class_{name}'] = n_class
            summary[f'n_correct_{name}'] = n_correct_class
            summary[f'cls_acc_{name}'] = float(n_correct_class / n_class) if n_class > 0 else float('nan')
            if n_class > 0:
                recall_values.append(summary[f'cls_acc_{name}'])

        if recall_values:
            summary['cls_macro_acc'] = float(np.mean(recall_values))

        return summary

    def _aggregate_slide_metrics(self, per_slide_metrics: Dict[str, dict]) -> dict:
        if not per_slide_metrics:
            return {}

        aggregated = {}
        for key in self.AGGREGATE_METRIC_KEYS:
            values = np.array([m.get(key, np.nan) for m in per_slide_metrics.values()], dtype=np.float64)
            aggregated[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')
        return aggregated

    def _slides_for_current_rank(self) -> List[str]:
        return [slide_id for idx, slide_id in enumerate(self.slides_train) if idx % self.world_size == self.rank]

    def _extract_p_true_np(self, spots_s: pd.DataFrame) -> np.ndarray:
        n = len(spots_s)
        p_true_np = np.full(n, np.nan, dtype=np.float32)

        if 'proportion' in spots_s.columns:
            p_true_np = pd.to_numeric(spots_s['proportion'], errors='coerce').to_numpy(dtype=np.float32)
        elif 'fetal' in spots_s.columns:
            p_true_np = pd.to_numeric(spots_s['fetal'], errors='coerce').to_numpy(dtype=np.float32)
        elif 'fetal_cells' in spots_s.columns and 'composition_total_cells' in spots_s.columns:
            fetal_np = pd.to_numeric(spots_s['fetal_cells'], errors='coerce').to_numpy(dtype=np.float32)
            total_np = pd.to_numeric(spots_s['composition_total_cells'], errors='coerce').to_numpy(dtype=np.float32)
            valid = np.isfinite(fetal_np) & np.isfinite(total_np) & (total_np > 0)
            p_true_np[valid] = fetal_np[valid] / total_np[valid]
        elif 'fetal_cells' in spots_s.columns and 'maternal_cells' in spots_s.columns:
            fetal_np = pd.to_numeric(spots_s['fetal_cells'], errors='coerce').to_numpy(dtype=np.float32)
            maternal_np = pd.to_numeric(spots_s['maternal_cells'], errors='coerce').to_numpy(dtype=np.float32)
            denom = fetal_np + maternal_np
            valid = np.isfinite(fetal_np) & np.isfinite(denom) & (denom > 0)
            p_true_np[valid] = fetal_np[valid] / denom[valid]

        return np.clip(p_true_np, 0.0, 1.0)

    def _extract_p_true(self, spots_s: pd.DataFrame, batch_indices: np.ndarray) -> Optional[torch.Tensor]:
        p_true_np = self._extract_p_true_np(spots_s.iloc[batch_indices])
        if np.isfinite(p_true_np).sum() < 1:
            return None
        return torch.from_numpy(p_true_np).to(self.device)

    def _build_class_labels_np(self, spots_s: pd.DataFrame) -> np.ndarray:
        p_true_np = self._extract_p_true_np(spots_s)
        labels = np.full(len(spots_s), -1, dtype=np.int64)
        valid = np.isfinite(p_true_np)
        labels[valid & (p_true_np <= float(self.cfg.class_lo_thr))] = 0
        labels[valid & (p_true_np >= float(self.cfg.class_hi_thr))] = 1
        labels[valid & (p_true_np > float(self.cfg.class_lo_thr)) & (p_true_np < float(self.cfg.class_hi_thr))] = 2
        return labels

    def _aac_source_column(self, spots_s: pd.DataFrame) -> Optional[str]:
        requested = str(getattr(self.cfg, 'aac_column', 'auto')).strip()
        if requested and requested.lower() != 'auto' and requested in spots_s.columns:
            return requested
        for col in ['alt_count_sum', 'aac', 'alt_reads']:
            if col in spots_s.columns:
                return col
        return None

    def _attach_aac_features(self, spots_s: pd.DataFrame) -> pd.DataFrame:
        spots_s = spots_s.copy()
        source_col = self._aac_source_column(spots_s)
        if source_col is None:
            spots_s['aac_valid'] = 0.0
            spots_s['aac_z'] = 0.0
            spots_s['aac_pct'] = 0.5
            spots_s['aac_purity'] = 0.0
            return spots_s

        raw = pd.to_numeric(spots_s[source_col], errors='coerce').astype(np.float32)
        valid = np.isfinite(raw.to_numpy())
        z = np.zeros(len(spots_s), dtype=np.float32)
        pct = np.full(len(spots_s), 0.5, dtype=np.float32)

        if valid.any():
            raw_valid = np.maximum(raw.to_numpy()[valid], 0.0)
            log_valid = np.log1p(raw_valid).astype(np.float32)
            if log_valid.size >= 2:
                std = float(np.std(log_valid))
                if std > 1e-8:
                    z[valid] = ((log_valid - float(np.mean(log_valid))) / std).astype(np.float32)
                rank = pd.Series(log_valid).rank(method='average', pct=True).to_numpy(dtype=np.float32)
                pct[valid] = rank
            else:
                pct[valid] = 1.0

        purity = np.abs(pct - 0.5).astype(np.float32) * 2.0
        spots_s['aac_valid'] = valid.astype(np.float32)
        spots_s['aac_z'] = z
        spots_s['aac_pct'] = pct
        spots_s['aac_purity'] = purity
        return spots_s

    def _extract_aac_features(self, spots_s: pd.DataFrame, batch_indices: np.ndarray) -> torch.Tensor:
        cols = ['aac_valid', 'aac_z', 'aac_pct', 'aac_purity']
        if not all(col in spots_s.columns for col in cols):
            arr = np.zeros((len(batch_indices), 4), dtype=np.float32)
        else:
            arr = spots_s.iloc[batch_indices][cols].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(arr).to(self.device)

    def _cache_slide_data(self, sid: str, spots_s: pd.DataFrame, M_s) -> None:
        """Store per-slide data. Override in subclasses for custom caching strategies."""
        self.per_slide[sid] = (spots_s, M_s)

    @staticmethod
    def _iter_batch_indices(n_items: int, batch_size: int):
        for start in range(0, n_items, batch_size):
            yield np.arange(start, min(start + batch_size, n_items))

    def _build_prediction_dataframe(
        self,
        spots_s: pd.DataFrame,
        p_true_np: np.ndarray,
        pred_class: np.ndarray,
        p_pred: np.ndarray,
        probs: np.ndarray,
        fetal_score: np.ndarray,
    ) -> pd.DataFrame:
        n_spots = len(spots_s)
        pred_df = pd.DataFrame({
            'barcode': spots_s['barcode'].to_numpy() if 'barcode' in spots_s.columns else np.arange(n_spots),
            'pred_class': pred_class,
            'p_pred': p_pred,
            'prob_maternal': probs[:, 0],
            'prob_fetal': probs[:, 1],
            'prob_mix': probs[:, 2],
            'fetal_score': fetal_score,
        })

        for col in ['x', 'y', 'array_row', 'array_col']:
            if col in spots_s.columns:
                pred_df[col] = spots_s[col].to_numpy()

        if np.isfinite(p_true_np).any():
            pred_df['p_true'] = p_true_np
            pred_df['true_class'] = self._build_class_labels_np(spots_s)

        for col in ['alt_count_sum', 'aac_valid', 'aac_z', 'aac_pct', 'aac_purity']:
            if col in spots_s.columns:
                pred_df[col] = spots_s[col].to_numpy()

        return pred_df

    def prepare_data(self):
        self.spots, self.genes = load_spot_and_gene_metadata(self.out_root)
        if 'slide_id' in self.spots.columns:
            self.spots['slide_id'] = self.spots['slide_id'].astype(str)

        if self.hvg_json is not None and self.hvg_json.exists():
            with open(self.hvg_json, 'r') as f:
                hvg_data = json.load(f)
            hvg_genes = hvg_data['hvg_genes']

            if 'gene_name' in self.genes.columns:
                gene_names = self.genes['gene_name'].values
            elif 'gene' in self.genes.columns:
                gene_names = self.genes['gene'].values
            else:
                gene_names = self.genes.index.values

            gene_name_to_idx = {name: idx for idx, name in enumerate(gene_names)}
            self.hvg_indices = np.array([gene_name_to_idx[g] for g in hvg_genes if g in gene_name_to_idx], dtype=np.int32)
            self.genes = self.genes.iloc[self.hvg_indices].reset_index(drop=True)
            self.n_genes = len(self.hvg_indices)
        else:
            self.hvg_indices = None
            self.n_genes = len(self.genes)

        self.cfg.n_genes = self.n_genes

        train_label_counts = np.zeros(3, dtype=np.int64)
        all_slides = self.slides_train + self.slides_val + self.slides_test

        for sid in all_slides:
            spots_s = self.spots[self.spots['slide_id'] == sid].reset_index(drop=True)
            M_s = load_slide_matrix_csr(self.out_root, sid, matrix_type=self.matrix_type)

            if self.hvg_indices is not None:
                M_s = M_s[:, self.hvg_indices]

            if len(spots_s) != M_s.shape[0]:
                min_len = min(len(spots_s), M_s.shape[0])
                spots_s = spots_s.iloc[:min_len].reset_index(drop=True)
                M_s = M_s[:min_len, :]

            spots_s = self._attach_aac_features(spots_s)
            self._cache_slide_data(sid, spots_s, M_s)

            if sid in self.slides_train:
                labels = self._build_class_labels_np(spots_s)
                for cls_idx in range(3):
                    train_label_counts[cls_idx] += int((labels == cls_idx).sum())

        if is_main_process(self.rank):
            total = int(train_label_counts.sum())
            if total > 0:
                fracs = train_label_counts / total
                print(
                    f"[ClassDist][train] maternal={train_label_counts[0]} ({fracs[0]:.4f}), "
                    f"fetal={train_label_counts[1]} ({fracs[1]:.4f}), "
                    f"mix={train_label_counts[2]} ({fracs[2]:.4f})"
                )

        default_weights = (
            abs(float(self.cfg.class_weight_maternal) - 1.0) < 1e-12 and
            abs(float(self.cfg.class_weight_fetal) - 1.0) < 1e-12 and
            abs(float(self.cfg.class_weight_mix) - 1.0) < 1e-12
        )
        total = int(train_label_counts.sum())
        if total > 0 and default_weights:
            counts = train_label_counts.astype(np.float64)
            inv = np.zeros_like(counts)
            nz = counts > 0
            inv[nz] = 1.0 / counts[nz]
            if inv[nz].sum() > 0:
                inv = inv / inv[nz].mean()
            self.cfg.class_weight_maternal = float(inv[0]) if counts[0] > 0 else 1.0
            self.cfg.class_weight_fetal = float(inv[1]) if counts[1] > 0 else 1.0
            self.cfg.class_weight_mix = float(inv[2]) if counts[2] > 0 else 1.0

            if is_main_process(self.rank):
                print(
                    f"[ClassWeight][auto] maternal={self.cfg.class_weight_maternal:.4f}, "
                    f"fetal={self.cfg.class_weight_fetal:.4f}, mix={self.cfg.class_weight_mix:.4f}"
                )

    def build_model(self):
        self.model = MultiTaskArchitecture(
            n_genes=self.n_genes,
            mlp_hidden_dims=self.cfg.mlp_hidden_dims,
            cls_hidden_dim=self.cfg.cls_hidden_dim,
            prop_hidden_dim=self.cfg.prop_hidden_dim,
            input_mode=self.cfg.input_mode,
            use_class_head=self.cfg.use_class_head,
            presence_dropout=self.cfg.presence_dropout,
            classifier_dropout=self.cfg.classifier_dropout,
            layer_norm_mode=self.cfg.layer_norm_mode,
        ).to(self.device)

        if self.cfg.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scaler = GradScaler() if self.cfg.use_amp else None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max' if self._higher_is_better(self.cfg.monitor) else 'min',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            threshold=0.001,
            threshold_mode='rel',
        )

        if self.load_checkpoint is not None and self.load_checkpoint.exists():
            checkpoint = torch.load(self.load_checkpoint, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
            unwrap(self.model).load_state_dict(state_dict, strict=True)
            if is_main_process(self.rank):
                print(f"Loaded checkpoint: {self.load_checkpoint}")

    def _train_step(
        self,
        x_batch: torch.Tensor,
        p_true_batch: Optional[torch.Tensor],
        aac_batch: torch.Tensor,
    ):
        """Single optimizer step. Handles AMP and non-AMP paths."""
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is not None:
            with autocast():
                loss, metrics = compute_multitask_loss(self.model, x_batch, self.cfg, p_true=p_true_batch, aac_features=aac_batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, metrics = compute_multitask_loss(self.model, x_batch, self.cfg, p_true=p_true_batch, aac_features=aac_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return loss, metrics

    def train_epoch(self, epoch: int) -> dict:
        del epoch
        self.model.train()

        raw_summary = self._empty_raw_summary()
        slides_this_rank = self._slides_for_current_rank()

        for slide_id in slides_this_rank:
            spots_s, M_s = self.per_slide[slide_id]
            n_spots = M_s.shape[0]
            batch_size = self.cfg.batch_size or n_spots
            shuffled_indices = np.random.permutation(n_spots)

            for batch_start in range(0, n_spots, batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
                x_sparse = csr_rows_to_torch_sparse_batch(M_s, batch_indices, self.device)
                p_true_batch = self._extract_p_true(spots_s, batch_indices)
                aac_batch = self._extract_aac_features(spots_s, batch_indices)

                loss, metrics = self._train_step(x_sparse, p_true_batch, aac_batch)
                self._accumulate_raw_summary(raw_summary, metrics)
                del x_sparse, p_true_batch, aac_batch, loss

        if self.cfg.distributed:
            gathered = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, raw_summary)
            raw_summary = self._merge_raw_summaries(gathered)
            ddp_barrier(self.cfg.distributed)

        return self._finalize_raw_summary(raw_summary)

    def evaluate_split(self, slide_ids: List[str]) -> Dict[str, dict]:
        if not is_main_process(self.rank):
            return {}

        self.model.eval()
        results = {}

        for sid in slide_ids:
            spots_s, M_s = self.per_slide[sid]
            n_spots = M_s.shape[0]
            batch_size = self.cfg.batch_size or n_spots
            raw_summary = self._empty_raw_summary()

            for batch_indices in self._iter_batch_indices(n_spots, batch_size):
                x_sparse = csr_rows_to_torch_sparse_batch(M_s, batch_indices, self.device)
                p_true_batch = self._extract_p_true(spots_s, batch_indices)
                aac_batch = self._extract_aac_features(spots_s, batch_indices)

                batch_metrics = compute_multitask_metrics(self.model, x_sparse, p_true=p_true_batch, cfg=self.cfg, aac_features=aac_batch)
                self._accumulate_raw_summary(raw_summary, batch_metrics)
                del x_sparse, p_true_batch, aac_batch

            results[sid] = self._finalize_raw_summary(raw_summary)

        return results

    def evaluate(self) -> Dict[str, dict]:
        return self.evaluate_split(self.slides_val)

    def save_predictions(self):
        if not is_main_process(self.rank):
            return

        self.model.eval()
        for slide_id in self.slides_train + self.slides_val + self.slides_test:
            spots_s, M_s = self.per_slide[slide_id]
            n_spots = M_s.shape[0]
            batch_size = self.cfg.batch_size or n_spots

            all_probs, all_classes, all_scores, all_p_pred = [], [], [], []

            for batch_indices in self._iter_batch_indices(n_spots, batch_size):
                x_sparse = csr_rows_to_torch_sparse_batch(M_s, batch_indices, self.device)
                aac_batch = self._extract_aac_features(spots_s, batch_indices)

                with torch.no_grad():
                    fetal_score, aux = unwrap(self.model)(x_sparse, aac_features=aac_batch, return_aux=True)
                    p_pred = aux['p_pred'].cpu().numpy()
                    probs_t = aux['probabilities']
                    if probs_t is None:
                        probs = np.full((len(batch_indices), 3), np.nan, dtype=np.float32)
                        pred_class = np.full(len(batch_indices), -1, dtype=np.int64)
                        score_np = np.full(len(batch_indices), np.nan, dtype=np.float32)
                    else:
                        probs = probs_t.cpu().numpy()
                        pred_class = np.argmax(probs, axis=1)
                        score_np = fetal_score.cpu().numpy()

                all_scores.append(score_np)
                all_probs.append(probs)
                all_classes.append(pred_class)
                all_p_pred.append(p_pred)
                del x_sparse, aac_batch

            probs_all = np.concatenate(all_probs, axis=0)
            pred_all = np.concatenate(all_classes, axis=0)
            score_all = np.concatenate(all_scores, axis=0)
            p_pred_all = np.concatenate(all_p_pred, axis=0)
            p_true_np = self._extract_p_true_np(spots_s)
            pred_df = self._build_prediction_dataframe(
                spots_s=spots_s,
                p_true_np=p_true_np,
                pred_class=pred_all,
                p_pred=p_pred_all,
                probs=probs_all,
                fetal_score=score_all,
            )

            pred_df.to_csv(self.exp_dir / f'pred_{slide_id}.csv', index=False)

        n_total = len(self.slides_train + self.slides_val + self.slides_test)
        print(f"  ✓ Predictions saved for {n_total} slides")

    def save_training_history(self):
        if not is_main_process(self.rank):
            return

        history_file = self.exp_dir / 'training_history.json'
        payload = {
            'metadata': {
                'best_epoch': self.best_epoch,
                'best_score': self.best_score,
                'monitor': self.cfg.monitor,
                'total_epochs': len(self.history['epoch']),
            },
            'history': self.history,
        }
        history_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"  ✓ Training history saved: {history_file}")

    def save_metrics(self):
        if not is_main_process(self.rank):
            return

        results = {
            'val': self.evaluate_split(self.slides_val),
            'test': self.evaluate_split(self.slides_test),
        }
        metrics_path = self.exp_dir / 'metrics.json'
        metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  ✓ Final metrics saved: {metrics_path}")

    def _checkpoint_payload(self) -> dict:
        return {
            'model_state_dict': unwrap(self.model).state_dict(),
            'config': {
                'task': self.cfg.task,
                'n_genes': self.n_genes,
                'mlp_hidden_dims': self.cfg.mlp_hidden_dims,
                'cls_hidden_dim': self.cfg.cls_hidden_dim,
                'prop_hidden_dim': self.cfg.prop_hidden_dim,
                'input_mode': self.cfg.input_mode,
                'use_class_head': self.cfg.use_class_head,
                'layer_norm_mode': self.cfg.layer_norm_mode,
                'matrix_type': self.matrix_type,
            },
        }

    def _record_history(self, epoch: int, train_summary: dict, val_summary: dict):
        if not is_main_process(self.rank):
            return

        self.history['epoch'].append(epoch + 1)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        self.history['train_loss_total'].append(train_summary.get('loss_total', float('nan')))
        self.history['train_cls_loss'].append(train_summary.get('cls_loss', float('nan')))
        self.history['train_cls_acc'].append(train_summary.get('cls_acc', float('nan')))
        self.history['train_cls_macro_acc'].append(train_summary.get('cls_macro_acc', float('nan')))
        self.history['train_prop_loss'].append(train_summary.get('prop_loss', float('nan')))
        self.history['train_prop_mae'].append(train_summary.get('prop_mae', float('nan')))
        self.history['train_prop_corr'].append(train_summary.get('prop_corr', float('nan')))
        self.history['val_loss_total'].append(val_summary.get('loss_total', float('nan')))
        self.history['val_cls_loss'].append(val_summary.get('cls_loss', float('nan')))
        self.history['val_cls_acc'].append(val_summary.get('cls_acc', float('nan')))
        self.history['val_cls_macro_acc'].append(val_summary.get('cls_macro_acc', float('nan')))
        self.history['val_cls_acc_maternal'].append(val_summary.get('cls_acc_maternal', float('nan')))
        self.history['val_cls_acc_fetal'].append(val_summary.get('cls_acc_fetal', float('nan')))
        self.history['val_cls_acc_mix'].append(val_summary.get('cls_acc_mix', float('nan')))
        self.history['val_prop_loss'].append(val_summary.get('prop_loss', float('nan')))
        self.history['val_prop_mae'].append(val_summary.get('prop_mae', float('nan')))
        self.history['val_prop_corr'].append(val_summary.get('prop_corr', float('nan')))

    def _log_epoch(self, epoch: int, train_summary: dict, val_summary: dict, patience_left: int, do_eval: bool):
        if not is_main_process(self.rank):
            return

        if do_eval and val_summary:
            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                f"train: total={train_summary.get('loss_total', float('nan')):.4f}, "
                f"cls={train_summary.get('cls_loss', float('nan')):.4f}, "
                f"prop_mae={train_summary.get('prop_mae', float('nan')):.4f}, "
                f"acc={train_summary.get('cls_acc', float('nan')):.4f}, "
                f"macro={train_summary.get('cls_macro_acc', float('nan')):.4f} | "
                f"val: total={val_summary.get('loss_total', float('nan')):.4f}, "
                f"cls={val_summary.get('cls_loss', float('nan')):.4f}, "
                f"prop_mae={val_summary.get('prop_mae', float('nan')):.4f}, "
                f"prop_corr={val_summary.get('prop_corr', float('nan')):.4f}, "
                f"acc={val_summary.get('cls_acc', float('nan')):.4f}, "
                f"macro={val_summary.get('cls_macro_acc', float('nan')):.4f}, "
                f"maternal={val_summary.get('cls_acc_maternal', float('nan')):.4f}, "
                f"fetal={val_summary.get('cls_acc_fetal', float('nan')):.4f}, "
                f"mix={val_summary.get('cls_acc_mix', float('nan')):.4f} | "
                f"patience={patience_left}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{self.cfg.epochs} | "
                f"train: total={train_summary.get('loss_total', float('nan')):.4f}, "
                f"cls={train_summary.get('cls_loss', float('nan')):.4f}, "
                f"prop_mae={train_summary.get('prop_mae', float('nan')):.4f}, "
                f"acc={train_summary.get('cls_acc', float('nan')):.4f}, "
                f"macro={train_summary.get('cls_macro_acc', float('nan')):.4f}"
            )

    def train(self) -> float:
        if (not bool(getattr(self.cfg, 'use_class_head', True))) and str(self.cfg.monitor).startswith('cls_'):
            raise ValueError('Classification monitors require use_class_head=True')

        self.prepare_data()
        self.build_model()
        patience_left = self.cfg.early_stop_patience

        if is_main_process(self.rank):
            batch_size_str = 'entire slide' if self.cfg.batch_size is None else str(self.cfg.batch_size)
            print(f"\n{'=' * 80}")
            print('Binary-mask DeepSet Classification')
            print(f"  Train slides: {len(self.slides_train)}")
            print(f"  Val slides: {len(self.slides_val)}")
            print(f"  Test slides: {len(self.slides_test)}")
            print(f"  Genes: {self.n_genes}")
            print(f"  Device: {self.device}")
            print(f"  Mixed Precision: {self.cfg.use_amp}")
            print(f"  Batch Size: {batch_size_str}")
            print('  Architecture:')
            mlp_dims_str = ' -> '.join(str(int(d)) for d in self.cfg.mlp_hidden_dims)
            print(f"    Input mode: {self.cfg.input_mode} ({self.n_genes} genes)")
            print(f"    Encoder (MLP): {self.n_genes} -> {mlp_dims_str}")
            print(f"    LayerNorm mode: {self.cfg.layer_norm_mode}")
            if self.cfg.use_class_head:
                print(f"    Classifier = MLP({self.cfg.mlp_hidden_dims[-1]} -> {self.cfg.cls_hidden_dim} -> 3)")
                print(f"    Proportion head = MLP({self.cfg.mlp_hidden_dims[-1]} + 3 + AAC[4] -> {self.cfg.prop_hidden_dim} -> {self.cfg.prop_hidden_dim // 2} -> 1)")
            else:
                print("    Classifier = disabled")
                print(f"    Proportion head = MLP({self.cfg.mlp_hidden_dims[-1]} + AAC[4] -> {self.cfg.prop_hidden_dim} -> {self.cfg.prop_hidden_dim // 2} -> 1)")
            print('  Labels:')
            print(f"    maternal if p_true <= {self.cfg.class_lo_thr}")
            print(f"    fetal if p_true >= {self.cfg.class_hi_thr}")
            print(f"    mix otherwise")
            print(
                f"  Class weights: maternal={self.cfg.class_weight_maternal}, "
                f"fetal={self.cfg.class_weight_fetal}, mix={self.cfg.class_weight_mix}"
            )
            print(
                f"  Multi-task weights: cls={self.cfg.cls_loss_weight}, prop={self.cfg.prop_loss_weight}, "
                f"consistency={self.cfg.prop_consistency_weight}, aac_purity={self.cfg.aac_purity_weight}"
            )
            print(f"  AAC column: {self.cfg.aac_column}")
            print(f"  Loss: {self.cfg.loss_type} (focal_gamma={self.cfg.focal_gamma})")
            print(f"  Monitor: {self.cfg.monitor} ({'higher' if self._higher_is_better(self.cfg.monitor) else 'lower'} is better)")
            print(f"  {get_gpu_memory_stats(self.device)}")
            print(f"{'=' * 80}\n")

        for epoch in range(self.cfg.epochs):
            train_summary = self.train_epoch(epoch)

            do_eval = ((epoch + 1) % self.cfg.eval_every == 0) and len(self.slides_val) > 0
            val_per_slide = self.evaluate() if do_eval else {}
            val_summary = self._aggregate_slide_metrics(val_per_slide) if do_eval else {}

            if is_main_process(self.rank) and do_eval:
                current_score = val_summary.get(self.cfg.monitor, float('nan'))
                improved = False

                if np.isfinite(current_score):
                    if self._higher_is_better(self.cfg.monitor):
                        improved = current_score > self.best_score + self.cfg.min_delta
                    else:
                        improved = current_score < self.best_score - self.cfg.min_delta

                if improved:
                    self.best_score = current_score
                    self.best_epoch = epoch + 1
                    patience_left = self.cfg.early_stop_patience
                    torch.save(self._checkpoint_payload(), self.exp_dir / 'best.ckpt')
                    print(f"  → New best model saved (epoch {self.best_epoch}, score={self.best_score:.4f})")
                elif np.isfinite(current_score):
                    patience_left -= 1

                if np.isfinite(current_score):
                    prev_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(current_score)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if new_lr < prev_lr:
                        print(f"  → Learning rate reduced: {prev_lr:.2e} → {new_lr:.2e}")

            if self.cfg.distributed and do_eval:
                self.best_score = bcast_float(self.best_score, src=0, device=self.device)
                patience_left = bcast_int(patience_left, src=0, device=self.device)

            should_stop = (
                len(self.slides_val) > 0 and
                do_eval and
                (epoch + 1) >= self.cfg.min_epochs and
                patience_left <= 0
            )
            if self.cfg.distributed:
                should_stop = bcast_flag(should_stop, src=0, device=self.device)

            self._record_history(epoch, train_summary, val_summary)
            self._log_epoch(epoch, train_summary, val_summary, patience_left, do_eval)

            if should_stop:
                if is_main_process(self.rank):
                    print('Early stopping triggered')
                break

        if is_main_process(self.rank):
            torch.save(self._checkpoint_payload(), self.exp_dir / 'last.ckpt')
            print(f"\n{'=' * 80}")
            print(f"  ✓ Last checkpoint saved: {self.exp_dir / 'last.ckpt'}")
            self.save_training_history()

        if is_main_process(self.rank) and (self.exp_dir / 'best.ckpt').exists():
            ckpt = torch.load(self.exp_dir / 'best.ckpt', map_location='cpu')
            unwrap(self.model).load_state_dict(ckpt['model_state_dict'], strict=True)
            print(f"  ✓ Loaded best checkpoint from epoch {self.best_epoch}")

        if self.cfg.distributed:
            ddp_barrier(self.cfg.distributed)

        self.model.to(self.device)
        self.save_predictions()
        self.save_metrics()

        if is_main_process(self.rank):
            print(f"{'=' * 80}\n")

        return self.best_score
