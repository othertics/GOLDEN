#!/usr/bin/env python3
"""Optimized training entrypoint for binary-mask DeepSet classification.

Optimizations over the base Trainer:
- Precompute dense binary uint8 matrices per slide on CPU (avoid sparse->dense each batch).
- Cache p_true and AAC features as NumPy arrays (no per-batch pandas iloc).
- No per-batch empty_cache() / cuda.synchronize() stalls.
- Optional torch.compile for faster throughput (PyTorch 2.0+).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model import TrainConfig, init_distributed, is_main_process, save_training_run_config, set_global_seed
from model.losses import compute_multitask_metrics
from model.trainer import Trainer
from model.utils import ddp_barrier, unwrap


def parse_dims(text: str) -> list[int]:
    vals = [int(v.strip()) for v in str(text).split(',') if v.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise argparse.ArgumentTypeError('mlp-dims must contain at least one positive integer')
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Optimized binary-mask DeepSet 3-class classifier',
        allow_abbrev=False,
        add_help=False,
    )
    p.add_argument('-h', '--h', action='help', help='Show this help message and exit.')

    # Data
    p.add_argument('--out-root', type=Path, required=True,
                   help='Preprocessed data root containing genes/spots and count matrices.')
    p.add_argument('--slides', nargs='+', required=True,
                   help='Training slide IDs (space-separated).')
    p.add_argument('--val-slides', nargs='*', default=[],
                   help='Validation slide IDs (optional).')
    p.add_argument('--test-slides', nargs='*', default=[],
                   help='Test slide IDs (optional).')
    p.add_argument('--hvg-json', type=Path, default=None,
                   help='Optional JSON file listing genes to use (e.g., HVGs).')
    p.add_argument('--load-checkpoint', type=Path, default=None,
                   help='Checkpoint to initialize or resume training from.')
    p.add_argument(
        '--matrix-type',
        type=str,
        default='binary',
        choices=['binary', 'raw','cp10k_log1p'],
        help='Expression source matrix. For input-mode=binary, use matrix-type=binary.',
    )

    # Optimization
    p.add_argument('--epochs', type=int, default=100,
                   help='Maximum number of training epochs.')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Initial learning rate for AdamW.')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='Weight decay (L2 regularization) for optimizer.')
    p.add_argument('--batch-size', type=int, default=256,
                   help='Spots per batch. Lower this if you use many genes.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducibility.')
    p.add_argument('--exp-dir', type=Path, default=Path('./runs/exp'),
                   help='Directory to save checkpoints, logs, and predictions.')
    p.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    p.add_argument('--compile', action='store_true',
                   help='Enable torch.compile for faster throughput (requires PyTorch >= 2.0).')

    # Model
    p.add_argument(
        '--mlp-dims',
        type=parse_dims,
        default=None,
        help='MLP encoder hidden dimensions, e.g. 1024,256,64 (default: 1024,256,64)',
    )
    p.add_argument('--cls-hidden-dim', type=int, default=64,
                   help='Hidden dimension of the classification head MLP.')
    p.add_argument('--prop-hidden-dim', type=int, default=128,
                   help='Hidden dimension of the proportion regression head MLP.')
    p.add_argument('--input-mode', type=str, default='binary', choices=['binary', 'expression'],
                   help='Model input interpretation: binary presence or expression values.')
    class_head_group = p.add_mutually_exclusive_group()
    class_head_group.add_argument('--use-class-head', dest='use_class_head', action='store_true', default=True,
                                  help='Enable maternal/fetal/mix classification head.')
    class_head_group.add_argument('--no-class-head', dest='use_class_head', action='store_false',
                                  help='Disable classification head and train proportion-only model.')
    p.add_argument('--presence-dropout', type=float, default=0.0,
                   help='Randomly hide observed genes during training for robustness.')
    p.add_argument('--classifier-dropout', type=float, default=0.1,
                   help='Dropout rate used in classifier/regression heads.')
    p.add_argument('--layer-norm-mode', type=str, default='second', choices=['none', 'all', 'first', 'second'],
                   help='LayerNorm placement in encoder hidden blocks.')

    # Labels
    p.add_argument('--class-lo-thr', type=float, default=0.1,
                   help='Lower threshold for assigning maternal class from p_true.')
    p.add_argument('--class-hi-thr', type=float, default=0.9,
                   help='Upper threshold for assigning fetal class from p_true.')
    p.add_argument('--class-label-smoothing', type=float, default=0.0,
                   help='Label smoothing factor for cross-entropy.')
    p.add_argument('--class-weight-maternal', type=float, default=1.0,
                   help='Class weight for maternal samples in classification loss.')
    p.add_argument('--class-weight-fetal', type=float, default=1.0,
                   help='Class weight for fetal samples in classification loss.')
    p.add_argument('--class-weight-mix', type=float, default=1.0,
                   help='Class weight for mix samples in classification loss.')
    p.add_argument(
        '--loss-type',
        type=str,
        default='focal',
        choices=['ce', 'focal'],
        help='Classification loss: standard cross-entropy (ce) or focal loss (focal).',
    )
    p.add_argument('--focal-gamma', type=float, default=2.0,
                   help='Focusing parameter for focal loss.')
    p.add_argument('--cls-loss-weight', type=float, default=1.0,
                   help='Weight of classification loss term.')
    p.add_argument('--prop-loss-weight', type=float, default=1.0,
                   help='Weight of proportion regression loss term.')
    p.add_argument('--prop-consistency-weight', type=float, default=0.2,
                   help='Weight of class-proportion consistency regularization.')
    p.add_argument('--aac-purity-weight', type=float, default=0.1,
                   help='Weight of AAC purity regularization term.')
    p.add_argument('--prop-loss-type', type=str, default='huber', choices=['huber', 'mse'],
                   help='Regression loss for fetal proportion prediction.')
    p.add_argument('--aac-column', type=str, default='auto',
                   help='AAC source column. Use auto to search alt_count_sum/aac/alt_reads.')

    # Early stopping
    p.add_argument('--early-stop-patience', type=int, default=15,
                   help='Stop training after this many non-improving evals.')
    p.add_argument(
        '--monitor',
        type=str,
        default='prop_corr',
        choices=['cls_acc','prop_mae', 'prop_corr'],
        help='Validation metric used for best-checkpoint selection and early stopping.',
    )
    p.add_argument('--min-epochs', type=int, default=0,
                   help='Minimum epochs to run before early stopping can trigger.')
    p.add_argument('--min-delta', type=float, default=0.0,
                   help='Minimum improvement required to reset early-stopping patience.')
    p.add_argument('--eval-every', type=int, default=1,
                   help='Run validation every N epochs.')

    return p.parse_args()


class OptimizedTrainer(Trainer):
    """Trainer with dense CPU cache, no per-batch GPU stalls, and optional torch.compile."""

    def __init__(self, *args, compile_model: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.compile_model = compile_model

    def build_model(self):
        super().build_model()
        if self.compile_model:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                if is_main_process(self.rank):
                    print("  [torch.compile] Model compiled for faster throughput.")
            elif is_main_process(self.rank):
                print("  [torch.compile] Skipped: torch.compile requires PyTorch >= 2.0.")

    def _cache_slide_data(self, slide_id: str, spots_s, M_s) -> None:
        """Precompute and cache dense model inputs / labels for zero-overhead batch extraction."""
        p_true_np = self._extract_p_true_np(spots_s)
        cols = ['aac_valid', 'aac_z', 'aac_pct', 'aac_purity']
        aac_np = (
            spots_s[cols].to_numpy(dtype=np.float32, copy=True)
            if all(c in spots_s.columns for c in cols)
            else np.zeros((len(spots_s), 4), dtype=np.float32)
        )
        if str(self.cfg.input_mode).lower() == 'binary':
            x_input_np = M_s.toarray().astype(np.uint8, copy=False)
        else:
            x_input_np = M_s.toarray().astype(np.float32, copy=False)
        self.per_slide[slide_id] = {
            'spots': spots_s,
            'x_input': x_input_np,
            'p_true': p_true_np,
            'aac': aac_np,
        }

    def _extract_p_true_cached(self, p_true_np: np.ndarray, batch_indices: np.ndarray) -> Optional[torch.Tensor]:
        batch = p_true_np[batch_indices]
        if np.isfinite(batch).sum() < 1:
            return None
        return torch.from_numpy(batch).to(self.device, non_blocking=True)

    def _extract_aac_cached(self, aac_np: np.ndarray, batch_indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(aac_np[batch_indices]).to(self.device, non_blocking=True)

    def _extract_dense_x(self, x_input_np: np.ndarray, batch_indices: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x_input_np[batch_indices]).to(self.device, dtype=torch.float32, non_blocking=True)

    def train_epoch(self, epoch: int) -> dict:
        del epoch
        self.model.train()

        raw_summary = self._empty_raw_summary()
        slides_this_rank = self._slides_for_current_rank()

        for slide_id in slides_this_rank:
            bundle = self.per_slide[slide_id]
            x_input_np = bundle['x_input']
            p_true_np = bundle['p_true']
            aac_np = bundle['aac']
            n_spots = x_input_np.shape[0]
            batch_size = self.cfg.batch_size or n_spots
            shuffled_indices = np.random.permutation(n_spots)

            for batch_start in range(0, n_spots, batch_size):
                batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
                x_batch = self._extract_dense_x(x_input_np, batch_indices)
                p_true_batch = self._extract_p_true_cached(p_true_np, batch_indices)
                aac_batch = self._extract_aac_cached(aac_np, batch_indices)

                loss, metrics = self._train_step(x_batch, p_true_batch, aac_batch)
                self._accumulate_raw_summary(raw_summary, metrics)
                del x_batch, p_true_batch, aac_batch, loss

        if self.cfg.distributed:
            gathered = [None] * self.world_size
            torch.distributed.all_gather_object(gathered, raw_summary)
            raw_summary = self._merge_raw_summaries(gathered)
            ddp_barrier(self.cfg.distributed)

        return self._finalize_raw_summary(raw_summary)

    def evaluate_split(self, slide_ids: list[str]) -> dict[str, dict]:
        if not is_main_process(self.rank):
            return {}

        self.model.eval()
        results = {}

        for slide_id in slide_ids:
            bundle = self.per_slide[slide_id]
            x_input_np = bundle['x_input']
            p_true_np = bundle['p_true']
            aac_np = bundle['aac']
            n_spots = x_input_np.shape[0]
            batch_size = self.cfg.batch_size or n_spots
            raw_summary = self._empty_raw_summary()

            for batch_indices in self._iter_batch_indices(n_spots, batch_size):
                x_batch = self._extract_dense_x(x_input_np, batch_indices)
                p_true_batch = self._extract_p_true_cached(p_true_np, batch_indices)
                aac_batch = self._extract_aac_cached(aac_np, batch_indices)

                batch_metrics = compute_multitask_metrics(self.model, x_batch, p_true=p_true_batch, cfg=self.cfg, aac_features=aac_batch)
                self._accumulate_raw_summary(raw_summary, batch_metrics)
                del x_batch, p_true_batch, aac_batch

            results[slide_id] = self._finalize_raw_summary(raw_summary)

        return results

    def save_predictions(self):
        if not is_main_process(self.rank):
            return

        self.model.eval()
        for slide_id in self.slides_train + self.slides_val + self.slides_test:
            bundle = self.per_slide[slide_id]
            spots_s = bundle['spots']
            x_input_np = bundle['x_input']
            p_true_np = bundle['p_true']
            aac_np = bundle['aac']
            n_spots = x_input_np.shape[0]
            batch_size = self.cfg.batch_size or n_spots

            all_probs, all_classes, all_scores, all_p_pred = [], [], [], []

            for batch_indices in self._iter_batch_indices(n_spots, batch_size):
                x_batch = self._extract_dense_x(x_input_np, batch_indices)
                aac_batch = self._extract_aac_cached(aac_np, batch_indices)

                with torch.no_grad():
                    fetal_score, aux = unwrap(self.model)(x_batch, aac_features=aac_batch, return_aux=True)
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
                del x_batch, aac_batch

            probs_all = np.concatenate(all_probs, axis=0)
            pred_all = np.concatenate(all_classes, axis=0)
            score_all = np.concatenate(all_scores, axis=0)
            p_pred_all = np.concatenate(all_p_pred, axis=0)
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


def main():
    args = parse_args()
    rank, world_size, device = init_distributed()
    applied_seed = set_global_seed(args.seed, rank=rank)
    mlp_dims = args.mlp_dims if args.mlp_dims is not None else [1024, 256, 64]

    try:
        cfg = TrainConfig(
            task='classification3',
            mlp_hidden_dims=mlp_dims,
            cls_hidden_dim=args.cls_hidden_dim,
            prop_hidden_dim=args.prop_hidden_dim,
            input_mode=args.input_mode,
            use_class_head=args.use_class_head,
            presence_dropout=args.presence_dropout,
            classifier_dropout=args.classifier_dropout,
            layer_norm_mode=args.layer_norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=(not args.no_amp and device.type == 'cuda'),
            batch_size=None if args.batch_size == 0 else args.batch_size,
            class_lo_thr=args.class_lo_thr,
            class_hi_thr=args.class_hi_thr,
            class_label_smoothing=args.class_label_smoothing,
            class_weight_maternal=args.class_weight_maternal,
            class_weight_fetal=args.class_weight_fetal,
            class_weight_mix=args.class_weight_mix,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            cls_loss_weight=args.cls_loss_weight,
            prop_loss_weight=args.prop_loss_weight,
            prop_consistency_weight=args.prop_consistency_weight,
            aac_purity_weight=args.aac_purity_weight,
            prop_loss_type=args.prop_loss_type,
            aac_column=args.aac_column,
            early_stop_patience=args.early_stop_patience,
            monitor=args.monitor,
            min_epochs=args.min_epochs,
            min_delta=args.min_delta,
            eval_every=args.eval_every,
            matrix_type=args.matrix_type,
            distributed=(world_size > 1),
            seed=args.seed,
        )

        if str(args.input_mode).lower() == 'binary' and str(args.matrix_type).lower() != 'binary':
            raise ValueError("input-mode=binary requires --matrix-type binary now that in-model binarization is disabled")

        if is_main_process(rank):
            if applied_seed is None:
                print('[Seed] No fixed seed provided')
            else:
                print(f'[Seed] Fixed seed enabled: base={args.seed}, rank0_effective={applied_seed}')
            save_training_run_config(args, cfg, args.exp_dir)

        trainer = OptimizedTrainer(
            out_root=args.out_root,
            slides_train=args.slides,
            slides_val=args.val_slides,
            slides_test=args.test_slides,
            cfg=cfg,
            device=device,
            exp_dir=args.exp_dir,
            rank=rank,
            world_size=world_size,
            load_checkpoint=args.load_checkpoint,
            matrix_type=args.matrix_type,
            hvg_json=args.hvg_json,
            compile_model=args.compile,
        )
        trainer.train()

    finally:
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
