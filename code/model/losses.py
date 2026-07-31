"""Losses and metrics for spot-state multi-task learning."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


SPOT_STATE_NAMES = ('maternal', 'fetal', 'mix')


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    loss_type = str(getattr(cfg, 'prop_loss_type', 'huber')).lower().strip()
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    return F.smooth_l1_loss(pred, target, beta=0.05)


def _proportion_consistency_loss(p_pred: torch.Tensor, labels: torch.Tensor, cfg) -> torch.Tensor:
    lo_thr = float(getattr(cfg, 'class_lo_thr', 0.1))
    hi_thr = float(getattr(cfg, 'class_hi_thr', 0.9))
    losses = []

    mat_mask = labels == 0
    if mat_mask.any():
        losses.append(torch.relu(p_pred[mat_mask] - lo_thr).pow(2).mean())

    fet_mask = labels == 1
    if fet_mask.any():
        losses.append(torch.relu(hi_thr - p_pred[fet_mask]).pow(2).mean())

    mix_mask = labels == 2
    if mix_mask.any():
        mix_pred = p_pred[mix_mask]
        mix_pen = torch.relu(lo_thr - mix_pred).pow(2) + torch.relu(mix_pred - hi_thr).pow(2)
        losses.append(mix_pen.mean())

    if not losses:
        return p_pred.sum() * 0.0
    return torch.stack(losses).mean()


def _aac_purity_loss(p_pred: torch.Tensor, aac_features: Optional[torch.Tensor]) -> torch.Tensor:
    if aac_features is None or aac_features.numel() == 0:
        return p_pred.sum() * 0.0
    valid = aac_features[:, 0] > 0.5
    if not valid.any():
        return p_pred.sum() * 0.0
    target_purity = aac_features[valid, 3].clamp(0.0, 1.0)
    pred_purity = (p_pred[valid] - 0.5).abs() * 2.0
    return F.smooth_l1_loss(pred_purity, target_purity, beta=0.05)


def build_spot_state_labels(
    p_true: torch.Tensor,
    lo_thr: float = 0.1,
    hi_thr: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map continuous fetal proportion targets into spot-state labels."""
    if not isinstance(p_true, torch.Tensor):
        raise TypeError('p_true must be a torch.Tensor')

    p_true = p_true.float()
    valid_mask = torch.isfinite(p_true)
    labels = torch.full_like(p_true, fill_value=-1, dtype=torch.long)

    labels[valid_mask & (p_true <= float(lo_thr))] = 0
    labels[valid_mask & (p_true >= float(hi_thr))] = 1
    labels[valid_mask & (p_true > float(lo_thr)) & (p_true < float(hi_thr))] = 2
    return labels, valid_mask


def _class_weight_tensor(cfg, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(getattr(cfg, 'class_weight_maternal', 1.0)),
            float(getattr(cfg, 'class_weight_fetal', 1.0)),
            float(getattr(cfg, 'class_weight_mix', 1.0)),
        ],
        dtype=dtype,
        device=device,
    )


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor, cfg) -> torch.Tensor:
    if logits is None:
        return labels.float().sum() * 0.0
    loss_type = str(getattr(cfg, 'loss_type', 'ce')).lower().strip()
    class_weights = _class_weight_tensor(cfg, dtype=logits.dtype, device=logits.device)

    if loss_type == 'focal':
        gamma = max(0.0, float(getattr(cfg, 'focal_gamma', 2.0)))
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        p_t = probs.gather(dim=1, index=labels.view(-1, 1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - p_t, gamma)
        ce_per_sample = F.cross_entropy(logits, labels, weight=class_weights, reduction='none', label_smoothing=0.0)
        return (focal_factor * ce_per_sample).mean()

    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=float(getattr(cfg, 'class_label_smoothing', 0.0)),
    )


def _empty_metrics() -> dict:
    metrics = {
        'loss_total': float('nan'),
        'cls_loss': float('nan'),
        'cls_acc': float('nan'),
        'cls_macro_acc': float('nan'),
        'cls_acc_maternal': float('nan'),
        'cls_acc_fetal': float('nan'),
        'cls_acc_mix': float('nan'),
        'prop_loss': float('nan'),
        'prop_consistency_loss': float('nan'),
        'aac_purity_loss': float('nan'),
        'prop_mae': float('nan'),
        'prop_rmse': float('nan'),
        'prop_corr': float('nan'),
        'mean_confidence': float('nan'),
        'n_labeled': 0,
        'n_cls_labeled': 0,
        'n_correct': 0,
        'n_prop': 0,
        '_prop_abs_err_sum': 0.0,
        '_prop_sq_err_sum': 0.0,
        '_prop_sum_true': 0.0,
        '_prop_sum_pred': 0.0,
        '_prop_sum_true2': 0.0,
        '_prop_sum_pred2': 0.0,
        '_prop_sum_true_pred': 0.0,
    }
    for name in SPOT_STATE_NAMES:
        metrics[f'n_class_{name}'] = 0
        metrics[f'n_correct_{name}'] = 0
    return metrics


def _classification_metrics_from_logits(logits: Optional[torch.Tensor], labels: torch.Tensor) -> dict:
    metrics = _empty_metrics()
    if logits is None or labels.numel() == 0:
        return metrics

    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1)

    n_labeled = int(labels.numel())
    n_correct = int((pred == labels).sum().item())
    metrics['n_cls_labeled'] = n_labeled
    metrics['n_correct'] = n_correct
    metrics['cls_acc'] = float(n_correct / n_labeled)
    metrics['mean_confidence'] = float(probs.max(dim=1).values.mean().item())

    recall_values = []
    for class_idx, name in enumerate(SPOT_STATE_NAMES):
        class_mask = labels == class_idx
        n_class = int(class_mask.sum().item())
        n_correct_class = int(((pred == labels) & class_mask).sum().item())
        metrics[f'n_class_{name}'] = n_class
        metrics[f'n_correct_{name}'] = n_correct_class
        if n_class > 0:
            recall = n_correct_class / n_class
            metrics[f'cls_acc_{name}'] = float(recall)
            recall_values.append(recall)

    if recall_values:
        metrics['cls_macro_acc'] = float(np.mean(recall_values))

    return metrics


def _add_proportion_metrics(metrics: dict, p_pred: torch.Tensor, p_true: torch.Tensor) -> dict:
    if p_pred.numel() == 0:
        return metrics

    err = p_pred - p_true
    n_prop = int(p_pred.numel())
    metrics['n_prop'] = n_prop
    metrics['_prop_abs_err_sum'] = float(err.abs().sum().item())
    metrics['_prop_sq_err_sum'] = float(err.pow(2).sum().item())
    metrics['_prop_sum_true'] = float(p_true.sum().item())
    metrics['_prop_sum_pred'] = float(p_pred.sum().item())
    metrics['_prop_sum_true2'] = float(p_true.pow(2).sum().item())
    metrics['_prop_sum_pred2'] = float(p_pred.pow(2).sum().item())
    metrics['_prop_sum_true_pred'] = float((p_true * p_pred).sum().item())
    metrics['prop_mae'] = float(err.abs().mean().item())
    metrics['prop_rmse'] = float(torch.sqrt(err.pow(2).mean()).item())

    if n_prop >= 2:
        p_true_c = p_true - p_true.mean()
        p_pred_c = p_pred - p_pred.mean()
        denom = torch.sqrt((p_true_c.pow(2).sum()) * (p_pred_c.pow(2).sum())).clamp(min=1e-8)
        metrics['prop_corr'] = float((p_true_c * p_pred_c).sum().item() / denom.item())
    return metrics


def _compute_all_losses(
    logits_v: Optional[torch.Tensor],
    labels_v: torch.Tensor,
    p_pred_v: torch.Tensor,
    p_true_v: torch.Tensor,
    aac_v: Optional[torch.Tensor],
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared multi-task loss computation. Returns (total, cls, prop, consistency, purity)."""
    use_class_head = bool(getattr(cfg, 'use_class_head', True))
    zero = p_pred_v.sum() * 0.0
    loss_cls = _classification_loss(logits_v, labels_v, cfg) if use_class_head else zero
    loss_prop = _regression_loss(p_pred_v, p_true_v, cfg)
    loss_cons = _proportion_consistency_loss(p_pred_v, labels_v, cfg) if use_class_head else zero
    loss_purity = _aac_purity_loss(p_pred_v, aac_v)
    total_loss = (
        float(getattr(cfg, 'cls_loss_weight', 1.0)) * loss_cls
        + float(getattr(cfg, 'prop_loss_weight', 1.0)) * loss_prop
        + float(getattr(cfg, 'prop_consistency_weight', 0.2)) * loss_cons
        + float(getattr(cfg, 'aac_purity_weight', 0.1)) * loss_purity
    )
    return total_loss, loss_cls, loss_prop, loss_cons, loss_purity


def _run_multitask_losses(
    cls_logits: Optional[torch.Tensor],
    p_pred: torch.Tensor,
    p_true: torch.Tensor,
    aac_features: Optional[torch.Tensor],
    cfg,
) -> Tuple[torch.Tensor, dict]:
    """Filter to valid samples, compute all losses, fill metrics dict.

    Shared implementation used by both compute_loss and compute_metrics.
    Returns (total_loss_tensor, metrics_dict).
    """
    metrics = _empty_metrics()
    device = cls_logits.device if cls_logits is not None else p_pred.device

    if not isinstance(p_true, torch.Tensor):
        p_true = torch.as_tensor(p_true, dtype=torch.float32, device=device)
    else:
        p_true = p_true.to(device=device, dtype=torch.float32)

    labels, valid_mask = build_spot_state_labels(
        p_true,
        lo_thr=float(getattr(cfg, 'class_lo_thr', 0.1)),
        hi_thr=float(getattr(cfg, 'class_hi_thr', 0.9)),
    )
    if not valid_mask.any():
        return p_pred.sum() * 0.0, metrics

    logits_v = cls_logits[valid_mask] if cls_logits is not None else None
    labels_v = labels[valid_mask]
    p_pred_v = p_pred[valid_mask]
    p_true_v = p_true[valid_mask]
    aac_v = aac_features[valid_mask] if aac_features is not None else None
    metrics['n_labeled'] = int(valid_mask.sum().item())

    total_loss, loss_cls, loss_prop, loss_cons, loss_purity = _compute_all_losses(
        logits_v, labels_v, p_pred_v, p_true_v, aac_v, cfg
    )

    cls_metrics = _classification_metrics_from_logits(logits_v, labels_v)
    for key, value in cls_metrics.items():
        if key != 'n_labeled':
            metrics[key] = value
    metrics = _add_proportion_metrics(metrics, p_pred_v, p_true_v)
    metrics['loss_total'] = float(total_loss.item())
    metrics['cls_loss'] = float(loss_cls.item())
    metrics['prop_loss'] = float(loss_prop.item())
    metrics['prop_consistency_loss'] = float(loss_cons.item())
    metrics['aac_purity_loss'] = float(loss_purity.item())
    return total_loss, metrics


def compute_multitask_loss(
    model,
    x_input: torch.Tensor,
    cfg,
    epoch: int = 0,
    p_true: Optional[torch.Tensor] = None,
    aac_features: Optional[torch.Tensor] = None,
    **_,
):
    """Compute AAC-aware multi-task loss for classification + fetal proportion."""
    del epoch
    _, aux = model(x_input, aac_features=aac_features, return_aux=True)

    if p_true is None:
        return aux['p_pred'].sum() * 0.0, _empty_metrics()

    return _run_multitask_losses(aux['cls_logits'], aux['p_pred'], p_true, aac_features, cfg)


def compute_multitask_metrics(
    model,
    x_input: torch.Tensor,
    p_true: Optional[torch.Tensor] = None,
    cfg=None,
    aac_features: Optional[torch.Tensor] = None,
):
    """Compute evaluation metrics for the spot-state multi-task model."""
    with torch.no_grad():
        _, aux = model(x_input, aac_features=aac_features, return_aux=True)

        if p_true is None:
            return _empty_metrics()

        _, metrics = _run_multitask_losses(aux['cls_logits'], aux['p_pred'], p_true, aac_features, cfg)
        return metrics

