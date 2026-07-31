"""Multi-task architecture for maternal/fetal/mix classification + proportion regression."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn


def _normalize_dims(dims: Iterable[int]) -> List[int]:
    hidden_dims = [int(d) for d in dims if int(d) > 0]
    if not hidden_dims:
        raise ValueError('mlp_hidden_dims must contain at least one positive integer')
    return hidden_dims


def _normalize_ln_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {'none', 'all', 'first', 'second'}:
        raise ValueError("layer_norm_mode must be one of: none, all, first, second")
    return m


def _normalize_input_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m not in {'binary', 'expression'}:
        raise ValueError("input_mode must be one of: binary, expression")
    return m


class GeneFeatureEncoder(nn.Module):
    """Dense encoder that maps gene-level inputs to latent spot features."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float = 0.1, layer_norm_mode: str = 'all'):
        super().__init__()
        dims = _normalize_dims(hidden_dims)
        ln_mode = _normalize_ln_mode(layer_norm_mode)
        layers = []
        prev_dim = int(input_dim)

        for idx, dim in enumerate(dims):
            layers.append(nn.Linear(prev_dim, dim))
            if idx < len(dims) - 1:
                use_ln = (ln_mode == 'all') or (ln_mode == 'first' and idx == 0) or (ln_mode == 'second' and idx == 1)
                if use_ln:
                    layers.append(nn.LayerNorm(dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.hidden_dims = dims
        self.output_dim = dims[-1]
        self.net = nn.Sequential(*layers)

    def forward(self, gene_input: torch.Tensor) -> torch.Tensor:
        return self.net(gene_input)


class SpotStateClassificationHead(nn.Module):
    """Classifier head for maternal/fetal/mix state prediction."""

    def __init__(self, z_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        layers = [nn.Linear(z_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 3))
        self.head = nn.Sequential(*layers)

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        return self.head(encoded_features)


class FetalProportionRegressionHead(nn.Module):
    """AAC-aware regression head producing fetal proportion in [0, 1]."""

    def __init__(
        self,
        z_dim: int,
        aac_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_class_probs: bool = True,
    ):
        super().__init__()
        self.use_class_probs = bool(use_class_probs)
        in_dim = int(z_dim) + int(aac_dim) + (3 if self.use_class_probs else 0)
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim // 2, 1))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        encoded_features: torch.Tensor,
        class_probabilities: Optional[torch.Tensor],
        aac_features: torch.Tensor,
    ) -> torch.Tensor:
        pieces = [encoded_features]
        if self.use_class_probs:
            if class_probabilities is None:
                raise ValueError('class_probabilities must be provided when use_class_probs=True')
            pieces.append(class_probabilities)
        pieces.append(aac_features)
        x = torch.cat(pieces, dim=1)
        return torch.sigmoid(self.head(x)).squeeze(1)


class MultiTaskArchitecture(nn.Module):
    """
    Multi-task classifier/regressor for maternal/fetal/mix prediction.

    Architecture: gene input -> MLP encoder (configurable LayerNorm) -> optional cls head + proportion head.

    Accepts both sparse COO tensors (base Trainer path) and dense uint8/float32 tensors
    (OptimizedTrainer path) without subclassing.
    """

    def __init__(
        self,
        n_genes: int = 5000,
        mlp_hidden_dims: Iterable[int] = (1024, 256, 64),
        cls_hidden_dim: int = 64,
        prop_hidden_dim: int = 128,
        input_mode: str = 'binary',
        use_class_head: bool = True,
        presence_dropout: float = 0.0,
        classifier_dropout: float = 0.1,
        layer_norm_mode: str = 'all',
        **_,  # absorb legacy kwargs silently
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.input_mode = _normalize_input_mode(input_mode)
        self.use_class_head = bool(use_class_head)
        self.presence_dropout = float(presence_dropout)
        self.mlp_hidden_dims = _normalize_dims(mlp_hidden_dims)
        self.layer_norm_mode = _normalize_ln_mode(layer_norm_mode)

        self.encoder = GeneFeatureEncoder(
            input_dim=self.n_genes,
            hidden_dims=self.mlp_hidden_dims,
            dropout=classifier_dropout,
            layer_norm_mode=self.layer_norm_mode,
        )
        self.z_dim = self.encoder.output_dim
        self.cls_head = None
        if self.use_class_head:
            self.cls_head = SpotStateClassificationHead(
                z_dim=self.z_dim,
                hidden_dim=int(cls_hidden_dim),
                dropout=classifier_dropout,
            )
        self.prop_head = FetalProportionRegressionHead(
            z_dim=self.z_dim,
            aac_dim=4,
            hidden_dim=int(prop_hidden_dim),
            dropout=classifier_dropout,
            use_class_probs=self.use_class_head,
        )

    def _prepare_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Convert any input format into the configured input representation."""
        if getattr(input_tensor, 'is_sparse', False):
            x_dense = input_tensor.to_dense().float()
        elif input_tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            x_dense = input_tensor.float()
        else:
            x_dense = input_tensor.float()

        if self.input_mode == 'binary':
            # Binary mode expects pre-binarized input (0/1) from preprocessing.
            x_out = x_dense
            if self.training and self.presence_dropout > 0:
                drop_mask = (torch.rand_like(x_out) < self.presence_dropout) & (x_out > 0.5)
                x_out = x_out.masked_fill(drop_mask, 0.0)
            return x_out

        return x_dense

    def forward(
        self,
        input_tensor: torch.Tensor,
        aac_features: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        """
        Returns:
            return_aux=False: fetal_score (B,)
            return_aux=True:  (fetal_score, aux_dict)
        """
        prepared_input = self._prepare_input(input_tensor)
        encoded_features = self.encoder(prepared_input)
        cls_logits = None
        class_probs = None
        fetal_score = torch.full(
            (prepared_input.shape[0],),
            float('nan'),
            dtype=prepared_input.dtype,
            device=prepared_input.device,
        )
        if self.use_class_head:
            cls_logits = self.cls_head(encoded_features)
            class_probs = torch.softmax(cls_logits, dim=1)
            fetal_score = class_probs[:, 1]
        if aac_features is None:
            aac_features = torch.zeros((prepared_input.shape[0], 4), dtype=prepared_input.dtype, device=prepared_input.device)
        p_pred = self.prop_head(encoded_features, class_probs, aac_features)

        if return_aux:
            return fetal_score, {
                'x_input_prepared': prepared_input,
                'z_spot': encoded_features,
                'cls_logits': cls_logits,
                'probabilities': class_probs,
                'p_pred': p_pred,
                'aac_features': aac_features,
            }
        return fetal_score
