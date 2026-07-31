#!/usr/bin/env python3
"""
DeepSetMA inference script.

Outputs per slide:
- pred_<slide>.csv
- predictions_<slide>.csv (compatibility copy)

Columns:
- barcode, x, y
- pred_class, p_pred
- prob_maternal, prob_fetal, prob_mix
- fetal_score
- (optional) AAC columns: aac_valid, aac_z, aac_pct, aac_purity, alt_count_sum
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


CODE_ROOT = Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dl_module = _load_module_from_path(
    'golden_last_data_loader_for_inf',
    CODE_ROOT / 'model' / 'data_loader.py'
)
_arch_module = _load_module_from_path(
    'golden_last_arch_for_inf',
    CODE_ROOT / 'model' / 'architecture.py'
)

load_spots_and_genes = _dl_module.load_spot_and_gene_metadata
load_csr = _dl_module.load_slide_matrix_csr
csr_to_torch_sparse_batch = _dl_module.csr_rows_to_torch_sparse_batch
DeepSetVAE = _arch_module.MultiTaskArchitecture
MLPEncoder = _arch_module.GeneFeatureEncoder
ClassificationHead = _arch_module.SpotStateClassificationHead
ProportionHead = _arch_module.FetalProportionRegressionHead


DOMAIN_SLIDE_IDS = {
    'human': [
        'Norm', 'Pept2', 'PlaCamb', 'WSPLAS0', 'WSPLAS5',
        'WSPLAS6', 'WSPLAS7', 'WSPLAS9', 'PlaHDBR', 'WSPLAS4',
    ],
    'cancer': ['DCIS08', 'IDCL97', 'ILCL95', 'IDCB14', 'IDCL98'],
    'covid': [
        'S01', 'S03', 'S04', 'S15', 'S16', 'S17', 'S18', 'S19',
        'S20', 'S21', 'S22', 'S23a', 'S23b', 'S24', 'S25', 'S26',
    ],
    'sim': ['292', '655', '88', '624', '862', '615'],
}


def resolve_slide_ids(
    slide_id: Optional[str],
    slides: Optional[list[str]],
    domain: Optional[str],
) -> list[str]:
    if slide_id is not None:
        return [str(slide_id)]
    if slides:
        return [str(s) for s in slides]
    if domain:
        return list(DOMAIN_SLIDE_IDS[str(domain)])
    raise ValueError('Either --slide-id, --slides, or --domain must be provided.')


def load_svd_features(svd_dir: Path, slide_id: str) -> np.ndarray:
    candidates = [
        svd_dir / f"counts_svd_{slide_id}.npz",
        svd_dir / f"counts_svd_{slide_id}.npy",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"SVD features not found for slide={slide_id} under {svd_dir}")

    if path.suffix == '.npy':
        x = np.load(path)
    else:
        obj = np.load(path)
        if 'x' in obj.files:
            x = obj['x']
        elif len(obj.files) == 1:
            x = obj[obj.files[0]]
        else:
            raise ValueError(f"NPZ must contain key 'x' or a single array: {path}")

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"SVD feature matrix must be 2D, got shape={x.shape} from {path}")
    return x


def _infer_classifier_dropout(model_state: dict, cfg: dict, default: float = 0.0) -> float:
    cfg_dropout = cfg.get('classifier_dropout', None)
    if cfg_dropout is not None:
        return float(cfg_dropout)

    has_dropout_layers = (
        ('cls_head.head.3.weight' in model_state)
        or ('prop_head.head.6.weight' in model_state)
    )
    return 0.1 if has_dropout_layers else float(default)


def _load_sidecar_run_config(checkpoint_path: Path) -> dict:
    run_cfg_path = checkpoint_path.parent / 'run_config.json'
    if not run_cfg_path.exists():
        return {}
    try:
        with open(run_cfg_path, 'r') as f:
            payload = json.load(f)
        merged = {}
        if isinstance(payload.get('cfg'), dict):
            merged.update(payload['cfg'])
        if isinstance(payload.get('args'), dict):
            for key, value in payload['args'].items():
                merged.setdefault(key, value)
        return merged
    except Exception:
        return {}


def _infer_model_kwargs(model_state: dict, cfg: dict) -> dict:
    enc_keys: List[str] = sorted(
        [k for k in model_state if k.startswith('encoder.net.') and k.endswith('.weight')],
        key=lambda k: int(k.split('.')[2]),
    )
    # LayerNorm weights are 1D; only Linear weights (2D) define the hidden dims.
    enc_linear_keys = [k for k in enc_keys if model_state[k].ndim == 2]
    ln_indices = [int(k.split('.')[2]) for k in enc_keys if model_state[k].ndim == 1]

    if not enc_linear_keys:
        n_genes = int(cfg.get('n_genes', 0))
        mlp_hidden_dims = list(cfg.get('mlp_hidden_dims', [1024, 256, 64]))
        if n_genes <= 0:
            raise ValueError(
                'Cannot infer n_genes: encoder.net.*.weight not found and config missing n_genes.'
            )
    else:
        n_genes = int(model_state[enc_linear_keys[0]].shape[1])
        mlp_hidden_dims = [int(model_state[k].shape[0]) for k in enc_linear_keys]

    cfg_ln_mode = str(cfg.get('layer_norm_mode', '')).strip().lower()
    if cfg_ln_mode in {'all', 'first', 'second', 'none'}:
        layer_norm_mode = cfg_ln_mode
    else:
        linear_indices = [int(k.split('.')[2]) for k in enc_linear_keys]
        if not ln_indices:
            layer_norm_mode = 'none'
        elif len(ln_indices) >= 2:
            layer_norm_mode = 'all'
        else:
            # single LN: first block if it appears before second Linear, else second block
            if len(linear_indices) >= 2 and ln_indices[0] < linear_indices[1]:
                layer_norm_mode = 'first'
            else:
                layer_norm_mode = 'second'

    cls_w = model_state.get('cls_head.head.0.weight')
    cls_hidden_dim = int(cls_w.shape[0]) if cls_w is not None else int(cfg.get('cls_hidden_dim', 64))
    use_class_head_cfg = cfg.get('use_class_head', None)
    if use_class_head_cfg is None:
        use_class_head = cls_w is not None
    else:
        use_class_head = bool(use_class_head_cfg)

    prop_w = model_state.get('prop_head.head.0.weight')
    prop_hidden_dim = int(prop_w.shape[0]) if prop_w is not None else int(cfg.get('prop_hidden_dim', 128))
    prop_head_in_dim = int(prop_w.shape[1]) if prop_w is not None else None

    classifier_dropout = _infer_classifier_dropout(model_state, cfg, default=0.0)
    expected_prop_in_dim = int(mlp_hidden_dims[-1]) + 3 + 4
    expected_prop_in_dim_no_cls = int(mlp_hidden_dims[-1]) + 4

    input_mode = str(cfg.get('input_mode', 'binary')).strip().lower()
    if input_mode not in {'binary', 'expression'}:
        input_mode = 'binary'

    return {
        'n_genes': n_genes,
        'mlp_hidden_dims': mlp_hidden_dims,
        'cls_hidden_dim': cls_hidden_dim,
        'prop_hidden_dim': prop_hidden_dim,
        'input_mode': input_mode,
        'use_class_head': use_class_head,
        'classifier_dropout': classifier_dropout,
        'presence_dropout': 0.0,
        'layer_norm_mode': layer_norm_mode,
    }


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
        cfg = ckpt.get('config', {})
    else:
        model_state = ckpt
        cfg = {}

    sidecar_cfg = _load_sidecar_run_config(checkpoint_path)
    if sidecar_cfg:
        merged_cfg = dict(sidecar_cfg)
        merged_cfg.update(cfg)
        cfg = merged_cfg

    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', '', 1): v for k, v in model_state.items()}

    model_kwargs = _infer_model_kwargs(model_state, cfg)
    print("\nModel configuration inferred from checkpoint:")
    for k in ['n_genes', 'mlp_hidden_dims', 'cls_hidden_dim', 'prop_hidden_dim', 'classifier_dropout', 'layer_norm_mode']:
        print(f"  {k}: {model_kwargs[k]}")
    print(f"  input_mode: {model_kwargs.get('input_mode', 'binary')}")
    print(f"  use_class_head: {model_kwargs.get('use_class_head', True)}")
    print("  architecture: current MultiTaskArchitecture only")

    model = DeepSetVAE(**model_kwargs)
    result = model.load_state_dict(model_state, strict=False)

    if result.missing_keys:
        print(f"⚠ Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"⚠ Unexpected keys: {result.unexpected_keys}")

    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    return model


def _extract_spots_for_slide(spots: pd.DataFrame, slide_id: str) -> pd.DataFrame:
    sid = str(slide_id)
    if 'slide' in spots.columns:
        sp = spots[spots['slide'].astype(str) == sid].copy().reset_index(drop=True)
    elif 'slide_id' in spots.columns:
        sp = spots[spots['slide_id'].astype(str) == sid].copy().reset_index(drop=True)
    else:
        sp = spots.copy().reset_index(drop=True)
    return sp


def _aac_source_column(spots_s: pd.DataFrame, requested: str) -> Optional[str]:
    req = str(requested).strip()
    if req and req.lower() != 'auto' and req in spots_s.columns:
        return req
    for col in ['alt_count_sum', 'aac', 'alt_reads']:
        if col in spots_s.columns:
            return col
    return None


def _attach_aac_features(spots_s: pd.DataFrame, aac_column: str = 'auto') -> pd.DataFrame:
    spots_s = spots_s.copy()
    source_col = _aac_source_column(spots_s, aac_column)

    if source_col is None:
        spots_s['aac_valid'] = 0.0
        spots_s['aac_log1p'] = 0.0
        spots_s['aac_z'] = 0.0
        spots_s['aac_pct'] = 0.5
        spots_s['aac_purity'] = 0.0
        return spots_s

    raw = pd.to_numeric(spots_s[source_col], errors='coerce').astype(np.float32)
    valid = np.isfinite(raw.to_numpy())

    logv = np.zeros(len(spots_s), dtype=np.float32)
    z = np.zeros(len(spots_s), dtype=np.float32)
    pct = np.full(len(spots_s), 0.5, dtype=np.float32)

    if valid.any():
        raw_valid = np.maximum(raw.to_numpy()[valid], 0.0)
        log_valid = np.log1p(raw_valid).astype(np.float32)
        logv[valid] = log_valid

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
    spots_s['aac_log1p'] = logv
    spots_s['aac_z'] = z
    spots_s['aac_pct'] = pct
    spots_s['aac_purity'] = purity
    return spots_s


@torch.no_grad()
def infer_slide(model, x_data, spots_s: pd.DataFrame, device: torch.device, batch_size: int, use_dense: bool = False):
    n_spots = x_data.shape[0]
    bs = batch_size if batch_size and batch_size > 0 else n_spots

    all_scores = []
    all_probs = []
    all_pred_class = []
    all_p_pred = []

    for batch_start in range(0, n_spots, bs):
        batch_indices = np.arange(batch_start, min(batch_start + bs, n_spots))
        if use_dense:
            x_batch = torch.from_numpy(x_data[batch_indices]).to(device=device, dtype=torch.float32)
        else:
            x_batch = csr_to_torch_sparse_batch(x_data, batch_indices, device)

        aac_cols = ['aac_valid', 'aac_z', 'aac_pct', 'aac_purity']
        if all(c in spots_s.columns for c in aac_cols):
            aac_batch_np = spots_s.iloc[batch_indices][aac_cols].to_numpy(dtype=np.float32, copy=True)
        else:
            aac_batch_np = np.zeros((len(batch_indices), 4), dtype=np.float32)
        aac_batch = torch.from_numpy(aac_batch_np).to(device)

        out = model(x_batch, aac_features=aac_batch, return_aux=True)
        if isinstance(out, tuple) and len(out) == 3:
            fetal_score, _, aux = out
        elif isinstance(out, tuple) and len(out) == 2:
            fetal_score, aux = out
        else:
            raise ValueError('Unexpected model return format during inference')

        probs_t = aux['probabilities']
        if probs_t is None:
            probs = np.full((len(batch_indices), 3), np.nan, dtype=np.float32)
            pred_class = np.full(len(batch_indices), -1, dtype=np.int64)
            fetal_score_np = np.full(len(batch_indices), np.nan, dtype=np.float32)
        else:
            probs = probs_t.cpu().numpy()
            pred_class = np.argmax(probs, axis=1)
            fetal_score_np = fetal_score.cpu().numpy()
        p_pred = aux['p_pred'].cpu().numpy()

        all_scores.append(fetal_score_np)
        all_probs.append(probs)
        all_pred_class.append(pred_class)
        all_p_pred.append(p_pred)

    return {
        'fetal_score': np.concatenate(all_scores, axis=0),
        'probs': np.concatenate(all_probs, axis=0),
        'pred_class': np.concatenate(all_pred_class, axis=0),
        'p_pred': np.concatenate(all_p_pred, axis=0),
    }


def run_inference(
    checkpoint_path: Path,
    preprocess_dir: Path,
    slide_id: str,
    output_dir: Path,
    matrix_type: str = 'binary',
    svd_dir: Optional[Path] = None,
    aac_column: str = 'auto',
    batch_size: int = 4096,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    spots, _genes = load_spots_and_genes(preprocess_dir)
    spots_slide = _extract_spots_for_slide(spots, str(slide_id))
    print(f"Spots for slide '{slide_id}': {len(spots_slide)}")

    model = load_model_from_checkpoint(checkpoint_path, device)
    checkpoint_n_genes = int(model.n_genes)
    checkpoint_input_mode = str(getattr(model, 'input_mode', 'binary')).strip().lower()

    if checkpoint_input_mode == 'binary' and matrix_type != 'binary' and svd_dir is None:
        raise ValueError(
            f"Checkpoint expects binary inputs, but --matrix-type {matrix_type!r} was provided. "
            "Use --matrix-type binary for inference on this checkpoint."
        )

    use_dense = svd_dir is not None
    if use_dense:
        X = load_svd_features(svd_dir, str(slide_id)).astype(np.float32, copy=False)
        print(f"SVD feature shape: {X.shape}")
        if X.shape[1] != checkpoint_n_genes:
            raise ValueError(
                f"Input SVD dimension ({X.shape[1]}) does not match checkpoint n_genes ({checkpoint_n_genes})."
            )
        n_rows = min(len(spots_slide), X.shape[0])
        if len(spots_slide) != X.shape[0]:
            print(
                f"⚠ Warning: spots rows ({len(spots_slide)}) != SVD rows ({X.shape[0]}), trimming to {n_rows}"
            )
            spots_slide = spots_slide.iloc[:n_rows].copy().reset_index(drop=True)
            X = X[:n_rows, :]
    else:
        M = load_csr(preprocess_dir, str(slide_id), matrix_type=matrix_type)
        print(f"Matrix shape: {M.shape}")

        if M.shape[1] != checkpoint_n_genes:
            raise ValueError(
                f"Input gene dimension ({M.shape[1]}) does not match checkpoint n_genes ({checkpoint_n_genes})."
            )

        n_rows = min(len(spots_slide), M.shape[0])
        if len(spots_slide) != M.shape[0]:
            print(
                f"⚠ Warning: spots rows ({len(spots_slide)}) != matrix rows ({M.shape[0]}), trimming to {n_rows}"
            )
            spots_slide = spots_slide.iloc[:n_rows].copy().reset_index(drop=True)
            M = M[:n_rows, :]

    if len(spots_slide) != n_rows:
        print(
            f"⚠ Warning: spots rows mismatch, trimming to {n_rows}"
        )
        spots_slide = spots_slide.iloc[:n_rows].copy().reset_index(drop=True)

    spots_slide = _attach_aac_features(spots_slide, aac_column=aac_column)

    print("Running inference...")
    out = infer_slide(
        model,
        X if use_dense else M,
        spots_slide,
        device=device,
        batch_size=batch_size,
        use_dense=use_dense,
    )

    probs = out['probs']
    pred_class = out['pred_class']
    p_pred = out['p_pred']

    print("\nClass distribution:")
    valid_pred_mask = pred_class >= 0
    if valid_pred_mask.any():
        for cls in [0, 1, 2]:
            ratio = float((pred_class[valid_pred_mask] == cls).mean()) if valid_pred_mask.any() else 0.0
            print(f"  class {cls}: {ratio:.2%}")
    else:
        print("  classification head disabled; no class probabilities available")

    barcode_col = 'barcode' if 'barcode' in spots_slide.columns else None
    x_col = 'x' if 'x' in spots_slide.columns else ('array_row' if 'array_row' in spots_slide.columns else None)
    y_col = 'y' if 'y' in spots_slide.columns else ('array_col' if 'array_col' in spots_slide.columns else None)

    barcodes = spots_slide[barcode_col].to_numpy() if barcode_col else np.array([f'spot_{i}' for i in range(n_rows)])
    xs = spots_slide[x_col].to_numpy() if x_col else np.full(n_rows, np.nan)
    ys = spots_slide[y_col].to_numpy() if y_col else np.full(n_rows, np.nan)

    pred_df = pd.DataFrame({
        'barcode': barcodes,
        'x': xs,
        'y': ys,
        'pred_class': pred_class.astype(np.int64),
        'p_pred': p_pred.astype(np.float32),
        'prob_maternal': probs[:, 0],
        'prob_fetal': probs[:, 1],
        'prob_mix': probs[:, 2],
        'fetal_score': out['fetal_score'],
    })

    for col in ['alt_count_sum', 'aac_valid', 'aac_z', 'aac_pct', 'aac_purity']:
        if col in spots_slide.columns:
            pred_df[col] = pd.to_numeric(spots_slide[col], errors='coerce').to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_main = output_dir / f"pred_{slide_id}.csv"
    pred_legacy = output_dir / f"predictions_{slide_id}.csv"

    pred_df.to_csv(pred_main, index=False)
    pred_df.to_csv(pred_legacy, index=False)
    print(f"\n✓ Saved predictions to {pred_main}")
    print(f"✓ Saved compatibility predictions to {pred_legacy}")


def main():
    parser = argparse.ArgumentParser(
        description='DeepSetMA/DeepFM/GatedAttn inference script',
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument('-h', '--h', action='help', help='Show this help message and exit.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to DeepSetMA/DeepFM/GatedAttn checkpoint')
    parser.add_argument('--preprocess-dir', type=Path, required=True, help='Directory containing preprocessed data')
    slide_group = parser.add_mutually_exclusive_group(required=True)
    slide_group.add_argument('--slide-id', type=str, help='Single slide ID to run inference on')
    slide_group.add_argument('--slides', nargs='+', help='One or more slide IDs to run inference on')
    slide_group.add_argument('--domain', type=str, choices=sorted(DOMAIN_SLIDE_IDS.keys()),
                             help='Use built-in slide list for a domain')
    parser.add_argument('--output-dir', type=Path, default=Path('./inference_results'), help='Output directory')
    parser.add_argument('--per-slide-subdir', action='store_true',
                        help='Save outputs under <output-dir>/<slide_id>')
    parser.add_argument(
        '--matrix-type',
        type=str,
        default='binary',
        choices=['binary', 'cp10k_log1p'],
        help='Preprocessed matrix type to use for inference. Use binary for base/class/noaac and cp10k_log1p for expression.',
    )
    parser.add_argument('--svd-dir', type=Path, default=None, help='Optional SVD feature directory with counts_svd_<slide>.npz')
    parser.add_argument('--aac-column', type=str, default='auto', help='AAC source column name or auto')
    parser.add_argument('--batch-size', type=int, default=4096, help='Inference batch size (0 = full slide)')
    args = parser.parse_args()

    print("=" * 80)
    print("DeepSetMA/DeepFM/GatedAttn Inference")
    print("=" * 80)
    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Preprocess dir:{args.preprocess_dir}")
    slide_ids = resolve_slide_ids(args.slide_id, args.slides, args.domain)
    print(f"Slides:        {slide_ids}")
    print(f"Matrix type:   {args.matrix_type}")
    print(f"SVD dir:       {args.svd_dir}")
    print(f"AAC column:    {args.aac_column}")
    print(f"Output dir:    {args.output_dir}")
    print("=" * 80)

    is_batch = len(slide_ids) > 1
    use_subdir = bool(args.per_slide_subdir or is_batch or args.domain is not None)
    success = 0

    for sid in slide_ids:
        print(f"\n[run] slide={sid}")
        try:
            target_output_dir = (args.output_dir / str(sid)) if use_subdir else args.output_dir
            run_inference(
                checkpoint_path=args.checkpoint,
                preprocess_dir=args.preprocess_dir,
                slide_id=sid,
                output_dir=target_output_dir,
                matrix_type=args.matrix_type,
                svd_dir=args.svd_dir,
                aac_column=args.aac_column,
                batch_size=args.batch_size,
            )
            success += 1
        except Exception as exc:
            print(f"[warn] slide={sid} failed: {exc}")

    if success == 0:
        raise SystemExit('No slides were processed successfully.')


if __name__ == '__main__':
    main()
