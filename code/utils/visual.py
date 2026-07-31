#!/usr/bin/env python3
"""
Visualization script for outputs.

Expected per-slide prediction CSV in inference/training output:
- pred_<slide>.csv (preferred) or predictions_<slide>.csv

Primary columns:
- barcode
- p_pred (or p_hat/p)
- pred_class (or class_pred / prob_*)
- x, y (or provided through --spots-root merge)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


CLASS_COLORS = ["#4575b4", "#d73027", "#9e9e9e"]  # maternal, fetal, mix
CLASS_NAMES = ["maternal(0)", "fetal(1)", "mix(2)"]


DOMAIN_SLIDE_IDS = {
    "human": [
        "Norm", "Pept2", "PlaCamb", "WSPLAS0", "WSPLAS5",
        "WSPLAS6", "WSPLAS7", "WSPLAS9", "PlaHDBR", "WSPLAS4",
    ],
    "cancer": ["DCIS08", "IDCL97", "ILCL95", "IDCB14", "IDCL98"],
    "covid": [
        "S01", "S03", "S04", "S15", "S16", "S17", "S18", "S19",
        "S20", "S21", "S22", "S23a", "S23b", "S24", "S25", "S26",
    ],
    "sim": ["292", "655", "88", "624", "862", "615"],
}


def resolve_slide_ids(slide_id: str | None, slides: list[str] | None, domain: str | None) -> list[str]:
    if slide_id is not None:
        return [str(slide_id)]
    if slides:
        return [str(s) for s in slides]
    if domain:
        return list(DOMAIN_SLIDE_IDS[str(domain)])
    raise ValueError("Either --slide-id, --slides, or --domain must be provided.")


def read_table_auto(base: Path) -> pd.DataFrame:
    for suf in (".parquet", ".csv", ".tsv"):
        p = base.with_suffix(suf)
        if p.exists():
            if suf == ".parquet":
                try:
                    return pd.read_parquet(p)
                except Exception:
                    continue
            sep = "," if suf == ".csv" else "\t"
            return pd.read_csv(p, sep=sep, low_memory=False)
    raise FileNotFoundError(f"{base}.[parquet|csv|tsv] not found")


def load_spots(spots_root: Path) -> pd.DataFrame:
    spots = read_table_auto(spots_root / "spots")
    need = {"slide_id", "barcode", "x", "y"}
    missing = need - set(spots.columns)
    if missing:
        raise ValueError(f"spots table missing columns: {missing}")
    return spots


def find_spots_for_slide(spots: pd.DataFrame, sid: str) -> pd.DataFrame:
    sp = spots.query("slide_id == @sid").copy()
    if len(sp) == 0:
        sp = spots[spots["slide_id"].astype(str) == str(sid)].copy()
    if len(sp) == 0:
        m = re.search(r"\d+", str(sid))
        if m:
            sid_num = m.group()
            sp = spots[spots["slide_id"].astype(str) == sid_num].copy()
    return sp


def load_pred_for_slide(inference_dir: Path, slide_id: str) -> pd.DataFrame:
    p1 = inference_dir / f"pred_{slide_id}.csv"
    p2 = inference_dir / f"predictions_{slide_id}.csv"
    path = p1 if p1.exists() else p2
    if not path.exists():
        raise FileNotFoundError(f"No prediction file for slide {slide_id}: {p1} or {p2}")

    df = pd.read_csv(path)
    if "barcode" not in df.columns:
        raise ValueError(f"{path.name} must include 'barcode'")

    if "p_pred" in df.columns:
        df["p_pred"] = pd.to_numeric(df["p_pred"], errors="coerce")
    elif "p_hat" in df.columns:
        df["p_pred"] = pd.to_numeric(df["p_hat"], errors="coerce")
    elif "p" in df.columns:
        df["p_pred"] = pd.to_numeric(df["p"], errors="coerce")

    if "pred_class" in df.columns:
        pass
    elif "class_pred" in df.columns:
        df = df.rename(columns={"class_pred": "pred_class"})
    elif {"prob_maternal", "prob_fetal", "prob_mix"}.issubset(df.columns):
        probs = df[["prob_maternal", "prob_fetal", "prob_mix"]].to_numpy(dtype=np.float32)
        df["pred_class"] = np.argmax(probs, axis=1).astype(np.int64)

    if "pred_class" in df.columns:
        df["pred_class"] = pd.to_numeric(df["pred_class"], errors="coerce").fillna(-1).astype(np.int64)

    if "p_pred" not in df.columns and "pred_class" not in df.columns:
        raise ValueError(f"{path.name} has no MA-compatible prediction columns")

    return df


def resolve_inference_dir(base_dir: Path, slide_id: str, use_slide_subdir: bool) -> Path:
    if use_slide_subdir:
        return base_dir / str(slide_id)
    return base_dir


def plot_one(df: pd.DataFrame, sid: str, save_path: Path, point_size: float) -> None:
    has_prop = "p_pred" in df.columns
    has_cls = "pred_class" in df.columns

    ncols = 2 if has_prop and has_cls else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    idx = 0
    if has_prop:
        ax = axes[idx]
        idx += 1
        sc = ax.scatter(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            c=df["p_pred"].to_numpy(),
            cmap="RdYlBu_r",
            s=point_size * 1.8,
            edgecolors="none",
            vmin=0.0,
            vmax=1.0,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{sid} proportion (p_pred)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("fetal proportion")

    if has_cls:
        ax = axes[idx]
        cmap = ListedColormap(CLASS_COLORS)
        sc = ax.scatter(
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            c=df["pred_class"].to_numpy(),
            cmap=cmap,
            s=point_size * 1.8,
            edgecolors="none",
            vmin=0,
            vmax=2,
        )
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{sid} class (pred_class)")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(CLASS_NAMES)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize outputs (saves plots to <slide_dir>/plots/)',
        allow_abbrev=False,
        add_help=False,
    )
    p.add_argument('-h', '--h', action='help', help='Show this help message and exit.')
    p.add_argument('--inference-dir', type=Path, required=True,
                   help='Directory with prediction CSVs, or base dir containing per-slide subdirs')
    slide_group = p.add_mutually_exclusive_group(required=True)
    slide_group.add_argument('--slide-id', type=str, help='Single slide ID to visualize')
    slide_group.add_argument('--slides', nargs='+', help='One or more slide IDs to visualize')
    slide_group.add_argument('--domain', type=str, choices=sorted(DOMAIN_SLIDE_IDS.keys()),
                             help='Use built-in slide list for a domain')
    p.add_argument('--inference-per-slide-subdir', action='store_true',
                   help='Read predictions from <inference-dir>/<slide_id>/pred_<slide>.csv')
    p.add_argument('--spots-root', type=Path, default=None,
                   help='Root containing spots.* (needed if x/y not in prediction files)')
    p.add_argument('--in-tissue-only', action='store_true',
                   help='Keep only in_tissue==1 when spots table is available')
    p.add_argument('--point-size', type=float, default=9.0)
    return p.parse_args()


def main():
    args = parse_args()
    slide_ids = resolve_slide_ids(args.slide_id, args.slides, args.domain)

    spots = None
    if args.spots_root is not None:
        spots = load_spots(args.spots_root)

    use_slide_subdir = bool(args.inference_per_slide_subdir or len(slide_ids) > 1 or args.domain is not None)
    plotted = 0
    for sid in slide_ids:
        inference_dir = resolve_inference_dir(args.inference_dir, sid, use_slide_subdir)
        
        # Create per-slide plots directory
        slide_plots_dir = inference_dir / 'plots'
        slide_plots_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            pred = load_pred_for_slide(inference_dir, sid)
        except Exception as e:
            print(f"[warn] {sid}: {e}")
            continue

        df = pred.copy()

        if not {"x", "y"}.issubset(df.columns):
            if spots is None:
                print(f"[warn] {sid}: x/y missing and --spots-root not provided")
                continue
            sp = find_spots_for_slide(spots, sid)
            if sp.empty:
                print(f"[warn] {sid}: not found in spots")
                continue
            df = sp.merge(df, on='barcode', how='inner')

        if args.in_tissue_only and ('in_tissue' in df.columns):
            df = df[df['in_tissue'] == 1]

        if df.empty:
            print(f"[warn] {sid}: no rows after merge/filter")
            continue

        out_path = slide_plots_dir / "prediction.png"
        plot_one(df, sid=sid, save_path=out_path, point_size=args.point_size)
        print(f"saved: {out_path}")
        plotted += 1

    if plotted == 0:
        raise SystemExit("No plots generated.")


if __name__ == '__main__':
    main()
