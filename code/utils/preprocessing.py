#!/usr/bin/env python3
"""
ST Preprocessing Pipeline

Inputs (per slide directory under data_root):
  - features.(tsv|txt|csv)[.gz]
  - barcodes.(tsv|txt|csv)[.gz]
  - matrix.mtx[.gz]    (Matrix Market; usually genes x spots)
  - tissue_positions.csv
  - composition.csv    (optional; per-spot maternal/fetal counts)
Common input:
  - geneInfo.(tab|tsv|csv) containing a gene identifier column (gene_name or gene_id)

Outputs (under out_root):
  - genes.csv
  - spots.csv
  - AAC feature columns can be disabled via --no-aac-features
  - counts_binary_<slide>.npz           (optional via --save-formats binary)
  - counts_raw_<slide>.npz              (optional via --save-formats raw)
  - counts_cp10k_log1p_<slide>.npz      (optional via --save-formats cp10k_log1p)
  - manifest.json

Design:
  - Meta (spots) in a single Parquet table
  - Expression matrices saved per slide as CSR sparse matrices (spots x genes)
  - All matrices reindexed so their columns (genes) follow geneInfo order

Normalization formats:
  - binary: Presence-only matrix (1 if expression > 0 else 0)
  - raw: Raw UMI counts (no normalization)
  - cp10k_log1p: Counts per 10K + log1p transformation


Usage:
  python preprocessing.py \
    --data-root /path/to/data \
    --geneinfo /path/to/geneInfo.tab \
    --out-root /path/to/preproc_out \
    --save-formats binary raw cp10k_log1p

Tip: Install deps:  pip install pandas numpy scipy pyarrow
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.stats import rankdata


# --------------------------- Logging ---------------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


# --------------------------- Config ----------------------------------

@dataclass
class Paths:
    data_root: Path
    geneinfo_path: Path
    out_root: Path


# --------------------------- I/O Utils -------------------------------

def _read_table_auto(path: Path, *, header: Optional[int] = None, names: Optional[List[str]] = None) -> pd.DataFrame:
    """Read a small text table with flexible separators/headers.
    - Supports .tsv/.csv/.txt (+ optional .gz)
    - Uses sep inferred from extension
    """
    p = str(path)
    if path.suffix.lower() == ".csv" or p.endswith(".csv.gz"):
        sep = ","
    else:
        sep = "\t"
    df = pd.read_csv(path, sep=sep, header=header, names=names)
    return df


def read_features(path: Path) -> pd.DataFrame:
    """Return DataFrame with at least column 'gene_name'.
    Accepts tsv/txt/csv; if multiple columns, prefer 'gene_name'/'gene_id'.
    If single column, treat as gene_name.
    """
    # Try common filenames
    cand = [
        path / "features.tsv",
        path / "features.tsv.gz",
        path / "features.txt",
        path / "features.txt.gz",
        path / "features.csv",
        path / "features.csv.gz",
    ]
    file = next((c for c in cand if c.exists()), None)
    if file is None:
        raise FileNotFoundError(f"features file not found under {path}")

    # Many 10x feature files have 3 columns: gene_id, gene_name, feature_type
    df = _read_table_auto(file, header=None)
    if df.shape[1] >= 2:
        # Heuristics: if columns >=2, map first two to gene_id/gene_name
        df = df.rename(columns={0: "gene_id", 1: "gene_name"})
    else:
        df = df.rename(columns={0: "gene_name"})

    # Ensure gene_name exists; if not, fallback to gene_id
    if "gene_name" not in df.columns:
        if "gene_id" in df.columns:
            df["gene_name"] = df["gene_id"].astype(str)
        else:
            raise ValueError("features file lacks gene_name/gene_id columns")

    # Keep only relevant columns
    return df[["gene_name"]].reset_index(drop=True)


def read_barcodes(path: Path) -> pd.DataFrame:
    cand = [
        path / "barcodes.tsv",
        path / "barcodes.tsv.gz",
        path / "barcodes.txt",
        path / "barcodes.txt.gz",
        path / "barcodes.csv",
        path / "barcodes.csv.gz",
    ]
    file = next((c for c in cand if c.exists()), None)
    if file is None:
        raise FileNotFoundError(f"barcodes file not found under {path}")
    df = _read_table_auto(file, header=None)
    df = df.rename(columns={0: "barcode"})[["barcode"]]
    return df.reset_index(drop=True)


def read_matrix(path: Path) -> sparse.coo_matrix:
    """Read matrix.mtx(.gz) and return COO (not CSR)."""
    cand = [path / "matrix.mtx", path / "matrix.mtx.gz"]
    file = next((c for c in cand if c.exists()), None)
    if file is None:
        raise FileNotFoundError(f"matrix.mtx(.gz) not found under {path}")
    M = mmread(file)
    if not sparse.isspmatrix_coo(M):
        M = sparse.coo_matrix(M)
    return M


def read_tissue_positions(path: Path) -> pd.DataFrame:
    file = path / "tissue_positions.csv"
    if not file.exists():
        raise FileNotFoundError(f"tissue_positions.csv not found under {path}")
    required = ["barcode", "in_tissue", "array_row", "array_col", "x", "y"]

    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in df.columns}

        rename_map = {
            "spot_id": "barcode",
            "pxl_col_in_fullres": "x",
            "pxl_row_in_fullres": "y",
        }
        ren = {}
        for src, dst in rename_map.items():
            if src in lower_cols:
                src_col = lower_cols[src]
                if src_col != dst and dst not in df.columns:
                    ren[src_col] = dst
        if ren:
            df = df.rename(columns=ren)
        return df

    # 1) Normal CSV with header (existing behavior)
    df = _standardize_columns(pd.read_csv(file))
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return df

    # 2) Fallback: headerless CSV
    df_no_header = pd.read_csv(file, header=None)
    if df_no_header.shape[1] < 6:
        raise ValueError(f"tissue_positions.csv missing columns: {missing}")

    # Expected order for headerless Visium-style file:
    # barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    df_no_header = df_no_header.iloc[:, :6].copy()
    df_no_header.columns = ["barcode", "in_tissue", "array_row", "array_col", "y", "x"]

    # If a header row was accidentally read as data, drop it
    first_row = [str(v).strip().lower() for v in df_no_header.iloc[0].tolist()]
    if len(first_row) >= 6 and first_row[:4] == ["barcode", "in_tissue", "array_row", "array_col"]:
        df_no_header = df_no_header.iloc[1:].reset_index(drop=True)

    for col in ["in_tissue", "array_row", "array_col", "x", "y"]:
        df_no_header[col] = pd.to_numeric(df_no_header[col], errors="coerce")

    missing_fallback = [c for c in required if c not in df_no_header.columns]
    if missing_fallback:
        raise ValueError(f"tissue_positions.csv missing columns: {missing_fallback}")

    return df_no_header


def read_alt_reads(path: Path) -> pd.DataFrame:
    """Read optional alt-read file and standardize to columns: barcode, alt_count_sum.

    Supported files (first found is used):
      - alt_reads.csv
      - barcode_alt_sum.csv

    Supported key columns:
      - barcode
      - spot_id
      - barcode_key
    Supported value columns:
      - alt_count_sum
      - alt_reads
    """
    candidates = [path / "alt_reads.csv", path / "barcode_alt_sum.csv"]
    file = next((f for f in candidates if f.exists()), None)
    if file is None:
        raise FileNotFoundError(f"alt-read file not found under {path} (tried: alt_reads.csv, barcode_alt_sum.csv)")

    # Allow csv/tsv-like files regardless of extension (e.g., barcode_alt_sum.csv with tab delimiter)
    df = pd.read_csv(file, sep=None, engine="python")
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(file, sep="\t")

    lower_cols = {c.lower(): c for c in df.columns}

    if "barcode" in lower_cols:
        key_col = lower_cols["barcode"]
    elif "spot_id" in lower_cols:
        key_col = lower_cols["spot_id"]
    elif "barcode_key" in lower_cols:
        key_col = lower_cols["barcode_key"]
    else:
        raise ValueError(f"{file.name} missing key column (barcode/spot_id/barcode_key): {file}")

    if "alt_count_sum" in lower_cols:
        val_col = lower_cols["alt_count_sum"]
    elif "alt_reads" in lower_cols:
        val_col = lower_cols["alt_reads"]
    else:
        raise ValueError(f"{file.name} missing value column (alt_count_sum/alt_reads): {file}")

    out = df[[key_col, val_col]].rename(columns={key_col: "barcode", val_col: "alt_count_sum"}).copy()
    out["barcode"] = out["barcode"].astype(str)
    return out


def read_composition_counts(path: Path) -> pd.DataFrame:
    """Read optional composition.csv and standardize to columns:
    barcode, maternal_cells, fetal_cells, composition_total_cells.

    Supported formats:
      1) Wide format (rows are Maternal/Fetal, columns are spot IDs)
         ,Spotx1,Spotx2,...
         Maternal,3,1,...
         Fetal,0,2,...
      2) Long format with columns including spot_id/barcode + maternal + fetal
    """
    file = path / "composition.csv"
    if not file.exists():
        raise FileNotFoundError(f"composition.csv not found under {path}")

    df = pd.read_csv(file)
    lower_cols = {c.lower(): c for c in df.columns}

    # Long format: spot_id/barcode + maternal + fetal
    has_maternal = "maternal" in lower_cols
    has_fetal = "fetal" in lower_cols
    has_key = ("spot_id" in lower_cols) or ("barcode" in lower_cols)
    if has_key and has_maternal and has_fetal:
        key_col = lower_cols.get("spot_id", lower_cols.get("barcode"))
        out = df[[key_col, lower_cols["maternal"], lower_cols["fetal"]]].rename(
            columns={
                key_col: "barcode",
                lower_cols["maternal"]: "maternal_cells",
                lower_cols["fetal"]: "fetal_cells",
            }
        )
        out["barcode"] = out["barcode"].astype(str)
        out["maternal_cells"] = pd.to_numeric(out["maternal_cells"], errors="coerce").fillna(0.0)
        out["fetal_cells"] = pd.to_numeric(out["fetal_cells"], errors="coerce").fillna(0.0)
        out["composition_total_cells"] = out["maternal_cells"] + out["fetal_cells"]
        return out

    # Wide format: first column contains row labels (Maternal/Fetal)
    if df.shape[1] < 2:
        raise ValueError(f"composition.csv has unsupported shape: {df.shape}")

    row_label_col = df.columns[0]
    row_labels = df[row_label_col].astype(str).str.strip().str.lower()
    if not ((row_labels == "maternal").any() and (row_labels == "fetal").any()):
        raise ValueError("composition.csv does not contain Maternal/Fetal rows")

    maternal_row = df.loc[row_labels == "maternal"].iloc[0, 1:]
    fetal_row = df.loc[row_labels == "fetal"].iloc[0, 1:]
    spot_ids = [str(c) for c in df.columns[1:]]

    out = pd.DataFrame(
        {
            "barcode": spot_ids,
            "maternal_cells": pd.to_numeric(maternal_row.values, errors="coerce"),
            "fetal_cells": pd.to_numeric(fetal_row.values, errors="coerce"),
        }
    )
    out["maternal_cells"] = out["maternal_cells"].fillna(0.0)
    out["fetal_cells"] = out["fetal_cells"].fillna(0.0)
    out["composition_total_cells"] = out["maternal_cells"] + out["fetal_cells"]
    return out


def read_geneinfo(path: Path) -> pd.DataFrame:
    """Read geneInfo with or without header.
    Supported formats:
      - With header containing one of: gene_name/gene/symbol/gene_id
      - No header, whitespace- or tab-delimited: e.g.
        ENSG00000290825 DDX11L2 lncRNA
        ENSG00000243485 MIR1302-2HG lncRNA
    Heuristic when no header:
      - If first column looks like Ensembl IDs (>=50% rows start with 'ENS'),
        use the SECOND column as gene_name (symbol). Otherwise use the FIRST column.
    Returns a DataFrame with columns: idx (0..G-1), gene_name (unique).
    """
    # Try to read with header first
    try:
        df = _read_table_auto(path, header=0)
        # If the auto-read produced unnamed columns or only numeric headers, fall back
        if all(str(c).startswith("Unnamed") for c in df.columns):
            raise ValueError("Unnamed columns; treating as no-header")
        header_mode = True
    except Exception:
        header_mode = False

    if header_mode:
        # Locate a gene identifier column
        gene_col = None
        for c in ["gene_name", "gene", "symbol", "gene_id", "Gene", "GENE"]:
            if c in df.columns:
                gene_col = c
                break
        if gene_col is None:
            # If only one column, use it
            if df.shape[1] == 1:
                gene_col = df.columns[0]
            else:
                # Fall back to no-header logic
                header_mode = False
        if header_mode:
            out = (df[[gene_col]]
                   .rename(columns={gene_col: "gene_name"})
                   .dropna()
                   .astype(str)
                   )
    if not header_mode:
        # No header: robust whitespace/CSV inference
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#", dtype=str)
        df = df.dropna(how="all")
        ncol = df.shape[1]
        # Strip
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        if ncol >= 2:
            col0 = df.iloc[:, 0]
            col1 = df.iloc[:, 1]
            prop_ens = col0.str.upper().str.startswith(("ENS", "ENSG", "ENST")).mean()
            gene_series = col1 if prop_ens >= 0.5 else col0
        else:
            gene_series = df.iloc[:, 0]
        out = pd.DataFrame({"gene_name": gene_series})

    out = out[["gene_name"]].dropna().drop_duplicates().reset_index(drop=True)
    out.insert(0, "idx", np.arange(len(out), dtype=int))
    return out


# --------------------------- Core logic ------------------------------

@dataclass
class SlideResult:
    slide_id: str
    n_spots: int
    counts_binary: Optional[str]
    counts_raw: Optional[str]
    counts_cp10k_log1p: Optional[str]
    counts_cp10k_int: Optional[str]
    counts_spot_max_bin: Optional[str]
    counts_spot_rank_norm: Optional[str]
    warnings: List[str]


def cp10k_log1p(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """CP10K normalization with log1p transformation."""
    # Per-row library size
    row_sums = np.asarray(csr.sum(axis=1)).reshape(-1, 1)
    row_sums[row_sums == 0] = 1.0
    norm = csr.multiply(1e4)
    norm = norm.multiply(1.0 / row_sums)
    # log1p only on data
    norm.data = np.log1p(norm.data)
    return norm.tocsr()


def binary_presence(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """Convert counts to binary presence matrix (non-zero -> 1)."""
    out = csr.copy().tocsr()
    if out.data.size:
        out.data = np.ones_like(out.data, dtype=np.uint8)
    return out


def cp10k_int(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """CP10K normalization without log transformation (returns integers)."""
    # Per-row library size
    row_sums = np.asarray(csr.sum(axis=1)).reshape(-1, 1)
    row_sums[row_sums == 0] = 1.0
    norm = csr.multiply(1e4)
    norm = norm.multiply(1.0 / row_sums)
    # Round to integers
    norm.data = np.round(norm.data)
    return norm.tocsr()


def spot_max_bin(csr: sparse.csr_matrix, n_bins: int = 10, cap_percentile: float = 99.0) -> sparse.csr_matrix:
    """Per-spot percentile-capped normalization followed by uniform binning.

    Steps:
      1. Compute the per-row ``cap_percentile``-th percentile over the stored
         (non-zero) values only.  This caps the influence of extreme outliers
         that would otherwise compress the dynamic range of the whole spot.
         Spots with all-zero expression are left as zero.
      2. Divide each row by its cap value; values above the cap are clipped
         to 1.0, so the resulting range is [0, 1].
      3. Map every non-zero value v to ceil(v * n_bins) / n_bins.
         This gives discrete levels 0.1, 0.2, …, 1.0 (for n_bins=10).
         Exact zero values remain 0 (structural sparsity is preserved).

    Args:
        csr:            Spots × genes CSR matrix of raw counts.
        n_bins:         Number of uniform bins (default 10).
        cap_percentile: Percentile used as the per-spot normalization cap
                        (default 99.0).  Set to 100 to recover strict-max
                        behaviour.

    Returns:
        CSR matrix with float32 values in {0.1, 0.2, …, 1.0}.
    """
    out = csr.astype(np.float32, copy=True).tocsr()
    if out.data.size == 0:
        return out

    nnz_per_row = np.diff(out.indptr)
    nonempty = np.where(nnz_per_row > 0)[0]
    if nonempty.size == 0:
        return out

    # Per-row cap: percentile over non-zero elements of each row.
    # We iterate only non-empty rows — typically fast because a spot rarely
    # has more than a few thousand expressed genes.
    row_cap = np.zeros(out.shape[0], dtype=np.float32)
    for i in nonempty:
        start, end = int(out.indptr[i]), int(out.indptr[i + 1])
        row_cap[i] = np.percentile(out.data[start:end], cap_percentile)
    # Guard against degenerate rows where all stored values are identical
    # (percentile could be 0 only if all stored values are literally 0,
    #  which shouldn't happen in CSR; guard anyway).
    row_cap = np.where(row_cap > 0, row_cap, 1.0).astype(np.float32)

    # Scale and clip: build row index array once, then broadcast
    row_idx = np.repeat(np.arange(out.shape[0], dtype=np.int64), nnz_per_row)
    out.data /= row_cap[row_idx]
    out.data = out.data.clip(0.0, 1.0)

    # Bin: map v -> ceil(v * n_bins) / n_bins, clipped to [1/n_bins, 1.0]
    out.data = (np.ceil(out.data * n_bins).clip(1, n_bins) / n_bins).astype(np.float32)
    return out


def spot_rank_norm(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """Per-spot rank-based normalization to [0.1, 1.0].

    Steps:
      1. For each spot (row), keep zeros as zero.
      2. For non-zero values, compute ranks (highest value = rank 1, lowest = rank n).
      3. Map each non-zero value to its normalized rank in [0.1, 1.0]:
         - rank 1 (highest) → 1.0
         - rank n (lowest) → 0.1
         - formula: 0.1 + (n - rank) / (n - 1) * 0.9

    This preserves the ordinal relationship of expression values within each spot
    while ensuring maximum expression is 1.0 and prevents dynamic range compression
    from extreme values.

    Returns a CSR matrix with float32 values in {0.1, 0.2, …, 1.0} (and
    structural zeros treated as 0).
    """
    out = csr.astype(np.float32, copy=True).tocsr()
    if out.data.size == 0:
        return out

    nnz_per_row = np.diff(out.indptr)
    nonempty = np.where(nnz_per_row > 0)[0]
    if nonempty.size == 0:
        return out

    # Per-row rank normalization: iterate only non-empty rows
    for i in nonempty:
        start, end = int(out.indptr[i]), int(out.indptr[i + 1])
        n_nonzero = end - start
        if n_nonzero == 1:
            # Single non-zero value → rank 1 → 1.0
            out.data[start:end] = 1.0
        else:
            # Compute ranks: higher values get lower rank numbers (rank 1 = highest)
            values = out.data[start:end]
            ranks = rankdata(-values, method='ordinal')  # negate to reverse order
            # Normalize: rank i -> 0.1 + (n - i) / (n - 1) * 0.9
            normalized = 0.1 + (n_nonzero - ranks) / (n_nonzero - 1) * 0.9
            out.data[start:end] = normalized.astype(np.float32)

    return out


def reorder_genes_to_geneinfo(M_spots_x_genes: sparse.csr_matrix, features: pd.DataFrame, geneinfo: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str]]:
    """Align to FULL geneInfo width (N x G_full CSR).
    - Keep geneInfo order exactly (0..G_full-1)
    - Genes absent in the slide remain zero columns
    """
    warnings: List[str] = []
    # features: add positional index for current matrix columns
    feats = features.reset_index().rename(columns={"index": "feat_idx"})[["feat_idx", "gene_name"]]
    # Map features to global geneInfo indices (inner join on gene_name)
    map_df = feats.merge(geneinfo[["idx", "gene_name"]], on="gene_name", how="inner")
    if map_df.empty:
        raise ValueError("No overlapping genes between features and geneInfo")

    # Extract only overlapping feature columns and convert to COO
    col_sel = map_df["feat_idx"].to_numpy()
    sub = M_spots_x_genes[:, col_sel].tocoo(copy=False)

    # Rewrite columns to global geneInfo idx
    new_cols = map_df["idx"].to_numpy()[sub.col]
    N = M_spots_x_genes.shape[0]
    G_full = int(geneinfo.shape[0])
    M = sparse.csr_matrix((sub.data, (sub.row, new_cols)), shape=(N, G_full))

    # Warnings (info)
    missing_in_gi = set(features["gene_name"]) - set(map_df["gene_name"])
    missing_in_feats = set(geneinfo["gene_name"]) - set(map_df["gene_name"])
    if missing_in_gi:
        warnings.append(f"{len(missing_in_gi)} features not in geneInfo (ignored)")
    if missing_in_feats:
        warnings.append(f"{len(missing_in_feats)} geneInfo genes absent in this slide (kept as zeros)")

    return M, warnings


def compute_qc_columns(M_spots_x_genes: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    total_counts = np.asarray(M_spots_x_genes.sum(axis=1)).ravel()
    # Row nnz: number of non-zero genes per spot
    n_genes = np.diff(M_spots_x_genes.indptr)
    return total_counts, n_genes


def process_single_slide(
    slide_dir: Path,
    geneinfo: pd.DataFrame,
    out_root: Path,
    save_formats: List[str],
    include_aac_features: bool = True,
) -> Tuple[pd.DataFrame, SlideResult]:
    slide_id = slide_dir.name
    logging.info(f"Processing slide: {slide_id}")

    # 1) Load
    features = read_features(slide_dir)
    barcodes = read_barcodes(slide_dir)
    M = read_matrix(slide_dir)

    # Matrix orientation: expect spots x genes
    if M.shape[0] == len(features) and M.shape[1] == len(barcodes):
        logging.info("Detected genes x spots; transposing to spots x genes (COO swap)")
        # COO transpose via (row<->col) swap, then CSR once
        M = sparse.coo_matrix((M.data, (M.col, M.row)), shape=(M.shape[1], M.shape[0]))
        M = M.tocsr()
    elif M.shape[0] == len(barcodes) and M.shape[1] == len(features):
        M = M.tocsr()
    else:
        raise ValueError(
            f"Matrix shape {M.shape} inconsistent with features({len(features)}) and barcodes({len(barcodes)})"
        )

    # 2) Gene reorder to geneInfo
    M, warn_list = reorder_genes_to_geneinfo(M, features, geneinfo)

    # 3) Tissue positions join
    pos = read_tissue_positions(slide_dir)

    # Barcode uniqueness check
    if barcodes["barcode"].duplicated().any():
        dup_cnt = int(barcodes["barcode"].duplicated().sum())
        raise ValueError(f"barcodes contain duplicates: {dup_cnt}")

    spots_df = barcodes.merge(pos, on="barcode", how="left")
    spots_df["slide_id"] = slide_id

    # Normalized barcode key for optional joins (remove 10x suffix like '-1')
    spots_df["barcode_key"] = spots_df["barcode"].astype(str).str.replace(r"-\d+$", "", regex=True)

    # 3.1) Optional alt reads join by barcode (with barcode normalization)
    if include_aac_features:
        try:
            alt_df = read_alt_reads(slide_dir)

            # Normalize both sides: remove 10x suffix like '-1'
            alt_df["barcode_key"] = alt_df["barcode"].astype(str).str.replace(r"-\d+$", "", regex=True)

            if alt_df["barcode_key"].duplicated().any():
                alt_df = alt_df.groupby("barcode_key", as_index=False)["alt_count_sum"].sum()
            else:
                alt_df = alt_df[["barcode_key", "alt_count_sum"]]

            spots_df = spots_df.merge(alt_df, on="barcode_key", how="left")
        except FileNotFoundError:
            warn_list.append("alt-read file not found (tried alt_reads.csv, barcode_alt_sum.csv; alt_count_sum column filled with NaN)")
            spots_df["alt_count_sum"] = np.nan
        except Exception as e:
            warn_list.append(f"alt-read file read/join failed: {e} (alt_count_sum column filled with NaN)")
            spots_df["alt_count_sum"] = np.nan
    else:
        warn_list.append("AAC/alt-read feature columns disabled by --no-aac-features")

    # 3.2) Optional composition join by barcode (maternal/fetal cell counts)
    try:
        comp_df = read_composition_counts(slide_dir)
        comp_df["barcode_key"] = comp_df["barcode"].astype(str).str.replace(r"-\d+$", "", regex=True)

        if comp_df["barcode_key"].duplicated().any():
            comp_df = comp_df.groupby("barcode_key", as_index=False)[
                ["maternal_cells", "fetal_cells", "composition_total_cells"]
            ].sum()
        else:
            comp_df = comp_df[["barcode_key", "maternal_cells", "fetal_cells", "composition_total_cells"]]

        spots_df = spots_df.merge(comp_df, on="barcode_key", how="left")
    except FileNotFoundError:
        warn_list.append("composition.csv not found (maternal/fetal composition columns filled with NaN)")
        spots_df["maternal_cells"] = np.nan
        spots_df["fetal_cells"] = np.nan
        spots_df["composition_total_cells"] = np.nan
    except Exception as e:
        warn_list.append(
            f"composition.csv read/join failed: {e} (maternal/fetal composition columns filled with NaN)"
        )
        spots_df["maternal_cells"] = np.nan
        spots_df["fetal_cells"] = np.nan
        spots_df["composition_total_cells"] = np.nan

    # 3.3) Derived AAC ratio: alt_count_sum / composition_total_cells
    # If denominator is missing/zero, keep NaN to avoid misleading infinities.
    if include_aac_features:
        denom = pd.to_numeric(spots_df.get("composition_total_cells", np.nan), errors="coerce")
        numer = pd.to_numeric(spots_df.get("alt_count_sum", np.nan), errors="coerce")
        spots_df["alt_count_sum_per_composition_total_cells"] = np.where(denom > 0, numer / denom, np.nan)

    # Drop join helper column
    spots_df = spots_df.drop(columns=["barcode_key"], errors="ignore")

    # 4) QC columns (always computed from raw counts)
    total_counts, n_genes = compute_qc_columns(M)
    spots_df["total_counts"] = total_counts
    spots_df["n_genes"] = n_genes

    # 5) Save matrices based on requested formats
    counts_binary_path = None
    counts_raw_path = None
    counts_cp10k_log1p_path = None
    counts_cp10k_int_path = None
    counts_spot_max_bin_path = None
    counts_spot_rank_norm_path = None

    if "binary" in save_formats:
        M_binary = binary_presence(M)
        counts_binary_path = out_root / f"counts_binary_{slide_id}.npz"
        sparse.save_npz(counts_binary_path, M_binary)
        logging.info(f"  Saved binary presence: {counts_binary_path.name}")

    if "raw" in save_formats:
        counts_raw_path = out_root / f"counts_raw_{slide_id}.npz"
        sparse.save_npz(counts_raw_path, M)
        logging.info(f"  Saved raw counts: {counts_raw_path.name}")

    if "cp10k_log1p" in save_formats:
        M_cp10k_log1p = cp10k_log1p(M)
        counts_cp10k_log1p_path = out_root / f"counts_cp10k_log1p_{slide_id}.npz"
        sparse.save_npz(counts_cp10k_log1p_path, M_cp10k_log1p)
        logging.info(f"  Saved CP10K+log1p: {counts_cp10k_log1p_path.name}")

    if "cp10k_int" in save_formats:
        M_cp10k_int = cp10k_int(M)
        counts_cp10k_int_path = out_root / f"counts_cp10k_int_{slide_id}.npz"
        sparse.save_npz(counts_cp10k_int_path, M_cp10k_int)
        logging.info(f"  Saved CP10K integer: {counts_cp10k_int_path.name}")

    if "spot_max_bin" in save_formats:
        M_spot_max_bin = spot_max_bin(M)
        counts_spot_max_bin_path = out_root / f"counts_spot_max_bin_{slide_id}.npz"
        sparse.save_npz(counts_spot_max_bin_path, M_spot_max_bin)
        logging.info(f"  Saved spot-max-bin: {counts_spot_max_bin_path.name}")

    if "spot_rank_norm" in save_formats:
        M_spot_rank_norm = spot_rank_norm(M)
        counts_spot_rank_norm_path = out_root / f"counts_spot_rank_norm_{slide_id}.npz"
        sparse.save_npz(counts_spot_rank_norm_path, M_spot_rank_norm)
        logging.info(f"  Saved spot-rank-norm: {counts_spot_rank_norm_path.name}")

    # Build manifest entry
    res = SlideResult(
        slide_id=slide_id,
        n_spots=M.shape[0],
        counts_binary=str(counts_binary_path.name) if counts_binary_path else None,
        counts_raw=str(counts_raw_path.name) if counts_raw_path else None,
        counts_cp10k_log1p=str(counts_cp10k_log1p_path.name) if counts_cp10k_log1p_path else None,
        counts_cp10k_int=str(counts_cp10k_int_path.name) if counts_cp10k_int_path else None,
        counts_spot_max_bin=str(counts_spot_max_bin_path.name) if counts_spot_max_bin_path else None,
        counts_spot_rank_norm=str(counts_spot_rank_norm_path.name) if counts_spot_rank_norm_path else None,
        warnings=warn_list,
    )

    # Column ordering for readability
    base_cols = [
        "slide_id",
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "x",
        "y",
        "maternal_cells",
        "fetal_cells",
        "composition_total_cells",
        "total_counts",
        "n_genes",
    ]
    if include_aac_features:
        base_cols.insert(10, "alt_count_sum")
        base_cols.insert(11, "alt_count_sum_per_composition_total_cells")
    # Keep extras too.
    extra_cols = [c for c in spots_df.columns if c not in base_cols]
    ordered = [c for c in base_cols if c in spots_df.columns] + extra_cols
    spots_df = spots_df[ordered]

    return spots_df, res


# --------------------------- Table saver (Parquet/CSV fallback) -----

def _save_table(df: pd.DataFrame, out_path: Path, preferred_fmt: str = "parquet") -> Path:
    preferred_fmt = preferred_fmt.lower()
    if preferred_fmt == "parquet":
        try:
            df.to_parquet(out_path.with_suffix(".parquet"), index=False)
            return out_path.with_suffix(".parquet")
        except Exception as e:
            logging.warning(f"Parquet not available ({e}); falling back to CSV")
            df.to_csv(out_path.with_suffix(".csv"), index=False)
            return out_path.with_suffix(".csv")
    elif preferred_fmt in {"csv", "tsv"}:
        sep = "," if preferred_fmt == "csv" else "	"
        df.to_csv(out_path.with_suffix(f".{preferred_fmt}"), index=False, sep=sep)
        return out_path.with_suffix(f".{preferred_fmt}")
    elif preferred_fmt == "feather":
        try:
            df.to_feather(out_path.with_suffix(".feather"))
            return out_path.with_suffix(".feather")
        except Exception as e:
            logging.warning(f"Feather not available ({e}); falling back to CSV")
            df.to_csv(out_path.with_suffix(".csv"), index=False)
            return out_path.with_suffix(".csv")
    else:
        logging.warning(f"Unknown table format '{preferred_fmt}', using CSV")
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        return out_path.with_suffix(".csv")


# --------------------------- Orchestrator ----------------------------

def preprocess(
    paths: Paths,
    *,
    save_formats: List[str],
    table_format: str = "parquet",
    include_aac_features: bool = True,
) -> None:
    out = paths.out_root
    out.mkdir(parents=True, exist_ok=True)

    # 1) Read geneInfo -> genes table
    geneinfo = read_geneinfo(paths.geneinfo_path)
    genes_path = _save_table(geneinfo, out / "genes", preferred_fmt=table_format)
    logging.info(f"Saved genes table: {genes_path} ({len(geneinfo)} genes)")

    # 2) Iterate slides
    slide_dirs = sorted([p for p in paths.data_root.iterdir() if p.is_dir()])
    if not slide_dirs:
        raise FileNotFoundError(f"No slide directories found under {paths.data_root}")

    all_spots: List[pd.DataFrame] = []
    manifest: List[Dict] = []

    logging.info(f"Saving formats: {', '.join(save_formats)}")

    for sd in slide_dirs:
        try:
            spots_df, slide_res = process_single_slide(
                sd,
                geneinfo,
                out,
                save_formats,
                include_aac_features=include_aac_features,
            )
        except Exception as e:
            logging.error(f"Failed processing {sd.name}: {e}")
            raise

        all_spots.append(spots_df)
        manifest.append(asdict(slide_res))

    # 3) Concatenate and save spots table
    spots = pd.concat(all_spots, ignore_index=True)

    # Uniqueness check for (slide_id, barcode)
    if spots.duplicated(subset=["slide_id", "barcode"]).any():
        dup = spots[spots.duplicated(subset=["slide_id", "barcode"], keep=False)]
        raise ValueError(f"Duplicate (slide_id, barcode) pairs found:{dup.head()}")

    spots_path = _save_table(spots, out / "spots", preferred_fmt=table_format)
    logging.info(f"Saved spots table: {spots_path} ({len(spots)} rows)")

    # 4) Save manifest.json
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logging.info(f"Wrote manifest.json with {len(manifest)} slides")


# --------------------------- CLI ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ST preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        add_help=False,
    )
    p.add_argument("-h", "--h", action="help", help="Show this help message and exit.")
    p.add_argument("--data-root", required=True, type=Path, help="Path to directory containing slide folders")
    p.add_argument("--geneinfo", required=True, type=Path, help="Path to geneInfo file (tab/tsv/csv)")
    p.add_argument("--out-root", required=True, type=Path, help="Path to output directory")
    p.add_argument("--save-formats", default=['binary'], nargs='+', choices=['binary', 'raw', 'cp10k_log1p'], help="Output matrix formats")
    p.add_argument(
        "--no-aac-features",
        action="store_true",
        help="Do not create AAC/alt-read feature columns in spots table (alt_count_sum and derived ratio).",
    )
    p.add_argument("--table-format", default="csv", choices=["csv", "tsv"], help="Output table format for genes/spots")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    paths = Paths(data_root=args.data_root, geneinfo_path=args.geneinfo, out_root=args.out_root)
    preprocess(
        paths,
        save_formats=args.save_formats,
        table_format=args.table_format,
        include_aac_features=(not args.no_aac_features),
    )


if __name__ == "__main__":
    main()

