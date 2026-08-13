#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare the downsampled single-cell input for one sample.

Pipeline position
    Script 01 of ``run_pipeline.sh``, run once per sample, before
    ``02_build_cell_pool.py``.

What it does
    1. Loads the Cell Ranger H5 matrix and computes QC metrics.
    2. Merges the per-barcode Vartrix alternate-allele totals into the cell
       metadata as ``alt_count_sum``.
    3. Downsamples expression so each cell carries at most --target_umi UMI,
       chosen so that a few cells summed into one spot land in the UMI range
       observed on real Visium spots.
    4. Swaps gene symbols for Ensembl IDs and prefixes barcodes with the sample
       name, then writes the H5AD and a metadata CSV.

    The Vartrix merge runs before the barcode prefix is applied, so the join
    happens on the raw 10x barcodes present in both files.

Inputs
    <data_dir>/<sample>/filtered_feature_bc_matrix.h5   Cell Ranger output
    --metadata                                          merged_metadata.csv, cell annotations
    --vartrix_csv                                       barcode_alt_sum.csv, from sum_alt_counts.py

Outputs
    <data_dir>/<sample>/<sample>_downsampled.h5ad
    <data_dir>/<sample>/<sample>_downsampled_metadata.csv
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import warnings
import argparse
from scipy import sparse

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# =========================================================================
# Vartrix merge
# =========================================================================
def merge_vartrix_data(adata, vartrix_path):
    """Attach per-barcode alternate-allele totals to ``adata.obs``.

    Reads the table written by ``sum_alt_counts.py`` and joins it onto the cell
    index as ``alt_count_sum``. The source table only lists barcodes that
    carried at least one alternate read, so cells absent from it are filled
    with 0. Any failure leaves the column present and set to 0 so that
    downstream steps still run.
    """
    if not vartrix_path or not os.path.exists(vartrix_path):
        print(f"   [Warning] Vartrix file not found or not provided: {vartrix_path}")
        print(f"             Setting 'alt_count_sum' to 0 for all cells.")
        adata.obs['alt_count_sum'] = 0
        return adata

    print(f"   [Vartrix] Loading Alt Counts from: {vartrix_path}")

    try:
        if vartrix_path.endswith('.tsv') or vartrix_path.endswith('.txt'):
            sep = '\t'
        else:
            sep = ','

        # sum_alt_counts.py writes tab-separated output regardless of the .csv
        # extension, so the file is read as TSV.
        df_alt = pd.read_csv(vartrix_path, sep='\t')

        # Fall back to positional lookup if the header is missing.
        if 'barcode' not in df_alt.columns:
            df_alt.rename(columns={df_alt.columns[0]: 'barcode'}, inplace=True)

        df_alt.set_index('barcode', inplace=True)

        # Joined on the raw 10x barcodes: the sample prefix has not been applied
        # to adata.obs_names yet at this point.
        original_len = len(adata)
        adata.obs = adata.obs.join(df_alt['alt_count_sum'], how='left')

        fill_count = adata.obs['alt_count_sum'].isna().sum()
        adata.obs['alt_count_sum'] = adata.obs['alt_count_sum'].fillna(0).astype(int)

        print(f"     -> Merged successfully.")
        print(f"     -> Cells with Alt Count: {original_len - fill_count}/{original_len}")
        print(f"     -> Max Alt Count: {adata.obs['alt_count_sum'].max()}")

    except Exception as e:
        print(f"     -> ❌ Error merging Vartrix data: {e}")
        print("     -> Proceeding with alt_count_sum = 0")
        adata.obs['alt_count_sum'] = 0

    return adata


# =========================================================================
# Downsampling
# =========================================================================
def apply_downsampling(adata, sample_name, target_umi):
    """Cap each cell at ``target_umi`` UMI by subsampling without replacement.

    Cells already below the target are left untouched, so the target acts as a
    ceiling rather than a fixed depth.
    """
    print(f"   [Downsampling] Processing {sample_name}...")

    # downsample_counts requires integer counts.
    if not np.issubdtype(adata.X.dtype, np.integer):
        if sparse.issparse(adata.X):
            adata.X = adata.X.astype(int)
        else:
            adata.X = adata.X.astype(int)

    original_mean = adata.obs['total_counts'].mean()
    print(f"      -> Original Mean: {original_mean:.1f}")
    print(f"      -> Applying Target UMI per Cell: {target_umi}")

    try:
        sc.pp.downsample_counts(adata, counts_per_cell=target_umi, replace=False)
    except Exception as e:
        print(f"      -> ❌ Downsampling Error: {e}")
        return adata

    # QC metrics are recomputed because total_counts is now stale.
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    new_mean = adata.obs['total_counts'].mean()
    print(f"      -> Result Mean UMI: {original_mean:.1f} => {new_mean:.1f}")

    return adata


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 1: Downsample scRNA-seq + Vartrix Feature")

    parser.add_argument('--data_dir', required=True, help="Path to data directory")
    parser.add_argument('--metadata', required=True, help="Path to merged_metadata.csv")
    parser.add_argument('--sample', required=True, help="Target Sample Name")
    parser.add_argument('--target_umi', type=int, default=2600, help="Target UMI counts")
    parser.add_argument('--vartrix_csv', required=False, default=None, help="Path to barcode_alt_sum.csv (Tab separated)")

    args = parser.parse_args()

    BASE_FOLDER = args.data_dir
    METADATA_PATH = args.metadata
    SAMPLE = args.sample
    TARGET_UMI = args.target_umi
    VARTRIX_PATH = args.vartrix_csv

    print("========================================================")
    print(f"   [01] Processing: {SAMPLE}")
    print("========================================================")

    # --- Global cell annotations, shared across samples ---
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Error: Metadata file not found at {METADATA_PATH}")
        return
    metadata = pd.read_csv(METADATA_PATH, index_col=0)

    # --- Expression matrix for this sample ---
    h5_path = os.path.join(BASE_FOLDER, SAMPLE, "filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        print(f"   -> ⚠️ File not found: {h5_path}")
        return

    try:
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
    except Exception as e:
        print(f"   -> ❌ Error reading H5: {e}")
        return

    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

    # --- Genotype signal, merged while barcodes are still raw ---
    if VARTRIX_PATH:
        adata = merge_vartrix_data(adata, VARTRIX_PATH)
    else:
        print("   [Info] No Vartrix file provided. 'alt_count_sum' will be 0.")
        adata.obs['alt_count_sum'] = 0

    adata = apply_downsampling(adata, SAMPLE, target_umi=TARGET_UMI)

    # --- Gene symbols -> Ensembl IDs, so downstream files key on stable IDs ---
    adata.var['gene_symbols'] = adata.var_names
    adata.var_names = adata.var['gene_ids']
    del adata.var['gene_ids']
    adata.var_names.name = "gene_ids"
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    # --- Prefix barcodes with the sample name so cells stay unique once
    #     samples are pooled. alt_count_sum already lives in obs, so renaming
    #     the index does not disturb it.
    sample_prefix = SAMPLE.upper()
    adata.obs_names = [f"{sample_prefix}{x.split('-')[0]}" for x in adata.obs_names]

    # --- Write outputs ---
    save_path = os.path.join(BASE_FOLDER, SAMPLE, f"{SAMPLE}_downsampled.h5ad")
    try:
        adata.write_h5ad(save_path)
        print(f"   -> Saved H5AD: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"   -> ❌ Error saving H5AD: {e}")
        return

    # Metadata CSV: cell annotations joined onto the barcodes that survived
    # loading, restricted to the columns the simulation needs.
    obs_df = adata.obs.copy()
    obs_df.reset_index(inplace=True)
    obs_df.rename(columns={'index': 'cell_barcode'}, inplace=True)

    merged_df = obs_df.merge(metadata, how='inner', left_on='cell_barcode', right_index=True)

    meta_cols = ['predict', 'tissueDese', 'sampleDesc', 'inferredCellOrigin', 'alt_count_sum']
    final_cols = ['cell_barcode'] + [c for c in meta_cols if c in merged_df.columns]

    csv_path = os.path.join(BASE_FOLDER, SAMPLE, f"{SAMPLE}_downsampled_metadata.csv")
    merged_df.to_csv(csv_path, columns=final_cols, index=False)

    print(f"\n✅ Step 01 Completed for {SAMPLE}!")


if __name__ == "__main__":
    main()
