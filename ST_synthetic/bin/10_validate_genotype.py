#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Check that the genotype signal tracks fetal cell content, before and after simulation.

Pipeline position
    Script 10 of ``run_pipeline.sh``, run once per sample, last.

What it does
    Part 1, single-cell level
        Joins the raw Vartrix totals to the cell origin labels and compares the
        alternate-allele distributions of fetal and maternal cells. This
        establishes that the genotype signal separates the two origins at all,
        independently of the simulation.

    Part 2, spot level
        For each simulated slide, joins the aggregated genotype signal to the
        ground-truth composition, rebuilds the grid coordinates, and plots the
        fetal cell map beside the genotype map. If the simulation preserved the
        signal, the two maps should overlap.

Inputs
    <data_dir>/merged_metadata.csv                         cell origin labels
    <data_dir>/<sample>/barcode_alt_sum.csv                raw per-cell totals
    <data_dir>/<sample>/out/synthetic_ST_seed*_1_alt_reads.csv
    <data_dir>/<sample>/out/synthetic_ST_seed*_1_composition.csv

Outputs (in <data_dir>/<sample>/out/)
    raw_data_qc.png                                        Part 1 figure
    seed<seed>_fetal_ratio_dist.png                        fetal fraction histogram
    seed<seed>_spatial_analysis.png                        side-by-side spatial maps
    synthetic_ST_seed<seed>_1_alt_reads_sumNorm.csv        per-spot table with
                                                           coordinates and ratios
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import csv

# ------------------------------------------------------------------------------
# Arguments and paths
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
parser.add_argument('--sample', type=str, required=True, help='Sample Name')
args = parser.parse_args()

DATA_DIR = args.data_dir
SAMPLE_NAME = args.sample

METADATA_PATH = os.path.join(DATA_DIR, "merged_metadata.csv")
SAMPLE_DIR = os.path.join(DATA_DIR, SAMPLE_NAME)

# The Vartrix table has been named several ways across runs, so fall back to any
# CSV in the sample folder with "alt" in its name.
if os.path.exists(os.path.join(SAMPLE_DIR, "barcode_alt_sum.csv")):
    VARTRIX_PATH = os.path.join(SAMPLE_DIR, "barcode_alt_sum.csv")
else:
    possible_files = glob.glob(os.path.join(SAMPLE_DIR, "*alt*.csv"))
    VARTRIX_PATH = possible_files[0] if possible_files else os.path.join(SAMPLE_DIR, "barcode_alt_sum.csv")

SIMULATION_OUT_DIR = os.path.join(SAMPLE_DIR, "out")

print(f"🚀 Integrated Analysis Started for: {SAMPLE_NAME}")
print(f"📂 Data Directory: {DATA_DIR}")
print(f"📂 Output Directory: {SIMULATION_OUT_DIR}")

# ------------------------------------------------------------------------------
# [PART 1] Single-cell level check
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("📌 [PART 1] Raw Single-Cell Statistical Analysis")
print("="*80)

print(f"   - Loading Metadata: {os.path.basename(METADATA_PATH)}")
if not os.path.exists(METADATA_PATH):
    print("❌ Metadata file not found. Skipping Part 1.")
else:
    meta_df = pd.read_csv(METADATA_PATH, index_col=0)

    if 'inferredCellOrigin' in meta_df.columns:
        meta_df = meta_df[['inferredCellOrigin']]
    else:
        print("⚠️ 'inferredCellOrigin' column missing. Skipping Part 1.")
        meta_df = pd.DataFrame() # Empty to skip

    print(f"   - Loading Vartrix:  {os.path.basename(VARTRIX_PATH)}")

    # The file is tab-separated but its fields are sometimes quoted, so quoting
    # is disabled and the quote characters are stripped by hand. A plain
    # comma-separated read is the fallback.
    try:
        alt_df = pd.read_csv(VARTRIX_PATH, sep='\t', quoting=3)
        alt_df.columns = alt_df.columns.str.replace('"', '').str.strip()
        if alt_df['barcode'].dtype == object:
            alt_df['barcode'] = alt_df['barcode'].str.replace('"', '').str.strip()
        if 'alt_count_sum' in alt_df.columns and alt_df['alt_count_sum'].dtype == object:
             alt_df['alt_count_sum'] = alt_df['alt_count_sum'].astype(str).str.replace('"', '').str.strip()
             alt_df['alt_count_sum'] = pd.to_numeric(alt_df['alt_count_sum'])
    except Exception as e:
        print(f"⚠️ Tab-separated read failed ({e}), trying comma separator...")
        try:
            alt_df = pd.read_csv(VARTRIX_PATH, sep=',')
        except:
             alt_df = pd.DataFrame() # Fail gracefully

    if not meta_df.empty and not alt_df.empty:
        # Barcode formats differ between the two files: metadata carries the
        # sample prefix added in step1, Vartrix carries the raw 10x barcode with
        # its -1 suffix. Strip both down to the bare barcode before joining.
        sample_prefix = SAMPLE_NAME.upper()
        current_meta = meta_df[meta_df.index.str.startswith(sample_prefix)].copy()
        if len(current_meta) == 0:
             # No prefix found: assume the metadata covers this sample only.
             current_meta = meta_df.copy()
        else:
             prefix_len = len(sample_prefix)
             current_meta.index = current_meta.index.str[prefix_len:]

        if 'barcode' in alt_df.columns:
            alt_df.set_index('barcode', inplace=True)
        alt_df.index = alt_df.index.str.replace('-1$', '', regex=True)

        raw_df = current_meta.join(alt_df, how='inner')
        raw_df['alt_count_sum'] = raw_df['alt_count_sum'].fillna(0)
        print(f"✅ Merged Single Cells: {len(raw_df)} cells")

        def classify_origin(label):
            """Collapse the origin annotation to Fetal or Maternal."""
            if pd.isna(label): return "Unknown"
            label_lower = str(label).lower()
            fetal_keywords = ['fetal', 'trophoblast', 'evt', 'sct', 'vct']
            if any(k in label_lower for k in fetal_keywords):
                return "Fetal"
            else:
                return "Maternal"

        raw_df['Group'] = raw_df['inferredCellOrigin'].apply(classify_origin)

        # Log scale: alt counts span orders of magnitude across cells.
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle(f"[PART 1] Raw Analysis - {SAMPLE_NAME}", fontsize=16)
        sns.boxplot(data=raw_df, x='Group', y='alt_count_sum', ax=axes1[0], palette=['#ff9999', '#66b3ff'])
        axes1[0].set_yscale("log")
        try:
            sns.kdeplot(data=raw_df, x='alt_count_sum', hue='Group', fill=True, ax=axes1[1],
                        palette=['#ff9999', '#66b3ff'], common_norm=False, clip=(0, raw_df['alt_count_sum'].quantile(0.99)))
        except:
            pass # kdeplot fails when a group is empty or nearly constant

        plt.tight_layout()
        raw_viz_path = os.path.join(SIMULATION_OUT_DIR, "raw_data_qc.png")
        plt.savefig(raw_viz_path)
        print(f"   -> Saved Raw Data QC: {os.path.basename(raw_viz_path)}")
        plt.close() # Headless environment: never show, always close

# ------------------------------------------------------------------------------
# [PART 2] Spot level check
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("📌 [PART 2] Spatial Simulation Analysis")
print("="*80)

# Seeds are discovered from the genotype files. The "_alt_counts" pattern is the
# older naming, kept as a fallback for slides generated before the rename.
alt_files = glob.glob(os.path.join(SIMULATION_OUT_DIR, "*_alt_reads.csv"))
if not alt_files:
    alt_files = glob.glob(os.path.join(SIMULATION_OUT_DIR, "*_alt_counts.csv"))
    if not alt_files:
        print("❌ Alt Count CSV files not found. Check previous steps.")
        exit(1)

seeds = sorted(list(set([re.search(r"seed(\d+)", f).group(1) for f in alt_files if re.search(r"seed(\d+)", f)])))
print(f"🌱 Detected {len(seeds)} Seeds: {seeds}")

for SEED in seeds:
    print("\n" + "-"*60)
    print(f"🚀 Processing Seed: {SEED}")
    print("-"*60)

    try:
        path_alt = os.path.join(SIMULATION_OUT_DIR, f"synthetic_ST_seed{SEED}_1_alt_reads.csv")
        path_comp = os.path.join(SIMULATION_OUT_DIR, f"synthetic_ST_seed{SEED}_1_composition.csv")

        df_alt = pd.read_csv(path_alt)
        df_comp = pd.read_csv(path_comp, index_col=0)

        # Composition is cell-type-major on disk; transpose to one row per spot.
        if 'Spot' not in df_comp.index[0] and 'Spot' in df_comp.columns[0]:
            df_comp = df_comp.T

        df_comp.index.name = 'spot_id'
        df_comp.reset_index(inplace=True)
        spatial_df = pd.merge(df_alt, df_comp, left_on='spot_id', right_on=df_comp.columns[0])

        # Spots are named Spotx1..SpotxN, so subtracting 1 recovers the
        # row-major index used to place them on the grid.
        n_spots = len(spatial_df)
        grid_size = int(np.ceil(np.sqrt(n_spots)))

        def extract_spot_num(x):
            return int(re.search(r"(\d+)", x).group(1)) - 1

        spatial_df['spot_num'] = spatial_df['spot_id'].apply(extract_spot_num)
        spatial_df = spatial_df.sort_values('spot_num')
        spatial_df['x'] = spatial_df['spot_num'] % grid_size
        # Negated so that row 0 plots at the top.
        spatial_df['y'] = -(spatial_df['spot_num'] // grid_size)

        # Cell type columns are whatever remains once the bookkeeping columns
        # are excluded.
        fetal_col = [c for c in spatial_df.columns if 'Fetal' in c or 'Trophoblast' in c][0]
        cell_cols = [c for c in spatial_df.columns if c not in ['spot_id', 'alt_count_sum', 'spot_num', 'x', 'y'] and 'Spot' not in c]

        spatial_df['total_cells'] = spatial_df[cell_cols].sum(axis=1)
        # replace(0, 1) guards the division; spots with no cells score 0 anyway.
        spatial_df['Fetal_Ratio'] = spatial_df[fetal_col] / spatial_df['total_cells'].replace(0, 1)
        spatial_df['alt_mean'] = spatial_df['alt_count_sum'] / spatial_df['total_cells'].replace(0, 1)

        output_path = os.path.join(SIMULATION_OUT_DIR, f"synthetic_ST_seed{SEED}_1_alt_reads_sumNorm.csv")
        spatial_df.to_csv(output_path)
        print(f"   -> Saved Expanded QC File: {os.path.basename(output_path)}")

        # --- Distribution of fetal fraction ---
        fig_dist, ax_dist = plt.subplots(1, 2, figsize=(14, 5))
        fig_dist.suptitle(f"Fetal Cell Ratio (Seed {SEED})", fontsize=16)
        # Log y: most spots sit at ratio 0, which would flatten everything else.
        sns.histplot(spatial_df['Fetal_Ratio'], bins=20, kde=False, ax=ax_dist[0], color='skyblue')
        ax_dist[0].set_yscale('log')
        ax_dist[0].set_title("All Spots (Log Scale)")

        fetal_existing = spatial_df[spatial_df[fetal_col] > 0]
        if len(fetal_existing) > 0:
            sns.histplot(fetal_existing['Fetal_Ratio'], bins=20, kde=True, ax=ax_dist[1], color='salmon')
            ax_dist[1].set_title(f"Active Spots (n={len(fetal_existing)})")

        plt.tight_layout()
        dist_save_path = os.path.join(SIMULATION_OUT_DIR, f"seed{SEED}_fetal_ratio_dist.png")
        plt.savefig(dist_save_path)
        plt.close()

        # --- Ground truth beside the genotype signal ---
        fig_map, axes_map = plt.subplots(1, 3, figsize=(20, 6))
        fig_map.suptitle(f"Spatial Simulation Analysis - Seed {SEED}", fontsize=16, fontweight='bold')

        scatter1 = axes_map[0].scatter(spatial_df['x'], spatial_df['y'], c=spatial_df[fetal_col], cmap='Blues', s=20, alpha=0.9)
        axes_map[0].set_title(f"Ground Truth: {fetal_col}")
        plt.colorbar(scatter1, ax=axes_map[0])
        axes_map[0].axis('off')

        # Capped at the 95th percentile so a few extreme spots do not wash out
        # the colour scale.
        vmax_val = spatial_df['alt_mean'].quantile(0.95)
        scatter2 = axes_map[1].scatter(spatial_df['x'], spatial_df['y'], c=spatial_df['alt_mean'], cmap='Reds', s=20, alpha=0.9, vmax=vmax_val)
        axes_map[1].set_title("Simulated Feature: Mean Alt")
        plt.colorbar(scatter2, ax=axes_map[1])
        axes_map[1].axis('off')

        sns.boxplot(data=spatial_df, x=fetal_col, y='alt_mean', ax=axes_map[2], palette='Reds', showfliers=False)
        axes_map[2].set_title("Correlation Check")

        plt.tight_layout()
        viz_path = os.path.join(SIMULATION_OUT_DIR, f"seed{SEED}_spatial_analysis.png")
        plt.savefig(viz_path)
        print(f"   -> Saved Visualization: {os.path.basename(viz_path)}")
        plt.close()

    except Exception as e:
        print(f"❌ Error processing Seed {SEED}: {e}")

print("\n✅ All Validation Steps Completed.")
