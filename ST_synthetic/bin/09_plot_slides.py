#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot the ground-truth spatial layout of each simulated slide.

Pipeline position
    Script 09 of ``run_pipeline.sh``, run once per sample.

What it does
    For every slide, pairs the Cell Ranger directory with its composition table,
    reconstructs the grid coordinates, and colours each spot by which origin
    dominates it. The result is a quick visual check that the invasion zone
    formed where the composition step intended.

    A spot is labelled Maternal or Fetal by whichever count is larger, Balanced
    on a tie, and Empty when both are zero.

Inputs (per slide, under <data_dir>/<sample>/out/)
    cellranger_format_seed<seed>/            from 06_to_cellranger.py
    synthetic_ST_seed<seed>_1_composition.csv

Output
    <data_dir>/<sample>/out/<sample>_plot_seed<seed>.png
"""

import pandas as pd
import scipy.io
from scipy import sparse
from anndata import AnnData
import anndata as ad
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scanpy as sc
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

# Subdirectory of the sample folder that holds simulation results.
TARGET_OUT_DIRS = ['out']


# =========================================================================
# Per-slide plotting
# =========================================================================
def process_sample(sample_name, seed, cellranger_dir, composition_file, output_dir_name):
    """Load one slide, assign a dominant-origin label per spot, and plot it."""
    print(f"   Processing Seed: {seed}")

    try:
        mtx = os.path.join(cellranger_dir, "matrix.mtx.gz")
        features = os.path.join(cellranger_dir, "features.tsv.gz")
        barcodes = os.path.join(cellranger_dir, "barcodes.tsv.gz")

        # Fall back to uncompressed files if the gzipped set is absent.
        if not (os.path.exists(mtx) and os.path.exists(features) and os.path.exists(barcodes)):
            mtx = mtx.replace(".gz", "")
            features = features.replace(".gz", "")
            barcodes = barcodes.replace(".gz", "")

            if not (os.path.exists(mtx) and os.path.exists(features) and os.path.exists(barcodes)):
                print(f"      -> Skipping: Cellranger files missing in {cellranger_dir}")
                return

        # Stored genes x spots, so transpose back to spots x genes for AnnData.
        X = scipy.io.mmread(mtx).T.tocsr()
        var = pd.read_csv(features, header=None, sep='\t')
        obs = pd.read_csv(barcodes, header=None, sep='\t')

        adata = AnnData(
            X=X,
            obs=pd.DataFrame(index=obs[0]),
            var=pd.DataFrame(index=var[1])
        )

        # Composition is cell-type-major; transpose to one row per spot.
        composition_df = pd.read_csv(composition_file, index_col=0)
        composition_df = composition_df.T

        if len(adata.obs) != len(composition_df):
            print(f"      -> Error: Spot count mismatch. Skipping.")
            return

        # Both are in the same row-major spot order, so indices can be aligned
        # positionally: the barcodes carry a -1 suffix the composition lacks.
        composition_df.index = adata.obs.index
        adata.obs.index.name = None
        adata.obs = adata.obs.join(composition_df)

        # Rebuild the grid coordinates used by 04_place_cells.py.
        n_spots = len(adata.obs)
        grid_size = int(np.ceil(np.sqrt(n_spots)))

        coords = pd.DataFrame(index=adata.obs.index)
        coords['x'] = [i % grid_size for i in range(n_spots)]
        coords['y'] = [i // grid_size for i in range(n_spots)]

        adata.obsm['spatial'] = coords[['x', 'y']].to_numpy()

        def assign_label(row):
            maternal_count = row.get('Maternal', 0)
            fetal_count = row.get('Fetal', 0)
            if maternal_count > fetal_count: return 'Maternal'
            elif fetal_count > maternal_count: return 'Fetal'
            else: return 'Empty' if maternal_count == 0 else 'Balanced'

        adata.obs['Label'] = adata.obs.apply(assign_label, axis=1)
        top1_celltype = adata.obs['Label']

        # Only the labels actually present are kept in the legend.
        unique_celltypes = sorted(top1_celltype.unique())
        color_map = {'Maternal': '#1f77b4', 'Fetal': '#d62728', 'Balanced': '#9467bd', 'Empty': '#c7c7c7'}
        color_map = {ct: color_map[ct] for ct in unique_celltypes if ct in color_map}

        coords_arr = adata.obsm['spatial']
        x = coords_arr[:, 0]
        y = coords_arr[:, 1]
        colors = top1_celltype.astype(str).map(color_map).values

        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, c=colors, s=15, edgecolor='none')
        plt.xlabel("x (virtual)")
        plt.ylabel("y (virtual)")
        plt.title(f"[{sample_name}] Seed {seed}\nSimulated Composition")
        # Inverted so row 0 sits at the top, as in the generating grid.
        plt.gca().invert_yaxis()

        legend_elements = [Patch(facecolor=color_map[ct], label=ct) for ct in unique_celltypes if ct in color_map]
        plt.legend(handles=legend_elements, title="Dominant Origin", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.dirname(composition_file)
        save_filename = os.path.join(save_path, f"{sample_name}_plot_seed{seed}.png")
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"      -> Plot saved: {os.path.basename(save_filename)}")

    except Exception as e:
        print(f"      -> ❌ Error processing seed {seed}: {e}")


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Visualize simulation results (Single Sample).")

    parser.add_argument('--data_dir', required=True, help="Root data directory")
    parser.add_argument('--sample', required=True, help="Target Sample Name (e.g., norm_Endo2)")

    args = parser.parse_args()
    BASE_DIR = args.data_dir
    SAMPLE = args.sample

    print("========================================================")
    print(f"   [09] Visualization Started for: {SAMPLE}")
    print("========================================================")

    full_sample_path = os.path.join(BASE_DIR, SAMPLE)

    if not os.path.exists(full_sample_path):
        print(f"❌ Error: Sample folder not found: {full_sample_path}")
        return

    # Seeds are discovered from the composition filenames.
    seed_pattern = re.compile(r"synthetic_ST_seed(\d+)_1_composition\.csv")

    for target_out in TARGET_OUT_DIRS:
        out_dir = os.path.join(full_sample_path, target_out)

        if os.path.exists(out_dir):
            print(f" -> Scanning directory: {target_out}")

            found_count = 0
            for filename in os.listdir(out_dir):
                match = seed_pattern.match(filename)

                if match:
                    seed = match.group(1)
                    comp_file_path = os.path.join(out_dir, filename)

                    # A slide is only plotted if step3 produced its 10x folder.
                    cellranger_dir_name = f"cellranger_format_seed{seed}"
                    cellranger_full_path = os.path.join(out_dir, cellranger_dir_name)

                    if os.path.exists(cellranger_full_path):
                        process_sample(
                            sample_name=SAMPLE,
                            seed=seed,
                            cellranger_dir=cellranger_full_path,
                            composition_file=comp_file_path,
                            output_dir_name=target_out
                        )
                        found_count += 1

            if found_count == 0:
                print(f"    (No simulation results found in {target_out})")
        else:
            print(f" -> Directory not found: {target_out} (Skipping)")

    print(f"\n✅ Step 09 Completed for {SAMPLE}!")


if __name__ == "__main__":
    main()
