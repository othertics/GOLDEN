#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize per-spot statistics for every simulated slide of one sample.

Pipeline position
    Script 08 of ``run_pipeline.sh``, run once per sample after the slides have
    been assembled.

What it does
    Pairs each slide's count matrix with its composition table and reports the
    distributions that determine whether the simulation is realistic: UMI per
    spot, detected genes per spot, cells per spot, and overall sparsity. One row
    per seed is written to a single summary file, and a coarse sanity check on
    the mean cell count is printed at the end.

Inputs (in --dir)
    synthetic_ST_seed<seed>_1_counts.csv         spot x gene
    synthetic_ST_seed<seed>_1_composition.csv    cell type x spot

Output
    --dir/simulation_statistics_summary.csv      one row per seed
"""

import pandas as pd
import glob
import re
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate QC statistics for simulation results.")
    parser.add_argument('--dir', required=True, help="Target directory containing simulation CSVs")

    args = parser.parse_args()
    work_dir = args.dir

    print("========================================================")
    print("   [08] QC & Statistics Analysis Started")
    print(f"   Target Dir: {work_dir}")
    print("========================================================")

    summary_data = []

    # Matches the assemble_id=1 slides written by run_pipeline.sh.
    search_pattern = os.path.join(work_dir, "synthetic_ST_seed*_1_counts.csv")
    count_files = glob.glob(search_pattern)
    count_files.sort()

    print(f" -> Found {len(count_files)} files. Analyzing...\n")

    if not count_files:
        print("⚠️ No counts CSV files found. Skipping step.")
        return

    for count_file in count_files:
        try:
            filename = os.path.basename(count_file)
            match = re.search(r"seed(\d+)_", filename)

            if match:
                seed = match.group(1)
            else:
                print(f"Skipping: {filename} (Cannot extract seed number)")
                continue

            # --- Load the pair, aligning their orientations ---

            # Counts are already spot-major.
            counts = pd.read_csv(count_file, index_col=0)

            comp_filename = f"synthetic_ST_seed{seed}_1_composition.csv"
            comp_file = os.path.join(work_dir, comp_filename)

            if not os.path.exists(comp_file):
                print(f"Warning: Composition file missing for Seed {seed}")
                continue

            comp_raw = pd.read_csv(comp_file, index_col=0)

            # Composition is written cell-type-major, so transpose it to match.
            comp = comp_raw.T

            if counts.shape[0] != comp.shape[0]:
                print(f"[Warning] Seed {seed}: Spot count mismatch (Counts: {counts.shape[0]}, Comp: {comp.shape[0]})")

            # --- Per-spot statistics ---

            umis_per_spot = counts.sum(axis=1)
            genes_per_spot = (counts > 0).sum(axis=1)

            # After the transpose above, summing across columns totals all cell
            # types within one spot.
            cells_per_spot = comp.sum(axis=1)

            sparsity = (counts == 0).sum().sum() / counts.size

            stats_row = {
                "Sample_ID": f"cellranger_format_seed{seed}",
                "Seed": seed,

                "Mean_UMI": umis_per_spot.mean(),
                "Median_UMI": umis_per_spot.median(),
                "Std_UMI": umis_per_spot.std(),
                "Min_UMI": umis_per_spot.min(),
                "Max_UMI": umis_per_spot.max(),

                "Mean_Genes": genes_per_spot.mean(),
                "Median_Genes": genes_per_spot.median(),
                "Min_Genes": genes_per_spot.min(),
                "Max_Genes": genes_per_spot.max(),

                "Mean_Cells": cells_per_spot.mean(),
                "Max_Cells": cells_per_spot.max(),
                "Min_Cells": cells_per_spot.min(),

                "Sparsity": sparsity
            }

            summary_data.append(stats_row)

        except Exception as e:
            print(f"Error processing seed {seed}: {e}")

    if summary_data:
        df_summary = pd.DataFrame(summary_data)

        output_filename = "simulation_statistics_summary.csv"
        output_path = os.path.join(work_dir, output_filename)

        df_summary.to_csv(output_path, index=False)
        print("-" * 30)
        print(f"✅ Statistics saved to: {output_filename}")

        # Sanity check: simulated spots should hold a handful of cells, in line
        # with nuclear segmentation counts on real Visium spots.
        avg_cell_count = df_summary['Mean_Cells'].mean()
        print(f" -> Average Cells per Spot (All Seeds): {avg_cell_count:.2f}")
        if avg_cell_count > 100:
             print("    ⚠️  Check Check: Cell counts seem unusually high!")
        elif avg_cell_count < 1:
             print("    ⚠️  Check Check: Cell counts seem unusually low!")
        else:
             print("    OK: Cell counts look reasonable.")
    else:
        print("No data processed.")


if __name__ == "__main__":
    main()
