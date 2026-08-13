#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Synthesize the expression matrix and genotype signal for one simulated slide.

Pipeline position
    Script 05 of ``run_pipeline.sh``, run once per seed, after
    ``04_place_cells.py``.

What it does
    For every spot, draws the cells prescribed by the composition table from the
    matching single-cell pool (with replacement), then sums two quantities over
    the drawn cells:

    1. gene expression, giving a pseudobulk UMI vector per spot;
    2. ``alt_count_sum``, giving an aggregated genotype signal per spot.

Inputs (resolved from --out_dir, not passed explicitly)
    labels_generation_<seed>.p                              annotations + alt_count_sum
    counts_generation_<seed>.p                              cell x gene expression
    synthetic_ST_seed<seed>_<assemble_id>_composition.csv   cell type x spot counts

Outputs
    synthetic_ST_seed<seed>_<assemble_id>_counts.csv        spot x gene UMI matrix
    synthetic_ST_seed<seed>_<assemble_id>_alt_reads.csv     spot_id, alt_count_sum
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.sparse
import os

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('seed', type=int, help='random seed of split')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='./',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int, default=1, help='ID of ST assembly')
parser.add_argument('--annotation_col', dest='anno_col', type=str, default="annotation_1",
                    help='Name of column to use in annotation file (default: annotation_1)')

args = parser.parse_args()

seed = args.seed
out_dir = args.out_dir
assemble_id = args.assemble_id
anno_col = args.anno_col

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

print(f"Loading generation data for Seed {seed}...")
lbl_gen_file = out_dir + "labels_generation_" + str(seed) + ".p"
count_gen_file = out_dir + "counts_generation_" + str(seed) + ".p"
spots_members_file = out_dir + "synthetic_ST_seed" + str(seed) + "_" + str(assemble_id) + "_composition.csv"

# Cell metadata: carries the annotation column and alt_count_sum.
lbl_generation = pickle.load(open(lbl_gen_file, "rb"))

# Expression: cells as rows, genes as columns.
cnt_generation = pickle.load(open(count_gen_file, "rb"))

# Composition: cell types as rows, spots as columns.
spots_members = pd.read_csv(spots_members_file, index_col=0)

# ---------------------------------------------------------------------------
# Prepare lookup structures
# ---------------------------------------------------------------------------
# Summing hundreds of thousands of cell draws through pandas .loc is slow, so
# expression goes into a CSR matrix and cell names are mapped to row positions
# once, up front.

if isinstance(cnt_generation, pd.DataFrame):
    gene_names = cnt_generation.columns
    cnt_matrix = scipy.sparse.csr_matrix(cnt_generation.values)
    cell_indices_map = {name: i for i, name in enumerate(cnt_generation.index)}
else:
    # Not reached with the current split_sc output, which is always a DataFrame.
    gene_names = cnt_generation.columns
    cnt_matrix = cnt_generation
    cell_indices_map = {name: i for i, name in enumerate(cnt_generation.index)}

if isinstance(lbl_generation, pd.Series):
    lbl_generation = lbl_generation.to_frame()

# 01_prepare_input.py adds this column, filling it with zeros when no Vartrix
# file is available.
if 'alt_count_sum' not in lbl_generation.columns:
    print("⚠️ Warning: 'alt_count_sum' not found in metadata. Creating dummy 0s.")
    lbl_generation['alt_count_sum'] = 0

# Group cell row positions by type once, so each spot samples from a plain array
# instead of filtering the label frame again.
cell_type_indices = {}
unique_types = lbl_generation[anno_col].unique()

for ctype in unique_types:
    cells_of_type = lbl_generation.index[lbl_generation[anno_col] == ctype]
    indices = [cell_indices_map[c] for c in cells_of_type if c in cell_indices_map]
    cell_type_indices[ctype] = np.array(indices)

# ---------------------------------------------------------------------------
# Assemble spots
# ---------------------------------------------------------------------------

print(f"Assembling {spots_members.shape[1]} spots...")

np.random.seed(seed)

synth_rows = []
synth_alt_sums = []
spot_names = spots_members.columns # Spotx1, Spotx2...

for spot in spot_names:
    chosen_indices = []

    for ctype in unique_types:
        if ctype in spots_members.index:
            n_needed = int(spots_members.loc[ctype, spot])

            if n_needed > 0:
                # Sampling with replacement, so a pool smaller than n_needed is
                # still valid.
                pool = cell_type_indices.get(ctype, [])
                if len(pool) > 0:
                    selected = np.random.choice(pool, n_needed, replace=True)
                    chosen_indices.extend(selected)

    if len(chosen_indices) > 0:
        # A. Pseudobulk expression for the spot.
        spot_vec = cnt_matrix[chosen_indices, :].sum(axis=0)
        spot_vec = np.asarray(spot_vec).flatten()

        # B. Aggregated genotype signal over the same drawn cells.
        spot_alt = lbl_generation.iloc[chosen_indices]['alt_count_sum'].sum()

        synth_rows.append(spot_vec)
        synth_alt_sums.append(spot_alt)

    else:
        # Reached only if every cell type resolved to an empty pool; the
        # composition step otherwise guarantees at least one cell per spot.
        synth_rows.append(np.zeros(cnt_matrix.shape[1]))
        synth_alt_sums.append(0)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

st_cnt_df = pd.DataFrame(synth_rows, index=spot_names, columns=gene_names)
counts_out_name = out_dir + "synthetic_ST_seed" + str(seed) + "_" + str(assemble_id) + "_counts.csv"
st_cnt_df.to_csv(counts_out_name, sep=",", index=True, header=True)
print(f"Saved counts to: {counts_out_name}")

# The "_alt_reads" suffix is deliberate: 06_to_cellranger.py collects expression
# matrices with a "*_counts.csv" glob, which would also match "_alt_counts.csv".
st_alt_df = pd.DataFrame({'spot_id': spot_names, 'alt_count_sum': synth_alt_sums})
alt_out_name = out_dir + "synthetic_ST_seed" + str(seed) + "_" + str(assemble_id) + "_alt_reads.csv"
st_alt_df.to_csv(alt_out_name, sep=",", index=False, header=True)
print(f"Saved alt reads to: {alt_out_name}")

print("Done.")
