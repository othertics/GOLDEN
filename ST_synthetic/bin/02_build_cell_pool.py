### SPLIT SINGLE-CELL DATASET IN GENERATION AND VALIDATION SET ###

"""Build the single-cell pool that spots are drawn from.

Pipeline position
    Script 02 of ``run_pipeline.sh``, run once per sample, before
    ``03_draw_design.py``.

What it does
    Pairs the expression matrix with its annotations, keeps only cells that
    carry both, drops cell types with too few cells to sample from, and writes
    the result as the "generation" set.

    This is the no-split variant of the upstream script: the whole dataset is
    used for generation and the validation set is written as an empty frame.
    It is kept for file-layout compatibility with the downstream scripts, which
    still expect all four pickles.

    Three seeds are drawn per run. Each produces an identical copy of the
    generation set; the seed value is what distinguishes the slides assembled
    from it in the later steps.

Inputs
    h5ad_file           <sample>_downsampled.h5ad, from 01_prepare_input.py
    annotation_file     <sample>_downsampled_metadata.csv, from 01_prepare_input.py

Outputs (one set per seed, in --out_dir)
    labels_generation_<seed>.p      annotation column + alt_count_sum
    counts_generation_<seed>.p      cell x gene expression
    labels_validation_<seed>.p      empty
    counts_validation_<seed>.p      empty
"""

import argparse
import pickle
import random
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument('h5ad_file', type=str, help='path to h5ad file')
parser.add_argument('annotation_file', type=str, help='path to csv file')
parser.add_argument('--annotation_col', dest='anno_col', type=str, default="annotation_1")
parser.add_argument('--out_dir', dest='out_dir', type=str, default=".")
args = parser.parse_args()

adata_file = args.h5ad_file
annotation_file = args.annotation_file
anno_col = args.anno_col
out_dir = args.out_dir

### Load input single-cell data and annotations ###
print(f"Loading H5AD: {adata_file}")
adata_raw = sc.read_h5ad(adata_file)

# Annotations come from the CSV, but alt_count_sum lives in the H5AD written by
# step1. Both are indexed by the prefixed barcode, so the column is copied
# across by index alignment.
labels_from_csv = pd.read_csv(annotation_file, index_col=0)

obs_from_h5ad = adata_raw.obs

if 'alt_count_sum' in obs_from_h5ad.columns:
    print("Found 'alt_count_sum' in H5AD. Preserving it.")
    labels_from_csv['alt_count_sum'] = obs_from_h5ad['alt_count_sum']
else:
    print("⚠️ Warning: 'alt_count_sum' NOT found in H5AD.")
    labels_from_csv['alt_count_sum'] = 0

labels = labels_from_csv

# Densify and orient as (cell x gene); 05_assemble_spots.py expects this layout.
if sp.issparse(adata_raw.X):
    X_array = adata_raw.X.T.toarray()
else:
    X_array = adata_raw.X.T

adata_df = pd.DataFrame(X_array, columns=adata_raw.obs_names, index=adata_raw.var_names)
adata_df = adata_df.T
adata_df.index.name = "cell"

### Subset to cells with label ###
# Both frames are reindexed to the same cell order here. 05_assemble_spots.py
# relies on that shared order when it indexes labels by matrix position.
common_cells = adata_df.index.intersection(labels.index)
print(f"matched total labeled cells: {len(common_cells)} / h5ad: {len(adata_df.index)} / labels: {len(labels.index)}")

adata_df = adata_df.loc[common_cells, :]
labels = labels.loc[common_cells, :]

### Split generation and validation set ###
sc_cnt = adata_df

# Carry alt_count_sum alongside the annotation column so the genotype signal
# survives into the generation pickle.
cols_to_keep = [anno_col]
if 'alt_count_sum' in labels.columns:
    cols_to_keep.append('alt_count_sum')

sc_lbl = labels[cols_to_keep].copy()

# Drop unannotated cells from both frames in the same operation order.
valid_mask = sc_lbl[anno_col].notna()
print(f"Valid annotations: {valid_mask.sum()} / {len(valid_mask)}")

sc_lbl = sc_lbl[valid_mask]
sc_cnt = sc_cnt[valid_mask]

# Drop cell types with too few cells for repeated sampling to be meaningful.
labels_series = sc_lbl[anno_col].values
uni_labs, uni_counts = np.unique(labels_series, return_counts=True)

keep_types = uni_counts > 40
keep_cells = np.isin(labels_series, uni_labs[keep_types])

labels_series = labels_series[keep_cells]
sc_cnt = sc_cnt.iloc[keep_cells, :]
sc_lbl = sc_lbl.iloc[keep_cells, :]

n_types = uni_labs.shape[0]

seeds = random.sample(range(1000), 3)

for seed in seeds:
    random.seed(seed)
    print("Seed " + str(seed) + " (Using 100% Data - No Split)")

    # No split: the generation set is the full filtered dataset, so no shuffling
    # or index selection is needed.
    cnt_generation = sc_cnt
    lbl_generation = sc_lbl

    # Written empty, but written, because downstream tooling expects the files.
    cnt_validation = pd.DataFrame()
    lbl_validation = pd.DataFrame()

    pickle.dump(lbl_generation, open(out_dir + "labels_generation_" + str(seed) + ".p", "wb"))
    pickle.dump(cnt_generation, open(out_dir + "counts_generation_" + str(seed) + ".p", "wb"))
    pickle.dump(lbl_validation, open(out_dir + "labels_validation_" + str(seed) + ".p", "wb"))
    pickle.dump(cnt_validation, open(out_dir + "counts_validation_" + str(seed) + ".p", "wb"))

print("Processing complete with 100% data usage (Alt Count Preserved).")
