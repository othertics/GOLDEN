#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert simulated count matrices into 10x Cell Ranger directories.

Pipeline position
    Script 06 of ``run_pipeline.sh``, run once per sample over every slide in
    the output directory.

What it does
    Finds each ``*_counts.csv`` produced by ``05_assemble_spots.py`` and writes it
    out in the format downstream deconvolution tools expect: a gzipped Matrix
    Market file plus feature and barcode tables. Gene symbols are recovered from
    the sample's original Cell Ranger H5, since the simulated matrices are keyed
    on Ensembl IDs. A synthetic ``tissue_positions.csv`` is generated so the
    slides carry Visium-style coordinates.

Inputs
    --dir/*_counts.csv                        spot x gene matrices
    <parent of --dir>/filtered_feature_bc_matrix.h5   Ensembl ID to symbol map

Output (one directory per slide, inside --dir)
    cellranger_format_seed<seed>/
        matrix.mtx.gz      genes x spots, integer, CSC
        features.tsv.gz    id, name, type
        barcodes.tsv.gz    spot barcodes with a -1 suffix
        tissue_positions.csv
"""

import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os
import glob
import re
import gzip
import shutil
import h5py
import argparse


def load_gene_map_from_h5(h5_path):
    """Build an Ensembl ID to gene symbol mapping from a Cell Ranger H5."""
    print(f"   [Ref] Loading gene map from: {os.path.basename(h5_path)}")
    id_to_symbol = {}

    try:
        with h5py.File(h5_path, 'r') as f:
            # Cell Ranger layout: matrix/features/id and matrix/features/name.
            if 'matrix' in f and 'features' in f['matrix']:
                feature_group = f['matrix']['features']
                ids = feature_group['id'][:]
                names = feature_group['name'][:]

                for gene_id, gene_name in zip(ids, names):
                    # HDF5 returns bytes for string datasets.
                    if isinstance(gene_id, bytes): gene_id = gene_id.decode('utf-8')
                    if isinstance(gene_name, bytes): gene_name = gene_name.decode('utf-8')
                    id_to_symbol[gene_id] = gene_name
            else:
                print("   ❌ Error: H5 structure is not standard Cell Ranger format.")
                return {}
    except Exception as e:
        print(f"   ❌ Failed to read H5: {e}")
        return {}

    return id_to_symbol


def save_10x_mtx(counts_df, output_dir, id_to_symbol):
    """Write a spot x gene frame as a 10x Cell Ranger triplet of files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- matrix.mtx.gz ---
    matrix_path = os.path.join(output_dir, "matrix.mtx")

    # Transposed to genes x spots and cast to int, matching the Cell Ranger
    # convention. CSC keeps the file column-ordered by spot.
    sparse_mat = scipy.sparse.csc_matrix(counts_df.values.T.astype(int))
    scipy.io.mmwrite(matrix_path, sparse_mat)

    # mmwrite labels integer matrices as "real"; some readers reject that.
    with open(matrix_path, 'r') as f:
        lines = f.readlines()
    if "real" in lines[0]:
        lines[0] = lines[0].replace("real", "integer")
        with open(matrix_path, 'w') as f:
            f.writelines(lines)

    with open(matrix_path, 'rb') as f_in:
        with gzip.open(matrix_path + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(matrix_path)

    # --- features.tsv.gz ---
    features_path = os.path.join(output_dir, "features.tsv.gz")
    genes_ids = counts_df.columns.tolist()

    # Genes missing from the reference keep their Ensembl ID as the name.
    gene_symbols = [id_to_symbol.get(g, g) for g in genes_ids]

    features_df = pd.DataFrame({
        'id': genes_ids,
        'name': gene_symbols,
        'type': 'Gene Expression'
    })
    features_df.to_csv(features_path, sep='\t', header=False, index=False, compression='gzip')

    # --- barcodes.tsv.gz ---
    barcodes_path = os.path.join(output_dir, "barcodes.tsv.gz")
    barcodes = counts_df.index.tolist()
    barcodes_formatted = [f"{b}-1" if not str(b).endswith("-1") else str(b) for b in barcodes]

    with gzip.open(barcodes_path, 'wt') as f:
        for b in barcodes_formatted:
            f.write(f"{b}\n")

    print(f"   -> Saved 10x files to: {os.path.basename(output_dir)}")
    return barcodes_formatted


def create_tissue_positions_file(barcodes, output_dir):
    """Lay the spots out on a square grid and write Visium-style coordinates."""
    n_spots = len(barcodes)
    grid_size = int(np.ceil(np.sqrt(n_spots)))

    # Row-major order, matching how 04_place_cells.py placed the spots.
    positions_df = pd.DataFrame(index=range(n_spots))
    positions_df['barcode'] = barcodes
    positions_df['in_tissue'] = 1
    positions_df['array_row'] = [i // grid_size for i in range(n_spots)]
    positions_df['array_col'] = [i % grid_size for i in range(n_spots)]

    # Arbitrary pixel scale; only the relative layout matters for plotting.
    positions_df['pxl_row_in_fullres'] = positions_df['array_row'] * 100
    positions_df['pxl_col_in_fullres'] = positions_df['array_col'] * 100

    output_path = os.path.join(output_dir, "tissue_positions.csv")
    positions_df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Convert simulation CSVs to Cell Ranger format.")
    parser.add_argument('--dir', required=True, help="Target directory containing *_counts.csv (e.g., .../downsampled_v2)")
    args = parser.parse_args()

    current_dir = args.dir
    print(f"Scanning for counts CSVs in: {os.path.abspath(current_dir)}")

    # Only expression matrices are picked up here. The genotype file is named
    # "_alt_reads.csv" by 05_assemble_spots.py so that this glob skips it.
    count_files = glob.glob(os.path.join(current_dir, "*_counts.csv"))

    if not count_files:
        print("No *_counts.csv files found.")
        return

    # The reference H5 sits one level up, in data/<SampleName>/.
    parent_dir = os.path.dirname(os.path.abspath(current_dir))
    ref_h5_path = os.path.join(parent_dir, "filtered_feature_bc_matrix.h5")

    if os.path.exists(ref_h5_path):
        id_to_symbol = load_gene_map_from_h5(ref_h5_path)
    else:
        print(f"⚠️ Critical Warning: Reference H5 not found at {ref_h5_path}")
        print("   Proceeding without symbol mapping (IDs will be used as names).")
        id_to_symbol = {}

    print(f"Found {len(count_files)} CSV file(s). Processing...\n")

    for input_csv in count_files:
        filename = os.path.basename(input_csv)
        print(f"Processing: {filename}")

        try:
            counts_df = pd.read_csv(input_csv, index_col=0)
        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            continue

        # Output folder is named after the seed embedded in the filename.
        match = re.search(r"seed(\d+)", filename)
        if match:
            folder_name = f"cellranger_format_seed{match.group(1)}"
        else:
            folder_name = f"cellranger_format_{filename.replace('.csv', '').replace('_counts', '')}"

        output_dir = os.path.join(current_dir, folder_name)

        # Slides carry 4,000 spots, so a frame with a different row count is
        # gene-major and has to be flipped to spots x genes.
        if counts_df.shape[0] != 4000:
             counts_df = counts_df.T

        barcodes = save_10x_mtx(counts_df, output_dir, id_to_symbol)
        create_tissue_positions_file(barcodes, output_dir)
        print("-" * 50)

    print("\n✅ All conversions completed successfully.")


if __name__ == "__main__":
    main()
