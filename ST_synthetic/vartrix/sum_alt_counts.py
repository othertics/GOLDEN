#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collapse a Vartrix coverage matrix into one alternate-allele total per cell.

Pipeline position
    Runs before the simulation pipeline, once per sample, driven by
    ``run_batch.sh``. Its output feeds the ``--vartrix_csv``
    argument of ``01_prepare_input.py``.

What it does
    Vartrix emits a sparse (variant x barcode) matrix. This script maps each
    stored entry back to its barcode and sums across all queried SNP loci,
    giving a single ``alt_count_sum`` per cell.

    Because the matrix is sparse, barcodes with no alternate reads are absent
    from the output rather than listed as 0; step1 fills them in during its
    left join.

Inputs
    --mtx           Vartrix alternate-allele coverage matrix (.mtx)
    --barcodes      the barcodes.tsv passed to Vartrix, in the same column order

Output
    --output        tab-separated, columns: barcode, alt_count_sum
"""

import pandas as pd
import scipy.io
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate barcode_alt_sum.csv from Vartrix outputs")
    parser.add_argument('--mtx', required=True, help="Path to Vartrix .mtx file (e.g., Alt_Frac_counts.mtx)")
    parser.add_argument('--barcodes', required=True, help="Path to barcodes.tsv used for Vartrix")
    parser.add_argument('--output', required=True, help="Path to save the output CSV")
    
    args = parser.parse_args()
    
    mtx_path = args.mtx
    barcodes_path = args.barcodes
    out_path = args.output
    
    print(f"Processing...")
    print(f"  - MTX: {mtx_path}")
    print(f"  - Barcodes: {barcodes_path}")

    if not os.path.exists(mtx_path):
        print(f"❌ Error: MTX file not found.")
        sys.exit(1)
    if not os.path.exists(barcodes_path):
        print(f"❌ Error: Barcodes file not found.")
        sys.exit(1)

    try:
        # COO form exposes .col (barcode index) and .data (count) directly.
        # mmread returns 0-based indices.
        mtx = scipy.io.mmread(mtx_path).tocoo()
        
        # Written without a header by Cell Ranger; only the first column is used.
        barcodes_df = pd.read_csv(barcodes_path, header=None, sep='\t')
        barcodes = barcodes_df[0].values
        
        # Guard against a barcodes file that does not belong to this matrix.
        # Vartrix creates one column per barcode, so the declared matrix width
        # must equal the barcode count exactly. shape[1] is compared rather
        # than col.max(): the latter is the highest index carrying a stored
        # value, which is legitimately smaller whenever the trailing columns
        # are all zero, and it cannot detect a barcode file that is too long.
        if mtx.shape[1] != len(barcodes):
            print(f"❌ Error: Matrix has {mtx.shape[1]} columns but the barcode list has {len(barcodes)} entries.")
            print("   Please check if the correct barcodes.tsv is used.")
            sys.exit(1)

        # Replace column indices with barcode strings, one row per stored entry.
        df_temp = pd.DataFrame({
            "barcode": barcodes[mtx.col],
            "alt_count": mtx.data
        })
        
        # One cell can carry alternate reads at several loci; sum across them.
        print("  - Aggregating counts by barcode...")
        barcode_alt_sum = (
            df_temp
            .groupby("barcode", as_index=False)["alt_count"]
            .sum()
            .rename(columns={"alt_count": "alt_count_sum"})
        )
        
        # Tab-separated regardless of the .csv extension, matching how
        # 01_prepare_input.py reads the file.
        barcode_alt_sum.to_csv(out_path, sep='\t', index=False)
        
        print(f"✅ Done! Saved to: {out_path}")
        print(f"   (Total cells with Alt counts: {len(barcode_alt_sum)})")
        print(f"   (Max Alt Sum: {barcode_alt_sum['alt_count_sum'].max()})")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
