#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Reconcile gene names in the generated features tables with the reference.

Pipeline position
    Script 07 of ``run_pipeline.sh``, run once per sample over every Cell Ranger
    directory written by ``06_to_cellranger.py``.

What it does
    Some genes have no usable symbol, so the reference annotation stores the
    Ensembl ID in the name field as well. Step 3 resolves names from the sample's
    own H5, which may disagree. This step finds every gene whose reference entry
    uses ID-as-name and rewrites the name column of each ``features.tsv.gz`` to
    match, keeping the generated slides consistent with the annotation the
    downstream tools were built against.

Inputs
    --dir       directory searched recursively for features.tsv.gz
    --ref       geneInfo.tab, tab-separated, no header: id, name, type

Effect
    Every matching features.tsv.gz is rewritten in place, still gzipped.
"""

import os
import pandas as pd
import glob
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description="Fix gene symbols in features.tsv using geneInfo reference.")
    parser.add_argument('--dir', required=True, help="Target directory containing 10x output folders")
    parser.add_argument('--ref', required=True, help="Path to geneInfo.tab reference file")

    args = parser.parse_args()

    BASE_DIR = args.dir
    GENE_INFO_PATH = args.ref

    print("========================================================")
    print("   [07] Features Compatibility Fix Started")
    print(f"   Target Dir: {BASE_DIR}")
    print(f"   Reference : {GENE_INFO_PATH}")
    print("========================================================")

    if not os.path.exists(GENE_INFO_PATH):
        print(f"❌ Error: geneInfo.tab not found at {GENE_INFO_PATH}")
        return

    print(f"Loading Reference: {os.path.basename(GENE_INFO_PATH)}...")

    try:
        ref_df = pd.read_csv(GENE_INFO_PATH, sep='\t', header=None, names=['id', 'name', 'type'])
    except Exception as e:
        print(f"❌ Error reading geneInfo.tab: {e}")
        return

    # A reference row whose name equals its ID marks a gene with no distinct
    # symbol. Those are the genes whose names have to be reverted to IDs.
    ids_to_convert_set = set(ref_df[ref_df['id'] == ref_df['name']]['id'])

    print(f" -> Reference Check: Total {len(ref_df)} genes.")
    print(f" -> Target Genes: {len(ids_to_convert_set)} genes use ID as Name.")
    print("    (These symbols will be reverted to IDs in features.tsv)")

    print(f"\nScanning for features.tsv.gz in: {BASE_DIR}")
    # Recursive: one features.tsv.gz per cellranger_format_seed* directory.
    target_files = glob.glob(os.path.join(BASE_DIR, '**', 'features.tsv.gz'), recursive=True)

    if not target_files:
        print("⚠️ No features.tsv.gz files found. Check your directory path.")
        return

    print(f" -> Found {len(target_files)} files. Processing...\n")

    for file_path in target_files:
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'name', 'type'], compression='gzip')

            mask = df['id'].isin(ids_to_convert_set)
            changed_count = mask.sum()

            if changed_count > 0:
                df.loc[mask, 'name'] = df.loc[mask, 'id']

                # Rewritten in place, preserving gzip compression.
                df.to_csv(file_path, sep='\t', header=False, index=False, compression='gzip')

                parent_folder = os.path.basename(os.path.dirname(file_path))
                print(f"   [Fixed] {parent_folder} : {changed_count} genes updated.")
            else:
                parent_folder = os.path.basename(os.path.dirname(file_path))
                print(f"   [Skip]  {parent_folder} : No changes needed.")

        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")

    print("\n✅ All features.tsv files have been synchronized with geneInfo.tab!")


if __name__ == "__main__":
    main()
