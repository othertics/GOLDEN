#!/bin/bash

# ==============================================================================
# End-to-end driver for the synthetic ST simulation pipeline.
#
# Runs every stage for a single sample: input preparation and genotype merge,
# per-seed slide simulation, conversion to Cell Ranger format, QC, and
# validation reporting.
#
# Location: bin/run_pipeline.sh
# Usage:    ./bin/run_pipeline.sh <SampleName>
#
# Expected layout, relative to the repository root:
#   data/merged_metadata.csv                     cell annotations, all samples
#   data/geneInfo.tab                            gene symbol reference
#   data/<SampleName>/filtered_feature_bc_matrix.h5
#   data/<SampleName>/barcode_alt_sum.csv        from vartrix/run_batch.sh
#
# All results are written to data/<SampleName>/out/. Results from a previous
# run of the same sample are moved to data/<SampleName>/out_archived_<date>/
# before this run begins.
# ==============================================================================

# Abort on the first failing command.
set -e

SAMPLE_NAME=$1

if [ -z "$SAMPLE_NAME" ]; then
  echo "❌ Error: Please provide a sample name."
  echo "Usage: ./run_pipeline.sh <SampleName>"
  exit 1
fi

# Paths are derived from this script's own location, so the pipeline can be
# invoked from any working directory.
BIN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$BIN_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Shared reference files
METADATA_FILE="${DATA_DIR}/merged_metadata.csv"
GENE_INFO_FILE="${DATA_DIR}/geneInfo.tab"

# Per-sample paths
SAMPLE_DIR="${DATA_DIR}/${SAMPLE_NAME}"
OUT_DIR="${SAMPLE_DIR}/out/"

# Per-barcode alternate-allele totals. vartrix/run_batch.sh writes
# this file directly to the path below, so no copying is needed; it only has to
# have been run before this pipeline.
VARTRIX_FILE="${SAMPLE_DIR}/barcode_alt_sum.csv"

echo "=================================================================="
echo " 🚀 Pipeline Started for: $SAMPLE_NAME"
echo "    Bin Dir      : $BIN_DIR"
echo "    Data Dir     : $DATA_DIR"
echo "    Vartrix File : $VARTRIX_FILE"
echo "=================================================================="

# ------------------------------------------------------------------
# [01] Downsampling and genotype merge
# ------------------------------------------------------------------
echo ""
echo ">>> [01] Running Downsampling & Merging Vartrix Data..."

if [ ! -f "$VARTRIX_FILE" ]; then
    echo "⚠️  Warning: Vartrix file not found at $VARTRIX_FILE"
    echo "             'alt_count_sum' will be set to 0 for all cells."
fi

python3 "${BIN_DIR}/01_prepare_input.py" \
    --data_dir "$DATA_DIR" \
    --metadata "$METADATA_FILE" \
    --sample "$SAMPLE_NAME" \
    --vartrix_csv "$VARTRIX_FILE"

# ------------------------------------------------------------------
# [02-05] Slide simulation
# ------------------------------------------------------------------
echo ""
echo ">>> [02-05] Running Simulation..."

# Outputs of 01, used as the single-cell pool.
MYscRNA="${SAMPLE_DIR}/${SAMPLE_NAME}_downsampled.h5ad"
MYlabel="${SAMPLE_DIR}/${SAMPLE_NAME}_downsampled_metadata.csv"

# Every step from here on discovers its inputs by globbing OUT_DIR, so results
# left over from an earlier run would be picked up and mixed into this one.
# Anything already present is moved aside rather than deleted, so nothing is
# lost; the archived folder is not named "out" and so is ignored by later steps.
if [ -d "${OUT_DIR%/}" ] && [ -n "$(ls -A "${OUT_DIR%/}" 2>/dev/null)" ]; then
    ARCHIVE_DIR="${SAMPLE_DIR}/out_archived_$(date +%Y%m%d_%H%M%S)"
    mv "${OUT_DIR%/}" "$ARCHIVE_DIR"
    echo "    Previous results moved to: $(basename "$ARCHIVE_DIR")"
fi

mkdir -p "$OUT_DIR"

# 02. Build the generation set. No-split mode: all cells are used.
python3 "${BIN_DIR}/02_build_cell_pool.py" "$MYscRNA" "$MYlabel" \
    --annotation_col inferredCellOrigin \
    --out_dir "$OUT_DIR"

# 03-05. Recover the seeds that split_sc just wrote, then build one slide each.
n_spots=4000
seeds=$(ls "${OUT_DIR}"labels_generation* | sed 's/.*_//' | sed 's/.p//')

if [ -z "$seeds" ]; then
    echo "❌ Error: Seeds not found. Check split_sc script output."
    exit 1
fi

for seed in $seeds; do
    echo "    Processing Seed: $seed"
    
    # Design: per-cell-type extent and density, drawn from gamma distributions.
    python3 "${BIN_DIR}/03_draw_design.py" "$seed" \
        --annotation_col inferredCellOrigin \
        --tot_spots $n_spots \
        --mean_high 3 --mean_low 1 \
        --percent_sparse 45 \
        --out_dir "$OUT_DIR"

    # Composition: place cells on the grid via the displacement + gradient model.
    python3 "${BIN_DIR}/04_place_cells.py" "$seed" \
        --annotation_col inferredCellOrigin \
        --out_dir "$OUT_DIR" \
        --tot_spots $n_spots \
        --assemble_id 1

    # Assembly: sum UMI and alt_count_sum over the cells drawn for each spot.
    python3 "${BIN_DIR}/05_assemble_spots.py" "$seed" \
        --annotation_col inferredCellOrigin \
        --out_dir "$OUT_DIR" \
        --assemble_id 1
done

# ------------------------------------------------------------------
# [06] Convert to Cell Ranger format
# ------------------------------------------------------------------
echo ""
echo ">>> [06] Converting Formats..."
python3 "${BIN_DIR}/06_to_cellranger.py" --dir "$OUT_DIR"

# ------------------------------------------------------------------
# [07] Reconcile gene symbols with the reference
# ------------------------------------------------------------------
echo ""
echo ">>> [07] Fixing Gene Symbols..."
python3 "${BIN_DIR}/07_fix_gene_symbols.py" --dir "$OUT_DIR" --ref "$GENE_INFO_FILE"

# ------------------------------------------------------------------
# [08] QC statistics
# ------------------------------------------------------------------
echo ""
echo ">>> [08] Quality Control..."
python3 "${BIN_DIR}/08_qc_stats.py" --dir "$OUT_DIR"

# ------------------------------------------------------------------
# [09] Visualization
# ------------------------------------------------------------------
echo ""
echo ">>> [09] Visualizing Results..."
python3 "${BIN_DIR}/09_plot_slides.py" --data_dir "$DATA_DIR" --sample "$SAMPLE_NAME"

echo ""
echo "=================================================================="
echo " 🎉 All Steps Completed Successfully for $SAMPLE_NAME"
echo "=================================================================="

# ------------------------------------------------------------------
# [10] Validation and reporting
# ------------------------------------------------------------------
echo ""
echo ">>> [10] Running Final Validation & Reporting..."
python3 "${BIN_DIR}/10_validate_genotype.py" \
    --data_dir "$DATA_DIR" \
    --sample "$SAMPLE_NAME"

echo ""
echo "=================================================================="
echo " 🎉 All Steps & Validation Completed Successfully for $SAMPLE_NAME"
echo "=================================================================="
