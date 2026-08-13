#!/bin/bash

# ==============================================================================
# Batch driver for sum_alt_counts.py.
#
# Walks every Vartrix alternate-allele coverage matrix in VARTRIX_DIR, locates
# the barcodes.tsv that was used to run Vartrix for that sample, and collapses
# the matrix into one alt_count_sum per cell barcode.
#
# Results are written straight into the simulation data tree, at the path
# run_pipeline.sh reads from:
#     data/<SampleName>/barcode_alt_sum.csv
#
# The script is written to fail loudly rather than guess. A sample is skipped,
# with a warning, whenever its data folder is missing, its barcode file cannot
# be identified unambiguously, or the conversion itself fails. This matters
# because run_pipeline.sh treats a missing genotype file as "no genotype
# data" and completes normally with alt_count_sum set to 0, which is difficult
# to notice afterwards.
#
# Location: vartrix/run_batch.sh
# Usage:    BARCODE_BASE_DIR=/path/to/cellranger ./run_batch.sh
# ==============================================================================

# Resolved from this script's own location, matching run_pipeline.sh.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# ------------------------------------------------------------------------------
# Site-specific paths. Both may be set in the environment, so the script itself
# does not need editing:
#
#     BARCODE_BASE_DIR=/path/to/cellranger ./run_batch.sh
#
# BARCODE_BASE_DIR is the directory holding the Cell Ranger outputs. Its
# subdirectories are expected to be named <accession>_<SampleName>, matching the
# GEO download layout. There is no default: an unset value stops the run rather
# than silently searching the wrong place.
#
# VARTRIX_DIR holds the Vartrix coverage matrices, named
# <SampleName>_Alt_coverage.mtx. It defaults to VartrixRunOut/ beside this
# script.
# ------------------------------------------------------------------------------
BARCODE_BASE_DIR="${BARCODE_BASE_DIR:-}"
VARTRIX_DIR="${VARTRIX_DIR:-${SCRIPT_DIR}/VartrixRunOut}"

if [ -z "$BARCODE_BASE_DIR" ]; then
    echo "❌ Error: BARCODE_BASE_DIR is not set."
    echo "   Point it at the Cell Ranger output root, for example:"
    echo "       BARCODE_BASE_DIR=/path/to/cellranger ./run_batch.sh"
    exit 1
fi

if [ ! -d "$BARCODE_BASE_DIR" ]; then
    echo "❌ Error: BARCODE_BASE_DIR does not exist: $BARCODE_BASE_DIR"
    exit 1
fi

if [ ! -d "$VARTRIX_DIR" ]; then
    echo "❌ Error: VARTRIX_DIR does not exist: $VARTRIX_DIR"
    echo "   Set VARTRIX_DIR if the Vartrix matrices live elsewhere."
    exit 1
fi

# Glob matching the accession prefix of the Cell Ranger folders; GEO names them
# <GSM accession>_<sample>. Used with ${var#pattern} below, never as a path
# glob. Adjust for a directory layout that is not a GEO download.
ACCESSION_PREFIX="GSM[0-9]*_"


# ------------------------------------------------------------------------------
# Locate the barcodes.tsv.gz belonging to exactly one sample.
#
# Each candidate directory has its accession prefix stripped and the remainder
# compared to the sample name as a plain string. A path glob is deliberately
# avoided here: in "*_Endo2" the wildcard also spans underscores, so the folder
# "GSM111_norm_Endo2" matches sample Endo2 as well, and Endo2 would take
# norm_Endo2's barcodes. Stripping with '#' removes only the shortest matching
# prefix, so "GSM111_norm_Endo2" reduces to "norm_Endo2" and the comparison
# fails as it should.
#
# Prints every match, one per line; the caller decides what to do with a count
# other than one.
# ------------------------------------------------------------------------------
find_barcode_files() {
    local sample="$1"

    shopt -s nullglob
    local dir base remainder barcode
    for dir in "$BARCODE_BASE_DIR"/*/; do
        base=$(basename "$dir")
        remainder="${base#$ACCESSION_PREFIX}"
        [ "$remainder" = "$sample" ] || continue

        barcode="${dir}filtered_feature_bc_matrix/barcodes.tsv.gz"
        [ -f "$barcode" ] && printf '%s\n' "$barcode"
    done
    shopt -u nullglob
}


echo "🚀 Starting batch conversion of Vartrix results..."
echo "   Data directory: $DATA_DIR"

skipped=0
converted=0

for mtx_file in "$VARTRIX_DIR"/*_Alt_coverage.mtx; do

    # Sample name is the filename with the matrix suffix removed.
    filename=$(basename "$mtx_file")
    sample_name=${filename%_Alt_coverage.mtx}

    echo "--------------------------------------------------"
    echo "▶️ Processing: $sample_name"

    # The destination folder must already exist in the data tree; otherwise the
    # sample name does not correspond to a pipeline sample.
    sample_dir="${DATA_DIR}/${sample_name}"
    if [ ! -d "$sample_dir" ]; then
        echo "   ⚠️ Warning: no data folder at $sample_dir, skipping."
        echo "      (Vartrix matrix names must match the folder names under data/)"
        skipped=$((skipped + 1))
        continue
    fi

    # Collected into an array so that an ambiguous result is reported rather
    # than silently resolved to whichever entry happens to come first.
    barcode_matches=()
    while IFS= read -r line; do
        barcode_matches+=( "$line" )
    done < <(find_barcode_files "$sample_name")

    if [ ${#barcode_matches[@]} -eq 0 ]; then
        echo "   ⚠️ Warning: no barcode file found for $sample_name, skipping."
        echo "      Looked under: $BARCODE_BASE_DIR/<accession>_${sample_name}/"
        skipped=$((skipped + 1))
        continue
    fi

    if [ ${#barcode_matches[@]} -gt 1 ]; then
        echo "   ❌ Error: ${#barcode_matches[@]} barcode files match $sample_name, skipping."
        echo "      Assigning the wrong one would silently misattribute variants."
        printf '      - %s\n' "${barcode_matches[@]}"
        skipped=$((skipped + 1))
        continue
    fi

    barcode_file="${barcode_matches[0]}"
    echo "   🔍 Barcode file: $barcode_file"

    out_csv="${sample_dir}/barcode_alt_sum.csv"

    # Exit status is checked so that a failed conversion is counted rather than
    # passed over. The loop continues, so one bad sample does not stop the batch.
    if python3 "${SCRIPT_DIR}/sum_alt_counts.py" \
        --mtx "$mtx_file" \
        --barcodes "$barcode_file" \
        --output "$out_csv"; then
        echo "   ✅ Done: $out_csv"
        converted=$((converted + 1))
    else
        echo "   ❌ Error: sum_alt_counts.py failed for $sample_name."
        skipped=$((skipped + 1))
    fi

done

echo "--------------------------------------------------"
echo "📂 Converted $converted sample(s) into $DATA_DIR/<SampleName>/barcode_alt_sum.csv"

if [ "$skipped" -gt 0 ]; then
    echo "⚠️ $skipped sample(s) were skipped. Review the warnings above before"
    echo "   running the pipeline: a skipped sample runs with alt_count_sum = 0."
    exit 1
fi

echo "🎉 All samples processed."
