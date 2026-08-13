# golden-simulation

Generates synthetic spatial transcriptomics (ST) slides from single-cell RNA-seq
data, with matched cell-type ground truth and a genotype signal for every spot.

## What it is for

Real ST data does not tell you what cells are in each spot. That makes it
unusable both as a training target for a supervised method and as a yardstick
for measuring one. This simulator produces data where the composition of every
spot is known exactly, so a method can be fitted against real labels and then
scored against them.

It is built specifically for methods that combine expression with allelic
information. Two quantities are generated per spot, from the same underlying
cells:

- **cell-type composition** — the ground truth;
- **summed alternate-allele read counts** — a genotype signal, for testing
  whether maternal and fetal origin can be separated when expression alone is
  ambiguous.

Spatial layout is not random. Fetal cells are confined to an invasion zone
anchored on one edge of the slide, their proportion falling from the centre of
that zone towards its perimeter, and they displace maternal cells rather than
adding to them — after the biology of trophoblast invasion. This puts a
realistic boundary region in every slide, where the two origins are genuinely
mixed and inference is hardest.

Each slide holds 4,000 spots, laid out in row-major order on a grid
ceil(sqrt(N)) columns wide — 64 columns for the default 4,000, leaving the last
row partly filled. Spots are filled by sampling real cells from an annotated
scRNA-seq dataset and summing their expression.

Because every spot carries an exact label, the output serves as a supervised
dataset and not only as a test set. In the accompanying manuscript the generated
slides were split into training, validation and test sets, and the model was
fitted on the training portion.

The simulation core is adapted from the `ST_simulation` code accompanying
Andersson et al. The invasion-zone model, the genotype signal, and the
surrounding pipeline are additions.

---

## Requirements

```bash
conda env create -f environment.yml
conda activate golden-simulation
```

Python 3.10 with scanpy, anndata, h5py, numpy, pandas, scipy, matplotlib and
seaborn. Versions are pinned to the minor release; see `environment.yml`.

[VarTrix](https://github.com/10XGenomics/vartrix) is needed only if you want the
genotype signal. It is a standalone binary, run before this pipeline.

---

## Repository layout

```
golden-simulation/
├── bin/          pipeline scripts, run in numbered order
├── vartrix/      genotype pre-processing
└── data/         inputs and outputs (not tracked by git)
```

`bin/` and `data/` must stay siblings: `run_pipeline.sh` locates the project root
from its own path and expects `data/` next to `bin/`.

---

## Preparing inputs

Everything lives under `data/`. One subdirectory per sample, named however you
like; that name is the argument you pass to the pipeline.

```
data/
├── merged_metadata.csv                    required, all samples
├── geneInfo.tab                           included in this repository
└── <SampleName>/
    ├── filtered_feature_bc_matrix.h5      required
    └── barcode_alt_sum.csv                optional, see below
```

### `filtered_feature_bc_matrix.h5`

Cell Ranger output for that sample, unmodified.

### `merged_metadata.csv`

Cell annotations for every sample in one table. In practice this is the `.obs`
frame of a standard scanpy workflow, exported after the samples have been
concatenated and annotated. Only two things about it are load-bearing:

**The index must be `<SAMPLE NAME IN UPPER CASE><10x barcode without the -1
suffix>`.** For a sample directory named `norm_Endo2`, a row index looks like
`NORM_ENDO2AAACCCAAGTTCCGGC`. The pipeline rebuilds this string itself and joins
on it, so a different format produces an empty result — with no error. If your
first run yields an empty metadata file, check this first.

**An `inferredCellOrigin` column, holding `Maternal` or `Fetal`.** These two
labels are hard-coded throughout the simulation; other values are not supported.

Three further columns are copied through to the downsampled metadata if present,
and silently skipped if not: `predict`, `tissueDese`, `sampleDesc`. Any other
columns are ignored.

### `geneInfo.tab` (included)

Tab-separated, no header, three columns: Ensembl ID, gene symbol, gene type. A
copy is included in this repository, so no action is needed unless your data
comes from a different reference.

Used to normalise gene naming: rows whose symbol equals their ID mark genes with
no distinct symbol, and those genes have their names reverted to IDs in the
generated `features.tsv.gz`. This keeps the simulated slides consistent with
whatever annotation your downstream tools were built against.

**Replace it if you ran Cell Ranger with a different reference package.** The
included file reconciles against one specific gene set; using it alongside a
different Cell Ranger reference would rename the wrong genes.

If the file is removed, step 07 is skipped and the pipeline completes normally;
gene names then come straight from the Cell Ranger H5.

### `barcode_alt_sum.csv` (optional)

Per-cell totals of alternate-allele reads, produced from VarTrix output. Without
it the pipeline still runs, and `alt_count_sum` is set to 0 for every cell —
expression simulation is unaffected, but there is no genotype signal to work
with.

Generate it after running VarTrix in `coverage` scoring mode:

```bash
BARCODE_BASE_DIR=/path/to/cellranger/outputs ./vartrix/run_batch.sh
```

`BARCODE_BASE_DIR` holds the Cell Ranger output directories, whose names are
expected to be `<accession>_<SampleName>` — the GEO download layout. The sample
name after the accession must match the `data/` subdirectory name exactly.

`VARTRIX_DIR` defaults to `vartrix/VartrixRunOut/` and holds the matrices, named
`<SampleName>_Alt_coverage.mtx`. Override it if they live elsewhere:

```bash
BARCODE_BASE_DIR=/path/to/cellranger VARTRIX_DIR=/path/to/matrices ./vartrix/run_batch.sh
```

Results are written straight to `data/<SampleName>/barcode_alt_sum.csv`. Samples
whose data directory is missing, whose barcode file is ambiguous, or whose
conversion fails are reported and skipped, and the script exits non-zero — a
skipped sample would otherwise run with no genotype signal and look normal.

---

## Running

One sample per invocation:

```bash
./bin/run_pipeline.sh norm_Endo2
```

For several samples, loop over them:

```bash
for s in norm_Endo2 norm_Endo7 RPL_Endo3; do
    ./bin/run_pipeline.sh "$s"
done
```

Each run produces three slides, from three independently drawn random seeds.

Results from a previous run of the same sample are moved to
`data/<SampleName>/out_archived_<timestamp>/` before the new run starts. Nothing
is deleted, but these accumulate; clear them out when disk space matters.

---

## Outputs

Everything lands in `data/<SampleName>/out/`. Per seed:

| File | Contents |
|---|---|
| `synthetic_ST_seed<seed>_1_composition.csv` | **ground truth** — cell types × spots |
| `synthetic_ST_seed<seed>_1_counts.csv` | expression — spots × genes |
| `synthetic_ST_seed<seed>_1_alt_reads.csv` | genotype — alt read total per spot |
| `cellranger_format_seed<seed>/` | the same expression matrix in 10x format |
| `<SampleName>_plot_seed<seed>.png` | spot map coloured by dominant origin |
| `seed<seed>_spatial_analysis.png` | fetal content beside genotype signal |
| `seed<seed>_fetal_ratio_dist.png` | distribution of fetal fraction |
| `synthetic_ST_seed<seed>_1_alt_reads_sumNorm.csv` | per-spot table with grid coordinates and ratios |

Once per sample:

| File | Contents |
|---|---|
| `simulation_statistics_summary.csv` | per-slide UMI, gene, cell and sparsity statistics |
| `raw_data_qc.png` | single-cell genotype signal, fetal vs maternal |

`cellranger_format_seed<seed>/` contains `matrix.mtx.gz`, `features.tsv.gz`,
`barcodes.tsv.gz` and `tissue_positions.csv`, so tools that read Visium output
can consume the simulated slides directly.

Intermediate files (`labels_generation_*.p`, `counts_generation_*.p`,
`*_design.csv`) also remain in `out/`. The `*_validation_*.p` files are written
empty: this pipeline uses the whole dataset for generation and keeps those files
only for layout compatibility.

---

## Pipeline

`run_pipeline.sh` calls the numbered scripts in order. Each is a standalone
command-line program and can be run on its own.

| Script | Does |
|---|---|
| `01_prepare_input.py` | merges genotype totals into the cell metadata, then downsamples expression to 2,600 UMI per cell |
| `02_build_cell_pool.py` | pairs expression with annotations, filters, draws three seeds |
| `03_draw_design.py` | samples per-cell-type spatial extent and density from gamma distributions |
| `04_place_cells.py` | places cells on the 4,000-spot grid; invasion zone with a linear fetal gradient |
| `05_assemble_spots.py` | sums expression and alt counts over the cells drawn for each spot |
| `06_to_cellranger.py` | writes 10x-format directories with synthetic Visium coordinates |
| `07_fix_gene_symbols.py` | reconciles gene naming against `geneInfo.tab` |
| `08_qc_stats.py` | per-slide summary statistics |
| `09_plot_slides.py` | spot maps of the ground-truth composition |
| `10_validate_genotype.py` | checks that the genotype signal tracks fetal content, at cell and spot level |

Steps 03–05 run once per seed; the rest run once per sample.

Simulation parameters are set in `run_pipeline.sh`: 4,000 spots, a mean of 3
cells per spot, and an invasion zone targeting 45% of spots.

---

## Constraints

**Two cell classes.** The `--annotation_col` argument chooses which metadata
column is read as the cell type, but the values inside that column must be
`Maternal` and `Fetal`: those two strings are hard-coded in `03_draw_design.py`
and `04_place_cells.py`. Renaming the column works; renaming the labels does not.

**4,000 spots.** `06_to_cellranger.py` uses the spot count to decide whether a
matrix needs transposing. Changing `n_spots` in `run_pipeline.sh` requires
editing that script to match.

**`alt_count_sum` is a raw read count.** It is the number of alternate-allele
reads per cell at the original sequencing depth, summed across all queried SNP
loci. It is not normalised by coverage, and it is not an allele fraction.

---

## Reproducibility

The simulation is stochastic by design. Each run draws three fresh random seeds
and samples the design, the invasion zone and the cells within each spot anew, so
two runs on the same input produce different slides. This is intentional — the
seeds exist to measure variability across replicate slides, not to reproduce one
particular slide.

Consequently, re-running this code will not regenerate the exact datasets used in
the accompanying manuscript.
