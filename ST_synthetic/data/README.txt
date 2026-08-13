Input data goes here. Nothing in this directory is tracked by git.

    data/
    ├── merged_metadata.csv                     cell annotations, all samples
    ├── geneInfo.tab                            optional
    └── <SampleName>/
        ├── filtered_feature_bc_matrix.h5       Cell Ranger output
        └── barcode_alt_sum.csv                 from vartrix/run_batch.sh

<SampleName> is the argument you pass to bin/run_pipeline.sh, and it must match
the sample prefix used in the merged_metadata.csv index. Results are written to
data/<SampleName>/out/; a previous run's results are moved aside to
data/<SampleName>/out_archived_<timestamp>/ before a new run starts.

See the repository README for the format each input file has to follow — in
particular the index format expected in merged_metadata.csv, which produces an
empty result rather than an error when it does not match.
