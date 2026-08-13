### Make design of simulated ST datasets from single-cell data

"""Draw the per-cell-type design parameters for one simulated slide.

Pipeline position
    Script 03 of ``run_pipeline.sh``, run once per seed, after
    ``02_build_cell_pool.py`` and before ``04_place_cells.py``.

What it does
    Classifies each cell type along two axes and then samples its spatial
    extent and density from gamma distributions:

    uniform
        1 = present across the whole slide, 0 = confined to a subset of spots.
        Maternal cells are uniform, fetal cells are regional.

    density
        0 = high density, 1 = low density. Both cell types are set to high
        density here, so ``--mean_low`` has no effect on the output.

    ``nspots`` is then drawn with mean ``--percent_uniform`` or
    ``--percent_sparse`` percent of the slide, and ``mean_ncells`` with mean
    ``--mean_high`` or ``--mean_low``. Sampling the design per seed is what
    makes cell density vary between slides.

Inputs (resolved from --out_dir, not passed explicitly)
    labels_generation_<seed>.p     cell annotations, from 02_build_cell_pool.py
    counts_generation_<seed>.p     cell x gene expression, from 02_build_cell_pool.py

Output
    synthetic_ST_seed<seed>_design.csv
        One row per cell type, columns: uniform, density, nspots, mean_ncells.
"""

import argparse
import pickle
import numpy as np 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int,
                    help='random seed of split')
parser.add_argument('--tot_spots', dest='tot_spots', type=int,
                    default=1000,
                    help='Total number of spots to simulate')
parser.add_argument('--mean_high', dest='mean_high', type=float,
                    default=2.5,
                    help='Mean cell density for high-density cell types')
parser.add_argument('--mean_low', dest='mean_low', type=float,
                    default=0.8,
                    help='Mean cell density for low-density cell types')
parser.add_argument('--percent_uniform', dest='percent_uniform', type=float,
                    default=100,
                    help='Sparsity of uniform cell types (% non-zero spots of total spots)')
parser.add_argument('--percent_sparse', dest='percent_sparse', type=float,
                    default=20,
                    help='Sparsity of sparse cell types (% non-zero spots of total spots)')
parser.add_argument('--annotation_col', dest='anno_col', type=str,
                    default="annotation_1",
                    help='Name of column to use in annotation file (default: annotation_1)')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='./',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int,
                    default=1,
                    help='ID of ST assembly')

args = parser.parse_args()

seed = args.seed
tot_spots = args.tot_spots
mean_high = args.mean_high
mean_low = args.mean_low
percent_uniform = args.percent_uniform
percent_sparse = args.percent_sparse
out_dir = args.out_dir
assemble_id = args.assemble_id
anno_col = args.anno_col

### Load input data ### 
lbl_gen_file = out_dir + "labels_generation_" + str(seed) + ".p"
count_gen_file = out_dir + "counts_generation_" + str(seed) + ".p"

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
cnt_generation = pickle.load(open(count_gen_file, "rb"))

uni_labels = lbl_generation[anno_col].unique()
labels = lbl_generation
cnt = cnt_generation

### Define uniform VS sparse cell types (w more sparse = 0)
# Maternal cells form the tissue-wide background; fetal cells are regional,
# confined to the invasion zone drawn by 04_place_cells.py.
design_map = {
    'Maternal': 1, 
    'Fetal': 0
}
uniform_ct = np.array([design_map.get(label, 0) for label in uni_labels])

#### Define low VS high density cell types (w more low density = 1)

design_df = pd.DataFrame({'uniform': uniform_ct}, index=uni_labels)

design_df['density'] = np.nan

# Both cell types are pinned to high density, so mean_ncells is always drawn
# from the mean_high distribution below.
print("Forcing density to HIGH (0) for all cell types.")
if 'Maternal' in design_df.index:
    design_df.loc['Maternal', 'density'] = 0 # 0 = High density
if 'Fetal' in design_df.index:
    design_df.loc['Fetal', 'density'] = 0 # 0 = High density


### Generate no of spots per cell type 
# Gamma parameters are derived from a target mean and a variance fixed at
# mean/0.3, then converted to the shape/scale form numpy expects.
mean_unif = round((tot_spots / 100) * percent_uniform)
mean_sparse = round((tot_spots / 100) * percent_sparse)
sigma_unif = np.sqrt(mean_unif / 0.3)
sigma_sparse = np.sqrt(mean_sparse / 0.3)

shape_unif = mean_unif ** 2 / sigma_unif ** 2
scale_unif = sigma_unif ** 2 / mean_unif
shape_sparse = mean_sparse ** 2 / sigma_sparse ** 2
scale_sparse = sigma_sparse ** 2 / mean_sparse

unif_nspots = np.round(np.random.gamma(shape=shape_unif, scale=scale_unif, size=sum(design_df.uniform == 1)))
sparse_nspots = np.round(np.random.gamma(shape=shape_sparse, scale=scale_sparse, size=sum(design_df.uniform == 0)))
# if samples n spots is greater than total number of spots trim to the total
if (unif_nspots > tot_spots).sum() >= 1:
    unif_nspots[unif_nspots > tot_spots] = tot_spots
if (sparse_nspots > tot_spots).sum() >= 1:
    sparse_nspots[sparse_nspots > tot_spots] = tot_spots


design_df['nspots'] = np.nan
design_df.loc[design_df.index[design_df.uniform == 1], 'nspots'] = unif_nspots
design_df.loc[design_df.index[design_df.uniform == 0], 'nspots'] = sparse_nspots

### Generate avg density per spot per cell type
# Same construction as above, with variance fixed at mean/2. This mean is the
# lambda that 04_place_cells.py feeds to its Poisson draw.
sigma_low = np.sqrt(mean_low / 2)
sigma_high = np.sqrt(mean_high / 2)

shape_low = mean_low ** 2 / sigma_low ** 2
scale_low = sigma_low ** 2 / mean_low
shape_high = mean_high ** 2 / sigma_high ** 2
scale_high = sigma_high ** 2 / mean_high

low_ncells_mean = np.random.gamma(shape=shape_low, scale=scale_low, size=sum(design_df.density == 1))
high_ncells_mean = np.random.gamma(shape=shape_high, scale=scale_high, size=sum(design_df.density == 0))

design_df['mean_ncells'] = np.nan
design_df.loc[design_df.index[design_df.density == 1], 'mean_ncells'] = low_ncells_mean
design_df.loc[design_df.index[design_df.density == 0], 'mean_ncells'] = high_ncells_mean

# Filename carries only the seed: the design is shared by every assembly made
# from it, so assemble_id is not part of the name.
out_name = out_dir + "synthetic_ST_seed" + lbl_gen_file.split("_")[-1].rstrip(".p") + "_" + "design" + ".csv"
design_df.to_csv(out_name, sep=",", index=True, header=True)
