#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Assign a per-spot cell type composition for one simulated ST slide.

Pipeline position
    Script 03 of ``run_pipeline.sh``, run once per seed, after
    ``03_draw_design.py`` and before ``05_assemble_spots.py``.

Model
    Maternal cells form a tissue-wide background; fetal cells are confined to a
    contiguous invasion zone anchored on a randomly chosen grid edge, after the
    biology of trophoblast invasion. Two mechanisms define the zone:

    Displacement
        A spot's total cell capacity is drawn once from a Poisson distribution
        and is not increased inside the invasion zone. Fetal cells therefore
        displace maternal cells rather than adding to them.

    Gradient
        Within the zone, the per-cell probability of being fetal falls linearly
        from 1.0 at the zone centre to 0.0 at its perimeter, and the fetal count
        is drawn from Binomial(total_cells, that probability).

Inputs (resolved from --out_dir, not passed explicitly)
    labels_generation_<seed>.p            cell annotations, from 02_build_cell_pool.py
    synthetic_ST_seed<seed>_design.csv    per-cell-type design, from 03_draw_design.py

Output
    synthetic_ST_seed<seed>_<assemble_id>_composition.csv
        Cell types as rows, spots as columns; values are integer cell counts.
"""

import argparse
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int,
                    help='random seed of split')
parser.add_argument('--tot_spots', dest='tot_spots', type=int,
                    default=1000,
                    help='Total number of spots to simulate')
parser.add_argument('--out_dir', dest='out_dir', type=str,
                    default='./',
                    help='Output directory')
parser.add_argument('--assemble_id', dest='assemble_id', type=int,
                    default=1,
                    help='ID of ST assembly')
parser.add_argument('--annotation_col', dest='anno_col', type=str,
                    default="annotation_1",
                    help='Name of column to use in annotation file (default: annotation_1)')

args = parser.parse_args()

seed = args.seed
tot_spots = args.tot_spots
out_dir = args.out_dir
assemble_id = args.assemble_id
anno_col = args.anno_col

# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
# Input paths are rebuilt from out_dir rather than passed in, so out_dir must be
# the same directory that split_sc and assemble_design wrote to.

lbl_gen_file = out_dir + "labels_generation_" + str(seed) + ".p"
design_file = out_dir + "synthetic_ST_seed" + str(seed) + "_design.csv"

lbl_generation = pickle.load(open(lbl_gen_file, "rb"))
uni_labels = lbl_generation[anno_col].unique()

design_df = pd.read_csv(design_file, index_col=0)
design_df = abs(design_df)

# ---------------------------------------------------------------------------
# Build the spatial composition
# ---------------------------------------------------------------------------

print(f"--- Starting Manual Spatial Composition v6 (Displacement & Gradient) ---")

# Allocate an empty (spot x cell type) frame.
spot_names = [f"Spotx{i+1}" for i in range(tot_spots)]
composition_df = pd.DataFrame(0, index=spot_names, columns=uni_labels)

# Square grid side, used both for spot coordinates and for placing the zone.
grid_size = int(np.ceil(np.sqrt(tot_spots)))
print(f"Total spots: {tot_spots} (Grid size: {grid_size}x{grid_size})")

maternal_design = design_df.loc['Maternal']
fetal_design = design_df.loc['Fetal']

# Anchor the invasion zone on a random edge, at a random position along it.
random_edge = np.random.randint(0, 4) # 0=Top, 1=Bottom, 2=Left, 3=Right
random_pos = np.random.randint(0, grid_size)

if random_edge == 0: # Top Edge
    cx, cy = random_pos, 0
elif random_edge == 1: # Bottom Edge
    cx, cy = random_pos, grid_size
elif random_edge == 2: # Left Edge
    cx, cy = 0, random_pos
else: # Right Edge (random_edge == 3)
    cx, cy = grid_size, random_pos

# Area compensation: only part of a circle centred on the boundary falls inside
# the grid. Near a corner roughly a quarter does, elsewhere along an edge
# roughly a half, so the target area is inflated by 4x or 2x before the radius
# is derived. "Near a corner" means the outer 20% of the edge at either end.
corner_threshold = grid_size * 0.2
is_near_corner = (random_pos < corner_threshold) or (random_pos > grid_size - corner_threshold)

if is_near_corner:
    area_factor = 4.0  # Quarter circle
    shape_desc = "Corner (Quarter-circle)"
else:
    area_factor = 2.0  # Semicircle
    shape_desc = "Edge (Semi-circle)"

compensated_nspots = fetal_design['nspots'] * area_factor

# Invert area = pi * r^2 to get the radius for the compensated target.
fetal_radius = np.sqrt(compensated_nspots / np.pi)

print(f"Fetal Zone Strategy: {shape_desc}")
print(f" - Center: ({cx}, {cy})")
print(f" - Original Target: {fetal_design['nspots']:.0f} spots")
print(f" - Compensated Target: {compensated_nspots:.0f} spots (Factor x{area_factor})")
print(f" - Calculated Radius: {fetal_radius:.2f}")

# Walk every spot in row-major order and assign its cell counts.
count_fetal_spots = 0

for i in range(tot_spots):
    spot_id = spot_names[i]
    x = i % grid_size
    y = i // grid_size

    # Total capacity comes from the maternal design for every spot, inside and
    # outside the invasion zone alike. This is what makes the model
    # displacement-based: fetal cells take slots from maternal cells instead of
    # raising the spot's cell count.
    total_cells = np.random.poisson(maternal_design['mean_ncells'])

    # Guarantee a non-empty spot.
    if total_cells == 0:
        total_cells = 1

    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    m_cells = total_cells
    f_cells = 0

    if distance < fetal_radius:
        # Inside the invasion zone.

        # Linear gradient: 1.0 at the zone centre, 0.0 at the perimeter.
        prob_fetal = 1.0 - (distance / fetal_radius)
        prob_fetal = np.clip(prob_fetal, 0.0, 1.0)

        f_cells = np.random.binomial(n=total_cells, p=prob_fetal)

        # Force at least one fetal cell so the zone contains no holes.
        if f_cells == 0:
             f_cells = 1

        m_cells = total_cells - f_cells

        if m_cells < 0:
            m_cells = 0

        count_fetal_spots += 1

    composition_df.loc[spot_id, 'Maternal'] = m_cells
    composition_df.loc[spot_id, 'Fetal'] = f_cells

print(f"--- Spatial Composition Finished. ---")
print(f"Total spots with Fetal Cells > 0: {count_fetal_spots}")

spots_members = composition_df

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
# Transposed on write, so the file is (cell type x spot). Downstream readers
# transpose it back: 08_qc_stats.py, 09_plot_slides.py and 10_validate_genotype.py all
# expect this orientation.

synthetic_st = {"composition": spots_members.T}

for k, v in synthetic_st.items():
    out_name = out_dir + "synthetic_ST_seed" + str(seed) + "_" + str(
        assemble_id) + "_" + k + ".csv"
    v.to_csv(out_name, sep=",", index=True, header=True)
