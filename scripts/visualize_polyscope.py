"""
visualize_polyscope.py

Proof-of-concept: view a GPR .pkl sample (GT vs. Step1 prediction, colored by
material class) interactively with Polyscope, as an alternative to the
existing matplotlib voxel-cube scripts (scripts/visualize_stage2_*.py).

Usage:
    python scripts/visualize_polyscope.py --pkl <path-to-sample.pkl>
"""

import argparse
from pathlib import Path

import fvdb
import numpy as np
import polyscope as ps
import torch

# Same palette-ish idea as matplotlib's tab10, just picked by hand so material
# classes are visually distinct: 0=grey, 1=blue, 2=orange, 3=green.
MATERIAL_COLORS = np.array([
    [0.6, 0.6, 0.6],
    [0.2, 0.4, 0.9],
    [0.9, 0.5, 0.1],
    [0.2, 0.7, 0.3],
])


def grid_world_xyz(grid: fvdb.GridBatch) -> np.ndarray:
    return grid.grid_to_world(grid.ijk.float()).jdata.cpu().numpy()


def material_colors(material: torch.Tensor) -> np.ndarray:
    return MATERIAL_COLORS[material.cpu().numpy().astype(np.int64)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True, type=Path)
    args = p.parse_args()

    sample = torch.load(args.pkl, weights_only=False)

    gt_xyz = grid_world_xyz(sample["target_grid"])
    input_xyz = grid_world_xyz(sample["input_grid"])

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    gt_cloud = ps.register_point_cloud("Ground Truth", gt_xyz, radius=0.0035, point_render_mode="quad")
    input_cloud = ps.register_point_cloud("Step1 Prediction", input_xyz, radius=0.0035, point_render_mode="quad")

    if "target_material" in sample:
        gt_cloud.add_color_quantity("material", material_colors(sample["target_material"]), enabled=True)
    if "input_material" in sample:
        input_cloud.add_color_quantity("material", material_colors(sample["input_material"]), enabled=True)
    if "input_prob" in sample:
        input_cloud.add_scalar_quantity("confidence", sample["input_prob"].squeeze(-1).cpu().numpy())

    print(f"GT voxels: {gt_xyz.shape[0]}, Step1 prediction voxels: {input_xyz.shape[0]}")
    ps.show()


if __name__ == "__main__":
    main()
