"""
preprocess_gpr_corr_medium.py

Converts the corr_medium Step1 voxel-radius HDF5 pair (10,000 samples, GT vs.
Step1 model prediction) into the same .pkl format preprocess_gpr.py produces
for the original TEUNet/GT dataset, so it can be consumed by the existing
GPRDataset / Stage 1 / Stage 2 training code unchanged.

Source: corr_medium_gt_voxel_radius/README_corr_medium_voxel_radius.md
Key differences from the original TEUNet source data:
  - Single H5 file per split (GT, prediction) holding all 10,000 samples,
    rather than one .h5 file per sample.
  - Volume axis order is (D=z, H=y, W=x) = (48, 64, 64); the original TEUNet
    pipeline used (X, Y, Z) = (64, 64, 48). Transposed to (X, Y, Z) here to
    match GPRDataset's existing convention.
  - Occupancy is already boolean (`pipe_mask`) on both sides -- no probability
    threshold needed. The prediction side's `confidence` field (dense
    pipe-probability value at occupied voxels) fills the role TEUNet's
    continuous probability played before (`input_prob`).
  - Isotropic 0.005m voxel spacing, vs. the original's near-isotropic but not
    exactly equal per-axis spacing.
"""

import argparse
import random
import sys
from pathlib import Path

import fvdb
import h5py
import numpy as np
import torch

VOXEL_SIZE = [0.005, 0.005, 0.005]  # meters, X Y Z (isotropic)
GRID_DIMS = (64, 64, 48)            # voxels, X Y Z, after transposing from (Z, Y, X)


def to_xyz(arr: np.ndarray) -> np.ndarray:
    """(Z, Y, X) -> (X, Y, Z), matching GPRDataset's existing axis convention."""
    return np.transpose(arr, (2, 1, 0))


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = (pred_mask & gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


def build_sample(gt_mask: np.ndarray, pred_mask: np.ndarray, pred_confidence: np.ndarray,
                  gt_material: np.ndarray, pred_material: np.ndarray):
    """
    Convert one sample's raw arrays (already transposed to X,Y,Z) to an
    fvdb-based sample dict. Returns None (with a reason string) if the
    sample should be skipped.
    """
    if gt_mask.shape != GRID_DIMS or pred_mask.shape != GRID_DIMS:
        return None, (
            f"shape mismatch: gt={gt_mask.shape}, pred={pred_mask.shape}, "
            f"expected {GRID_DIMS}"
        )

    gt_ijk = np.argwhere(gt_mask).astype(np.int32)
    if gt_ijk.shape[0] == 0:
        return None, "GT has no occupied voxels"

    input_ijk = np.argwhere(pred_mask).astype(np.int32)
    if input_ijk.shape[0] == 0:
        return None, "Step1 prediction has no occupied voxels"

    gt_ijk_t = torch.tensor(gt_ijk, dtype=torch.int32)
    input_ijk_t = torch.tensor(input_ijk, dtype=torch.int32)

    input_vals = pred_confidence.astype(np.float32)[pred_mask]  # (M,)
    input_vals = torch.tensor(input_vals).unsqueeze(-1)          # (M, 1)

    # Material class index at each occupied voxel (boolean-mask indexing walks
    # the array in the same C-order as np.argwhere, so these align 1:1 with
    # gt_ijk/input_ijk respectively). Always a real class (0-3) at occupied
    # voxels -- the -1 sentinel only ever occurs outside pipe_mask. Stored as
    # int8 (only 4 classes) rather than int64 to keep dataset size down;
    # base_encoder.py casts to .long() itself before the nn.Embedding lookup.
    target_material_vals = torch.tensor(gt_material[gt_mask].astype(np.int8))     # (N_gt,)
    input_material_vals = torch.tensor(pred_material[pred_mask].astype(np.int8))  # (M,)

    grid_origin = [vs / 2.0 for vs in VOXEL_SIZE]
    gt_grid = fvdb.GridBatch()
    gt_grid.set_from_ijk(
        fvdb.JaggedTensor(gt_ijk_t),
        voxel_sizes=VOXEL_SIZE,
        origins=grid_origin,
    )
    input_grid = fvdb.GridBatch()
    input_grid.set_from_ijk(
        fvdb.JaggedTensor(input_ijk_t),
        voxel_sizes=VOXEL_SIZE,
        origins=grid_origin,
    )

    sample = {
        'target_grid': gt_grid,               # GridBatch: GT occupied voxels
        'input_grid': input_grid,             # GridBatch: Step1-predicted occupied voxels
        'input_prob': input_vals,             # (M, 1) tensor: Step1 confidence at each input voxel
        'target_material': target_material_vals,  # (N_gt,) tensor: GT material class per GT voxel
        'input_material': input_material_vals,    # (M,) tensor: Step1 material class per input voxel
    }
    return sample, None


def assign_tier(score: float) -> str:
    if score > 0.8:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    return 'poor'


def stratified_split(stems_and_scores: list, seed: int = 42):
    tiers = {'good': [], 'moderate': [], 'poor': []}
    for stem, score in stems_and_scores:
        tiers[assign_tier(score)].append(stem)

    train, val, test = [], [], []
    rng = random.Random(seed)

    for tier_name, names in tiers.items():
        rng.shuffle(names)
        n = len(names)
        n_val = max(1, round(n * 0.1)) if n >= 3 else 0
        n_test = max(1, round(n * 0.1)) if n >= 3 else 0
        n_train = n - n_val - n_test

        train.extend(names[:n_train])
        val.extend(names[n_train:n_train + n_val])
        test.extend(names[n_train + n_val:])

        print(
            f"  tier={tier_name:8s}  total={n:5d}  "
            f"train={n_train:5d}  val={n_val:4d}  test={n - n_train - n_val:4d}"
        )

    return train, val, test


REQUIRED_KEYS = {'target_grid', 'input_grid', 'input_prob', 'target_material', 'input_material'}


def spot_check(pkl_dir: Path, stems: list, n: int = 5):
    sample_stems = random.sample(stems, min(n, len(stems)))
    print(f"\nSpot-checking {len(sample_stems)} .pkl files...")
    all_ok = True
    for stem in sample_stems:
        path = pkl_dir / f"{stem}.pkl"
        try:
            obj = torch.load(path, weights_only=False)
            missing = REQUIRED_KEYS - set(obj.keys())
            if missing:
                print(f"  FAIL {stem}: missing keys {missing}")
                all_ok = False
                continue
            prob = obj['input_prob']
            if prob.dim() != 2 or prob.shape[1] != 1:
                print(f"  FAIL {stem}: input_prob shape {prob.shape}")
                all_ok = False
                continue
            input_mat = obj['input_material']
            target_mat = obj['target_material']
            if input_mat.shape[0] != prob.shape[0]:
                print(f"  FAIL {stem}: input_material shape {input_mat.shape} != input_prob {prob.shape}")
                all_ok = False
                continue
            if target_mat.shape[0] != obj['target_grid'].total_voxels:
                print(f"  FAIL {stem}: target_material shape {target_mat.shape} != target_grid voxels {obj['target_grid'].total_voxels}")
                all_ok = False
                continue
            print(f"  OK   {stem}  (M={prob.shape[0]}, N_gt={obj['target_grid'].total_voxels})")
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
            all_ok = False
    return all_ok


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess corr_medium GT/Step1-prediction HDF5 pair into XCube-compatible .pkl files."
    )
    p.add_argument('--gt_h5', required=True, type=Path,
                   help="Path to corr_medium_gt_voxel_radius.h5")
    p.add_argument('--pred_h5', required=True, type=Path,
                   help="Path to corr_medium_step1_pred_voxel_radius.h5")
    p.add_argument('--output_dir', required=True, type=Path,
                   help="Root output directory (dataset/resolution subdirs created automatically)")
    p.add_argument('--custom_name', type=str, default="gpr_corr_medium",
                   help="Dataset subdirectory name (default gpr_corr_medium)")
    p.add_argument('--seed', type=int, default=42,
                   help="Random seed for stratified split (default 42)")
    p.add_argument('--limit', type=int, default=None,
                   help="Process only the first N samples (for test runs)")
    return p.parse_args()


def main():
    args = parse_args()

    resolution = "0.005"
    pkl_dir = args.output_dir / args.custom_name / resolution
    pkl_dir.mkdir(parents=True, exist_ok=True)

    print(f"GT h5      : {args.gt_h5}")
    print(f"Pred h5    : {args.pred_h5}")
    print(f"Output dir : {pkl_dir}")
    print(f"Seed       : {args.seed}\n")

    with h5py.File(args.gt_h5, 'r') as gt_f, h5py.File(args.pred_h5, 'r') as pred_f:
        n_gt = gt_f['sample_id'].shape[0]
        n_pred = pred_f['sample_id'].shape[0]
        if n_gt != n_pred:
            print(f"ERROR: GT has {n_gt} samples, prediction has {n_pred}", file=sys.stderr)
            sys.exit(1)

        n = n_gt if args.limit is None else min(args.limit, n_gt)
        print(f"Processing {n} of {n_gt} samples.\n")

        stems_and_scores = []
        skipped = 0

        for i in range(n):
            gt_id = gt_f['sample_id'][i]
            pred_id = pred_f['sample_id'][i]
            gt_id = gt_id.decode() if isinstance(gt_id, bytes) else gt_id
            pred_id = pred_id.decode() if isinstance(pred_id, bytes) else pred_id
            if gt_id != pred_id:
                print(f"  WARN  skipping index {i}: sample_id mismatch (gt={gt_id}, pred={pred_id})")
                skipped += 1
                continue
            stem = gt_id

            if not bool(gt_f['done'][i]) or not bool(pred_f['done'][i]):
                print(f"  WARN  skipping {stem}: done=False")
                skipped += 1
                continue

            gt_mask = to_xyz(gt_f['pipe_mask'][i])
            pred_mask = to_xyz(pred_f['pipe_mask'][i])
            pred_confidence = to_xyz(pred_f['confidence'][i])
            gt_material = to_xyz(gt_f['material'][i])
            pred_material = to_xyz(pred_f['material'][i])

            sample, reason = build_sample(gt_mask, pred_mask, pred_confidence, gt_material, pred_material)
            if sample is None:
                print(f"  WARN  skipping {stem}: {reason}")
                skipped += 1
                continue

            score = dice_score(pred_mask, gt_mask)
            torch.save(sample, pkl_dir / f"{stem}.pkl")
            stems_and_scores.append((stem, score))

            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{n} processed")

    saved = len(stems_and_scores)
    print(f"\nSaved {saved} .pkl files, skipped {skipped}.\n")

    if saved == 0:
        print("ERROR: no samples were saved -- nothing to split.", file=sys.stderr)
        sys.exit(1)

    print("Stratified split:")
    train_stems, val_stems, test_stems = stratified_split(stems_and_scores, seed=args.seed)

    lst_dir = args.output_dir / args.custom_name
    for split_name, split_stems in [
        ("train", train_stems),
        ("val", val_stems),
        ("test", test_stems),
    ]:
        lst_path = lst_dir / f"{split_name}.lst"
        lst_path.write_text("\n".join(split_stems) + "\n")
        print(f"  wrote {lst_path}  ({len(split_stems)} entries)")

    train_set = set(train_stems)
    val_set = set(val_stems)
    test_set = set(test_stems)
    overlaps = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlaps:
        print(f"\nWARN: {len(overlaps)} stems appear in more than one split: {overlaps}")
    else:
        print("\nSplit overlap check: OK (no overlap)")

    all_ok = spot_check(pkl_dir, [s for s, _ in stems_and_scores], n=5)
    if not all_ok:
        print("\nWARN: one or more spot-checks failed -- inspect the files above.")
    else:
        print("\nAll spot-checks passed.")


if __name__ == "__main__":
    main()
