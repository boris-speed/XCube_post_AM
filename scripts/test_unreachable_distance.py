"""
Follow-up to test_vae_structural_confinement.py: that script measured HOW MUCH
of GT's coarse structure is architecturally unreachable given Step1's own
coarse footprint (up to 48.2% on poor tier). It didn't say anything about
WHERE that missing structure sits relative to what Step1 did find.

This matters because the coarse-dilation fix (PROGRESS.md, 2026-07-15) only
adds a small fixed margin (1 voxel at kernel_size=3) around Step1's existing
footprint -- a "near miss" fix. If most of the unreachable structure sits far
from Step1's footprint (a genuinely separate missed region, e.g. the README's
noted "pipe near patch bottom missed entirely" failure mode), no reasonable
margin recovers it, and widening the margin further just reintroduces the
free-generation failure mode already ruled out (TEUNet fix attempt 4, Part 1:
IoU collapsed to ~0.004 when the decoder was given a fully open candidate
region).

This script measures, per test sample, the distance (in coarse-voxel units)
from every unreachable GT coarse cell to the nearest cell in Step1's own
coarse footprint, and buckets that distance against the margins the dilation
fix could plausibly use (1, 2, 3, 5 voxels) to show how much of the missing
structure each margin size could actually reach.

Usage:
    python scripts/test_unreachable_distance.py [n_samples]
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from xcube.utils import exp
from xcube.data.base import DatasetSpec as DS

_orig_load = torch.load
def _trusted_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _trusted_load

custom_pickle = pickle
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "featurevdb._Cpp":
            module = "fvdb._Cpp"
        return super().find_class(module, name)
custom_pickle.Unpickler = CustomUnpickler

# Deliberately the ORIGINAL (non-dilated) corr_medium VAE -- this question is
# about the geometry of Step1's data vs. GT, not about any particular decode()
# fix, so it should use the same checkpoint the 48.2%-unreachable number came
# from (test_vae_structural_confinement.py), not either coarse-dilation VAE.
VAE_CONFIG = Path('/home/ameliacatala/Documents/XCube/configs/gpr/gpr_vae_corr_medium.yaml')
VAE_CKPT_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/VAE_stage1_corr_medium/version_1/checkpoints')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

ckpts = sorted(VAE_CKPT_DIR.glob('epoch=*.ckpt'), key=lambda p: p.stat().st_mtime)
VAE_CKPT = ckpts[-1]
print('Using checkpoint:', VAE_CKPT)

model_args = exp.parse_config_yaml(VAE_CONFIG)
net_module = importlib.import_module("xcube.models." + model_args.model).Model
vae = net_module.load_from_checkpoint(VAE_CKPT, hparams=model_args)
vae = vae.cuda().eval()

COARSE_FACTOR = 2
MARGIN_BUCKETS = [1, 2, 3, 5]  # voxel margins corresponding to kernel_size 3, 5, 7, 11

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else len(stems)
stems = stems[:n_samples]
print(f'{len(stems)} test samples.\n')

def grid_dice(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return 2 * inter / (upi[0] + 1e-6)

def assign_tier(score):
    if score > 0.8:
        return 'good'
    elif score >= 0.5:
        return 'moderate'
    return 'poor'

# results[tier] = list of (n_unreachable, counts_within_each_margin_bucket, min_dist_sum)
results = []

for i, stem in enumerate(stems):
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    tier = assign_tier(grid_dice(gt_grid, step1_grid))

    with torch.no_grad():
        latent = vae._encode({DS.INPUT_PC: step1_grid, DS.INPUT_MATERIAL: step1_material}, use_mode=True)
    reach_grid = latent.grid

    gt_coarse = gt_grid.coarsened_grid(COARSE_FACTOR)
    total_gt_coarse = gt_coarse.total_voxels
    if total_gt_coarse == 0:
        continue

    reach_idx = reach_grid.ijk_to_index(gt_coarse.ijk).jdata
    reach_mask = reach_idx >= 0
    n_unreachable = (~reach_mask).sum().item()
    if n_unreachable == 0:
        results.append((tier, 0, [0] * len(MARGIN_BUCKETS), 0.0))
        if i % 100 == 0:
            print(f'[{tier:8s}] {i:3d}/{len(stems)}  n_unreachable=0 (fully reachable)')
        continue

    unreachable_ijk = gt_coarse.ijk.jdata[~reach_mask].cpu().numpy().astype(np.float64)
    footprint_ijk = reach_grid.ijk.jdata.cpu().numpy().astype(np.float64)

    if footprint_ijk.shape[0] == 0:
        # Degenerate: Step1 found nothing at all -- every unreachable cell is
        # "infinitely" far. Bucket all as beyond the largest margin.
        counts = [0] * len(MARGIN_BUCKETS)
        results.append((tier, n_unreachable, counts, float('inf')))
        continue

    tree = cKDTree(footprint_ijk)
    # Chebyshev (chessboard) distance matches how a cubic dilation margin
    # actually reaches neighbors -- a voxel at offset (1,1,1) is reached by a
    # 1-voxel cubic margin just as much as one at (1,0,0).
    dists, _ = tree.query(unreachable_ijk, k=1, p=np.inf)

    counts = [int((dists <= m).sum()) for m in MARGIN_BUCKETS]
    results.append((tier, n_unreachable, counts, float(dists.mean())))

    if i % 100 == 0:
        pct_within = [f'<={m}:{c/n_unreachable:.2f}' for m, c in zip(MARGIN_BUCKETS, counts)]
        print(f'[{tier:8s}] {i:3d}/{len(stems)}  n_unreachable={n_unreachable}  '
              f'mean_dist={dists.mean():.2f}  frac_within {" ".join(pct_within)}')

print()
print('=' * 100)
for tier in ['good', 'moderate', 'poor']:
    sub = [r for r in results if r[0] == tier]
    if not sub:
        continue
    total_unreachable = sum(r[1] for r in sub)
    if total_unreachable == 0:
        print(f'{tier:8s} (n={len(sub)}): no unreachable structure in this tier')
        continue
    print(f'{tier:8s} (n={len(sub)}, total unreachable coarse cells={total_unreachable}):')
    for bi, m in enumerate(MARGIN_BUCKETS):
        recovered = sum(r[2][bi] for r in sub)
        print(f'    within margin={m} (kernel_size={2*m+1}): '
              f'{recovered}/{total_unreachable} = {recovered/total_unreachable:.3f} of missing structure reachable')
    finite_dists = [r[3] for r in sub if r[1] > 0 and r[3] != float('inf')]
    if finite_dists:
        print(f'    avg mean-distance-to-footprint across samples: {sum(finite_dists)/len(finite_dists):.2f} coarse voxels')

print()
all_unreachable = sum(r[1] for r in results)
print(f'OVERALL total unreachable coarse cells across {len(results)} samples: {all_unreachable}')
for bi, m in enumerate(MARGIN_BUCKETS):
    recovered = sum(r[2][bi] for r in results)
    frac = recovered / all_unreachable if all_unreachable else float('nan')
    print(f'  within margin={m} (kernel_size={2*m+1}): {recovered}/{all_unreachable} = {frac:.3f}')
