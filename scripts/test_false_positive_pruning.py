"""
Complementary diagnostic to test_vae_structural_confinement.py and
test_unreachable_distance.py. Those two measure the "not enough room" side of
the problem (structure the decoder can never reach). This one measures the
opposite side: does the decoder correctly DROP coarse candidate blocks from
Step1's own footprint that don't correspond to any real GT structure, or does
it wrongly keep them?

Why this might be a separate, untested gap (see PROGRESS.md, 2026-07-15,
"what if it needs to remove coarse blocks" discussion): the decoder's
struct_conv already makes a keep/discard decision at every level including
the coarsest one, so mechanically it CAN prune wrong blocks. But Stage 1
trains by self-reconstructing ground truth (input_key: "target_grid"), so the
coarse footprint it's handed during training is always exactly correct --
the only "wrong" cells it ever practices pruning are the artificial dilation
margin (uniformly close to real structure, by construction). It never
practices pruning a genuinely false-positive block sitting somewhere with no
real structure nearby, the way Step1 sometimes actually produces (e.g. the
"blocky, disconnected cluster" pattern seen in the visual audit, sample #987).

This script measures, per test sample: of the coarse cells in Step1's own
(undilated) footprint that do NOT correspond to real GT coarse structure, what
fraction survives into the decoder's coarsest-level keep/discard output
(res.structure_grid[1], i.e. genuinely wrongly kept) vs. gets correctly
dropped.

Usage:
    python scripts/test_false_positive_pruning.py [n_samples]
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import torch

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

# Same original (non-dilated) checkpoint as test_vae_structural_confinement.py
# and test_unreachable_distance.py -- this is about whether the ALREADY-
# EXISTING pruning mechanism handles Step1's genuine false positives, not
# about the dilation fix.
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
COARSE_FEAT_DEPTH = 1  # num_blocks - 1 for tree_depth=2 -- matches struct-acc-1 / coarsened_grid(2)

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

results = []  # (tier, n_false_positive_coarse_cells, n_survived)

for i, stem in enumerate(stems):
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    step1_grid = sample['input_grid'].to('cuda')
    step1_material = sample['input_material'].to('cuda')

    tier = assign_tier(grid_dice(gt_grid, step1_grid))

    with torch.no_grad():
        latent = vae._encode({DS.INPUT_PC: step1_grid, DS.INPUT_MATERIAL: step1_material}, use_mode=True)
        res = vae.unet.FeaturesSet()
        res, output_x = vae.unet.decode(res, latent, is_testing=True)
    footprint_grid = latent.grid  # Step1's own (undilated) coarse footprint
    coarse_decision_grid = res.structure_grid[COARSE_FEAT_DEPTH]  # decoder's kept-at-coarsest-level output

    gt_coarse = gt_grid.coarsened_grid(COARSE_FACTOR)

    n_footprint = footprint_grid.total_voxels
    if n_footprint == 0:
        continue

    # Which of Step1's own coarse cells are NOT part of GT's true coarse structure?
    true_idx = gt_coarse.ijk_to_index(footprint_grid.ijk).jdata
    is_false_positive = true_idx < 0
    n_fp = is_false_positive.sum().item()
    if n_fp == 0:
        results.append((tier, 0, 0))
        if i % 100 == 0:
            print(f'[{tier:8s}] {i:3d}/{len(stems)}  no false-positive coarse cells in Step1 footprint')
        continue

    fp_ijk = footprint_grid.ijk.jdata[is_false_positive]
    survived_idx = coarse_decision_grid.ijk_to_index(fp_ijk.contiguous()).jdata
    n_survived = (survived_idx >= 0).sum().item()

    results.append((tier, n_fp, n_survived))

    if i % 100 == 0:
        print(f'[{tier:8s}] {i:3d}/{len(stems)}  false_positive_cells={n_fp}  '
              f'wrongly_kept={n_survived} ({n_survived/n_fp:.3f})')

print()
print('=' * 100)
for tier in ['good', 'moderate', 'poor']:
    sub = [r for r in results if r[0] == tier]
    if not sub:
        continue
    total_fp = sum(r[1] for r in sub)
    total_survived = sum(r[2] for r in sub)
    if total_fp == 0:
        print(f'{tier:8s} (n={len(sub)}): no false-positive coarse cells found in this tier')
        continue
    print(f'{tier:8s} (n={len(sub)}, total false-positive coarse cells={total_fp}): '
          f'wrongly kept={total_survived} ({total_survived/total_fp:.3f})  '
          f'correctly pruned={total_fp-total_survived} ({1-total_survived/total_fp:.3f})')

all_fp = sum(r[1] for r in results)
all_survived = sum(r[2] for r in results)
print()
if all_fp:
    print(f'OVERALL: {all_fp} false-positive coarse cells across {len(results)} samples, '
          f'{all_survived} ({all_survived/all_fp:.3f}) wrongly kept by the decoder, '
          f'{all_fp-all_survived} ({1-all_survived/all_fp:.3f}) correctly pruned.')
else:
    print('OVERALL: no false-positive coarse cells found in Step1\'s footprint across all samples.')
