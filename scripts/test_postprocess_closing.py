"""
Cheap, no-retraining test: does simple morphological closing (bridge small
gaps, i.e. dilate then erode) on the v1 model's existing output voxels
recover any IoU, especially on poor-tier (fragmented) samples? If yes, that
tells us the model's output fragments are "almost right" and just need
gap-bridging. If no, the fragments are genuinely too far apart / missing
too much structure for a cheap fix -- consistent with an information gap
rather than a fixable-by-postprocessing shape defect.

Usage:
    python scripts/test_postprocess_closing.py [idx1 idx2 ...]

Defaults to the full 198-sample test set if no indices are given.
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2/version_2')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
print(f'{len(stems)} test samples total. Tiers (by index): 0-125 good, 126-171 moderate, 172-197 poor.\n')
indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(len(stems)))
tier_of = lambda i: 'good' if i < 126 else ('moderate' if i < 172 else 'poor')

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

def ijk_to_dense(ijk, mins, shape):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

def dense_to_ijk(arr, mins):
    coords = np.argwhere(arr)
    return coords + mins

def numpy_iou(ijk_a, ijk_b):
    set_a = set(map(tuple, ijk_a))
    set_b = set(map(tuple, ijk_b))
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / (union + 1e-6)

PAD = 3  # margin so closing can bridge gaps near the current bbox edge
struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

results = []
for i in indices:
    stem = stems[i]
    tier = tier_of(i)
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')

    with torch.no_grad():
        cond_latent = model.vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid = res.structure_grid[0]

    pred_ijk = to_ijk(pred_grid)
    gt_ijk = to_ijk(gt_grid)

    iou_before = numpy_iou(pred_ijk, gt_ijk)

    mins = np.minimum(pred_ijk.min(axis=0), gt_ijk.min(axis=0)) - PAD
    maxs = np.maximum(pred_ijk.max(axis=0), gt_ijk.max(axis=0)) + PAD
    shape = tuple(maxs - mins + 1)

    dense = ijk_to_dense(pred_ijk, mins, shape)
    closed_1 = ndimage.binary_closing(dense, structure=struct, iterations=1)
    closed_2 = ndimage.binary_closing(dense, structure=struct, iterations=2)

    closed_1_ijk = dense_to_ijk(closed_1, mins)
    closed_2_ijk = dense_to_ijk(closed_2, mins)

    iou_closed_1 = numpy_iou(closed_1_ijk, gt_ijk)
    iou_closed_2 = numpy_iou(closed_2_ijk, gt_ijk)

    results.append((tier, iou_before, iou_closed_1, iou_closed_2))
    print(f'[{tier:8s}] {stem:20s} IoU before={iou_before:.3f}  closed(1)={iou_closed_1:.3f} ({iou_closed_1-iou_before:+.3f})  closed(2)={iou_closed_2:.3f} ({iou_closed_2-iou_before:+.3f})')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(b, c1, c2) for tr, b, c1, c2 in results if tr == tier]
    if not sub:
        continue
    n = len(sub)
    avg_b = sum(b for b, c1, c2 in sub) / n
    avg_c1 = sum(c1 for b, c1, c2 in sub) / n
    avg_c2 = sum(c2 for b, c1, c2 in sub) / n
    print(f'{tier:8s} (n={n}): before={avg_b:.3f}  closed(1)={avg_c1:.3f} ({avg_c1-avg_b:+.3f})  closed(2)={avg_c2:.3f} ({avg_c2-avg_b:+.3f})')

n = len(results)
avg_b = sum(b for _, b, c1, c2 in results) / n
avg_c1 = sum(c1 for _, b, c1, c2 in results) / n
avg_c2 = sum(c2 for _, b, c1, c2 in results) / n
print(f'\nOVERALL (n={n}): before={avg_b:.3f}  closed(1)={avg_c1:.3f} ({avg_c1-avg_b:+.3f})  closed(2)={avg_c2:.3f} ({avg_c2-avg_b:+.3f})')
