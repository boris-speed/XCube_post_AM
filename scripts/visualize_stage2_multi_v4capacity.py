"""
Same as visualize_stage2_multi.py, but points at the v4-capacity checkpoint
(trained with increased network capacity -- model_channels 32->64,
channel_mult [1,2]->[1,2,4], num_res_blocks 1->2 -- everything else identical
to v1: plain GT footprint, no classifier-free dropout, no attention).

Usage:
    python scripts/visualize_stage2_multi_v4capacity.py [idx1 idx2 idx3 ...]
"""
import sys
sys.path.insert(0, '/home/ameliacatala/Documents/XCube')
import pickle
import importlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v4/version_0')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [60, 140, 185]
tier_of = lambda i: 'good' if i < 126 else ('moderate' if i < 172 else 'poor')

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

def to_dense(ijk, mins, shape):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

fig = plt.figure(figsize=(15, 5 * len(indices)))
col_titles = ['TEUNet Input (flawed)', 'XCube Stage 2 Output (v4-capacity)', 'Ground Truth']

for row, idx in enumerate(indices):
    stem = stems[idx]
    tier = tier_of(idx)
    print(f'[{row+1}/{len(indices)}] sample #{idx} ({tier}): {stem}')

    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')

    with torch.no_grad():
        cond_latent = model.vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid = res.structure_grid[0]

    teunet_ijk = to_ijk(teunet_grid)
    pred_ijk = to_ijk(pred_grid)
    gt_ijk = to_ijk(gt_grid)

    mins = gt_ijk.min(axis=0)
    maxs = gt_ijk.max(axis=0)
    shape = tuple(maxs - mins + 1)

    for col, (ijk, base_title) in enumerate(zip([teunet_ijk, pred_ijk, gt_ijk], col_titles)):
        ax = fig.add_subplot(len(indices), 3, row * 3 + col + 1, projection='3d')
        ax.voxels(to_dense(ijk, mins, shape), edgecolor='k', linewidth=0.1)
        title = base_title if row == 0 else ''
        ax.set_title(f'{title}\n[{tier}] {ijk.shape[0]} voxels'.strip())
        ax.set_box_aspect(shape)
        ax.view_init(elev=20, azim=-60)

plt.tight_layout()
out_path = '/home/ameliacatala/Documents/stage2_visual_comparison_multi_v4capacity.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
