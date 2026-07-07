"""
Side-by-side comparison of TEUNet input / v1 output / v4-capacity output /
ground truth, for specific test-set indices.

Usage:
    python scripts/visualize_stage2_v1_vs_v4capacity.py [idx1 idx2 ...]

Defaults to samples 175 and 185 (both poor-tier) if no indices are given.
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

V1_VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2/version_2')
V4_VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v4/version_0')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

def load_model(version_dir):
    model_args = exp.parse_config_yaml(version_dir / 'hparams.yaml')
    net_module = importlib.import_module("xcube.models." + model_args.model).Model
    model = net_module.load_from_checkpoint(version_dir / 'checkpoints' / 'last.ckpt', hparams=model_args)
    return model.cuda().eval()

print('Loading v1 model...')
model_v1 = load_model(V1_VERSION_DIR)
print('Loading v4-capacity model...')
model_v4 = load_model(V4_VERSION_DIR)

stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [175, 185]
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

def run_model(model, teunet_grid):
    with torch.no_grad():
        cond_latent = model.vae._encode({DS.INPUT_PC: teunet_grid}, use_mode=True)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    return res.structure_grid[0]

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

fig = plt.figure(figsize=(20, 5 * len(indices)))
col_titles = ['TEUNet Input (flawed)', 'v1 Output', 'v4-capacity Output', 'Ground Truth']

for row, idx in enumerate(indices):
    stem = stems[idx]
    tier = tier_of(idx)
    print(f'[{row+1}/{len(indices)}] sample #{idx} ({tier}): {stem}')

    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')

    pred_v1 = run_model(model_v1, teunet_grid)
    pred_v4 = run_model(model_v4, teunet_grid)

    iou_teunet = grid_iou(gt_grid, teunet_grid)
    iou_v1 = grid_iou(gt_grid, pred_v1)
    iou_v4 = grid_iou(gt_grid, pred_v4)
    print(f'  IoU(TEUNet,GT)={iou_teunet:.3f}  IoU(v1,GT)={iou_v1:.3f}  IoU(v4-capacity,GT)={iou_v4:.3f}')

    teunet_ijk = to_ijk(teunet_grid)
    v1_ijk = to_ijk(pred_v1)
    v4_ijk = to_ijk(pred_v4)
    gt_ijk = to_ijk(gt_grid)

    mins = gt_ijk.min(axis=0)
    maxs = gt_ijk.max(axis=0)
    shape = tuple(maxs - mins + 1)

    ious = [iou_teunet, iou_v1, iou_v4, None]
    for col, (ijk, base_title, iou) in enumerate(zip([teunet_ijk, v1_ijk, v4_ijk, gt_ijk], col_titles, ious)):
        ax = fig.add_subplot(len(indices), 4, row * 4 + col + 1, projection='3d')
        ax.voxels(to_dense(ijk, mins, shape), edgecolor='k', linewidth=0.1)
        title = base_title if row == 0 else ''
        iou_str = f' (IoU={iou:.3f})' if iou is not None else ''
        ax.set_title(f'{title}\n[{tier}] #{idx} {ijk.shape[0]} voxels{iou_str}'.strip())
        ax.set_box_aspect(shape)
        ax.view_init(elev=20, azim=-60)

plt.tight_layout()
out_path = '/home/ameliacatala/Documents/stage2_visual_v1_vs_v4capacity_175_185.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
