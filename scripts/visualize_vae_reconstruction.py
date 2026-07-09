"""
Saves a side-by-side 3D picture of one GPR pipe sample as it passes through
the trained Stage 1 VAE: the ground-truth input, the coarse intermediate
structure the network predicts first, and the final fine-resolution
reconstruction it decodes back out to.

Usage:
    python scripts/visualize_vae_reconstruction.py [test_set_index]
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/VAE_stage1_corr_medium/version_1')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
stems = [s for s in (DATA_DIR / 'test.lst').read_text().split('\n') if s]
stem = stems[idx]
print('Visualizing sample:', stem)

sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
gt_grid = sample['target_grid'].to('cuda')
gt_material = sample['target_material'].to('cuda')

with torch.no_grad():
    batch = {DS.INPUT_PC: gt_grid, DS.INPUT_MATERIAL: gt_material}
    out = model(batch, {})
tree = out['tree']  # dict keyed by tree depth: 0 = finest, 1 = coarser (tree_depth=2)

def to_ijk(grid):
    return grid.ijk[0].jdata.cpu().numpy()

before_ijk = to_ijk(gt_grid)               # ground-truth input
during_ijk = to_ijk(tree[1])               # coarse structure predicted first
after_ijk = to_ijk(tree[0])                # final fine-resolution reconstruction

# Shared bounding box (fine-grid units) so all three panels render at the
# same physical scale. The coarse ("during") grid is in half-resolution
# units, so its ijk is scaled up by 2x to align with the fine grid's frame.
during_ijk_fine = during_ijk * 2

mins = before_ijk.min(axis=0)
maxs = before_ijk.max(axis=0)
shape = tuple(maxs - mins + 1)

def to_dense(ijk, cell=1):
    arr = np.zeros(shape, dtype=bool)
    local = ijk - mins
    valid = np.all((local >= 0) & (local < np.array(shape)), axis=1)
    local = local[valid]
    if cell > 1:
        # Fill the whole cell-sized block each coarse voxel covers, so it
        # reads as visibly blockier rather than a sparse sub-sampled dot.
        for dx in range(cell):
            for dy in range(cell):
                for dz in range(cell):
                    off = local + np.array([dx, dy, dz])
                    ok = np.all(off < np.array(shape), axis=1)
                    o = off[ok]
                    arr[o[:, 0], o[:, 1], o[:, 2]] = True
    else:
        arr[local[:, 0], local[:, 1], local[:, 2]] = True
    return arr

fig = plt.figure(figsize=(15, 5))
panels = [
    ('Before\n(Ground Truth input)', before_ijk, 1),
    ('During\n(coarse structure, depth 1)', during_ijk_fine, 2),
    ('After\n(VAE reconstruction, depth 0)', after_ijk, 1),
]
for i, (title, ijk, cell) in enumerate(panels):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.voxels(to_dense(ijk, cell=cell), edgecolor='k', linewidth=0.1)
    ax.set_title(f'{title}\n({ijk.shape[0]} voxels)')
    ax.set_box_aspect(shape)
    ax.view_init(elev=20, azim=-60)

plt.suptitle(f'Sample: {stem}')
plt.tight_layout()
out_path = '/home/ameliacatala/Documents/XCube/vae_reconstruction_visual.png'
plt.savefig(out_path, dpi=130)
print('Saved to', out_path)
