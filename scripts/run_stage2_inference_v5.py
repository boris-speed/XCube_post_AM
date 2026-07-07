"""
Same as run_stage2_inference_v4.py, but points at the v5 checkpoint (trained
with hardness-based loss reweighting -- see PROGRESS.md and
configs/gpr/gpr_diffusion_v5.yaml). Same v1 architecture/capacity and plain
GT footprint otherwise, so this isolates reweighting as the only new
variable vs v1.

Usage:
    python scripts/run_stage2_inference_v5.py
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

VERSION_DIR = Path('/home/ameliacatala/Documents/checkpoints/gpr/Diffusion_stage2_v5/version_0')
DATA_DIR = Path('/home/ameliacatala/Documents/preprocess/data_full/gpr')

model_args = exp.parse_config_yaml(VERSION_DIR / 'hparams.yaml')
net_module = importlib.import_module("xcube.models." + model_args.model).Model
model = net_module.load_from_checkpoint(VERSION_DIR / 'checkpoints' / 'last.ckpt', hparams=model_args)
model = model.cuda().eval()

test_lst = DATA_DIR / 'test.lst'
stems = [s for s in test_lst.read_text().split('\n') if s]
print(f'{len(stems)} test samples total. Tiers (by index): 0-125 good, 126-171 moderate, 172-197 poor.\n')

indices = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else list(range(len(stems)))

def grid_iou(gt, pd):
    idx = pd.ijk_to_index(gt.ijk)
    upi = (pd.num_voxels + gt.num_voxels).cpu().numpy().tolist()
    inter = torch.sum(idx[0].jdata >= 0).item()
    return inter / (upi[0] - inter + 1e-6)

tier_of = lambda i: 'good' if i < 126 else ('moderate' if i < 172 else 'poor')

results = []
for i in indices:
    stem = stems[i]
    sample = torch.load(DATA_DIR / '0.005' / f'{stem}.pkl', pickle_module=custom_pickle)
    gt_grid = sample['target_grid'].to('cuda')
    teunet_grid = sample['input_grid'].to('cuda')

    with torch.no_grad():
        cond_latent = model.encode_cond_grid(teunet_grid)
        res, output_x = model.evaluation_api(
            batch={DS.COND_PC: teunet_grid}, grids=cond_latent.grid,
            use_ddim=True, ddim_step=100, use_ema=False)
    pred_grid = res.structure_grid[0]

    iou_pred = grid_iou(gt_grid, pred_grid)
    iou_teunet = grid_iou(gt_grid, teunet_grid)
    tier = tier_of(i)
    results.append((tier, iou_pred, iou_teunet))
    print(f'[{tier:8s}] {stem:20s} IoU(pred,GT)={iou_pred:.3f}  IoU(TEUNet,GT)={iou_teunet:.3f}  diff={iou_pred-iou_teunet:+.3f}')

print()
for tier in ['good', 'moderate', 'poor']:
    sub = [(p, t) for tr, p, t in results if tr == tier]
    if not sub:
        continue
    avg_pred = sum(p for p, t in sub) / len(sub)
    avg_teunet = sum(t for p, t in sub) / len(sub)
    print(f'{tier:8s} (n={len(sub)}): avg IoU(pred,GT)={avg_pred:.3f}  avg IoU(TEUNet,GT)={avg_teunet:.3f}  diff={avg_pred-avg_teunet:+.3f}')

avg_pred_all = sum(p for _, p, t in results) / len(results)
avg_teunet_all = sum(t for _, p, t in results) / len(results)
print(f'\nOVERALL (n={len(results)}): avg IoU(pred,GT)={avg_pred_all:.3f}  avg IoU(TEUNet,GT)={avg_teunet_all:.3f}  diff={avg_pred_all-avg_teunet_all:+.3f}')
