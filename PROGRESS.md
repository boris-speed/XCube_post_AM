# GPR Diffusion-Restoration Project — Progress Log

## Goal

A conditional diffusion model that denoises random noise into the correct
ground-truth (GT) tube/pipe shape, conditioned on TEUNet's flawed reconstruction
of the same subsurface GPR scan. Two stages:

- **Stage 1**: a VAE learns a latent space of real tube geometry by
  self-reconstructing GT shapes.
- **Stage 2**: a conditional diffusion model denoises into that latent space,
  conditioned on TEUNet's (imperfect) output.

The research theme specifically calls for diffusion rather than plain
regression, since regression tends to average over multiple plausible
corrections into one blurry compromise, while diffusion can commit to one
sharp, plausible answer per sample.

## Source data

`/home/ameliacatala/Documents/preprocess/transfer_data/{teunet,gt}/` — 1973
paired patches (TEUNet probability grid + binary GT grid), 64x64x48 voxels,
~5mm resolution. Dice score distribution: good (>0.8): 1257, moderate
(0.5-0.8): 455, poor (<0.5): 261.

## Environment setup

- Working conda env: **`preproc`** (not `base`, not `gpr310` — those had an
  unrelated PyPI package literally named `fvdb`, a FAISS-based vector DB tool,
  shadowing the real NVIDIA fVDB library).
- The publicly installable `fvdb_core` package is missing `fvdb.nn.VDBTensor`,
  required throughout XCube. Fixed by building the real fVDB from source, from
  a specific historical PR of `AcademySoftwareFoundation/openvdb`
  (`pull/1808/head`, branch `feature/fvdb`), cloned into `openvdb/` (gitignored,
  not part of this repo — a separate third-party library).
- Building fVDB from source needed 4 patches for libtorch-version drift
  (`torch::linalg::inv` removed, a CuBLAS reduction option enum change, a
  duplicate pybind11 type caster, and CUDA include-path environment variables).
- **Required env var**: `LD_PRELOAD=.../envs/preproc/lib/libstdc++.so.6` (system
  libstdc++ is too old). Originally set via `LD_LIBRARY_PATH` instead, which
  caused a hard-to-diagnose `CUBLAS_STATUS_NOT_INITIALIZED` bug by shadowing
  torch's own bundled cuBLAS. This is now set **automatically** on every
  `conda activate preproc`, via a fixed conda activation hook script
  (`envs/preproc/etc/conda/activate.d/libstdcxx.sh`) — no longer needs to be
  set manually per command.
- CUDA 12.4 JIT compilation needs `PATH=/usr/local/cuda-12.4/bin:$PATH` and
  `CPATH=/usr/local/cuda-12.4/targets/x86_64-linux/include:$CPATH`.

## Preprocessing (`datagen/preprocess_gpr.py`)

Converts the raw TEUNet/GT `.h5` pairs into `.pkl` files (fvdb `GridBatch`
objects) plus stratified train/val/test split lists (80/10/10 per dice tier).

Bugs fixed:
- The real fVDB build has no `GridBatch.from_ijk(...)` classmethod (that's the
  newer `fvdb_core` API) — replaced with `GridBatch()` + `.set_from_ijk(...)`.
- Grid origin convention: XCube expects `origins = voxel_size / 2`, not
  `[0, 0, 0]`. Mismatched origins caused an assertion failure in
  `base_loss.py` during Stage 1 training.

Final dataset: `/home/ameliacatala/Documents/preprocess/data_full/gpr/` — all
1973 samples processed successfully, 0 skipped, all spot-checks passed
(including "moderate"/"poor" tier samples).

## Stage 1: VAE (`configs/gpr/gpr_vae.yaml`, `xcube/models/autoencoder.py`)

Self-reconstructs GT shapes to learn a latent space of valid tube geometry.
Network scaled down from XCube's default (Waymo-scale) sizes to fit GPR's
tiny 64x64x48 grids: `cut_ratio=4`, `f_maps=32`, `c_dim=32`.

Bug fixed: Lightning's automatic `batch_size` inference for `self.log(...)`
fails when a batch contains only custom fvdb objects (no plain tensors) —
fixed by passing `batch_size=out['gt_grid'].grid_count` explicitly everywhere.

**Result (100 epochs, full 1973-sample dataset):**
- Validation loss: 1.53 → 0.27
- Structure accuracy (voxel-level filled/empty correctness): 97.3% → 99.68%
- Curve flattened in the second half — 100 epochs was sufficient, not wasteful.
- Checkpoint: `checkpoints/gpr/VAE_stage1/version_4/checkpoints/last.ckpt`

## Stage 2: Conditional Diffusion (`configs/gpr/gpr_diffusion.yaml`, `xcube/models/diffusion.py`)

### Conditioning design

XCube's diffusion code always treats "the main shape being learned" (read from
`DS.INPUT_PC`) and "the conditioning hint" as two separate fields. GPR's
dataset originally only exposed two fields (`INPUT_PC`, `GT_DENSE_PC`),
neither cleanly mapping to "GT is the target, TEUNet is a separate hint." Fix:

1. Added a new field `DS.COND_PC` (`xcube/data/base.py`) for "a second,
   separate conditioning grid."
2. `xcube/data/gpr.py`: populates `COND_PC` with TEUNet's grid
   (`input_data['input_grid']`), independent of the `input_key` used for the
   main `INPUT_PC` field (which the diffusion config sets to `"target_grid"`,
   i.e. GT, since `extract_latent` always treats `INPUT_PC` as the thing to
   noise/denoise).
3. `xcube/models/diffusion.py`: added `use_cond_grid_concat_cond` — encodes
   TEUNet's grid (`batch[DS.COND_PC]`) through the **same frozen Stage 1 VAE
   encoder** used for the main latent, aligns it onto the noisy latent's own
   sparse grid via `fill_to_grid` (TEUNet's grid generally occupies different
   voxels than GT), then concatenates it as extra channels before denoising.
   Added in 4 places: hparams default, `_forward_cond` (training/sampling),
   `get_dataset_spec`, and `evaluation_api` (real inference, no GT available).

### Other bugs fixed along the way

- Missing dependency `torch_scatter` — not available as a prebuilt wheel for
  this torch/CUDA combo (too new), built from source via pip using the same
  CUDA toolchain env vars as the fVDB build.
- Same Lightning `batch_size` inference bug as Stage 1, fixed the same way
  (`out['log_batch_size'] = bsz` in `forward()`, passed through
  `train_val_step`).
- `torch.load` in PyTorch >=2.6 defaults to `weights_only=True`, which blocks
  loading our own checkpoints (they contain an OmegaConf settings object).
  Fixed in 3 places: `diffusion.py`'s VAE loader, `train.py`'s manual resume
  checkpoint load, and `train.py`'s `trainer.fit(..., ckpt_path=...)` call
  (via a temporarily-scoped `torch.load` monkeypatch, since that one happens
  deep inside the pytorch-lightning library). Safe since it's always our own
  locally-trained checkpoint, never a downloaded one.

### Training results

Ran in two stages (50 epochs, then resumed to 100) on the full dataset:

- Epochs 0-50: validation loss dropped from ~0.99 (a freshly-initialized
  diffusion model's loss is expected to start near 1.0, the variance of
  the noise it's learning to predict) down to a plateau around 0.30.
- Epochs 50-100 (resumed from the epoch-50 checkpoint, no retraining from
  scratch needed): loss stayed in the same ~0.25-0.40 noisy plateau — little
  further improvement, but also **no overfitting** at any point: training and
  validation loss tracked closely together the entire 100 epochs.
- Practical lesson: this setup converges by ~epoch 25-30; the remaining ~75
  epochs of compute mostly just confirmed stability.
- Checkpoint: `checkpoints/gpr/Diffusion_stage2/version_2/checkpoints/last.ckpt`
- Curve plots: `stage1_training_curve.png`,
  `stage2_diffusion_curve.png` (50 epochs), `stage2_diffusion_curve_100ep.png`
  (full 100 epochs) in `~/Documents/`.

## Testing / Evaluation (`scripts/run_stage2_inference.py`)

Ran the trained Stage 2 model on real test samples, comparing its output
against ground truth using IoU (intersection-over-union of occupied voxels),
and against TEUNet's raw output as a baseline.

**Preliminary check (11 samples spread across tiers)** suggested a promising
pattern: roughly even on "good"/"moderate" tiers, but a meaningful improvement
on "poor" tier (+0.038 avg IoU vs. TEUNet baseline).

**Full test set (all 198 samples, run 2026-06-25)** — the real, trustworthy
number; raw output saved in `scripts/results/stage2_full_test_198samples.txt`:

| Tier | n | Avg IoU (model output vs. truth) | Avg IoU (TEUNet vs. truth) | Difference |
|---|---|---|---|---|
| Good | 126 | 0.802 | 0.816 | -0.013 |
| Moderate | 46 | 0.572 | 0.580 | -0.008 |
| Poor | 26 | 0.090 | 0.085 | +0.004 |
| **Overall** | **198** | **0.655** | **0.665** | **-0.010** |

**Honest conclusion: the small-sample improvement on "poor" tier did not hold
up at full scale** (+0.038 on n=3 shrank to +0.004 on n=26) — at full scale,
the model is essentially on par with, or very slightly behind, just using
TEUNet's raw output directly, across every tier. Several poor-tier samples
show TEUNet finding zero overlapping voxels with ground truth at all — cases
where TEUNet's reconstruction failed almost completely, leaving little for the
diffusion model's conditioning signal to work from.

This is a genuine negative result for the current setup, not yet a successful
improvement over baseline.

## Diagnosis (`scripts/test_vae_roundtrip.py`)

To find the actual cause, ran a diagnostic: pass TEUNet's grid through the
frozen Stage 1 VAE's encoder + decoder directly — zero noise, zero diffusion
process at all — and compare to ground truth the same way.

| Tier | TEUNet baseline | VAE round-trip only (no diffusion) | Full diffusion model |
|---|---|---|---|
| Good | 0.816 | 0.808 | 0.802 |
| Moderate | 0.580 | 0.580 | 0.572 |
| Poor | 0.085 | 0.088 | 0.090 |
| **Overall** | **0.665** | **0.661** | **0.655** |

**The VAE round-trip alone performs almost identically to the full trained
diffusion model.** The diffusion process isn't doing meaningful denoising —
it's behaving like it learned to mostly pass the condition straight through.

Confirmed visually too (`scripts/visualize_stage2_sample.py`, now rendering
actual solid voxel cubes instead of scattered dots — much clearer): on a
poor-tier sample, TEUNet's input gets the pipe's right-hand section right but
the left-hand section disintegrates into a scattered, broken cluster. The
model's output keeps the good section, adds a modest number of voxels
overall, but the broken left-hand section stays just as scattered — it never
reorganizes that region into the smooth tube ground truth actually has.

**Root-cause theory (structural confinement):** the VAE's decoder
(`sunet.py:467-512`) grows structure level-by-level, only ever subdividing
cells that are *already part of* the coarse footprint it's handed — it can
never invent occupied space outside that starting footprint (like zooming
into a map: you can add detail inside a region you're already looking at, but
can't discover a city that wasn't on the map at all). During training,
`extract_latent` always uses `DS.INPUT_PC` = ground truth (`input_key:
"target_grid"`), so the model only ever practices "refine an already-correct
neighborhood" — it never sees a *wrong* starting footprint during training.
At real test time, the starting footprint instead comes from encoding
TEUNet's own (possibly wrong) grid. On good/moderate tiers TEUNet's footprint
roughly overlaps GT's, so this barely bites; on poor tier, where TEUNet's
footprint diverges most, the model is structurally boxed into TEUNet's wrong
neighborhood and can't escape it — matching the measured results exactly.

**Fix attempt 1 (tried 2026-06-27): test-time dilation — did not work.**
`scripts/test_dilation_fix.py` dilates TEUNet's grid via fvdb's
`GridBatch.conv_grid(kernel_size, stride=1)` before encoding/conditioning,
giving the decoder a wider candidate region to grow structure into. Tested
both a 1-voxel margin (`kernel_size=3`) and a 3-voxel margin (`kernel_size=7`)
on the same 11-sample tier spread — neither changed results meaningfully
(poor tier: 0.230 baseline vs 0.229 dilated at kernel=7; essentially flat
across all tiers, no improvement at any margin tested).

**Why this null result is itself informative**: it rules out "not enough
room" as the mechanism, and sharpens the diagnosis — the model doesn't just
need more space, it never learned *how* to use an uncertain/wide candidate
region productively. It only ever practiced refining a region it could
already trust was exactly correct (GT's). Given extra room at test time, it
has no learned behavior for filling it in, so it mostly predicts "not
occupied" regardless. This confirms the real fix has to change what the model
practices on during training, not just what it's given at test time.

**Fix attempt 2 (tried 2026-06-27/28): retrain on TEUNet's dilated footprint
— also did NOT work.** Implemented in `xcube/models/diffusion.py`: new
hparams `train_cond_footprint` and `cond_grid_dilation_kernel`, a shared
`encode_cond_grid()` helper (dilates via `conv_grid` then encodes through the
frozen VAE, used consistently in training and at real inference), and a new
branch in `forward()` that uses the dilated-TEUNet encode as the structural
topology + noising target (GT's true features aligned onto it via
`fill_to_grid`) instead of GT's own footprint. New config
`configs/gpr/gpr_diffusion_v2.yaml` (`cond_grid_dilation_kernel: 5`, ~2-voxel
margin), trained 50 epochs on the full dataset (`checkpoints/gpr/Diffusion_stage2_v2/version_2`,
val loss plateaued ~0.28-0.33, comparable to v1). Full 198-sample evaluation
(`scripts/run_stage2_inference_v2.py`):

| Tier | n | v2 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.800 | 0.816 | -0.016 |
| Moderate | 46 | 0.565 | 0.580 | -0.015 |
| Poor | 26 | 0.089 | 0.085 | +0.004 |
| **Overall** | **198** | **0.652** | **0.665** | **-0.013** |

**Essentially identical to v1** (-0.010 → -0.013 overall, poor tier +0.004
unchanged). Two structural-footprint fixes in a row (test-time dilation, and
now retraining on a dilated footprint) produced no improvement at all.

**Updated theory**: combined with the earlier VAE-round-trip finding (frozen
VAE encode/decode alone, zero diffusion, already matches the full model's
performance), the evidence now points less at "the decoder doesn't have
enough room" and more at **the diffusion model learning to just pass the
conditioning hint straight through rather than doing real generative
correction** — changing the footprint doesn't matter if the model was never
forced to rely on anything besides copying its hint in the first place.

**Fix attempt 3 (tried 2026-06-28/29): classifier-free conditioning dropout
— also did NOT meaningfully change anything.** `use_classifier_free` and
`classifier_free_prob` already existed as hparams (unused before); the
dropout mechanism (`conduct_classifier_free`) was already fully wired into
the `use_cond_grid_concat_cond` branch. New config
`configs/gpr/gpr_diffusion_v3.yaml` (built on v1's plain footprint, not v2's
dilated one, to isolate dropout as the only new variable), trained 50 epochs
full dataset (`checkpoints/gpr/Diffusion_stage2_v3/version_0`, val loss
plateau ~0.30-0.34, same range as v1/v2). Full 198-sample evaluation
(`scripts/run_stage2_inference_v3.py`):

| Tier | n | v3 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.806 | 0.816 | -0.010 |
| Moderate | 46 | 0.570 | 0.580 | -0.010 |
| Poor | 26 | 0.090 | 0.085 | +0.005 |
| **Overall** | **198** | **0.657** | **0.665** | **-0.008** |

**The real finding now is the consistency itself**: three structurally
different fixes (test-time dilation, retraining on a dilated footprint,
classifier-free dropout) all land within noise of each other (-0.010, -0.013,
-0.008 overall; poor tier always +0.004 to +0.005). None of our three
theories about the specific mechanism turned out to be the deciding factor.
This looks less like "haven't found the right tweak" and more like this
conditioning setup (concat-based, single frozen-VAE encode of TEUNet's grid,
small network: `model_channels=32`, `channel_mult=[1,2]`, `num_res_blocks=1`,
no attention) has a real ceiling around 0.65-0.66 overall IoU — just shy of
TEUNet's own baseline — regardless of these three training-time
interventions.

**Fix attempt 4 (tried 2026-06-30): free structural generation at inference
— catastrophic failure, IoU ≈ 0.** Implemented in
`scripts/run_stage2_inference_v4.py` (uses the v1 checkpoint, no
retraining). Instead of encoding TEUNet's grid and passing its topology as
`grids`, manually constructs a fully-dense coarse grid matching the VAE's
bottom level (`feat_depth = tree_depth-1 = 1`, `gap_stride = 2`,
`voxel_bound = [32, 32, 24]`, `voxel_sizes = voxel_size * gap_stride`,
`origins = voxel_sizes / 2`) and passes that as `grids`, while still passing
TEUNet's grid as a conditioning hint via `batch={DS.COND_PC: teunet_grid}`.
Full 198-sample eval (`scripts/results/stage2_full_test_v4_198samples.txt`):

| Tier | n | v4 IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.004 | 0.816 | −0.811 |
| Moderate | 46 | 0.002 | 0.580 | −0.578 |
| Poor | 26 | 0.009 | 0.085 | −0.077 |
| **Overall** | **198** | **0.004** | **0.665** | **−0.661** |

The model filled the entire domain with a solid block (~165k voxels)
regardless of sample — visible in `~/Documents/stage2_visual_comparison_multi_v4.png`.
Root cause: the decoder was only ever trained to subdivide GT-shaped sparse
grids; given a fully-dense starting grid it has no learned behavior for
pruning and activates every voxel. This definitively rules out structural
confinement as the fixable bottleneck — the model had full freedom and
performed catastrophically worse, not better.

## Poor-Tier Visual Audit (2026-07-02)

All 26 poor-tier test samples (indices 172–197) rendered side-by-side
(TEUNet input / v1 model output / ground truth) in four batches saved to
`~/Documents/poor_batch{1-4}_*.png`. Four distinct failure patterns
identified:

**Pattern A — Complete spatial miss (IoU = 0.000): 10 of 26 samples (38%)**
TEUNet places voxels in entirely the wrong physical location or finds
near-nothing. Model output stays at 0.000 — no correction is possible
because the conditioning signal carries zero spatial information about the
pipe's actual location. Includes cases (#176, #177, #183, #191) where
TEUNet outputs a thick rectangular slab in a small corner of the domain
while GT has two large parallel cylinders spanning the full length.

**Pattern B — Over-segmentation / wrong shape (IoU ≈ 0.05): 4 of 26 (15%)**
TEUNet finds a large amorphous blob roughly where the pipe is but produces
the wrong shape entirely (flat slab instead of cylinder). Model copies this
with marginal voxel count changes and no structural correction.

**Pattern C — Fragmentation (IoU = 0.10–0.30): 4 of 26 (15%)**
TEUNet finds voxels in the right region but they are scattered and
disconnected. **The only category where the model shows any meaningful
improvement**: #175 (0.242→0.295), #185 (0.241→0.296). Slight
consolidation of the fragmented signal occurs, but never a full clean
tube reconstruction.

**Pattern D — Complex geometry (IoU = 0.08–0.29): 8 of 26 (31%)**
GT contains multi-pipe arrangements, L/T junctions, or curved paths.
TEUNet partially captures these but gets the topology wrong. Model output
is near-identical to TEUNet (±0.01 IoU). Best cases in the entire poor
tier (#187 at 0.301, #192 at 0.278) fall here — TEUNet was close but
noise prevented it crossing the 0.3 threshold.

**Key finding**: Pattern A (38%) is entirely out of reach for any model
that conditions only on TEUNet's output — there is no information in the
conditioning signal about where the pipe is. Patterns B and D suggest the
bottleneck is shape/topology learning capacity. Only Pattern C shows the
model can do anything useful. Accessing pre-TEUNet signal (raw GPR scan or
intermediate representation) is the only realistic path to fixing Pattern A.

## Visualization scripts

| Script | Purpose |
|---|---|
| `scripts/visualize_stage2_sample.py` | Single sample: TEUNet / v1 output / GT |
| `scripts/visualize_stage2_multi.py` | Multi-sample batch: TEUNet / v1 output / GT |
| `scripts/visualize_stage2_multi_v4.py` | Same but uses free-gen coarse grid (fix 4) |

All use matplotlib `ax.voxels()` for filled-cube rendering with a shared
bounding box from GT and a fixed camera angle (elev=20, azim=-60).

## Capacity test (tried 2026-07-03): bigger network, still no improvement

Tested the "network is too small" theory from the earlier "not yet tried"
list: `configs/gpr/gpr_diffusion_v4.yaml` keeps v1's plain GT footprint (no
dilation, no classifier-free dropout, no attention) but quadruples rough
capacity — `model_channels` 32→64, one more depth level (`channel_mult`
[1,2]→[1,2,4]), `num_res_blocks` 1→2. Trained 50 epochs (`batch_size: 2`,
~19hr on the RTX 2000 Ada). Full 198-sample eval
(`scripts/results/stage2_full_test_v4capacity_198samples.txt`):

| Tier | n | v4-capacity IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.780 | 0.816 | −0.036 |
| Moderate | 46 | 0.527 | 0.580 | −0.053 |
| Poor | 26 | 0.089 | 0.085 | +0.003 |
| **Overall** | **198** | **0.630** | **0.665** | **−0.035** |

Not only did more capacity fail to beat TEUNet's baseline, it landed
*below* the ~0.65-0.66 overall IoU that all three training-regime variants
(v1/v2/v3) converged to — network size was not the bottleneck. Combined
with the audit's finding that 38% of poor-tier failures are complete
spatial misses with zero recoverable signal in TEUNet's output, this
closes out the architectural/training-regime angle entirely.

## Post-processing test (tried 2026-07-03): morphological closing on model output

Cheap, no-retraining check: does bridging small gaps (morphological closing,
`scipy.ndimage.binary_closing`, 1-2 iterations) on the v1 model's *output*
voxels recover any IoU, especially on poor-tier fragmented samples?
`scripts/test_postprocess_closing.py`, full 198-sample run
(`scripts/results/postprocess_closing_198samples.txt`): **essentially zero
change everywhere** (good +0.000, moderate -0.000, poor +0.000 to +0.001).
The model's fragmented outputs aren't "almost right, just needs bridging" —
confirms the gaps are too large/structurally different for cheap geometric
cleanup, consistent with an information gap rather than a fixable shape
defect.

## Fix attempt 6 (tried 2026-07-06): hardness-based loss reweighting — also did NOT help

Theory: with every training sample weighted equally, the ~64% "good" tier
(where blind copy-through of TEUNet's hint is already near-optimal)
dominates the average loss, leaving little gradient incentive to learn real
correction behavior on the ~13% "poor" tier where it actually matters. Kept
v1's exact architecture and plain GT footprint (capacity and structural
fixes already ruled out) and changed only the loss: `configs/gpr/gpr_diffusion_v5.yaml`
sets `use_hardness_reweight: true`, `hardness_scale: 3.0`. New
`hardness_scale` param on `GPRDataset` (`xcube/data/gpr.py`) computes
`weight = 1 + hardness_scale * (1 - IoU(TEUNet, GT))` per sample (range
~1.16-3.61 across a random subset, mean ~2.05 — sensible spread, not
degenerate); `use_hardness_reweight` in `xcube/models/diffusion.py` switches
`compute_loss` from a flat mean MSE to a per-voxel-weighted mean (voxel
weight looked up via `jidx` from its sample's `DS.LOSS_WEIGHT`). Trained 50
epochs (`checkpoints/gpr/Diffusion_stage2_v5/version_0`, ~8hr — faster than
v4-capacity since this reused v1's smaller network). Full 198-sample eval
(`scripts/results/stage2_full_test_v5reweight_198samples.txt`):

| Tier | n | v5-reweight IoU vs GT | TEUNet IoU vs GT | Diff |
|---|---|---|---|---|
| Good | 126 | 0.807 | 0.816 | −0.009 |
| Moderate | 46 | 0.573 | 0.580 | −0.007 |
| Poor | 26 | 0.089 | 0.085 | +0.004 |
| **Overall** | **198** | **0.658** | **0.665** | **−0.007** |

Overall lands within the same noise band as v1/v2/v3 (0.655/0.652/0.657) —
no regression like v4-capacity, but no real gain either. **Poor tier is the
key number: +0.004, identical to v1's +0.004 and v2's +0.004/v3's +0.005**
— despite up to ~3.6x more training loss weight on exactly these samples,
zero additional correction ability resulted. Strong evidence the model
isn't failing to prioritize hard cases (which reweighting would fix); it's
that TEUNet's signal genuinely contains no recoverable information for
those cases, so no amount of training emphasis can manufacture a correction
signal that isn't there.

## Status as of today (2026-07-06)

Six fix attempts completed (three training-regime variants, one structural
free-gen experiment, one capacity increase, one loss-reweighting) plus a
full poor-tier visual audit and a post-processing gap-bridging test. All six
training variants land at or below the ~0.65-0.66 ceiling, just shy of
TEUNet's own 0.665 baseline, and the post-processing test rules out cheap
geometric cleanup too. Combined with the audit finding 38% of poor-tier
samples are complete spatial misses with zero recoverable signal, the
conclusion holds firmly: the bottleneck is informational, not architectural,
training-regime, loss-weighting, or post-processing based.

**Waiting on**: raw GPR scan data access (pre-TEUNet signal), expected
~2026-07-05. This is the prerequisite for any meaningful next attempt on
Pattern A (complete miss) samples, which make up the largest fraction of
poor-tier failures. No further training-time or architecture tweaks are
worth trying against TEUNet's output alone until that data lands.

**Update (2026-07-09): the true raw signal would be the A-scan** (the raw
GPR waveform, pre-*any* neural network) — but the data owner declined to
provide A-scan access, for reasons not stated to us. What was provided
instead, below, is another model's (Step1's) processed predictions, not the
raw signal. This matters: it means the original hypothesis behind this whole
data request — that Pattern A misses are fixable given access to genuinely
raw, pre-model information — remains untested. corr_medium is a real new
data source and worth trying, but it does not by itself resolve the
information-gap diagnosis; see the in-distribution caveat below.

## New data source: corr_medium Step1 predictions (arrived 2026-07-07)

Requested as the raw-GPR-signal data; what actually landed was a GT/prediction
pair from a **Step1 model** (a different, earlier-pipeline-stage model — not
TEUNet, and not the raw A-scan — see note above):
`corr_medium_gt_voxel_radius.h5` and `corr_medium_step1_pred_voxel_radius.h5`,
10,000 paired samples, in
`/home/ameliacatala/Documents/corr_medium_gt_voxel_radius/`.

### Preprocessing (`datagen/preprocess_gpr_corr_medium.py`)

Converts this pair into the exact same `.pkl` format `preprocess_gpr.py`
produces, so it drops into the existing `GPRDataset`/Stage 1/Stage 2 code
unchanged. Key differences from the original TEUNet source format:

- Single `.h5` file per split holding all 10,000 samples (vs. one `.h5` file
  per sample originally).
- Volume axis order in the source is (D,H,W) = (48,64,64); transposed to
  (X,Y,Z) = (64,64,48) to match `GPRDataset`'s existing convention.
- Occupancy (`pipe_mask`) is already boolean on both GT and prediction sides
  — no probability threshold needed. The prediction's `confidence` field
  (dense probability at occupied voxels) fills the role TEUNet's continuous
  `input_prob` played before.
- Isotropic 0.005m voxel spacing (vs. TEUNet's near-isotropic but not exactly
  equal per-axis spacing).

Same per-sample Dice-score tiering and 80/10/10 stratified split as the
original pipeline.

**Result**: 10,000/10,000 samples processed, 0 skipped, all spot-checks
passed. Output: `/home/ameliacatala/Documents/preprocess/data_full/gpr_corr_medium/0.005/`
(6.7G), splits `train.lst`/`val.lst`/`test.lst` (8,002/999/999 entries).

Tier breakdown — notably different balance from the original TEUNet dataset:

| Tier | corr_medium (n=10,000) | Original TEUNet (n=1,973) |
|---|---|---|
| Good (>0.8) | 6,143 (61.4%) | 1,257 (63.7%) |
| Moderate (0.5-0.8) | 3,725 (37.3%) | 455 (23.1%) |
| Poor (<0.5) | 132 (1.3%) | 261 (13.2%) |

Poor tier is far smaller proportionally here (1.3% vs. 13.2%).

**Important caveat (from `README_corr_medium_voxel_radius.md`, packaged with
the source data): this is NOT the hoped-for harder/OOD signal.** The dataset
is explicitly medium-difficulty and **in-distribution** relative to the Step1
model's own training set (checkpoint `gpr_topo_step1/runs/real_v9/best.pt`) —
the README states plainly: *"the prediction quality is expected to look
strong. A diffusion model may not necessarily improve these results much."*
So the small poor-tier share above likely reflects a curated, easier subset,
**not** evidence that this Step1 model is a fundamentally richer signal than
TEUNet for the hard Pattern-A (complete spatial miss) cases this project
actually needs solved. The README also names two specific known failure
modes in this data — pipes near a patch's bottom sometimes missed entirely,
and pipes cut by the patch boundary becoming poor-quality half-pipes — both
flagged as likely hard for a diffusion model to fix too.

The README also documents two per-voxel fields present in both H5 files but
**not currently used** by the preprocessing script: `radius_m` (physical pipe
radius per voxel, float16, zero outside `pipe_mask`) and `material` (pipe
material class per voxel, int16, `-1` outside `pipe_mask`). Both are ignored
because the existing `.pkl` schema (`target_grid`/`input_grid`/`input_prob`),
inherited unchanged from the TEUNet pipeline, has no field for them — using
them would require extending the schema and the Stage 1/2 models themselves.

**Next step**: given the in-distribution caveat, treat this dataset as a
useful additional training/validation source rather than a confirmed fix for
Pattern-A failures. Worth checking whether the source data can also provide a
genuinely harder/OOD split before concluding it addresses the poor-tier
information gap identified earlier.

**Verified 2026-07-08**: checked a random 200-sample subset directly -- worst
Dice was 0.39, zero samples at exactly 0.000, consistent with the README's
overall worst of 0.234. Confirms no Pattern-A-style complete misses in this
dataset (unlike the original TEUNet data).

## Material conditioning (implemented 2026-07-08)

`corr_medium`'s H5 files also carry a per-voxel `material` class (int16, `-1`
outside `pipe_mask`; observed classes `{0,1,2,3}` across GT+prediction). A
third field, `radius_m`, was considered but skipped: `pipe_mask` is already a
radius-expanded mask, so `radius_m` is largely redundant with occupancy shape
-- `material` is the genuinely new signal, useful for tasks like Pattern-C
fragment consolidation (matching fragments by material continuity), though it
does **not** address Pattern-A complete misses (both `radius_m` and
`material` are undefined outside `pipe_mask`, so there's still nothing to
condition on where the prediction found zero voxels).

**Key finding that reshaped the implementation**: traced how conditioning
actually reaches the model and found `confidence`/`input_prob` -- present
since the original pipeline -- was **never actually consumed**. It's wired
through `DS.INPUT_INTENSITY`, gated by VAE hparam `use_input_intensity`
(`false` in `gpr_vae.yaml`), and even when enabled, Stage 2's
`encode_cond_grid()` (which reuses the **frozen** Stage 1 VAE encoder to
encode the conditioning grid) never passed intensity data to it at all --
only grid positions. This meant the frozen encoder's first layer (`mix_fc`)
was never trained to accept any extra per-voxel channel, so material
conditioning requires retraining Stage 1's encoder with the new input
dimension, not just a Stage-2-only change. Since Stage 1 needed retraining
anyway (new corr_medium data), this is folded into that retrain rather than
being extra overhead.

### Design

Mirrors the existing `use_input_semantic` pattern in
`xcube/modules/autoencoding/base_encoder.py` (categorical class index through
a learned `nn.Embedding`, concatenated onto the position-embedded features
before `mix_fc`) rather than one-hot encoding manually.

- `xcube/data/base.py`: two new `DatasetSpec` entries --
  `INPUT_MATERIAL` (aligned with whatever `INPUT_PC` currently is: GT's
  material when `input_key="target_grid"`, the Step1 prediction's material
  otherwise) and `COND_MATERIAL` (always the Step1 prediction's material,
  aligned with `COND_PC`, since Stage 2 encodes that grid separately from
  `INPUT_PC` through the same frozen encoder).
- `xcube/modules/autoencoding/base_encoder.py`: new `use_input_material`,
  `num_material`, `dim_material` hparams; `nn.Embedding(num_material,
  dim_material)` on the raw class index, concatenated into `unet_feat`.
- `xcube/models/autoencoder.py`: `get_dataset_spec()` requests
  `DS.INPUT_MATERIAL` when the flag is set.
- `xcube/data/gpr.py`: `_get_item` populates `DS.INPUT_MATERIAL` from
  `target_material` or `input_material` in the `.pkl` depending on
  `input_key`, and `DS.COND_MATERIAL` always from `input_material`.
- `xcube/models/diffusion.py`: new `use_cond_material` hparam;
  `encode_cond_grid()` takes an optional `cond_material` argument and passes
  `DS.INPUT_MATERIAL` through to the frozen VAE's `_encode` call alongside
  `DS.INPUT_PC`; all three call sites (`_forward_cond`, `forward`'s
  `train_cond_footprint` branch, `evaluation_api`) updated; `get_dataset_spec`
  requests `DS.COND_MATERIAL` when the flag is set.
- `datagen/preprocess_gpr_corr_medium.py`: `build_sample()` now also extracts
  `target_material` (GT's material at GT-occupied voxels) and
  `input_material` (Step1 prediction's material at its occupied voxels),
  stored as `int8` (only 4 classes -- keeps the dataset small; grew from 6.7G
  to 7.0G for all 10,000 samples). Regenerated the full dataset with this
  field 2026-07-08.
- New configs: `configs/gpr/gpr_vae_corr_medium.yaml` (Stage 1, retrained on
  corr_medium instead of the original 1973-sample TEUNet/GT set,
  `use_input_material: true`, `num_material: 4`, `dim_material: 8`) and
  `configs/gpr/gpr_diffusion_v6_material.yaml` (Stage 2, `use_cond_material:
  true`, otherwise v1's plain architecture/footprint -- isolates material
  conditioning as the one new variable on top of the new dataset).

Smoke-tested (5 train / 2 val batches) before committing to the full run --
material embedding shapes align correctly, loss decreases normally, no
crashes.

**Unrelated environment fix needed along the way**: the JIT CUDA extension
build (`ext/common`, used by `color_util.py`) started failing with `nvcc`
rejecting the conda env's own bundled `gcc` (14.3.0 -- CUDA 12.4 supports up
to 13) as the host compiler, even though a working cached `.so` already
existed. Root cause not fully pinned down, but coincides with the shared
machine's disk-space issue being resolved by another user (see below) --
possibly a system change invalidated cached intermediate build objects.
Fixed by forcing `CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12` (system compilers,
CUDA-12.4-compatible) when invoking `train.py`. Needed for any future
training run on this machine, not specific to material conditioning.

### Status as of 2026-07-08

Stage 1 retrain running in the user's own terminal (not backgrounded by
Claude this time), **50 epochs** (not 100 -- the original Stage 1 run's 100
epochs was overkill; the loss curve had already flattened well before that,
and 50 was judged sufficient here), `--wname corr_medium_material_50ep`, log
at `~/vae_corr_medium_material_50ep.log`. Measured smoke-test rate (~3.3 it/s,
4,501 steps/epoch) gives an estimate of **~19 hours total** (~22-23
min/epoch, corr_medium's 10,000 samples being ~5x the original dataset).
Also needed an unrelated environment fix along the way (see above): `nvcc`
rejecting the conda env's own gcc 14 as host compiler, fixed via
`CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12`.

Once finished, Stage 2 (`gpr_diffusion_v6_material.yaml`) needs its
`vae_checkpoint` path updated to the real `version_N` output directory
(under `checkpoints/gpr/VAE_stage1_corr_medium/`) before training.

## Aside: shared-machine disk-space incident (2026-07-08)

Unrelated to the research itself, but affected ability to run jobs during
this session: the shared machine's root filesystem hit 926G/926G (15M free),
traced to ~750G used by another user's data that `du`/`lsof` under a
non-root account couldn't fully diagnose. Freed ~12G in the meantime via
safe local cleanup (removed unused `herb-phenology` conda env, stale
training logs, an unrelated project folder). The other user's data was later
cleared on their end, restoring ~462G free. Not logged further here since
it's infra, not modeling -- noted only because it explains why the full
corr_medium preprocessing run initially failed and had to be redone.

## Stage 1 corr_medium+material retrain finished; Stage 2 v6 launched (2026-07-09)

**Stage 1 result**: `checkpoints/gpr/VAE_stage1_corr_medium/version_1`, 50
epochs on the full 10,000-sample corr_medium dataset with material
conditioning. Validation loss 0.85 -> 0.31 (train 1.34 -> 0.27), structure
accuracy (finest tree level, `struct-acc-0`) 98.8% -> 99.6%, still trending
down/up slightly at epoch 49 (not fully flattened, unlike the original
100-epoch Stage 1 run on the smaller dataset). The coarser tree level
(`struct-acc-1`) sits at a trivial flat 100% throughout. Curve plots:
`vae_stage1_corr_medium_training.png`. Before/during/after reconstruction
visual (`scripts/visualize_vae_reconstruction.py`, one GT input sample vs.
the coarse depth-1 structure vs. the final depth-0 reconstruction) saved to
`vae_reconstruction_visual.png` -- reconstruction closely matches the input,
consistent with the measured accuracy.

**Stage 2 v6 (`configs/gpr/gpr_diffusion_v6_material.yaml`)**: updated
`vae_checkpoint` to point at the real `version_1` checkpoint (was still a
`version_0` smoke-test placeholder). Smoke-tested (5 train/2 val batches)
against the new checkpoint -- material embedding + corr_medium dataset load
cleanly, loss behaves as expected for a fresh diffusion model (~0.997).

Launched the real 50-epoch run in the user's own terminal (nohup,
`~/v6_material_training.log`), `checkpoints/gpr/Diffusion_stage2_v6_material/version_0`.
Measured rate ~13.75 it/s, 40,260 steps/epoch -> ~49 min/epoch, so **~41
hours for 50 epochs** (corr_medium's 10,000 samples vs. the original
dataset's 1,973 is the main driver, consistent with Stage 1's own ~5x
slowdown). Given every prior Stage 2 attempt (v1-v5) converged/plateaued by
epoch ~25-30, plan is to check the in-progress checkpoint at that point
(~24h in, not the full 50) and decide whether to let it keep running or stop
-- `last.ckpt` updates continuously so stopping early loses nothing.

**Why this run might still land at the same ~0.65-0.66 ceiling as v1-v5**:
the strongest evidence so far that the model just passes its conditioning
input through rather than doing real generative correction is the VAE
round-trip test (zero diffusion, just encode/decode TEUNet's grid, scored
almost identically to the full trained model) and fix attempt 3 (classifier-free
dropout -- forcibly removing the conditioning signal during training some of
the time -- also didn't unlock any real correction behavior). Neither more
data of a similar statistical character, nor a new conditioning channel
(material class), changes *what the training loss rewards* -- copying
through is only an easy shortcut because the condition is usually already
close to correct, and corr_medium's Step1 predictions are, if anything, even
closer to correct on average than TEUNet's were (poor-tier share 13.2% ->
1.3%), so the shortcut may be even more attractive here, not less.

Countering that: the diagnosis above was built entirely from TEUNet-conditioned
training. If Step1's error profile is qualitatively different (more
small/local imperfections, less of TEUNet's catastrophic complete-miss
failure mode) rather than just quantitatively smaller, the model would be
training on a genuinely different distribution of correction problems than
the one that produced the "it just copies" diagnosis -- so it's a real open
question, not a settled one, until the checkpoint eval actually runs.

**If this also lands flat, remaining untried directions** (in rough order of
promise): (1) request actual A-scan (raw waveform, pre-any-model) access
again -- corr_medium was requested as this but the data owner declined to
provide the true raw signal and gave Step1's processed predictions instead,
so the original hypothesis behind the whole data request is still untested;
(2) train against a *corrupted* conditioning signal (not just occasionally
absent, as classifier-free dropout already tried) so blind copy-through
reliably produces high loss, forcing the network away from the shortcut;
(3) check whether the model has hidden correction capacity being averaged
away by deterministic single-sample DDIM evaluation, by sampling multiple
times per input and inspecting variance; (4) reframe the deliverable around
the diagnosis itself (a rigorous elimination of architecture/training-regime/
data-scale causes, isolating the bottleneck as informational) rather than a
successful IoU improvement, if that fits the project's actual goal.
