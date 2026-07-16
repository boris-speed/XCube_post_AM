# corr_medium Step1 voxel-radius H5 dataset

This package contains voxel-level pipe volumes for 10,000 newly generated
`corr_medium` Step1/FDTD samples. The two H5 files use the same sample order and
matching `sample_id` values.

## Files

- `corr_medium_gt_voxel_radius.h5`
  - Ground-truth voxel tube labels for the Step1 scan/core region only.
  - This is not the full larger FDTD simulation region.
- `corr_medium_step1_pred_voxel_radius.h5`
  - Step1 model predictions converted into final radius-expanded voxel tubes.
  - Generated with checkpoint `gpr_topo_step1/runs/real_v9/best.pt`.

## Dataset

- Source samples: `data_generation/datasets_corruption/corr_medium`
- Count: 10,000
- Sample IDs: `corr_medium_000000` to `corr_medium_009999`
- Seeds: `100000` to `109999`
- Difficulty: medium
- Distribution: in-distribution relative to the main Step1 training data; this is
  not an OOD set.
- Volume shape: `(48, 64, 64)`, stored as `[D=z, H=y, W=x]`
- Voxel spacing: `0.005 m`

## H5 Layout

Both H5 files contain:

- `sample_id`: sample identifier
- `sample_dir`: source sample directory
- `source_index`: source index
- `done`: whether the sample was successfully written
- `pipe_mask`: final voxel-level pipe mask, shape `(N, 48, 64, 64)`, bool
- `radius_m`: voxel-level pipe radius in meters, shape `(N, 48, 64, 64)`,
  float16; zero outside `pipe_mask`
- `material`: voxel-level material class, shape `(N, 48, 64, 64)`, int16; `-1`
  outside `pipe_mask`

The prediction H5 also contains:

- `confidence`: prediction confidence from the dense pipe probability field,
  shape `(N, 48, 64, 64)`, float16; zero outside `pipe_mask`

The GT and prediction files are aligned by `sample_id`; index `i` in the GT file
corresponds to index `i` in the prediction file.

## Important Notes

The final prediction dataset was not produced by structured graph generation of
the radii. Although the values come from a trained model, this exported dataset
uses only the dense-head outputs. The other structured components mainly helped
train the model better; they are not used here to construct a graph and then
generate the radius-expanded prediction volume.

This dataset contains only medium-difficulty samples. Medium data is the main
type in the Step1 training set, and this `corr_medium` set is in-distribution,
not OOD. I checked for direct data leakage carefully: train/test paths do not
overlap with `corr_medium`, seeds do not overlap, and exact normalized
`scene.json` hashes do not overlap. Even so, because the set is in-distribution,
the prediction quality is expected to look strong. A diffusion model may not
necessarily improve these results much.

Observed severe failure modes:

1. Pipes near the bottom of a patch are sometimes missed entirely.
2. Pipes cut by the patch boundary can appear as partial half-pipes with poor
   imaging quality.

Both failure modes may also be difficult for a diffusion model to identify or
repair reliably.

## Current Voxel Dice Check

For the full 10,000 samples, using the exported radius-expanded voxel masks:

- mean Dice: `0.805`
- median Dice: `0.829`
- p90 Dice: `0.912`
- p95 Dice: `0.925`
- best Dice: `0.990`
- worst Dice: `0.234`

These are voxel-level radius-expanded scores. They are more forgiving than
centerline, junction, or graph-topology metrics, especially because the
prediction volumes are slightly thicker on average than the GT volumes.