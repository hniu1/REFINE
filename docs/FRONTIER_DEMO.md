# REFINE Transformer: Frontier demo and team operations

## Frontier inference demo

Downscale daily Daymet `tmin`, `tmax`, and `prcp` from **1/4° to 1/24° (6×)**,
approximately 25 km to 4 km, using the supplied pretrained terrain-aware REFINE
transformer. This branch contains one downscaling stage.

**Repository authors:** Haoran Niu and Deeksha Rastogi.

**Data provenance:** Original [Daymet data](https://daymet.ornl.gov) (Thornton et al., 2021)
are available at 1 km resolution. For this demo, those data have been coarsened to approximately 4 km (1/24°) and 25 km (1/4°), providing the fine-resolution reference and coarse model inputs, respectively.

## Start here

The Daymet demo data are already staged on Frontier at
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1/daymet`.
Pretrained models are under the same demo directory's `artifacts/` folder;
use `artifacts/runs/refine_stage2_v1/best.pt` for the 25 → 4 km workflow.
No additional data download or preparation is needed for this demo.

Run from the shared demo repository root on Frontier:

```bash
cd /lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1
source scripts/frontier_env.sh
export MV_CHECKPOINT="$PWD/artifacts/runs/refine_stage2_v1/best.pt"
python scripts/check_demo.py
mkdir -p logs
sbatch slurm/04_infer.slurm
```

The inference job requests one node, one GPU, and 30 minutes. It processes
**1990-01-01** by default and writes `artifacts/demo/inference-JOBID.nc` plus a
JSON run record. Expect about 51 MB of uncompressed output per day. Use
`MV_END_INDEX=7 sbatch slurm/04_infer.slurm` for the first week; indexes are
zero-based and the end is exclusive. Full-year output is about 18.5 GB before
compression and needs a longer wall time. Submit only within your authorized
Frontier allocation (`cli138` in these launchers).

For a direct command inside an allocated GPU session:

```bash
python pipeline_04_infer.py \
  --data-dir daymet/prepared \
  --checkpoint artifacts/runs/refine_stage2_v1/best.pt \
  --input tmin=daymet/data/Daymet_ERA5_tmin_dy_1990_0p25deg.nc \
  --input tmax=daymet/data/Daymet_ERA5_tmax_dy_1990_0p25deg.nc \
  --input prcp=daymet/data/Daymet_ERA5_prcp_dy_1990_0p25deg.nc \
  --output artifacts/demo/day1.nc --start-date 1990-01-01 \
  --end-index 1 --batch-size 1 --amp --enforce-temperature-order
```

The output uses `time,y,x` dimensions, Celsius and mm/day; y/x are grid indexes,
not geographic coordinates. The native grid coordinates remain in the Daymet
files. Inputs must already use the trained grid and consecutive daily Gregorian
timestamps; `--start-date` describes input index zero. No regridding occurs.

## Shared assets and portable handoff

The following paths are relative to
`/lustre/orion/world-shared/cli138/haoran/GM_Downscaling_demo1`:

- `daymet/data/`: 1980–1990 coarse inputs and coarsened 4 km reference fields.
- `daymet/dem/`: copied DEMs at both resolutions.
- `daymet/prepared/`: full Stage 2 manifest, terrain, coordinates, masks and train/val/test time indexes.
- `daymet/normalization.json`: frozen training normalization matching the checkpoint.
- `artifacts/runs/refine_stage2_v1/best.pt`: pretrained 6× model for inference.
- `artifacts/runs/refine_stage2_v1/last.pt`: saved training state for resuming.
- `checkpoints/refine_6x.pt`: identical inference checkpoint copy used by the portable package and default CLI settings.
- `docs/reference_1990/`: historical evaluation metrics and compact spatial products.

Manifest source paths resolve relative to the manifest directory. CLI paths are
relative to the repository working directory. The package can be moved as a unit.
Large binary assets are ignored by Git; **a Git clone alone is not the complete
demo**. The team can use the shared Frontier assets directly. For a separate,
self-contained copy, give the team the archive produced by:

```bash
python scripts/package_demo.py
```

This includes current source, skills, data, weights and reference diagnostics,
excluding Git internals, old artifacts, archives and logs. See
[daymet/README.md](../daymet/README.md) for local asset paths and provenance. Run `python scripts/check_demo.py` after unpacking.

## Experiment workflow

1. Validate the package and input data.
2. Run one-day inference, then increase the interval as needed.
3. Evaluate against the included 1990 truth, reporting the interval and bilinear baseline:

```bash
python pipeline_03_evaluate.py --data-dir daymet/prepared \
  --checkpoint artifacts/runs/refine_stage2_v1/best.pt --output-dir artifacts/demo/evaluation \
  --split test --max-days 1 --batch-size 1 --amp --enforce-temperature-order
```

4. Produce comparison plots from those evaluation predictions:

```bash
python utility_plot_spatial_statistics.py --data-dir daymet/prepared \
  --evaluation-dir artifacts/demo/evaluation --split test --tile-rows 21
```

A one-day smoke test is not the historical full-year evaluation. Reference metrics
and their provenance are in [docs/EVALUATION_1990.md](EVALUATION_1990.md).
For full-year metrics without predictions use `slurm/03_evaluate.slurm` with
`MV_CHECKPOINT` exported as in the quickstart.
Use a fresh output directory for each experiment.

## Team skills and training

[skills/README.md](../skills/README.md) routes inference, evaluation, and Frontier
operations. [skill-authoring-kit](../skill-authoring-kit/README.md) supplies structured
request/result schemas and a model registry for agent integration. Its `stage2`
identifier is retained for checkpoint compatibility and means the sole 6× route.

Training remains available through `pipeline_01_prepare.py`,
`pipeline_02_train.py`, and the operations skill's
[training reference](../skills/refine-ops/references/training.md). The demo includes
the full Stage 2 index at `daymet/prepared/` and 1980–1990 paired inputs in
`daymet/data/`. All workflows use this same prepared directory.
The training launcher starts from scratch unless `MV_RESUME` is supplied; it has
no dependency on a 100→25 km model. Model changes belong in
`refine_downscaling/model.py`; preserve the released checkpoint for comparison.

## Environment and verification

`scripts/frontier_env.sh` activates the existing Frontier environment. The tested
stack is Python 3.12, ROCm 6.4.1 and PyTorch 2.8.0+rocm6.4. Direct dependencies
are pinned in `requirements.txt`; `requirements-lock.txt` records the original
full environment. Set `REFINE_ENV` to another compatible environment if needed.

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

Launchers accept `MV_BASE_DIR`, `MV_DATA_DIR`, `MV_RUN_DIR`, and `MV_CHECKPOINT`.
Inference also accepts `MV_TMIN_INPUT`, `MV_TMAX_INPUT`, `MV_PRCP_INPUT`,
`MV_OUTPUT`, `MV_START_INDEX`, and `MV_END_INDEX` (for the bundled 1990 year).

Data inputs must use the repository-local `daymet/` assets: `daymet/data/`,
`daymet/dem/`, and `daymet/prepared/` for all workflows. Use
`artifacts/runs/refine_stage2_v1/best.pt` in the shared demo, or the identical
`checkpoints/refine_6x.pt` copy in the portable package, for the released model.
Original paths in provenance records are not runtime inputs or fallbacks.
The shared Frontier software environment supplies dependencies only.

## References

Thornton, P.E., Shrestha, R., Thornton, M. et al. [Gridded daily weather data for North America with comprehensive uncertainty quantification](https://doi.org/10.1038/s41597-021-00973-0). Sci Data 8, 190 (2021).
