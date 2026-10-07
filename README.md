# REFINE

**Resolution-Enhancement Framework Integrating Artificial Intelligence for Natural and Energy Systems**

REFINE is a platform for AI-based spatial resolution enhancement and downscaling of environmental data. It is intended to bring multiple model products under one umbrella, with workflows for data preparation, training, evaluation, and inference for natural and energy systems applications.

The current release implements **REFINE Transformer**, a terrain-aware model that jointly downscales daily minimum temperature, maximum temperature, and precipitation. Additional products, including REFINE SRCNN and REFINE SRGAN, are part of the platform direction.

## Model products

| Product | Approach | Status in this repository |
| --- | --- | --- |
| **REFINE Transformer** | Multivariable transformer with terrain and seasonal information | Implemented; preparation, training, evaluation, and inference available |
| **REFINE SRCNN** | Super-resolution convolutional neural network | check my other repo |
| **REFINE SRGAN** | Super-resolution generative adversarial network | check my other repo |

The guide below describes REFINE Transformer. The current command-line pipeline performs a single **6× downscaling step**, from 1/4° to 1/24° (approximately 25 km to 4 km). The `stage2` name in some files and checkpoint metadata is a compatibility identifier; this workflow does not require an earlier model or stage.

## How REFINE Transformer works

The model combines coarse-resolution weather variables with elevation, normalized grid coordinates, and seasonal information. A shared variable-aware SwinV2-style encoder learns relationships across variables, terrain-aware upsampling reconstructs the finer grid, and separate reconstruction heads produce each output variable. Predictions are learned as residuals over a bilinearly interpolated input.

Training uses spatial patches with a surrounding halo to provide context. The objective combines reconstruction and spatial-gradient losses with penalties for temperature ordering and precipitation conservation. These penalties encourage physical consistency; they do not guarantee every constraint in the output. Evaluation and inference offer an optional temperature-order correction.

The example dataset is derived from [Daymet](https://daymet.ornl.gov/): the original 1 km data have been coarsened to approximately 25 km for inputs and 4 km for fine-resolution references. The model learns coarse-to-fine reconstruction from those paired fields.

## User guide

### 1. Clone the repository

```bash
git clone https://github.com/hniu1/REFINE.git
cd REFINE
```

Run the remaining commands from the repository root. A clone includes source code, normalization metadata, and reference reports. **Training data, terrain arrays, and pretrained weights must be obtained separately**; large binary assets are excluded from Git.

### 2. Set up the environment

The tested environment is Linux with Python 3.12, PyTorch 2.8.0, and AMD ROCm 6.4.1 on Frontier. A GPU is recommended for training; CPU execution is available for small checks. The CUDA and CPU installation options below have not been validated as the released Frontier environment.

Create an isolated Python 3.12 environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For the tested AMD ROCm stack:

```bash
python -m pip install -r requirements.txt
```

For an NVIDIA GPU or CPU environment, install the appropriate PyTorch build instead of the ROCm-specific `requirements.txt`:

```bash
# NVIDIA example: select a build compatible with your GPU driver
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# CPU alternative: use this instead of the NVIDIA command
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Remaining runtime dependencies
python -m pip install numpy==2.4.2 netCDF4==1.7.4 cftime==1.6.5 matplotlib==3.10.8
```

See the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v280) for supported builds and platforms. `requirements-lock.txt` records the original Frontier environment; it is a reproducibility reference rather than a portable installation recipe.

Check the active environment:

```bash
python -c "import torch, numpy, netCDF4; print('PyTorch:', torch.__version__); print('GPU available:', torch.cuda.is_available())"
```

PyTorch uses `torch.cuda` for both NVIDIA CUDA and AMD ROCm devices. On Frontier, users with access to an existing compatible environment can instead run `source scripts/frontier_env.sh`; set `REFINE_ENV` to their own environment path.

### 3. Arrange the data

Obtain the paired Daymet demo assets from the repository authors or an authorized shared installation. This repository does not provide a public data or checkpoint download endpoint. Downloading native Daymet alone does not produce the paired grids required below; coarsening and grid alignment must already have been performed.

For the default experiment, arrange the assets as follows:

```text
daymet/
├── data/
│   ├── Daymet_ERA5_tmin_dy_1980_0p25deg.nc
│   ├── Daymet_ERA5_tmin_dy_1980_trim.nc
│   └── ... paired files for tmin, tmax, prcp and each year through 1990
├── dem/
│   ├── VICa_DEM_0p25deg_fill0.nc
│   └── VICa_DEM_trim_fill0.nc
├── normalization.json
└── prepared/                    # populated by preparation
```

Each weather file must contain a variable named `{variable}_dy`, such as `tmin_dy`, ordered as `[time, y, x]`. The `0p25deg` files are coarse inputs; `trim` files are fine-resolution references. The fine grid must have exactly six times as many cells along each spatial dimension. Paired fields must share the region, grid orientation, and daily time ordering. All variables and years must use consistent grids. DEM files must contain a two-dimensional `DEM` field matching their respective grids.

Temperature units must be supported Celsius or Kelvin units; Kelvin values are converted to Celsius. Precipitation must be daily amounts in supported mm/day units. The reader clamps negative precipitation to zero and excludes missing values from the training loss.

The supplied `normalization.json` contains transforms fitted on the demo training years, 1980–1987. Temperature uses standardization; precipitation uses `log1p` followed by standardization. For a new domain or variable set, fit appropriate transforms on **training data only** and supply a matching normalization JSON. Preparation reuses supplied transforms; it does not fit them or regrid source data.

For demo asset locations and provenance, see [Daymet data documentation](daymet/README.md). Users with Frontier access can follow the [Frontier demo guide](docs/FRONTIER_DEMO.md).

### 4. Prepare the dataset

```bash
python pipeline_01_prepare.py \
  --data-root daymet/data \
  --dem-root daymet/dem \
  --normalization-manifest daymet/normalization.json \
  --output-dir daymet/prepared \
  --variables tmin tmax prcp \
  --train-start 1980 --train-end 1988 \
  --val-start 1988 --val-end 1990 \
  --test-start 1990 --test-end 1991
```

Year-end arguments are exclusive:

| Split | Years | Purpose |
| --- | --- | --- |
| Training | 1980–1987 | Learn model parameters |
| Validation | 1988–1989 | Select checkpoints and control early stopping |
| Test | 1990 | Evaluate held-out performance |

Preparation writes `daymet/prepared/manifest.json`, static terrain and coordinate arrays, masks, and split time indexes. Weather fields remain in source NetCDF files and are loaded as needed, avoiding a full duplicated dataset. Keep the source files available after preparation. Manifest source paths resolve relative to the prepared directory.

A complete matching prepared dataset can be reused. Keep held-out years out of normalization fitting and model selection.

### 5. Train REFINE Transformer

Start a single-GPU experiment from scratch:

```bash
python pipeline_02_train.py \
  --data-dir daymet/prepared \
  --run-dir artifacts/runs/refine_transformer_v1 \
  --epochs 100 --batch-size 4 \
  --patches-per-day 8 --validation-patches-per-day 4 \
  --core-size 8 --halo 2 \
  --embed-dim 96 \
  --num-groups 6 --blocks-per-group 6 \
  --num-heads 6 --window-size 8 \
  --learning-rate 2e-4 --warmup-epochs 5 \
  --gradient-clip 1.0 --amp
```

`--batch-size` is per process. Core size 8 with halo 2 gives a 12×12 coarse input patch and a 72×72 fine output patch; the loss is calculated on the central core. Reduce batch size if GPU memory is limited. `--amp` enables bfloat16 autocasting on GPU and can be omitted if the hardware does not support it. The default early-stop patience is 20 epochs.

Training uses all prepared variables by default. `--variables` can select a subset for a separate experiment; its order defines model channels. Use a new run directory for each experiment.

| Run artifact | Contents |
| --- | --- |
| `best.pt` | Checkpoint with the lowest validation objective |
| `last.pt` | Latest checkpoint, including optimizer and scheduler state |
| `history.jsonl` | Per-epoch training and validation losses |
| `run_config.json` | Model, data, and run configuration |

Resume the example with its original architecture and patch settings:

```bash
python pipeline_02_train.py \
  --data-dir daymet/prepared \
  --run-dir artifacts/runs/refine_transformer_v1 \
  --resume artifacts/runs/refine_transformer_v1/last.pt \
  --epochs 100 --amp
```

The defaults match the example above. If you changed model or patch settings, repeat them when resuming; the checkpoint model configuration must match. `--epochs` is the total target epoch count, not the number of additional epochs, and must exceed the saved epoch for training to continue.

For a fresh experiment initialized with compatible encoder weights, use `--init-backbone /path/to/checkpoint.pt` and a new run directory. This initializes compatible representation layers while leaving the reconstruction heads and optimizer fresh.

For multiple GPUs on one machine, use PyTorch DistributedDataParallel through `torchrun`. For example, on four GPUs:

```bash
torchrun --standalone --nproc-per-node=4 pipeline_02_train.py \
  --data-dir daymet/prepared \
  --run-dir artifacts/runs/refine_transformer_ddp \
  --epochs 100 --batch-size 4 --amp
```

Frontier launchers are provided in `slurm/`. Configure the account, partition, resources, and environment for your allocation before submitting. `slurm/02_train.slurm` currently requests two nodes with four GPU tasks per node; `submit_pipeline.sh` chains preparation, training, and evaluation. See the [Frontier demo guide](docs/FRONTIER_DEMO.md) for site-specific operations.

### 6. Evaluate the trained model

Start with a one-day check:

```bash
python pipeline_03_evaluate.py \
  --data-dir daymet/prepared \
  --checkpoint artifacts/runs/refine_transformer_v1/best.pt \
  --output-dir artifacts/evaluation/refine_transformer_v1_day1 \
  --split test --max-days 1 --batch-size 1 --amp \
  --enforce-temperature-order
```

Evaluation reports per-variable errors in physical units and compares the model with bilinear interpolation. It writes `evaluation_summary.json` and saves predictions unless `--no-save-predictions` is supplied. For the full held-out split, remove `--max-days 1` and choose a fresh output directory; a one-day check does not establish full-year performance.

Historical demo results and provenance are documented in [the 1990 evaluation report](docs/EVALUATION_1990.md). Comparisons should use the same data, time interval, normalization, and postprocessing settings.

### 7. Run inference

Use a trained checkpoint and the matching prepared manifest and static fields:

```bash
python pipeline_04_infer.py \
  --data-dir daymet/prepared \
  --checkpoint artifacts/runs/refine_transformer_v1/best.pt \
  --input tmin=daymet/data/Daymet_ERA5_tmin_dy_1990_0p25deg.nc \
  --input tmax=daymet/data/Daymet_ERA5_tmax_dy_1990_0p25deg.nc \
  --input prcp=daymet/data/Daymet_ERA5_prcp_dy_1990_0p25deg.nc \
  --output artifacts/inference/refine_transformer_v1_day1.nc \
  --start-date 1990-01-01 --end-index 1 \
  --batch-size 1 --amp --enforce-temperature-order
```

Supply one `--input` per checkpoint variable. Inputs must already match the trained coarse grid and contain consecutive daily Gregorian timesteps. `--start-date` describes input index zero; indexes are zero-based and `--end-index` is exclusive. No regridding occurs. Checkpoint and manifest normalization must match.

The NetCDF output uses `[time, y, x]` dimensions, Celsius for temperature, and mm/day for precipitation, with a companion JSON run record. Output `y` and `x` are grid indexes; retain the source-grid coordinates for geographic interpretation. Use a new output filename for each run.

## Repository layout

| Path | Purpose |
| --- | --- |
| `refine_downscaling/` | Transformer model, datasets, transforms, losses, and distributed helpers |
| `pipeline_01_prepare.py` | Prepare the paired-data index and static fields |
| `pipeline_02_train.py` | Train or resume a model |
| `pipeline_03_evaluate.py` | Evaluate against fine references and bilinear interpolation |
| `pipeline_04_infer.py` | Downscale coarse input files |
| `utility_plot_*.py` | Training-history and spatial diagnostic plotting utilities |
| `tests/` | Synthetic-data checks for the model and pipeline |
| `slurm/`, `scripts/` | Cluster launchers and demo packaging utilities |
| `docs/` | Evaluation reports and Frontier demo instructions |

For a local code check in the configured environment:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

## Automation and agent integration

The command-line workflow can be used directly or integrated into automation. Optional [skills](skills/README.md) document operations, inference, and evaluation. The [skill-authoring kit](skill-authoring-kit/README.md) provides request/result schemas and a model registry for agent integration. These resources complement the user guide; they are not required to train or run the model manually.

## Authors and references

Repository authors: **Haoran Niu and Deeksha Rastogi**.

Daymet data reference: Thornton, P. E., Shrestha, R., Thornton, M. et al. [Gridded daily weather data for North America with comprehensive uncertainty quantification](https://doi.org/10.1038/s41597-021-00973-0). *Scientific Data* **8**, 190 (2021).
