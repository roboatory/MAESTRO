# MAESTRO

<img src="assets/maestro-architecture.png" alt="MAESTRO architecture" width="300px" align="right"/>

**MA**sked **E**ncoding **S**et **TR**ansformer with self-distillati**O**n learns
one fixed-length representation of an entire cytometry sample from its unordered
set of cells. The representation supports patient-level prediction,
visualization, and cell-population interpretation without requiring cell labels
during pretraining.

<br clear="right"/>

## Repository layout

```text
maestro/
├── assets/                         # Figures used by project documentation
├── notebooks/                      # Exploration and analysis notebooks
├── scripts/                        # Standalone training and evaluation drivers
├── src/
│   ├── data/                       # Dataset loading and preprocessing logic
│   ├── models/                     # MAESTRO architecture
│   └── training/                   # Callbacks and distributed configuration
├── .pre-commit-config.yaml         # Ruff and repository hygiene hooks
├── .python-version                 # Python version used by uv
├── environment.yml                # Reproducible CUDA/Conda environment
├── pyproject.toml                  # Package and tooling configuration
└── uv.lock                         # Cross-platform locked dependencies
```

The local `data/`, `docs/`, and `.agents/` directories are ignored by Git.
Generated experiment outputs are also excluded from version control.

## Data organization

Current model-ready cohorts live under `data/processed/`:

- `amp/`: AMP response manifests and subject-level labels
- `cyto-diffusion-bridge/`: cyDM-corrected cohort representations
- `cyto-diffusion-prism/`: PRISM/imputed cohort representations

Original AMP B-cell and T-cell FCS files live under `data/raw/amp/`.

Directory labels use lowercase hyphenated names. Individual source-data
filenames remain unchanged. The entire `data/` directory is local-only and is
not included in the GitHub repository.

## Environment

Create the locked project environment with `uv`:

```bash
uv sync --all-groups
```

The lock targets the Python and core ML versions used by the original MAESTRO
implementation. The existing `environment.yml` remains available as a legacy,
Linux/CUDA-specific environment export.

Run repository checks through the locked environment:

```bash
uv run pre-commit run --all-files
uv run pytest
```

## Train MAESTRO

The Python entry point is a driver only; model, data, and training logic live
under `src/`. Cluster launchers live under `jobs/`.

```bash
bash scripts/train-maestro.sh
```

Submit the B200 smoke-test job with:

```bash
sbatch jobs/train-maestro-toy.slurm
```

Or invoke the driver directly:

```bash
uv run maestro-train \
    --project maestro-allof \
    --devices 0 \
    --data_dirs data/processed/cyto-diffusion-bridge/allof/allof-allof \
    --epochs 500 \
    --mode Train
```

Training outputs are written to `experiments/<project>/`.

## Notebooks

- `01-inventory-original-allof-prepro.ipynb`: inventory and reconcile the
  available legacy whole-blood pretraining snapshot
- `02-explore-original-allof-prepro-cells.ipynb`: memory-safe marker,
  population, projection, and subsampling EDA
- `analyze-maestro-results.ipynb`: reconstruct cells, extract embeddings, run
  downstream models, and inspect pooling attention

Launch notebooks from their directory so relative paths resolve consistently:

```bash
cd notebooks
uv run --group analysis jupyter notebook
```

## License

See [LICENSE](LICENSE).
