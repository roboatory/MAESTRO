# MAESTRO repository guide

This file is the shared operating context for coding agents and maintainers. Treat
the checked-out repository and its local data as the ground truth. Do not revive
files, paths, configurations, or assumptions from older chats without verifying
them against the current tree.

## Project purpose

MAESTRO (Masked Encoding Set Transformer with self-distillation) learns one
fixed-length representation of a complete cytometry sample. It treats each sample
as an unordered set of cells, trains without cell or patient labels, and combines
cell-set reconstruction with momentum-teacher self-distillation.

The immediate goal is a single, reproducible pretraining run that follows the
latest validated `allof + prepro` configuration. There are intentionally no sweep
files or alternate Hydra experiment variants in this repository.

## Source of truth and repository layout

- `configs/train.yaml`: the one supported Hydra training configuration.
- `jobs/train-maestro.slurm`: the one supported cluster launcher.
- `src/hydra_cli.py`: converts the resolved Hydra document into the training
  dataclass and starts training.
- `src/training/runner.py`: data construction, Lightning/DeepSpeed setup,
  checkpoints, CSV/W&B logging, and run provenance.
- `src/models/maestro.py`: masking, set-transformer, teacher/student losses, and
  Lightning module.
- `src/data/dataset.py`: HDF5 loading, marker canonicalization/intersection, and
  per-sample cell sampling.
- `src/training/callbacks.py`: DeepSpeed configuration, teacher EMA, and
  Sinkhorn-phase checkpoint behavior.
- `tests/test_hydra_config.py`: fixed-config and tracking/provenance tests.
- `notebooks/`: data inventory, cell-level EDA, and result analysis.
- `data/`, `docs/`, `experiments/`, `logs/`, and `output/` are local-only and
  ignored by Git. Never assume they exist in a fresh clone.

`README.md` is intentionally absent. Do not recreate it unless explicitly asked.
`CLAUDE.md` must remain a symlink to this file so both tools receive identical
instructions.

## Environment

- Python: `>=3.10` (the current uv environment uses Python 3.12).
- Dependency manager: `uv`; `uv.lock` is authoritative.
- Core training stack: PyTorch 2.7.1 with CUDA 12.8 wheels, Lightning 2.4.0,
  DeepSpeed 0.19.2, Hydra 1.3.2, GeomLoss/PyKeOps, and W&B.
- Install all local tooling with `uv sync --frozen --all-groups`.
- The Slurm job performs `uv sync --frozen --no-dev --inexact` before launching
  ranks. Keep `--inexact`: removing analysis packages from the shared `.venv`
  added about 50 seconds of avoidable startup time in an earlier run.

Before committing Python changes, run:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest -q
bash -n jobs/train-maestro.slurm
uv run maestro-run --cfg job --resolve
```

Ruff targets Python 3.10 with an 88-character line length. Preserve the existing
per-file naming exceptions for the model and callbacks.

## Current fixed training configuration

`configs/train.yaml` currently resolves to:

- Inputs: `data/raw/preprocessed-cyto-sources/allof` and
  `data/raw/preprocessed-cyto-sources/prepro`.
- Seed: 206.
- Teacher loader view: exactly 100,000 cells per sample.
- Student source subset: 40,000 cells.
- Shared input panel: 30 markers; fail if the inferred panel width differs.
- Five IPAB encoder blocks, 16 inducing prototypes per block, one attention head,
  hidden width 384, one pooling seed, and a 256-dimensional sample embedding.
- Two SAB decoder blocks and 5,000 reconstructed output cells.
- Layer normalization enabled.
- 500 epochs.
- AdamW, initial learning rate `1e-4`, weight decay `1e-3`, cosine decay to
  `1e-12`.
- Student temperature 0.10; teacher temperature 0.07.
- Teacher EMA 0.99; teacher-center momentum 0.99.
- Energy-distance reconstruction for epochs 0-24; Sinkhorn reconstruction from
  epoch 25 onward.
- BF16 mixed precision, DeepSpeed ZeRO stage 1, gradient clipping 5.0.
- Microbatch 3 per GPU and global batch 12 across four ranks.
- Checkpoints every ten epochs plus best/last selection after the Sinkhorn phase
  begins.
- CSV logging always enabled. W&B is enabled in offline mode by default; model
  artifact upload is disabled because DeepSpeed checkpoints are large.

Do not silently change these values. When a change is requested, update the YAML,
tests, and any affected Slurm assumptions together, and call out how the run is no
longer directly comparable with earlier results.

## Data and preprocessing context

The closest available paper-replication snapshot is the union of:

- `allof`: 1,692 HDF5 samples.
- `prepro`: 178 HDF5 samples.
- Combined: 1,870 samples and approximately 418.25 million cells.

The manuscript reports 1,792 samples, so the local run is an approximate rather
than byte-for-byte replication of the published training manifest.

Upstream preprocessing was performed before these HDF5 files reached this
repository: bead/dead-cell/pulse-shape filtering, doublet removal, and
`arcsinh(x / 5)` transformation. The whole-blood snapshot retains granulocytes
and is not cyDM-corrected or PRISM-imputed. Do not apply arcsinh or normalization
again in the loader.

Each file normally contains:

- `data`: cells by markers, used for model input.
- `feature_names`: marker names.
- `cell_types`: auxiliary gated annotations used for interpretation, not as
  encoder supervision.

The loader canonicalizes marker aliases (for example `CD8a -> CD8`), computes the
sorted marker intersection across input directories, downsamples samples above
100,000 cells without replacement, and upsamples smaller samples with
replacement. Duplicate HDF5 stems across input directories are an error.

Do not rename, rewrite, normalize, or delete local biological data without an
explicit request. Data paths are intentionally excluded from Git.

## Model and objective details

For each batch, the model draws a 40,000-cell student subset from the teacher's
100,000-cell view. One mask rate is selected for the entire batch from
`[0%, 20%, 40%, 60%, 80%]`. Cells are projected onto the first principal
component, sorted, and a contiguous block is removed from a randomly chosen
extreme. A 0% mask now masks zero cells.

The student encodes the masked subset and reconstructs an unordered cell
distribution. The teacher encodes the unmasked 100,000-cell view. Total loss is
the unweighted sum of reconstruction loss and KL self-distillation. Teacher
weights are an EMA of student weights, and the centered teacher projection is
temperature-sharpened. Teacher EMA and center momentum are wired through the
configuration; do not reintroduce older hard-coded values.

Important implementation details:

- The code currently chooses one random mask rate per batch, not five simultaneous
  masked copies per sample.
- Decoder output is 5,000 cells. Earlier paper analysis suggested 40,000 might be
  more manuscript-faithful, but the 40,000-output distributed run has not been
  validated and must not be substituted silently.
- The current configuration warms up with energy distance before switching to
  Sinkhorn. Treat this as an implementation choice when making paper claims.
- `Validate` mode is a train/validation split helper, not the manuscript's full
  downstream disease-classification evaluation.
- Paper-level classification claims require fold-specific pretraining so held-out
  specimens do not influence their encoder. The current launcher does not
  automate that procedure.

## Cluster execution

Submit the fixed run with:

```bash
mkdir -p output
sbatch jobs/train-maestro.slurm
```

The launcher is specific to the Penn cluster and requests one `dgx-b200` node,
four B200 GPUs, four Slurm tasks (one rank per GPU), 28 CPUs per task, 896 GB RAM,
and a five-day wall limit. Four tasks launched through `srun` are required;
allocating four GPUs to one task caused distributed initialization to stall.

The script loads CUDA 12.8.1 and adds
`$CUDA_HOME/targets/x86_64-linux/lib` to both `LD_LIBRARY_PATH` and
`LIBRARY_PATH`. Both are required for PyKeOps on B200: the runtime loader needs
the former and the JIT linker needs the latter. Keep the fail-fast PyKeOps GPU
preflight. Earlier missing library paths caused GeomLoss/KeOps to announce CPU
fallback and then segfault on CUDA tensors.

Do not submit, cancel, resize, or resubmit Slurm jobs unless the user explicitly
requests it. A shorter wall-time may backfill sooner, but it also risks truncating
later Sinkhorn epochs; do not alter the fixed five-day request without approval.

## Validated B200 run

The latest full-input smoke test was Slurm job `7151245` on four B200 GPUs. It
used the current 100,000-cell teacher view, 40,000-cell student subset, 5,000
outputs, and energy reconstruction, but ran for only one epoch.

- State/exit: `COMPLETED`, `0:0`.
- Queue wait: 44 minutes 19 seconds.
- Allocation runtime: 3 minutes 1 second.
- Lightning trainer runtime: 64.83 seconds; epoch compute was about 43 seconds.
- Epoch-0 total loss: 5.3206; reconstruction 1.9261; distillation about 3.395.
- Peak GPU memory: about 16.5 GiB per GPU; peak utilization reached 100%.

This validates the current data loader, exact-shape forward/backward path,
four-rank DeepSpeed topology, KeOps CUDA discovery, logging, and visualization.
It does not validate the later Sinkhorn phase, a complete 500-epoch run, or a
40,000-output decoder.

## Run provenance and outputs

For Slurm job `<JOB_ID>`, outputs are rooted at
`experiments/maestro-allof-prepro-<JOB_ID>/`. The harness records:

- resolved and base Hydra configuration;
- Git commit, status, and uncommitted binary diff;
- `uv.lock` SHA-256;
- copied submitted Slurm script;
- Slurm allocation, accounting, and numeric exit status;
- five-second GPU utilization/memory/power/temperature telemetry;
- exact sample inventory, resolved paths, sizes, shared marker order, and a
  path/size dataset fingerprint;
- checkpoints, reconstruction visualizations, and offline W&B files.

Lightning CSV metrics live under `logs/<run-name>/`; Slurm stdout/stderr lives
under `output/`. W&B offline runs can be uploaded later with `wandb sync`. Never
commit generated runs, local data, credentials, or W&B API keys.

## Change discipline

- Preserve unrelated user changes in a dirty worktree.
- Prefer the fixed Hydra path (`maestro-run`) for comparable runs; keep the legacy
  `maestro-train` CLI working unless explicitly removing it.
- Keep run-changing configuration in `configs/train.yaml`, not hidden constants
  in shell scripts.
- Keep model semantics unchanged when making tracking or orchestration changes.
- Record assumptions and distinguish manuscript settings, current implementation
  settings, and experimentally validated behavior.
- Use `rg`/`rg --files` for repository searches and focused tests for changes.
- Never commit secrets, local datasets, generated checkpoints, or ignored
  manuscripts/project notes.
