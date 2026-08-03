# MAESTRO repository guide

This file provides durable, high-level context for maintainers and coding agents.
Treat the checked-out repository as the source of truth and verify paths and
behavior against the current tree.

## Purpose

MAESTRO (Masked Encoding Set Transformer with self-distillation) learns one
fixed-length representation of a complete cytometry sample. Each sample is
treated as an unordered set of cells. Pretraining combines masked cell-set
reconstruction with momentum-teacher self-distillation and does not use cell or
patient labels as encoder supervision.

## System architecture

The training system has five main layers:

1. Hydra resolves the supported training configuration.
2. The HDF5 data layer discovers samples, aligns marker panels, and constructs
   fixed-size cell-set views.
3. The MAESTRO student encodes a masked cell subset and reconstructs a cell
   distribution, while an EMA teacher encodes the unmasked view.
4. The training layer combines reconstruction and distillation objectives and
   manages distributed execution, optimization, logging, and checkpoints.
5. The Slurm launcher provides the cluster-specific distributed entry point and
   stores every generated artifact beneath a self-contained experiment directory.

## Repository structure

- `configs/train.yaml`: supported Hydra training configuration.
- `jobs/train-maestro.slurm`: supported cluster launcher.
- `src/hydra_cli.py`: Hydra-to-training entry point.
- `src/data/`: HDF5 loading, marker alignment, and cell sampling.
- `src/models/`: MAESTRO model, masking, reconstruction, and self-distillation.
- `src/training/`: trainer construction, callbacks, logging, checkpoints, and
  run provenance.
- `notebooks/`: numbered exploratory and post-training analyses.
- `data/csv/` and `data/h5/`: tracked synthetic fixtures from the public release.
- `output/training/ToyModel/`: tracked public ToyModel checkpoint and config.
- `data/raw/`: ignored local biological data.
- `experiments/`: ignored, self-contained generated training runs.

`README.md` mirrors the public upstream project. `CLAUDE.md` must remain a
symlink to this file so both tools receive the same repository context.

## Data and artifact boundaries

Production biological data is local and must not be committed, rewritten, or
deleted without an explicit request. The tracked synthetic data and ToyModel are
documentation and analysis fixtures, not part of the production training
manifest. New training outputs belong under their owning directory in
`experiments/`; the tracked ToyModel directory is not an output destination.

## Development principles

- Preserve the packaged `src/` layout when porting behavior from the flat public
  repository.
- Keep the supported configuration and launcher as the visible sources of
  run-changing behavior.
- Make model-semantic changes deliberately and distinguish configured behavior
  from experimentally validated behavior.
- Preserve unrelated work in a dirty worktree and never commit credentials,
  local biological data, or generated experiment artifacts.
- Use `uv` for the environment and the repository's existing formatting and
  validation tooling for code changes.
