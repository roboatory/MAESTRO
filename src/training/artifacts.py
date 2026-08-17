"""Bound the generated artifacts retained by a MAESTRO training run."""

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactPruningSummary:
    """Count artifacts removed by a retention pass."""

    periodic_checkpoints: int = 0
    reconstruction_visualizations: int = 0


def _remove_path(path: Path) -> None:
    """Remove one generated file, link, or checkpoint directory."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_reconstruction_visualizations(figure_directory: Path) -> int:
    """Retain only the newest epoch reconstruction visualization."""
    visualization_paths = sorted(figure_directory.glob("epoch_*.pdf"))
    obsolete_paths = visualization_paths[:-1]
    for obsolete_path in obsolete_paths:
        _remove_path(obsolete_path)
    return len(obsolete_paths)


def prune_completed_run_artifacts(
    output_path: Path,
    *,
    retain_periodic_checkpoints: bool,
    retain_reconstruction_history: bool,
) -> ArtifactPruningSummary:
    """Apply the configured retention policy after successful training."""
    removed_checkpoints = 0
    if not retain_periodic_checkpoints:
        periodic_checkpoint_paths = sorted(output_path.glob("epoch=*.ckpt"))
        for checkpoint_path in periodic_checkpoint_paths:
            _remove_path(checkpoint_path)
        removed_checkpoints = len(periodic_checkpoint_paths)

    removed_visualizations = 0
    if not retain_reconstruction_history:
        removed_visualizations = prune_reconstruction_visualizations(
            output_path / "reconstruction_viz"
        )

    return ArtifactPruningSummary(
        periodic_checkpoints=removed_checkpoints,
        reconstruction_visualizations=removed_visualizations,
    )
