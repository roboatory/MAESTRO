"""Compose MAESTRO run configurations with Hydra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from src.training.runner import TrainingConfiguration


def _resolved_mapping(config: DictConfig, key: str) -> dict[str, Any]:
    """Resolve one Hydra mapping into plain Python values."""
    value = OmegaConf.to_container(config[key], resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Hydra configuration section '{key}' must be a mapping")
    return value


def configuration_from_hydra(config: DictConfig) -> TrainingConfiguration:
    """Convert the composed Hydra document into the training dataclass."""
    from src.training.runner import TrainingConfiguration

    run = _resolved_mapping(config, "run")
    training = _resolved_mapping(config, "training")
    tracking = _resolved_mapping(config, "tracking")

    for tuple_field in (
        "data_directories",
        "marker_directories",
        "removed_cell_types",
    ):
        training[tuple_field] = tuple(training.get(tuple_field, ()))

    return TrainingConfiguration(
        project_name=str(run["name"]),
        **training,
        wandb_enabled=bool(tracking["enabled"]),
        wandb_project=str(tracking["project"]),
        wandb_entity=tracking.get("entity"),
        wandb_mode=str(tracking["mode"]),
        wandb_group=tracking.get("group"),
        wandb_tags=tuple(tracking.get("tags", ())),
        wandb_run_id=tracking.get("run_id"),
        wandb_log_model=bool(tracking["log_model"]),
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def hydra_main(config: DictConfig) -> None:
    """Compose the selected configuration and start MAESTRO."""
    from src.training.runner import run_training

    run_training(configuration_from_hydra(config))


def main() -> None:
    """Run the Hydra-backed MAESTRO entry point."""
    hydra_main()


if __name__ == "__main__":
    main()
