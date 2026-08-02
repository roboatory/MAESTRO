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

    training["data_directories"] = tuple(training["data_directories"])

    return TrainingConfiguration(
        project_name=str(run["name"]),
        **training,
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
