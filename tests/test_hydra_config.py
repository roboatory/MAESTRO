"""Test Hydra composition and experiment tracking configuration."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from src.hydra_cli import configuration_from_hydra
from src.training.runner import (
    TrainingConfiguration,
    _create_loggers,
    _write_dataset_manifest,
)

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs"


def compose_training_configuration(
    *overrides: str,
) -> TrainingConfiguration:
    """Compose the repository configuration for a focused test."""
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(CONFIG_DIRECTORY),
    ):
        hydra_configuration = compose(
            config_name="train",
            overrides=list(overrides),
        )
    return configuration_from_hydra(hydra_configuration)


def test_paper_replication_configuration() -> None:
    """Hydra should reproduce the fixed paper-replication configuration."""
    configuration = compose_training_configuration()

    assert configuration.data_directories == (
        "data/raw/preprocessed-cyto-sources/allof",
        "data/raw/preprocessed-cyto-sources/prepro",
    )
    assert configuration.number_cells_subset == 40_000
    assert configuration.input_dimension == 30
    assert configuration.number_inducing_points == 16
    assert configuration.hidden_dimension == 384
    assert configuration.latent_dimension == 256
    assert configuration.number_attention_heads == 1
    assert configuration.layer_normalization
    assert configuration.number_outputs == 5_000
    assert configuration.number_epochs == 500
    assert configuration.sinkhorn_start_epoch == 25
    assert configuration.random_seed == 206
    assert configuration.initial_learning_rate == 1e-4
    assert configuration.minimum_learning_rate == 1e-12
    assert configuration.student_temperature == 0.10
    assert configuration.teacher_temperature == 0.07
    assert configuration.center_momentum == 0.99
    assert configuration.teacher_beta == 0.99
    assert configuration.wandb_enabled
    assert configuration.wandb_mode == "offline"
    assert configuration.wandb_group == "paper-reproduction"


def test_offline_mode_rejects_checkpoint_uploads() -> None:
    """Offline W&B runs should retain checkpoints locally without copying them."""
    with pytest.raises(ValueError, match="requires online mode"):
        compose_training_configuration("tracking.log_model=true")


def test_offline_wandb_logger_initializes_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default tracker should create a local run without credentials."""
    monkeypatch.setenv("WANDB_SILENT", "true")
    configuration = compose_training_configuration("run.name=offline-smoke-test")

    loggers = _create_loggers(configuration, tmp_path)
    loggers[-1].log_metrics({"smoke_test": 1.0}, step=0)
    for logger in loggers:
        logger.finalize("success")

    assert len(loggers) == 2
    assert list((tmp_path / "wandb").glob("offline-run-*"))


def test_dataset_manifest_records_inventory_and_marker_order(tmp_path: Path) -> None:
    """Every run should retain a cheap fingerprint of its exact input files."""
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    dataset = SimpleNamespace(
        file_paths={"first": first_path, "second": second_path},
        shared_markers=["CD3", "CD4"],
        subset_size=500,
    )

    parameters, artifacts = _write_dataset_manifest(dataset, tmp_path)

    assert parameters["dataset_sample_count"] == 2
    assert parameters["shared_markers"] == ["CD3", "CD4"]
    assert parameters["teacher_cell_count"] == 500
    assert len(str(parameters["dataset_fingerprint"])) == 64
    assert artifacts == (tmp_path / "dataset-manifest.json",)
