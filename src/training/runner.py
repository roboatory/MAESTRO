"""Configure and run MAESTRO training."""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import lightning
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader

from src.data.dataset import CyTOFDataset
from src.models.maestro import MAESTROLightning
from src.training.artifacts import prune_completed_run_artifacts
from src.training.callbacks import (
    SinkhornCheckpoint,
    UpdateTeacher,
    create_deep_speed_config,
)


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """Store the settings required for a training run."""

    project_name: str
    devices: str
    data_directories: tuple[str, ...]
    random_seed: int = 206
    number_cells_subset: int = 40_000
    input_dimension: int = 30
    number_inducing_points: int = 16
    hidden_dimension: int = 384
    latent_dimension: int = 256
    number_attention_heads: int = 1
    layer_normalization: bool = True
    initial_learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-12
    number_epochs: int = 500
    sinkhorn_start_epoch: int = 25
    number_outputs: int = 40_000
    student_temperature: float = 0.11
    teacher_temperature: float = 0.04
    teacher_temperature_step: float = 0.01
    center_momentum: float = 0.9
    teacher_beta: float = 0.99
    retain_periodic_checkpoints: bool = False
    retain_reconstruction_history: bool = False
    resume_checkpoint: str | None = None

    def __post_init__(self) -> None:
        """Validate scalar training settings."""
        positive_values = {
            "number_cells_subset": self.number_cells_subset,
            "input_dimension": self.input_dimension,
            "number_inducing_points": self.number_inducing_points,
            "hidden_dimension": self.hidden_dimension,
            "latent_dimension": self.latent_dimension,
            "number_attention_heads": self.number_attention_heads,
            "initial_learning_rate": self.initial_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "number_epochs": self.number_epochs,
            "number_outputs": self.number_outputs,
            "student_temperature": self.student_temperature,
            "teacher_temperature": self.teacher_temperature,
            "teacher_temperature_step": self.teacher_temperature_step,
        }
        invalid_names = [name for name, value in positive_values.items() if value <= 0]
        if invalid_names:
            raise ValueError(
                f"Training settings must be positive: {', '.join(invalid_names)}"
            )
        for name, value in {
            "center_momentum": self.center_momentum,
            "teacher_beta": self.teacher_beta,
        }.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in the interval [0, 1)")
        if self.sinkhorn_start_epoch < 0:
            raise ValueError("sinkhorn_start_epoch must be non-negative")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")


def _validate_input_dimension(
    configured_dimension: int,
    inferred_dimension: int,
) -> None:
    """Ensure the configured input width matches the shared marker panel."""
    if configured_dimension != inferred_dimension:
        raise ValueError(
            "Configured input dimension "
            f"{configured_dimension} does not match the "
            f"{inferred_dimension}-marker shared data panel"
        )


def _configure_warning_filters() -> None:
    """Suppress known third-party warnings emitted by distributed training."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    warnings.filterwarnings("ignore", message=".*Please use the new API settings.*")
    warnings.filterwarnings("ignore", message=".*you have set wrong precision.*")
    warnings.filterwarnings("ignore", message=".*CUDA device.*Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*Tensor Cores.*")


def _create_checkpoints(
    output_path: Path,
    sinkhorn_start_epoch: int,
    *,
    retain_periodic_checkpoints: bool,
) -> list[callbacks.Callback]:
    """Create bounded checkpoint callbacks for best and resumable state."""
    checkpoint_callbacks: list[callbacks.Callback] = [UpdateTeacher()]
    if retain_periodic_checkpoints:
        checkpoint_callbacks.append(
            callbacks.ModelCheckpoint(
                dirpath=output_path,
                filename="{epoch:03d}",
                every_n_epochs=10,
                save_top_k=-1,
                save_last=False,
                save_weights_only=False,
                verbose=True,
                save_on_train_epoch_end=True,
            )
        )
    best_checkpoint = SinkhornCheckpoint(
        sinkhorn_start=sinkhorn_start_epoch,
        dirpath=output_path,
        filename="best-{epoch:03d}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        verbose=True,
        save_on_train_epoch_end=True,
    )
    checkpoint_callbacks.append(best_checkpoint)
    return checkpoint_callbacks


def _create_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create a data loader with the settings used by MAESTRO."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )


def _write_dataset_manifest(
    dataset: CyTOFDataset,
    output_path: Path,
) -> None:
    """Fingerprint the exact sample inventory and record the marker order."""
    process_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    if process_rank != 0:
        return

    digest = hashlib.sha256()
    files = []
    total_bytes = 0
    for sample_name, file_path in sorted(dataset.file_paths.items()):
        size_bytes = file_path.stat().st_size
        total_bytes += size_bytes
        resolved_path = str(file_path.resolve())
        files.append(
            {
                "sample": sample_name,
                "path": resolved_path,
                "size_bytes": size_bytes,
            }
        )
        digest.update(f"{resolved_path}\0{size_bytes}\n".encode())

    fingerprint = digest.hexdigest()
    runtime_parameters = {
        "dataset_sample_count": len(files),
        "dataset_total_bytes": total_bytes,
        "dataset_fingerprint": fingerprint,
        "shared_markers": list(dataset.shared_markers),
        "teacher_cell_count": dataset.subset_size,
    }
    runtime_parameters.update(
        {
            name.lower(): value
            for name in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
            )
            if (value := os.environ.get(name)) is not None
        }
    )

    manifest = {
        **runtime_parameters,
        "fingerprint_method": "sha256(path, size_bytes)",
        "files": files,
    }
    manifest_path = output_path / "dataset-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _create_trainer(
    configuration: TrainingConfiguration,
    deep_speed_config: dict[str, object],
    output_path: Path,
) -> lightning.Trainer:
    """Create a Lightning trainer for the requested run."""
    trainer = lightning.Trainer(
        devices=configuration.devices,
        accelerator="cuda",
        strategy=DeepSpeedStrategy(config=deep_speed_config),
        precision="bf16-mixed",
        max_epochs=configuration.number_epochs,
        min_epochs=min(300, configuration.number_epochs),
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[
            *_create_checkpoints(
                output_path,
                configuration.sinkhorn_start_epoch,
                retain_periodic_checkpoints=(configuration.retain_periodic_checkpoints),
            ),
        ],
        log_every_n_steps=1,
        logger=CSVLogger(save_dir=str(output_path), name="", version=""),
    )
    trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    return trainer


def run_training(
    configuration: TrainingConfiguration,
) -> None:
    """Train MAESTRO using the supplied configuration."""
    _configure_warning_filters()
    output_path = Path("experiments") / configuration.project_name
    output_path.mkdir(parents=True, exist_ok=True)
    lightning.seed_everything(configuration.random_seed, workers=True)

    dataset = CyTOFDataset(
        configuration.data_directories,
        subset_size=100_000,
    )
    dim_input = len(dataset.shared_markers)
    _validate_input_dimension(configuration.input_dimension, dim_input)
    _write_dataset_manifest(dataset, output_path)

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(f"Project: {configuration.project_name}")
        print(f"Training {len(dataset)} samples")
        print(f"Input dimension inferred from shared markers: {dim_input}")

    model = MAESTROLightning(
        dim_input=dim_input,
        dim_output=dim_input,
        num_inds=configuration.number_inducing_points,
        dim_hidden=configuration.hidden_dimension,
        dim_latent=configuration.latent_dimension,
        num_heads=configuration.number_attention_heads,
        ln=configuration.layer_normalization,
        number_cells_subset=configuration.number_cells_subset,
        initial_lr=configuration.initial_learning_rate,
        min_lr=configuration.minimum_learning_rate,
        epochs=configuration.number_epochs,
        output_path=output_path,
        student_temperature=configuration.student_temperature,
        teacher_temperature=configuration.teacher_temperature,
        teacher_temperature_step=configuration.teacher_temperature_step,
        center_momentum=configuration.center_momentum,
        teacher_beta=configuration.teacher_beta,
        retain_reconstruction_history=(configuration.retain_reconstruction_history),
        num_outputs=configuration.number_outputs,
        sinkhorn_start=configuration.sinkhorn_start_epoch,
    )

    deep_speed_config = create_deep_speed_config()
    batch_size = int(deep_speed_config["train_micro_batch_size_per_gpu"])
    trainer = _create_trainer(
        configuration,
        deep_speed_config,
        output_path,
    )
    training_data_loader = _create_data_loader(
        dataset,
        batch_size,
        shuffle=True,
    )
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        ckpt_path=configuration.resume_checkpoint,
    )
    if trainer.is_global_zero:
        pruning_summary = prune_completed_run_artifacts(
            output_path,
            retain_periodic_checkpoints=(configuration.retain_periodic_checkpoints),
            retain_reconstruction_history=(configuration.retain_reconstruction_history),
        )
        print(
            "Artifact retention removed "
            f"{pruning_summary.periodic_checkpoints} periodic checkpoints and "
            f"{pruning_summary.reconstruction_visualizations} old reconstruction "
            "visualizations"
        )
