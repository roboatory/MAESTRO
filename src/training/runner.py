"""Configure and run MAESTRO training."""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader, random_split

from src.data import CyTOFDataset
from src.models import MAESTROLightning
from src.training.callbacks import (
    SinkhornCheckpoint,
    UpdateTeacher,
    create_deep_speed_config,
)

TrainingMode = Literal["Train", "Validate"]


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """Store the settings required for a training or validation run."""

    project_name: str
    devices: str
    data_directories: tuple[str, ...]
    marker_directories: tuple[str, ...] = ()
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
    number_outputs: int = 5_000
    student_temperature: float = 0.10
    teacher_temperature: float = 0.07
    center_momentum: float = 0.99
    teacher_beta: float = 0.99
    mode: TrainingMode = "Train"
    removed_cell_types: tuple[str, ...] = ()
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
) -> list[callbacks.Callback]:
    """Create periodic and best-loss checkpoint callbacks."""
    periodic_checkpoint = callbacks.ModelCheckpoint(
        dirpath=output_path,
        filename="{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,
        save_last=False,
        save_weights_only=False,
        verbose=True,
        save_on_train_epoch_end=True,
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
    return [UpdateTeacher(), periodic_checkpoint, best_checkpoint]


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


def _create_trainer(
    configuration: TrainingConfiguration,
    deep_speed_config: dict[str, object],
    output_path: Path,
) -> lightning.Trainer:
    """Create a Lightning trainer for the requested run."""
    logger = (
        CSVLogger(save_dir="logs/", name=configuration.project_name)
        if configuration.mode == "Train"
        else CSVLogger(save_dir="logs/")
    )
    trainer = lightning.Trainer(
        devices=configuration.devices,
        accelerator="cuda",
        strategy=DeepSpeedStrategy(config=deep_speed_config),
        precision="bf16-mixed",
        max_epochs=configuration.number_epochs,
        min_epochs=min(300, configuration.number_epochs),
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=_create_checkpoints(
            output_path,
            configuration.sinkhorn_start_epoch,
        ),
        log_every_n_steps=1,
        logger=logger,
    )
    trainer.strategy.config["zero_force_ds_cpu_optimizer"] = False
    return trainer


def run_training(
    configuration: TrainingConfiguration,
) -> None:
    """Train or validate MAESTRO using the supplied configuration."""
    _configure_warning_filters()
    output_path = Path("experiments") / configuration.project_name
    output_path.mkdir(parents=True, exist_ok=True)
    lightning.seed_everything(206, workers=True)

    dataset = CyTOFDataset(
        configuration.data_directories,
        subset_size=100_000,
        marker_dirs=configuration.marker_directories,
        cell_type_removal=configuration.removed_cell_types,
    )
    dim_input = len(dataset.shared_markers)
    _validate_input_dimension(configuration.input_dimension, dim_input)

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
        center_momentum=configuration.center_momentum,
        teacher_beta=configuration.teacher_beta,
        num_outputs=configuration.number_outputs,
        sinkhorn_start=configuration.sinkhorn_start_epoch,
    )

    deep_speed_config = create_deep_speed_config()
    batch_size = int(deep_speed_config["train_micro_batch_size_per_gpu"])
    if configuration.mode == "Train":
        trainer = _create_trainer(configuration, deep_speed_config, output_path)
        training_data_loader = _create_data_loader(
            dataset,
            batch_size,
            shuffle=True,
        )
        if configuration.resume_checkpoint is not None:
            trainer.fit(
                model=model,
                train_dataloaders=training_data_loader,
                ckpt_path=configuration.resume_checkpoint,
            )
        else:
            trainer.fit(
                model=model,
                train_dataloaders=training_data_loader,
            )
        return

    if configuration.mode != "Validate":
        return

    training_sample_count = int(len(dataset) * 0.9)
    validation_sample_count = len(dataset) - training_sample_count
    generator = torch.Generator().manual_seed(206)
    training_set, validation_set = random_split(
        dataset,
        [training_sample_count, validation_sample_count],
        generator=generator,
    )
    training_data_loader = _create_data_loader(
        training_set,
        batch_size,
        shuffle=True,
    )
    validation_data_loader = _create_data_loader(
        validation_set,
        batch_size,
        shuffle=False,
    )
    trainer = _create_trainer(configuration, deep_speed_config, output_path)
    trainer.fit(
        model=model,
        train_dataloaders=training_data_loader,
        val_dataloaders=validation_data_loader,
    )
