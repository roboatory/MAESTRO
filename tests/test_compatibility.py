"""Verify compatibility with the original MAESTRO public interfaces."""

import inspect
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from maestro.data import CyTOFDataset
from maestro.models import MAESTRO, MAESTROLightning, ruler_masking
from maestro.models.maestro import MAB
from maestro.training.cli import parse_args


def test_original_command_line_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the original training flags and their default values."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["maestro-train", "--data_dirs", "data/h5/dataA"],
    )

    configuration = parse_args()

    assert configuration.project_name is None
    assert configuration.devices == "0"
    assert configuration.data_directories == ("data/h5/dataA",)
    assert configuration.input_dimension == 30
    assert configuration.number_inducing_points == 16
    assert configuration.hidden_dimension == 384
    assert configuration.latent_dimension == 256
    assert configuration.number_attention_heads == 4
    assert configuration.layer_normalization is True
    assert configuration.mode == "Train"
    assert configuration.center_momentum == 0.99
    assert configuration.teacher_beta == 0.9995


def test_dataset_accepts_original_keywords(
    tmp_path: Path,
) -> None:
    """Load a sample through the original data_dirs constructor keyword."""
    sample_path = tmp_path / "sample.h5"
    with h5py.File(sample_path, "w") as h5_file:
        h5_file.create_dataset(
            "data",
            data=np.arange(12, dtype=np.float32).reshape(4, 3),
        )
        h5_file.create_dataset(
            "feature_names",
            data=np.array([b"CD8a", b"PD1", b"CD4"]),
        )
        h5_file.create_dataset(
            "cell_types",
            data=np.array([b"T", b"B", b"T", b"B"]),
        )

    dataset = CyTOFDataset(
        data_dirs=[tmp_path],
        subset_size=4,
    )
    features, cell_types, sample_name = dataset[0]

    assert dataset.shared_markers == ["CD4", "CD8", "PD-1"]
    assert features.shape == (4, 3)
    assert cell_types.shape == (4,)
    assert sample_name == "sample"
    assert dataset.get_num_cell_types() == dataset.get_number_cell_types()


def test_model_constructors_preserve_original_parameter_names() -> None:
    """Keep checkpoint and keyword-based model construction compatible."""
    maestro_parameters = inspect.signature(MAESTRO).parameters
    lightning_parameters = inspect.signature(MAESTROLightning).parameters

    expected_names = {
        "dim_input",
        "dim_output",
        "num_inds",
        "dim_hidden",
        "dim_latent",
        "num_heads",
        "num_outputs",
        "ln",
        "number_cells_subset",
        "student_temperature",
        "teacher_temperature",
        "sinkhorn_start",
    }
    assert expected_names <= set(maestro_parameters)
    assert expected_names <= set(lightning_parameters)
    assert {"initial_lr", "min_lr", "epochs", "output_path"} <= set(
        lightning_parameters
    )


def test_attention_block_preserves_original_options() -> None:
    """Accept the original attention constructor and return tensor shapes."""
    block = MAB(
        dim_Q=4,
        dim_K=4,
        dim_V=4,
        num_heads=2,
        ln=True,
        use_checkpoint=False,
        ent=1.2,
    )
    input_tensor = torch.randn(2, 3, 4)

    output, attention = block(input_tensor, input_tensor)

    assert output.shape == (2, 3, 4)
    assert attention.shape == (4, 3, 3)
    assert block.use_checkpoint is False
    assert block.ent == 1.2


def test_ruler_masking_preserves_mask_shape_and_count() -> None:
    """Mask the original number of cells for each sample."""
    input_tensor = torch.randn(2, 10, 4)

    mask = ruler_masking(input_tensor, mask_ratio=0.4)

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 10)
    assert torch.equal(mask.sum(dim=1), torch.tensor([4, 4]))
