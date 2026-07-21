"""Provide the MAESTRO training command-line interface."""

import argparse

from src.training.runner import TrainingConfiguration, run_training


def parse_args() -> TrainingConfiguration:
    """Parse command-line arguments into a training configuration."""
    parser = argparse.ArgumentParser(description="🎶MAESTRO🎶")
    # fmt: off
    parser.add_argument("--project", dest="project_name", type=str, required=True, help="Project name")  # noqa: E501
    parser.add_argument("--devices", type=str, default="0", help="GPU devices")
    parser.add_argument("--data_dirs", dest="data_directories", nargs="+", required=True, type=str, help="Data directories (used for training)")  # noqa: E501
    parser.add_argument("--marker_dirs", dest="marker_directories", nargs="+", default=None, type=str, help="Marker-only directories included in the shared-marker intersection but not used for training")  # noqa: E501
    parser.add_argument("--number_cells_subset", dest="number_cells_subset", type=int, default=40_000, help="Cells in the student subset")  # noqa: E501
    parser.add_argument("--dim_input", dest="input_dimension", type=int, default=30, help="Input dimension per cell")  # noqa: E501
    parser.add_argument("--num_inds", dest="number_inducing_points", type=int, default=16, help="IPAB inducing points")  # noqa: E501
    parser.add_argument("--dim_hidden", dest="hidden_dimension", type=int, default=384, help="Hidden dimension")  # noqa: E501
    parser.add_argument("--dim_latent", dest="latent_dimension", type=int, default=256, help="Latent dimension")  # noqa: E501
    parser.add_argument("--num_heads", dest="number_attention_heads", type=int, default=1, help="Attention heads")  # noqa: E501
    parser.add_argument("--ln", dest="layer_normalization", action=argparse.BooleanOptionalAction, default=True, help="Use layer normalization")  # noqa: E501
    parser.add_argument("--initial_lr", dest="initial_learning_rate", type=float, default=1e-4, help="Initial learning rate")  # noqa: E501
    parser.add_argument("--min_lr", dest="minimum_learning_rate", type=float, default=1e-12, help="Minimum learning rate")  # noqa: E501
    parser.add_argument("--epochs", dest="number_epochs", type=int, default=500, help="Number of epochs")  # noqa: E501
    parser.add_argument("--sinkhorn_start", dest="sinkhorn_start_epoch", type=int, default=25, help="Convert loss function to sampleloss Sinkhorn")  # noqa: E501
    parser.add_argument("--num_outputs", dest="number_outputs", type=int, default=5_000, help="Number of reconstructed output cells")  # noqa: E501
    parser.add_argument("--student_temperature", dest="student_temperature", type=float, default=0.10, help="Student softmax temperature")  # noqa: E501
    parser.add_argument("--teacher_temperature", dest="teacher_temperature", type=float, default=0.07, help="Teacher softmax temperature")  # noqa: E501
    parser.add_argument("--center_momentum", type=float, default=0.99, help="EMA momentum for teacher centering")  # noqa: E501
    parser.add_argument("--teacher_beta", type=float, default=0.99, help="EMA momentum for teacher weights")  # noqa: E501
    parser.add_argument("--mode", type=str, choices=("Train", "Validate"), default="Train", help="Train or Validate")  # noqa: E501
    parser.add_argument("--cell_type_removal", dest="removed_cell_types", type=str, nargs="+", default=None, help="Cell types to filter")  # noqa: E501
    parser.add_argument("--ckpt_resume", dest="resume_checkpoint", type=str, default=None, help="Checkpoint path to resume")  # noqa: E501
    # fmt: on
    arguments = parser.parse_args()
    return TrainingConfiguration(
        project_name=arguments.project_name,
        devices=arguments.devices,
        data_directories=tuple(arguments.data_directories),
        marker_directories=tuple(arguments.marker_directories or ()),
        number_cells_subset=arguments.number_cells_subset,
        input_dimension=arguments.input_dimension,
        number_inducing_points=arguments.number_inducing_points,
        hidden_dimension=arguments.hidden_dimension,
        latent_dimension=arguments.latent_dimension,
        number_attention_heads=arguments.number_attention_heads,
        layer_normalization=arguments.layer_normalization,
        initial_learning_rate=arguments.initial_learning_rate,
        minimum_learning_rate=arguments.minimum_learning_rate,
        number_epochs=arguments.number_epochs,
        sinkhorn_start_epoch=arguments.sinkhorn_start_epoch,
        number_outputs=arguments.number_outputs,
        student_temperature=arguments.student_temperature,
        teacher_temperature=arguments.teacher_temperature,
        center_momentum=arguments.center_momentum,
        teacher_beta=arguments.teacher_beta,
        mode=arguments.mode,
        removed_cell_types=tuple(arguments.removed_cell_types or ()),
        resume_checkpoint=arguments.resume_checkpoint,
    )


def main() -> None:
    """Run MAESTRO from command-line arguments."""
    run_training(parse_args())


if __name__ == "__main__":
    main()
