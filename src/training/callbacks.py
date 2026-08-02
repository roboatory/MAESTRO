"""Configure callbacks and distributed training."""

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


def create_deep_speed_config() -> dict[str, object]:
    """Create the DeepSpeed training configuration."""
    return {
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {
            "stage": 1,
            "contiguous_gradients": True,
            "overlap_comm": True,
        },
        "bf16": {
            "enabled": True,
        },
        "gradient_clipping": 5.0,
        "train_batch_size": 12,
        "train_micro_batch_size_per_gpu": 3,
    }


class UpdateTeacher(Callback):
    """Update teacher weights after each student optimization step."""

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        """Update the teacher after a training batch."""
        del trainer, outputs, batch, batch_idx
        pl_module.model._update_teacher()


class SinkhornCheckpoint(ModelCheckpoint):
    """Ignore warm-up epochs when selecting the best Sinkhorn checkpoint."""

    def __init__(
        self,
        sinkhorn_start: int = 0,
        **kwargs: object,
    ) -> None:
        """Initialize the checkpoint callback."""
        super().__init__(**kwargs)
        self.sinkhorn_start = sinkhorn_start
        self._switched = False

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Save checkpoints after the Sinkhorn phase begins."""
        if self.sinkhorn_start > 0 and trainer.current_epoch < self.sinkhorn_start:
            return

        if self.sinkhorn_start > 0 and not self._switched:
            self.best_model_score = torch.tensor(float("inf"))
            self.best_model_path = ""
            self._switched = True

        super().on_train_epoch_end(trainer, pl_module)
