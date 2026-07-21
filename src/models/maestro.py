"""Define MAESTRO models and their Lightning integration."""

import math
import os
import random
import time
import warnings
from pathlib import Path

import lightning
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from entmax import entmax_bisect
from geomloss import SamplesLoss
from pytorch_lightning.utilities import rank_zero_only
from torch import distributed, nn
from torch.nn import functional

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")


def ruler_masking(
    input_tensor: torch.Tensor,
    mask_ratio: float,
) -> torch.Tensor:
    """Mask cells from one extreme of the first principal component."""
    batch_size, number_cells, _ = input_tensor.shape
    device = input_tensor.device
    number_masked = int(number_cells * mask_ratio)

    if number_masked == 0:
        return torch.zeros(
            batch_size,
            number_cells,
            device=device,
            dtype=torch.bool,
        )

    float_input = input_tensor.float()
    centered_input = float_input - float_input.mean(dim=1, keepdim=True)

    _, _, right_singular_vectors = torch.linalg.svd(
        centered_input,
        full_matrices=False,
    )
    first_principal_component = right_singular_vectors[:, 0, :]

    one_dimensional_projection = torch.einsum(
        "bnd,bd->bn",
        centered_input,
        first_principal_component,
    )
    sorted_indices = one_dimensional_projection.argsort(dim=1)

    if torch.rand(1).item() > 0.5:
        mask_indices = sorted_indices[:, -number_masked:]
    else:
        mask_indices = sorted_indices[:, :number_masked]

    mask = torch.zeros(
        batch_size,
        number_cells,
        device=device,
        dtype=torch.bool,
    )
    mask.scatter_(1, mask_indices, True)
    return mask


class MAB(nn.Module):
    """Apply multi-head attention with residual feed-forward processing."""

    def __init__(
        self,
        dim_Q: int,
        dim_K: int,
        dim_V: int,
        num_heads: int,
        ln: bool = False,
        softmax_type: str = "regular",
        use_checkpoint: bool = True,
        ent: float = 1.15,
    ) -> None:
        """Initialize the attention block."""
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.dim_split = dim_V // num_heads
        self.softmax_type = softmax_type
        self.use_checkpoint = use_checkpoint
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.swig = SwiGLU(dim_V, dim_V)
        self.ent = ent

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        softmax_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform queries using attention over keys and values."""
        if softmax_type is None:
            softmax_type = self.softmax_type

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        S = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_split)

        if softmax_type == "regular":
            A = torch.softmax(S, dim=2)
        elif softmax_type == "sparse":
            A = entmax_bisect(S, alpha=self.ent, dim=2)
        else:
            raise ValueError(
                f"Unknown softmax_type: {softmax_type}, expected 'regular' or 'sparse'."
            )

        output = torch.cat((A.bmm(V_)).split(Q.size(0), 0), 2)
        output = Q + output
        output = output if getattr(self, "ln0", None) is None else self.ln0(output)
        output = output + self.swig(output)
        output = output if getattr(self, "ln1", None) is None else self.ln1(output)
        return output, A


class SAB(nn.Module):
    """Apply self-attention through a multi-head attention block."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        ln: bool = False,
        softmax_type: str = "regular",
    ) -> None:
        """Initialize the self-attention block."""
        super().__init__()
        self.mab = MAB(
            dim_in,
            dim_in,
            dim_out,
            num_heads,
            ln=ln,
            softmax_type=softmax_type,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention to the input tensor."""
        output, attention = self.mab(input_tensor, input_tensor)
        output = output + input_tensor
        return output, attention


class IPAB(nn.Module):
    """Encode large sets through learned inducing prototypes."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        num_inds: int,
        ln: bool = True,
        softmax_type: str = "sparse",
    ) -> None:
        """Initialize the induced-prototype attention block."""
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, int(num_inds), dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(
            dim_out,
            dim_in,
            dim_out,
            num_heads,
            ln=ln,
        )
        self.mab1 = MAB(
            dim_out,
            dim_in,
            dim_out,
            num_heads,
            ln=ln,
        )
        self.softmax_type = softmax_type

    def forward(
        self,
        input_tensor: torch.Tensor,
        use_self_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode a set using prototypes or direct self-attention."""
        if use_self_attention:
            hidden = input_tensor
            prototype_attention = None
        else:
            inducing_points = self.I.repeat(input_tensor.size(0), 1, 1)
            hidden, prototype_attention = self.mab0(
                inducing_points,
                input_tensor,
                self.softmax_type,
            )

        output, output_attention = self.mab1(
            input_tensor,
            hidden,
            self.softmax_type,
        )
        return output, output_attention, prototype_attention


class PMA(nn.Module):
    """Pool set elements through learned attention seeds."""

    def __init__(
        self,
        dim: int,
        dim_latent: int,
        num_heads: int,
        num_seeds: int,
        ln: bool = False,
        softmax_type: str = "regular",
    ) -> None:
        """Initialize pooling by multi-head attention."""
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_latent))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(
            dim_latent,
            dim,
            dim_latent,
            num_heads,
            ln=ln,
            softmax_type=softmax_type,
        )
        self.softmax_type = softmax_type

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool the input set into learned seed representations."""
        output, attention = self.mab(
            self.S.repeat(input_tensor.size(0), 1, 1),
            input_tensor,
            self.softmax_type,
        )
        return output, attention


class SwiGLU(nn.Module):
    """Apply a SwiGLU feed-forward transformation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        """Initialize the SwiGLU transformation."""
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Transform the input through a gated linear unit."""
        gate_input = self.w12(input_tensor)
        activation_input, linear_input = gate_input.chunk(2, dim=-1)
        hidden = functional.silu(activation_input) * linear_input
        return self.w3(hidden)


class SetTransformer(nn.Module):
    """Encode, pool, and reconstruct unordered cell sets."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_inds: int = 16,
        dim_hidden: int = 128,
        dim_latent: int = 256,
        num_heads: int = 1,
        num_seeds: int = 1,
        num_outputs: int = 30000,
        ln: bool = True,
    ) -> None:
        """Initialize the set-transformer encoder and decoder."""
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_input = dim_input
        self.num_seeds = num_seeds
        self.num_outputs = num_outputs

        self.enc1 = nn.Linear(dim_input, dim_hidden)
        self.enc2 = IPAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            num_inds,
            ln=ln,
            softmax_type="sparse",
        )
        self.enc3 = IPAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            num_inds,
            ln=ln,
            softmax_type="sparse",
        )
        self.enc4 = IPAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            num_inds,
            ln=ln,
            softmax_type="sparse",
        )
        self.enc5 = IPAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            num_inds,
            ln=ln,
            softmax_type="sparse",
        )
        self.enc6 = IPAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            num_inds,
            ln=ln,
            softmax_type="sparse",
        )

        self.pma = PMA(
            dim_hidden,
            dim_latent,
            num_heads,
            num_seeds,
            ln=ln,
            softmax_type="sparse",
        )
        self.project = nn.Linear(dim_latent, dim_latent)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim_latent))
        self.dec1a = PMA(
            dim_latent,
            dim_hidden,
            num_heads,
            num_outputs,
            ln=ln,
            softmax_type="regular",
        )
        self.dec2 = SAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            ln=ln,
            softmax_type="regular",
        )
        self.dec3 = SAB(
            dim_hidden,
            dim_hidden,
            num_heads,
            ln=ln,
            softmax_type="regular",
        )
        self.dec4 = SwiGLU(dim_hidden, dim_hidden, dim_output)

    def forward_encoder(
        self,
        input_tensor: torch.Tensor,
        use_self_attention: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Encode cells and return intermediate attention weights."""
        encoded = self.enc1(input_tensor)
        encoded, attention_1, prototype_attention_1 = self.enc2(
            encoded,
            use_self_attention=use_self_attention,
        )
        encoded, attention_2, prototype_attention_2 = self.enc3(
            encoded,
            use_self_attention=use_self_attention,
        )
        encoded, attention_3, prototype_attention_3 = self.enc4(
            encoded,
            use_self_attention=use_self_attention,
        )
        encoded, _, _ = self.enc5(
            encoded,
            use_self_attention=use_self_attention,
        )
        encoded, _, _ = self.enc6(
            encoded,
            use_self_attention=use_self_attention,
        )
        return (
            encoded,
            attention_1,
            attention_2,
            attention_3,
            prototype_attention_1,
            prototype_attention_2,
            prototype_attention_3,
        )

    def forward_pooling(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool encoded cells into a sample representation."""
        pooled, pooling_attention = self.pma(input_tensor)
        projection_logits = self.project(pooled)
        return pooled, projection_logits, pooling_attention

    def forward_decoder(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode a pooled representation into a reconstructed cell set."""
        batch_size, total_length = mask.shape
        number_unmasked = (~mask).sum(dim=1)[0]
        repeated_latent = latent.repeat(1, number_unmasked, 1)
        mask_tokens = self.mask_token.expand(
            batch_size,
            total_length,
            latent.shape[-1],
        )
        decoder_input = mask_tokens.clone()
        decoder_input[~mask] = repeated_latent.reshape(-1, latent.shape[-1])

        decoded, _ = self.dec1a(decoder_input)
        decoded, _ = self.dec2(decoded)
        decoded, _ = self.dec3(decoded)
        return self.dec4(decoded)

    def forward(
        self,
        input_tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Encode and reconstruct a masked cell set."""
        encoded, attention_1, attention_2, *_ = self.forward_encoder(input_tensor)
        latent, projection, pooling_attention = self.forward_pooling(encoded)
        prediction = self.forward_decoder(latent, mask)
        return (
            prediction,
            projection,
            latent,
            attention_1,
            attention_2,
            pooling_attention,
        )


class MAESTRO(nn.Module):
    """Train paired student and teacher set transformers."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_inds: int,
        dim_hidden: int,
        dim_latent: int,
        num_heads: int,
        num_outputs: int,
        ln: bool,
        number_cells_subset: int,
        student_temperature: float,
        teacher_temperature: float,
        sinkhorn_start: int = 0,
        center_momentum: float = 0.99,
        teacher_beta: float = 0.99,
    ) -> None:
        """Initialize student and momentum-teacher networks."""
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_inds = num_inds
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.num_heads = num_heads
        self.num_outputs = num_outputs
        self.ln = ln
        self.number_cells_subset = number_cells_subset
        self.sinkhorn_start = sinkhorn_start
        self.energy_loss = SamplesLoss(loss="energy", p=2, verbose=False)
        self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, verbose=False)
        self.use_sinkhorn = sinkhorn_start == 0
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.teacher_beta = teacher_beta
        self.register_buffer("center_latent", torch.ones((1, dim_latent)))
        self.cell_token = nn.Parameter(torch.zeros(1, dim_input))

        self.student = SetTransformer(
            dim_input=dim_input,
            dim_output=dim_output,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            num_heads=num_heads,
            num_outputs=num_outputs,
            ln=ln,
        )

        self.teacher = SetTransformer(
            dim_input=dim_input,
            dim_output=dim_output,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            dim_latent=dim_latent,
            num_heads=num_heads,
            num_outputs=num_outputs,
            ln=ln,
        )

        self._init_student()
        self._init_teacher()

    def _init_student(self) -> None:
        """Initialize trainable student layers."""
        for module in self.student.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                if module.weight is not None:
                    with torch.no_grad():
                        nn.init.trunc_normal_(module.weight, std=0.05)
                if module.bias is not None:
                    with torch.no_grad():
                        module.bias.data.fill_(0.0)
                if isinstance(module, nn.LayerNorm):
                    with torch.no_grad():
                        module.weight.data.fill_(1.0)

    def _init_teacher(self) -> None:
        """Copy student weights into the frozen teacher."""
        for student_parameter, teacher_parameter in zip(
            self.student.parameters(),
            self.teacher.parameters(),
            strict=True,
        ):
            teacher_parameter.data.copy_(student_parameter.data)
            teacher_parameter.requires_grad = False

    def calculate_kl_divergence(
        self,
        student_logits: torch.Tensor,
        teacher_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate divergence between student and teacher projections."""
        student_log_probabilities = functional.log_softmax(
            student_logits / self.student_temperature, dim=-1
        )
        return functional.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )

    def apply_centering_and_sharpening(
        self,
        teacher_output: torch.Tensor,
        teacher_center: torch.Tensor,
    ) -> torch.Tensor:
        """Center and sharpen a teacher projection."""
        centered_output = teacher_output - teacher_center.unsqueeze(0)
        sharpened_output = torch.softmax(
            centered_output / self.teacher_temperature, dim=-1
        )
        return sharpened_output

    @torch.no_grad()
    def _update_teacher(
        self,
        beta: float | None = None,
    ) -> None:
        """Update teacher parameters using student parameter momentum."""
        if beta is None:
            beta = self.teacher_beta
        for (_, student_parameter), (_, teacher_parameter) in zip(
            self.student.named_parameters(),
            self.teacher.named_parameters(),
            strict=True,
        ):
            student_parameter_data = student_parameter.data.to(teacher_parameter.device)
            teacher_parameter.data = (
                beta * teacher_parameter.data + (1.0 - beta) * student_parameter_data
            )

    @torch.no_grad()
    def update_center(
        self,
        teacher_output: torch.Tensor,
        teacher_center: torch.Tensor,
        momentum: float | None = None,
    ) -> None:
        """Update the teacher projection center using momentum."""
        if momentum is None:
            momentum = self.center_momentum

        teacher_sum = teacher_output.detach().float().sum(dim=0)
        teacher_count = torch.tensor(
            teacher_output.shape[0],
            device=teacher_output.device,
            dtype=torch.float32,
        )
        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(teacher_sum)
            distributed.all_reduce(teacher_count)

        current_mean = teacher_sum / teacher_count.clamp_min(1.0)
        teacher_center.mul_(momentum).add_(
            current_mean.to(teacher_center),
            alpha=1 - momentum,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Calculate reconstruction and self-distillation losses."""
        batch_size, total_cells, marker_count = input_tensor.shape
        subset_cell_count = min(self.number_cells_subset, total_cells)
        device = input_tensor.device

        if total_cells > subset_cell_count:
            indices = torch.stack(
                [
                    torch.randperm(total_cells, device=device)[:subset_cell_count]
                    for _ in range(batch_size)
                ]
            )
        else:
            indices = torch.stack(
                [
                    torch.randint(
                        0,
                        total_cells,
                        (subset_cell_count,),
                        device=device,
                    )
                    for _ in range(batch_size)
                ]
            )
        input_subset = input_tensor[
            torch.arange(batch_size, device=device).unsqueeze(1),
            indices,
        ]

        mask_rate = random.choice([0.0, 0.2, 0.4, 0.6, 0.8])
        mask = ruler_masking(input_subset, mask_rate)
        masked_input = input_subset[~mask].view(batch_size, -1, marker_count)

        student_prediction, student_projection_logits, *_ = self.student(
            masked_input,
            mask,
        )

        with torch.no_grad():
            teacher_latent, *_ = self.teacher.forward_encoder(input_tensor)
            _, teacher_projection_logits, _ = self.teacher.forward_pooling(
                teacher_latent
            )

        teacher_probabilities = self.apply_centering_and_sharpening(
            teacher_projection_logits,
            self.center_latent,
        )
        self.update_center(teacher_projection_logits, self.center_latent)

        reconstruction_loss_function = (
            self.sinkhorn_loss if self.use_sinkhorn else self.energy_loss
        )
        reconstruction_loss = reconstruction_loss_function(
            input_subset.to(torch.float32),
            student_prediction.to(torch.float32),
        ).mean()

        distillation_loss = self.calculate_kl_divergence(
            student_projection_logits,
            teacher_probabilities,
        )
        loss = reconstruction_loss + distillation_loss

        return (
            loss,
            reconstruction_loss,
            distillation_loss,
            masked_input,
            student_prediction,
        )


class MAESTROLightning(lightning.LightningModule):
    """Integrate MAESTRO with Lightning training infrastructure."""

    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_inds: int,
        dim_hidden: int,
        dim_latent: int,
        num_heads: int,
        num_outputs: int,
        ln: bool,
        number_cells_subset: int,
        initial_lr: float,
        min_lr: float,
        epochs: int,
        student_temperature: float,
        teacher_temperature: float,
        output_path: str | Path,
        sinkhorn_start: int = 0,
        center_momentum: float = 0.99,
        teacher_beta: float = 0.99,
    ) -> None:
        """Initialize the Lightning training module."""
        super().__init__()
        self.save_hyperparameters()
        self.model = MAESTRO(
            dim_input=self.hparams.dim_input,
            dim_output=self.hparams.dim_output,
            num_inds=self.hparams.num_inds,
            dim_hidden=self.hparams.dim_hidden,
            dim_latent=self.hparams.dim_latent,
            num_heads=self.hparams.num_heads,
            num_outputs=self.hparams.num_outputs,
            ln=self.hparams.ln,
            number_cells_subset=self.hparams.number_cells_subset,
            student_temperature=self.hparams.student_temperature,
            teacher_temperature=self.hparams.teacher_temperature,
            sinkhorn_start=self.hparams.sinkhorn_start,
            center_momentum=self.hparams.center_momentum,
            teacher_beta=self.hparams.teacher_beta,
        )
        self.epoch_start_time = 0
        self.epoch_loss = []
        self.epoch_sinkhorn = []
        self.epoch_distillation = []
        self.best_loss = 1000000
        self.model_start_time = None

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run a MAESTRO training forward pass."""
        return self.model(input_tensor)

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> torch.Tensor:
        """Calculate and log losses for one training batch."""
        del batch_idx
        data_tensor, *_ = batch
        sinkhorn_start = self.hparams.sinkhorn_start
        self.model.use_sinkhorn = (
            sinkhorn_start == 0 or self.current_epoch >= sinkhorn_start
        )
        loss, reconstruction_loss, distillation_loss, *_ = self.model(data_tensor)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=data_tensor.shape[0],
        )
        self.epoch_loss.append(loss.detach())
        self.epoch_sinkhorn.append(reconstruction_loss.detach())
        self.epoch_distillation.append(distillation_loss.detach())

        return loss

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[dict[str, object]]]:
        """Configure AdamW with cosine learning-rate annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.initial_lr, weight_decay=1e-3
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs,
                eta_min=self.hparams.min_lr,
            ),
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    @rank_zero_only
    def on_train_start(self) -> None:
        """Save configuration and start training-time measurement."""
        print("🎼 MAESTRO 🎼")
        print(
            f"🏋🏻‍♂️  Beginning training at 🗓️  {time.strftime('%a, %d %b %Y %H:%M:%S')}"
        )
        model_configuration = {
            "dim_input": self.hparams.dim_input,
            "dim_output": self.hparams.dim_output,
            "num_inds": self.hparams.num_inds,
            "dim_hidden": self.hparams.dim_hidden,
            "dim_latent": self.hparams.dim_latent,
            "num_heads": self.hparams.num_heads,
            "num_outputs": self.hparams.num_outputs,
            "ln": self.hparams.ln,
            "number_cells_subset": self.hparams.number_cells_subset,
            "initial_lr": self.hparams.initial_lr,
            "min_lr": self.hparams.min_lr,
            "epochs": self.hparams.epochs,
            "student_temperature": self.hparams.student_temperature,
            "teacher_temperature": self.hparams.teacher_temperature,
            "center_momentum": self.hparams.center_momentum,
            "teacher_beta": self.hparams.teacher_beta,
            "sinkhorn_start": self.hparams.sinkhorn_start,
            "output_path": self.hparams.output_path,
        }
        configuration_path = Path(self.hparams.output_path) / "config.pth"
        torch.save(model_configuration, configuration_path)
        self.model_start_time = time.time()

    @rank_zero_only
    def on_train_end(self) -> None:
        """Report total training duration."""
        model_end_time = time.time()
        model_duration = model_end_time - self.model_start_time
        print(f"🕰️ Time to train entire model was {model_duration:.2f} seconds 🕰️")
        print(f"Training finished at 🗓️ {time.strftime('%a, %d %b %Y %H:%M:%S')}")

    def on_train_epoch_start(self) -> None:
        """Start epoch-time measurement."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        """Clear epoch state and report metrics on the primary rank."""
        average_loss = torch.stack(self.epoch_loss).mean().item()
        average_reconstruction_loss = torch.stack(self.epoch_sinkhorn).mean().item()
        average_distillation_loss = torch.stack(self.epoch_distillation).mean().item()
        epoch_duration = (time.time() - self.epoch_start_time) / 60

        self.epoch_loss.clear()
        self.epoch_sinkhorn.clear()
        self.epoch_distillation.clear()

        if self.global_rank != 0:
            return

        loss_name = "sinkhorn" if self.model.use_sinkhorn else "energy"
        print(
            f"🎶 Epoch {self.current_epoch} [{loss_name}] | "
            f"⏱️  Duration: {epoch_duration:.2f} min | "
            f"💰 Loss: {average_loss} | "
            f"⚡️ Recon: {average_reconstruction_loss:.4f} | "
            f"⚗️ Distillation: {average_distillation_loss:.3e} 🎶\n"
        )

        if self.current_epoch % 10 == 0:
            try:
                self._visualize_reconstructions(self.current_epoch)
            except Exception as error:
                print(f"⚠️ Visualization failed at epoch {self.current_epoch}: {error}")

    @rank_zero_only
    @torch.no_grad()
    def _visualize_reconstructions(
        self,
        epoch: int,
        number_samples: int = 3,
        number_cells_visualized: int = 10000,
        mask_ratio: float = 0.5,
    ) -> None:
        """Save UMAP comparisons of target and reconstructed cell sets."""
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="umap")

        device = self.device
        model = self.model
        model.eval()

        cell_type_colors = {
            "T cell CD4 Naive": "#009874",
            "T cell CD4 Mem": "#FF6F61",
            "T cell CD4 EMRA": "#5A5B9F",
            "T cell CD8 Naive": "#BF1932",
            "T cell CD8 Mem": "#F5DF4D",
            "T cell gd": "#D94F70",
            "T cell DN": "#6667AB",
            "B cell": "#92A8D1",
            "Plasmablast": "#DECDBE",
            "Monocyte Classical": "#88B04B",
            "Monocyte Nonclassical": "#9B1B30",
            "mDC": "#F0C05A",
            "pDC": "#53B0AE",
            "Neutrophil": "#0F4C81",
            "Eosinophil": "#F7CAC9",
            "Basophil": "#5F4B8B",
            "NK cell": "#E2583E",
        }
        default_color = "#AAAAAA"
        status_colors = {
            "unmasked": "#FFB703",
            "target": "#E63946",
            "predicted": "#0077B6",
        }

        data_loader = self.trainer.train_dataloader
        batches, cell_type_batches, names = [], [], []
        for batch in data_loader:
            data_tensor, cell_types_batch, sample_names = batch
            for index in range(data_tensor.shape[0]):
                batches.append(data_tensor[index])
                cell_type_batches.append(cell_types_batch[index])
                names.append(sample_names[index])
            if len(batches) >= number_samples * 3:
                break

        if len(batches) < number_samples:
            model.train()
            return

        sample_indices = random.sample(range(len(batches)), number_samples)
        figure, axes = plt.subplots(
            number_samples,
            5,
            figsize=(20, 4 * number_samples),
        )

        column_titles = [
            "Cell Types",
            "All (by status)",
            "Unmasked",
            "Target (masked)",
            "Predicted",
        ]

        for row, sample_index in enumerate(sample_indices):
            input_data = batches[sample_index].to(device).unsqueeze(0)
            sample_cell_types = cell_type_batches[sample_index]
            sample_name = names[sample_index]

            total_cells = input_data.shape[1]
            subset_cell_count = min(number_cells_visualized, total_cells)
            if total_cells > subset_cell_count:
                subset_indices = torch.randperm(total_cells, device=device)[
                    :subset_cell_count
                ].unsqueeze(0)
                input_data = input_data.gather(
                    1,
                    subset_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        input_data.shape[-1],
                    ),
                )
                if isinstance(sample_cell_types, torch.Tensor):
                    sample_cell_types = sample_cell_types[
                        subset_indices.squeeze(0).cpu()
                    ]

            mask = ruler_masking(input_data, mask_ratio)

            batch_size, _, marker_count = input_data.shape
            number_unmasked = (~mask).sum(dim=1)[0].item()
            number_masked = mask.sum(dim=1)[0].item()

            if number_masked == 0 or number_unmasked == 0:
                for column in range(5):
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])
                continue

            model_dtype = next(model.parameters()).dtype
            unmasked = (
                input_data[~mask]
                .view(
                    batch_size,
                    number_unmasked,
                    marker_count,
                )
                .to(model_dtype)
            )
            masked_target = input_data[mask].view(
                batch_size,
                number_masked,
                marker_count,
            )

            encoded, *_ = model.student.forward_encoder(unmasked)
            latent, _, _ = model.student.forward_pooling(encoded)
            complete_prediction = model.student.forward_decoder(latent, mask)
            # PMA produces a fixed-size set, not one prediction per input cell.
            # Subsample masked points for visualization.
            all_predictions = complete_prediction[0]
            prediction_indices = torch.randperm(
                all_predictions.shape[0],
                device=all_predictions.device,
            )[:number_masked]
            prediction = all_predictions[prediction_indices].unsqueeze(0)

            unmasked_array = unmasked.squeeze(0).float().cpu().numpy()
            target_array = masked_target.squeeze(0).float().cpu().numpy()
            prediction_array = prediction.squeeze(0).float().cpu().numpy()

            del (
                complete_prediction,
                encoded,
                latent,
                masked_target,
                prediction,
                unmasked,
            )
            torch.cuda.empty_cache()

            combined = np.vstack([unmasked_array, target_array, prediction_array])
            unmasked_count = len(unmasked_array)
            target_count = len(target_array)
            prediction_count = len(prediction_array)

            cpu_mask = mask.squeeze(0).cpu()
            unmasked_indices = (~cpu_mask).nonzero(as_tuple=True)[0]
            masked_indices = cpu_mask.nonzero(as_tuple=True)[0]

            dataset = data_loader.dataset
            if isinstance(sample_cell_types, torch.Tensor):
                if hasattr(dataset, "get_cell_type_name"):
                    unmasked_cell_types = [
                        dataset.get_cell_type_name(sample_cell_types[index].item())
                        for index in unmasked_indices
                    ]
                    masked_cell_types = [
                        dataset.get_cell_type_name(sample_cell_types[index].item())
                        for index in masked_indices
                    ]
                else:
                    unmasked_cell_types = [
                        f"Unknown_{sample_cell_types[index].item()}"
                        for index in unmasked_indices
                    ]
                    masked_cell_types = [
                        f"Unknown_{sample_cell_types[index].item()}"
                        for index in masked_indices
                    ]
            else:
                unmasked_cell_types = [
                    sample_cell_types[index] for index in unmasked_indices
                ]
                masked_cell_types = [
                    sample_cell_types[index] for index in masked_indices
                ]

            try:
                reducer = umap.UMAP(n_neighbors=30, min_dist=0.3)
                embedding = reducer.fit_transform(combined)
            except Exception as error:
                print(f"⚠️ UMAP failed for {sample_name}: {error}")
                for column in range(5):
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])
                continue

            unmasked_embedding = embedding[:unmasked_count]
            target_embedding = embedding[unmasked_count : unmasked_count + target_count]
            prediction_embedding = embedding[unmasked_count + target_count :]

            x_minimum = embedding[:, 0].min() - 0.5
            x_maximum = embedding[:, 0].max() + 0.5
            y_minimum = embedding[:, 1].min() - 0.5
            y_maximum = embedding[:, 1].max() + 0.5

            axis = axes[row, 0]
            all_cell_types = (
                unmasked_cell_types
                + masked_cell_types
                + ["predicted"] * prediction_count
            )
            cell_type_plot_colors = [
                cell_type_colors.get(cell_type, default_color)
                for cell_type in all_cell_types
            ]
            axis.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=cell_type_plot_colors,
                s=1.5,
                alpha=0.5,
            )
            axis.set_xlim(x_minimum, x_maximum)
            axis.set_ylim(y_minimum, y_maximum)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_ylabel(f"{sample_name[:25]}", fontsize=8)
            if row == 0:
                axis.set_title(column_titles[0], fontsize=9)

            axis = axes[row, 1]
            axis.scatter(
                unmasked_embedding[:, 0],
                unmasked_embedding[:, 1],
                c=status_colors["unmasked"],
                s=1.5,
                alpha=0.4,
                label="unmasked",
            )
            axis.scatter(
                target_embedding[:, 0],
                target_embedding[:, 1],
                c=status_colors["target"],
                s=1.5,
                alpha=0.4,
                label="target",
            )
            axis.scatter(
                prediction_embedding[:, 0],
                prediction_embedding[:, 1],
                c=status_colors["predicted"],
                s=1.5,
                alpha=0.4,
                label="predicted",
            )
            axis.set_xlim(x_minimum, x_maximum)
            axis.set_ylim(y_minimum, y_maximum)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[1], fontsize=9)
                axis.legend(markerscale=5, fontsize=6, loc="upper right")

            axis = axes[row, 2]
            axis.scatter(
                unmasked_embedding[:, 0],
                unmasked_embedding[:, 1],
                c=status_colors["unmasked"],
                s=1.5,
                alpha=0.5,
            )
            axis.set_xlim(x_minimum, x_maximum)
            axis.set_ylim(y_minimum, y_maximum)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[2], fontsize=9)

            axis = axes[row, 3]
            axis.scatter(
                target_embedding[:, 0],
                target_embedding[:, 1],
                c=status_colors["target"],
                s=1.5,
                alpha=0.5,
            )
            axis.set_xlim(x_minimum, x_maximum)
            axis.set_ylim(y_minimum, y_maximum)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[3], fontsize=9)

            axis = axes[row, 4]
            axis.scatter(
                prediction_embedding[:, 0],
                prediction_embedding[:, 1],
                c=status_colors["predicted"],
                s=1.5,
                alpha=0.5,
            )
            axis.set_xlim(x_minimum, x_maximum)
            axis.set_ylim(y_minimum, y_maximum)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(column_titles[4], fontsize=9)

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=6,
                label=cell_type,
            )
            for cell_type, color in cell_type_colors.items()
        ]
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=default_color,
                markersize=6,
                label="Unknown",
            )
        )
        figure.legend(
            handles=legend_elements,
            loc="center right",
            bbox_to_anchor=(1.08, 0.5),
            fontsize=7,
            frameon=False,
        )

        figure_directory = Path(self.hparams.output_path) / "reconstruction_viz"
        figure_directory.mkdir(parents=True, exist_ok=True)
        figure_path = figure_directory / f"epoch_{epoch:04d}.pdf"

        plt.suptitle(f"Reconstruction — Epoch {epoch} (mask={mask_ratio})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 0.93, 0.97])
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

        print(f"📊 Saved reconstruction visualization to {figure_path}")
        model.train()
