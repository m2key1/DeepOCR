import functools
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import albumentations as A
import jiwer
import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import traceback
import torchvision
from albumentations.pytorch import ToTensorV2
from matplotlib import cm
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.backbones import build_backbone
from utils.tokenizer import DeepTokenizer

matplotlib.use('Agg')


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_display_image(images: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.as_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (images.detach().float().cpu() * std + mean).clamp(0, 1)


def _log_tensorboard_descriptions(writer: SummaryWriter):
    writer.add_text('guide/features', (
        '**norm_before_encoder** - Average L2 norm of feature vectors entering the MoE encoder. '
        'A collapse toward zero signals vanishing activations; sudden spikes suggest exploding gradients.\n\n'
        '**norm_after_encoder** - Average L2 norm after the MoE encoder. '
        'Compare with norm_before_encoder: a healthy encoder changes norms moderately. '
        'If both values track each other too closely the encoder may not be transforming features enough.'
    ))
    writer.add_text('guide/images', (
        '**input/sample** - The raw input image (de-normalized) used for the visualization batch.\n\n'
        '**validation_samples** - Up to 4 validation lines from a random batch, stacked vertically. '
        'Validation uses no augmentation, so this shows the resize, padding and normalisation only; '
        'for the augmentation stack see train_augmented.\n\n'
        '**train_augmented** - Four training lines exactly as the model receives them, augmentation '
        'applied, stacked one per row and logged once an epoch. This is where a transform that '
        'erases strokes or washes out the page shows up.\n\n'
        '**confidence_map** - Max predicted probability per CTC output step (a patch-column when '
        'the head collapses height, otherwise per patch), overlaid on the input as vertical bands. '
        'Bright regions mean the model is confident about its prediction there. '
        'Look for low confidence on text regions as a sign of underfitting.\n\n'
        '**confidence_margin_map** - Gap between the top-1 and top-2 predicted probabilities per output step. '
        'High margin means an unambiguous prediction; low margin means the model is torn between two characters. '
        'Persistently low margins on certain columns highlight where the model struggles most.\n\n'
        '**entropy_map** - Shannon entropy of the output distribution per output step. '
        'High entropy (bright in plasma colormap) = uncertain, spread-out distribution. '
        'Low entropy = peaked, decisive prediction. The inverse of confidence but more sensitive to '
        'how probability mass is distributed across all classes.\n\n'
        '**feature_norm_map_before / _after** - Per-patch L2 norm of features before and after the MoE encoder. '
        'Spatial uniformity is expected on padding; large norm variation on content patches can reveal '
        'which image regions the encoder amplifies or suppresses.\n\n'
        '**feature_norm_delta_map** - Absolute difference in per-patch norm before vs. after the encoder. '
        'Highlights where the encoder is making the largest changes. '
        'If delta is near zero everywhere the encoder layers may be under-contributing.\n\n'
        '**feature_mean_map_before / _after** - Per-patch mean activation before and after encoding. '
        'Useful for spotting bias shifts introduced by the encoder. '
        'Large systematic shifts may indicate layer-norm or initialization issues.\n\n'
        '**predicted_token_map** - Index of the highest-probability token per patch, shown as a heatmap. '
        'Uniform color over padding (usually the blank token) is expected. '
        'On text regions, variation indicates the model is predicting different characters.\n\n'
        '**span_mask** - The MAE-style column span mask, red where columns were replaced by the '
        'learned mask token so the encoder has to reconstruct them from context. Training only: '
        'validation runs unmasked, so there is nothing to show.\n\n'
        '**expert_routing_layer_N** - Which MoE expert was selected for each patch in layer N. '
        'Each color represents a different expert. Look for spatial structure: '
        'if one expert dominates everything, load balancing may be failing. '
        'Ideally experts should specialize across different spatial regions or content types.'
    ))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model: int, max_height: int = 50, max_width: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        self.height_embed = nn.Parameter(torch.randn(max_height, d_model // 2))
        self.width_embed = nn.Parameter(torch.randn(max_width, d_model // 2))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        _, seq_len, d_model = x.shape
        assert seq_len == height * width, f"Sequence length {seq_len} != {height} * {width}"

        h_embed = self.height_embed[:height]    # [H, d_model//2]
        w_embed = self.width_embed[:width]    # [W, d_model//2]

        h_grid = h_embed.unsqueeze(1).expand(height, width, -1)    # [H, W, d_model//2]
        w_grid = w_embed.unsqueeze(0).expand(height, width, -1)    # [H, W, d_model//2]

        pos_embed = torch.cat([h_grid, w_grid], dim=-1)    # [H, W, d_model]

        pos_embed = pos_embed.transpose(0, 1)

        pos_embed = pos_embed.reshape(width * height, d_model)    # [W*H, d_model]

        x = x + pos_embed.unsqueeze(0)    # [B, W*H, d_model]

        return self.dropout(x)


class Expert(nn.Module):

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


GROUPED_MM = hasattr(torch, '_grouped_mm')


@functools.lru_cache(maxsize=None)
def _grouped_mm_ok(device: torch.device) -> bool:
    """torch 2.8 exports _grouped_mm but raises below compute capability 9.0, while 2.10 also runs
    it on Ada and Blackwell, so the op is tried once per device instead of pinning a capability.
    The per-expert loop fallback is slower but computes the same thing."""
    if not GROUPED_MM:
        return False
    try:
        x = torch.zeros(2, 8, device=device, dtype=torch.bfloat16)
        w = torch.zeros(1, 8, 8, device=device, dtype=torch.bfloat16)
        torch._grouped_mm(x, w, offs=torch.tensor([2], device=device, dtype=torch.int32))
    except RuntimeError:
        return False
    return True


class SwiGLUExpert(nn.Module):
    """down(silu(gate(x)) * up(x)). No dropout, since sparse routing already regularises. At equal
    parameters the hidden width is two thirds of the GELU expert's."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SwiGLUExperts(nn.Module):
    """All routed SwiGLU experts of one block as stacked weights, one grouped matmul per projection.

    The grouped cost does not depend on routing imbalance, which is high early in training, whereas
    a per-expert loop or a batched matmul padded to the busiest expert pays for it. Without CUDA the
    loop is used."""

    def __init__(self, num_experts: int, d_model: int, hidden: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate_proj = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.up_proj = nn.Parameter(torch.empty(num_experts, d_model, hidden))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, d_model))
        # nn.Linear's default: uniform within 1/sqrt(fan_in)
        for weight, fan_in in ((self.gate_proj, d_model), (self.up_proj, d_model), (self.down_proj, hidden)):
            nn.init.uniform_(weight, -fan_in ** -0.5, fan_in ** -0.5)

    def forward(self, tokens: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """tokens [N, D]; expert_ids and weights [N, K]. Returns the weighted sum of each token's
        K expert outputs, [N, D]."""
        num_tokens, d_model = tokens.shape
        flat_expert = expert_ids.reshape(-1)
        flat_token = torch.arange(num_tokens, device=tokens.device).repeat_interleave(expert_ids.size(1))
        flat_weight = weights.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        sorted_tokens = tokens[flat_token[order]]
        counts = torch.bincount(flat_expert, minlength=self.num_experts)

        if tokens.is_cuda and _grouped_mm_ok(tokens.device):
            # autocast does not know this op, so the cast to its dtype is explicit
            dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else sorted_tokens.dtype
            xs = sorted_tokens.to(dtype)
            offsets = torch.cumsum(counts, 0).to(torch.int32)
            hidden = (F.silu(torch._grouped_mm(xs, self.gate_proj.to(dtype), offs=offsets))
                      * torch._grouped_mm(xs, self.up_proj.to(dtype), offs=offsets))
            out = torch._grouped_mm(hidden, self.down_proj.to(dtype), offs=offsets)
        else:
            chunks, start = [], 0
            for expert, count in enumerate(counts.tolist()):
                chunk = sorted_tokens[start:start + count]
                chunks.append((F.silu(chunk @ self.gate_proj[expert]) * (chunk @ self.up_proj[expert])) @ self.down_proj[expert])
                start += count
            out = torch.cat(chunks)

        # weight and accumulate in the tokens' precision, not the experts' bf16
        combined = tokens.new_zeros(num_tokens, d_model)
        combined.index_add_(0, flat_token[order], out.to(tokens.dtype) * flat_weight[order].unsqueeze(-1).to(tokens.dtype))
        return combined


class ConvExpert(nn.Module):
    """Conformer's convolution module as a routed expert. It is the only expert that sees
    neighbouring columns, which is what tells apart the stroke runs of Kurrent e, n, m and u.
    Kernel 15 spans about four characters at stride 8. LayerNorm instead of Conformer's BatchNorm
    because batch statistics would average over padded columns."""

    def __init__(self, d_model: int, kernel: int):
        super().__init__()
        self.pointwise_in = nn.Linear(d_model, 2 * d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """x [S, B, D], key_padding_mask [B, S]. Padding is zeroed before the conv so it never
        leaks into a line's last characters."""
        h = F.glu(self.pointwise_in(x), dim=-1)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)
        h = self.depthwise(h.permute(1, 2, 0)).permute(2, 0, 1)
        return self.pointwise_out(F.silu(self.norm(h)))


def build_expert(kind: str, d_model: int, hidden: int, dropout: float) -> nn.Module:
    if kind == 'gelu':
        return Expert(d_model, hidden, dropout)
    if kind == 'swiglu':
        return SwiGLUExpert(d_model, hidden)
    raise ValueError(f"Unknown expert_ffn '{kind}'; expected 'gelu' or 'swiglu'.")


class MoEBlock(nn.Module):

    def __init__(
        self, d_model: int, num_experts: int = 4, top_k: int = 2, dim_feedforward: int = 2048, dropout: float = 0.1,
        jitter_noise: float = 0.01, expert_ffn: str = 'gelu', num_shared_experts: int = 0,
        router_balance: str = 'aux_loss', router_bias_rate: float = 1e-3,
        num_conv_experts: int = 0, conv_expert_kernel: int = 15
    ):
        super().__init__()
        if num_conv_experts and expert_ffn != 'swiglu':
            raise ValueError("Convolutional experts need expert_ffn='swiglu'; the GELU loop dispatch does not route them.")
        self.num_ffn_experts = max(1, num_experts)
        self.num_conv_experts = num_conv_experts
        self.conv_expert_kernel = conv_expert_kernel
        # the router chooses among FFN and conv experts alike; conv ids follow the FFN ids
        self.num_experts = self.num_ffn_experts + num_conv_experts
        self.top_k = min(top_k, self.num_experts)
        self.jitter_noise = jitter_noise
        self.routing_prob = 1.0  # fraction of tokens routed by the learned gate, the rest at random
        self.expert_ffn = expert_ffn
        self.expert_hidden = dim_feedforward
        self.num_shared_experts = num_shared_experts
        if router_balance not in ('aux_loss', 'loss_free'):
            raise ValueError(f"Unknown router_balance '{router_balance}'; expected 'aux_loss' or 'loss_free'.")
        self.router_balance = router_balance
        self.router_bias_rate = router_bias_rate
        self._router_load = None
        self._last_max_violation = None

        if self.num_experts > 1:
            self.gate = nn.Linear(d_model, self.num_experts, bias=False)
        if self.router_balance == 'loss_free':
            # steers top-k selection only, so unlike a balance loss it adds no gradient
            self.register_buffer('router_bias', torch.zeros(self.num_experts))
        if expert_ffn == 'swiglu':
            self.experts = SwiGLUExperts(self.num_ffn_experts, d_model, dim_feedforward)
        else:
            self.experts = nn.ModuleList([build_expert(expert_ffn, d_model, dim_feedforward, dropout)
                                          for _ in range(self.num_ffn_experts)])
        self.conv_experts = nn.ModuleList([ConvExpert(d_model, conv_expert_kernel) for _ in range(num_conv_experts)])
        self.shared_experts = nn.ModuleList([build_expert(expert_ffn, d_model, dim_feedforward, dropout)
                                             for _ in range(num_shared_experts)])

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(x.shape)
        for expert in self.shared_experts:
            out = out + expert(x)
        return out

    def _conv_routed(self, x, key_padding_mask, valid_idx, topk_indices, topk_probs):
        """Conv experts run densely; each token takes their output scaled by its router weight."""
        out = None
        for offset, expert in enumerate(self.conv_experts):
            expert_id = self.num_ffn_experts + offset
            weight = (topk_probs * (topk_indices == expert_id)).sum(dim=-1)
            dense = expert(x, key_padding_mask).reshape(-1, x.size(-1))
            if valid_idx is not None:
                dense = dense[valid_idx]
            term = dense * weight.unsqueeze(-1).to(dense.dtype)
            out = term if out is None else out + term
        return out

    def forward(self,
                x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        num_tokens = x_flat.size(0)
        zero = x_flat.new_zeros(())

        if self.num_experts == 1:
            single = (self.experts(x_flat, torch.zeros(num_tokens, 1, dtype=torch.long, device=x.device),
                                   torch.ones(num_tokens, 1, device=x.device, dtype=x_flat.dtype))
                      if isinstance(self.experts, SwiGLUExperts) else self.experts[0](x_flat))
            out_flat = single + self._shared(x_flat)
            topk_full = torch.zeros(num_tokens, 1, dtype=torch.long, device=x.device)
            return out_flat.view(orig_shape), zero, zero, topk_full

        if key_padding_mask is not None:
            # x is [S, B, D] so the flat token index is s * B + b; the [B, S] mask must be
            # transposed before flattening to line up with that ordering
            valid_idx = (~key_padding_mask).transpose(0, 1).reshape(-1).nonzero(as_tuple=True)[0]
            x_valid = x_flat[valid_idx]
        else:
            valid_idx = None
            x_valid = x_flat

        num_valid = x_valid.size(0)
        if num_valid == 0:
            topk_full = torch.zeros(num_tokens, self.top_k, dtype=torch.long, device=x.device)
            return torch.zeros_like(x_flat).view(orig_shape), zero, zero, topk_full

        if self.training and self.jitter_noise > 0:
            jitter = torch.empty_like(x_valid).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            gate_input = x_valid * jitter
        else:
            gate_input = x_valid

        gate_logits = self.gate(gate_input)
        if self.router_balance == 'loss_free':
            # softmax is promoted to fp32 under autocast and sigmoid is not; the gating
            # weights deserve the same precision either way
            gate_probs = torch.sigmoid(gate_logits.float())
            _, learned_topk_indices = torch.topk(gate_probs + self.router_bias, self.top_k, dim=-1)
            learned_topk_probs = gate_probs.gather(-1, learned_topk_indices)
        else:
            gate_probs = F.softmax(gate_logits, dim=-1)
            learned_topk_probs, learned_topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        learned_topk_probs = learned_topk_probs / (learned_topk_probs.sum(dim=-1, keepdim=True) + 1e-6)

        if self.training and self.routing_prob < 1.0:
            noise = torch.rand(num_valid, self.num_experts, device=x.device)
            _, random_topk_indices = torch.topk(noise, self.top_k, dim=-1)

            random_topk_probs = torch.ones_like(learned_topk_probs) / self.top_k

            use_learned = torch.rand(num_valid, device=x.device) < self.routing_prob
            use_learned_mask = use_learned.unsqueeze(-1).expand_as(learned_topk_indices)

            topk_indices = torch.where(use_learned_mask, learned_topk_indices, random_topk_indices)
            topk_probs = torch.where(use_learned_mask, learned_topk_probs, random_topk_probs)
        else:
            topk_indices = learned_topk_indices
            topk_probs = learned_topk_probs

        flat_expert = topk_indices.reshape(-1)
        if isinstance(self.experts, SwiGLUExperts):
            ffn_indices, ffn_probs = topk_indices, topk_probs
            if self.conv_experts:
                # slots routed to a conv expert become weight-zero routes to FFN expert 0, so the
                # grouped dispatch never sees an id past its range
                is_conv = topk_indices >= self.num_ffn_experts
                ffn_indices = topk_indices.masked_fill(is_conv, 0)
                ffn_probs = topk_probs.masked_fill(is_conv, 0.0)
            out_valid = self.experts(x_valid, ffn_indices, ffn_probs)
            if self.conv_experts:
                out_valid = out_valid + self._conv_routed(x, key_padding_mask, valid_idx, topk_indices, topk_probs).to(out_valid.dtype)
        else:
            # one contiguous slice per expert instead of a full-length mask scan each
            flat_weight = topk_probs.reshape(-1)
            order = torch.argsort(flat_expert, stable=True)
            token_ids = order // self.top_k
            counts = torch.bincount(flat_expert, minlength=self.num_experts).tolist()

            out_valid = torch.zeros_like(x_valid)
            offset = 0
            for expert_id, count in enumerate(counts):
                if count > 0:
                    slice_order = order[offset:offset + count]
                    slice_tokens = token_ids[offset:offset + count]
                    expert_out = self.experts[expert_id](x_valid[slice_tokens])
                    out_valid.index_add_(0, slice_tokens, expert_out * flat_weight[slice_order].unsqueeze(-1))
                offset += count

        if self.shared_experts:
            out_valid = out_valid + self._shared(x_valid)

        if self.router_balance == 'loss_free':
            if self.training:
                self._router_load = torch.bincount(flat_expert, minlength=self.num_experts).float()
            balance_loss = zero
            z_loss = zero
        else:
            # Switch-style: dispatch fractions are counts, the gradient flows through mean probability
            top1 = topk_indices[:, 0]
            dispatch_fraction = torch.bincount(top1, minlength=self.num_experts).to(gate_probs.dtype) / num_valid
            mean_gate_probs = gate_probs.mean(dim=0)
            balance_loss = self.num_experts * (dispatch_fraction * mean_gate_probs).sum()
            z_loss = torch.logsumexp(gate_logits.float(), dim=-1).pow(2).mean()

        if valid_idx is not None:
            out_flat = torch.zeros_like(x_flat)
            out_flat[valid_idx] = out_valid
            topk_full = torch.zeros(num_tokens, self.top_k, dtype=torch.long, device=x.device)
            topk_full[valid_idx] = topk_indices
        else:
            out_flat = out_valid
            topk_full = topk_indices

        return out_flat.view(orig_shape), balance_loss, z_loss, topk_full

    @torch.no_grad()
    def update_router_bias(self):
        """Loss-free balancing (arXiv 2408.15664): a sign-based step toward the mean load, which the
        paper found better than proportional. Loads are summed across ranks so every DDP replica
        applies the same update."""
        if self.router_balance != 'loss_free' or self._router_load is None:
            return
        load = self._router_load
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        mean_load = load.mean()
        self._last_max_violation = (load.max() - mean_load) / mean_load.clamp(min=1)
        self.router_bias += self.router_bias_rate * torch.sign(mean_load - load)
        self._router_load = None


class DropPath(nn.Module):
    """Stochastic depth: zeroes the residual branch for whole samples during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [1] * x.ndim
        shape[1] = x.shape[1]
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask


class ColumnCollapse(nn.Module):
    """Collapses the patch-row (height) axis of a 2D token grid to a single row per column,
    producing a width-length sequence for the CTC head. kernel_width > 1 mixes neighbouring
    columns via overlapping receptive fields."""

    def __init__(self, d_model: int, grid_height: int, kernel_width: int = 3, depth: int = 1):
        super().__init__()
        convs = []
        for i in range(depth):
            is_last = i == depth - 1
            kh = grid_height if is_last else 1
            sh = grid_height if is_last else 1
            convs.append(nn.Conv2d(d_model, d_model, kernel_size=(kh, kernel_width),
                                   stride=(sh, 1), padding=(0, kernel_width // 2)))
            convs.append(nn.GELU())
        self.convs = nn.Sequential(*convs)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, batch_size: int, grid_height: int, grid_width: int) -> torch.Tensor:
        # x: [grid_width * grid_height, batch, d_model], column-major (index = w * grid_height + h)
        x = x.permute(1, 2, 0).reshape(batch_size, -1, grid_width, grid_height)    # [B, d, W, H]
        x = x.permute(0, 1, 3, 2).contiguous()                                     # [B, d, H, W]
        x = self.convs(x)                                                          # [B, d, 1, W]
        x = x.squeeze(2).permute(2, 0, 1).contiguous()                            # [W, B, d]
        return self.norm(x)


class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int = 4,
        top_k: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        drop_path: float = 0.0,
        expert_ffn: str = 'gelu',
        num_shared_experts: int = 0,
        router_balance: str = 'aux_loss',
        router_bias_rate: float = 1e-3,
        num_conv_experts: int = 0,
        conv_expert_kernel: int = 15,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.moe_block = MoEBlock(d_model, num_experts, top_k, dim_feedforward, dropout,
                                  expert_ffn=expert_ffn, num_shared_experts=num_shared_experts,
                                  router_balance=router_balance, router_bias_rate=router_bias_rate,
                                  num_conv_experts=num_conv_experts, conv_expert_kernel=conv_expert_kernel)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self,
                src: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, key_padding_mask=src_key_padding_mask)
        src = src + self.drop_path1(self.dropout1(src2))

        src2 = self.norm2(src)
        src2, balance_loss, z_loss, topk_indices = self.moe_block(src2, key_padding_mask=src_key_padding_mask)
        src = src + self.drop_path2(self.dropout2(src2))

        return src, balance_loss, z_loss, topk_indices


class DeepOCR(nn.Module):

    def __init__(
        self,
        tokenizer,
        target_height=96,
        weight_path: Optional[Path] = None,
        backbone_name: str = 'dinov3',
        cnn_width_stride: int = 16,
        cnn_patch_size: int = 16,
        dinov3_ckpt: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone: bool = False,
        use_2d_structure: bool = False,
        num_experts: int = 8,
        top_k_experts: int = 2,
        moe_penalty: float = 0.01,
        router_z_coef: float = 1e-3,
        expert_ffn: str = 'gelu',
        expert_hidden: int = 2048,
        num_shared_experts: int = 0,
        router_balance: str = 'aux_loss',
        router_bias_rate: float = 1e-3,
        num_conv_experts: int = 0,
        conv_expert_kernel: int = 15,
        convention_names: Optional[Sequence[str]] = None,
        convention_by_dataset: Optional[dict] = None,
        mask_ratio: float = 0.15,
        max_width: Optional[int] = None,
        num_encoder_layers: int = 3,
        drop_path: float = 0.0,
        collapse_height_for_ctc: bool = False,
        collapse_kernel_width: int = 3,
        collapse_depth: int = 1,
        use_decoder: bool = False,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        decoder_dim_feedforward: int = 2048,
        decoder_dropout: float = 0.1,
        ctc_loss_weight: float = 1.0,
        decoder_loss_weight: float = 1.0,
        decoder_label_smoothing: float = 0.1,
        decoder_eos_weight: float = 1.0,
        decoder_lm_path: Optional[Path] = None,
    ):
        super(DeepOCR, self).__init__()
        self.tokenizer = tokenizer
        self.num_classes = len(tokenizer)
        self.target_height = target_height
        self.max_width = max_width
        self.freeze_backbone = freeze_backbone
        self.use_2d_structure = use_2d_structure
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_penalty = moe_penalty
        self.router_z_coef = router_z_coef

        self.weight_path = None
        if weight_path is not None and weight_path.exists():
            self.weight_path = weight_path

        self.d_model = 512
        self.max_seq_len = 5000

        self.dinov3_ckpt = dinov3_ckpt
        self.backbone_name = backbone_name
        self.backbone = build_backbone(
            backbone_name, dinov3_ckpt=dinov3_ckpt, cnn_feature_dim=self.d_model,
            cnn_width_stride=cnn_width_stride, cnn_patch_size=cnn_patch_size
        )

        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze_backbone

        self.feature_dim = self.backbone.feature_dim    # e.g., 768 for dinov3-vitb16
        self.patch_size = self.backbone.patch_size    # vertical stride -> grid_height
        self.patch_size_w = self.backbone.patch_size_w    # horizontal stride -> grid_width

        self.grid_height = self.target_height // self.patch_size    # e.g., 96 / 16 = 6

        # https://arxiv.org/pdf/2409.08573
        self.use_span_mask = True
        self.mask_ratio = mask_ratio
        self.span_length = int(self.grid_height)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.mask_token, mean=0, std=0.02)

        if self.use_2d_structure:
            if self.feature_dim != self.d_model:
                self.feature_projection = nn.Linear(self.feature_dim, self.d_model)
            else:
                self.feature_projection = nn.Identity()

            self.pos_encoder = PositionalEncoding2D(
                d_model=self.d_model,
                max_height=self.grid_height,
                max_width=self.max_seq_len // self.grid_height,
                dropout=0.1
            )
        else:
            self.feature_projection = nn.Linear(self.feature_dim * self.grid_height, self.d_model)
            self.pos_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.max_seq_len, dropout=0.1)

        # Corpora disagree on long s; a per-line convention flag gives the same glyph two correct
        # targets instead of one contradictory one. Zeros mean the normalized convention, the
        # inference default.
        self.convention_names = list(convention_names or ())
        self.convention_by_dataset = dict(convention_by_dataset or {})
        self.convention_proj = (nn.Linear(len(self.convention_names), self.d_model, bias=False)
                                if self.convention_names else None)
        if self.convention_proj is not None:
            nn.init.zeros_(self.convention_proj.weight)

        self.num_layers = num_encoder_layers
        drop_path_rates = [drop_path * i / max(1, self.num_layers - 1) for i in range(self.num_layers)]
        self.transformer_layers = nn.ModuleList([
            MoETransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                num_experts=self.num_experts,
                top_k=self.top_k_experts,
                dim_feedforward=expert_hidden,
                dropout=0.2,
                drop_path=drop_path_rates[i],
                expert_ffn=expert_ffn,
                num_shared_experts=num_shared_experts,
                router_balance=router_balance,
                router_bias_rate=router_bias_rate,
                num_conv_experts=num_conv_experts,
                conv_expert_kernel=conv_expert_kernel,
            ) for i in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        self.output_layer = nn.Linear(self.d_model, self.num_classes)

        self.collapse_height_for_ctc = bool(collapse_height_for_ctc) and self.use_2d_structure
        if collapse_height_for_ctc and not self.use_2d_structure:
            print("collapse_height_for_ctc requires use_2d_structure=True; disabling.")
        self.collapse_kernel_width = collapse_kernel_width
        self.collapse_depth = collapse_depth
        if self.collapse_height_for_ctc:
            self.column_collapse = ColumnCollapse(
                d_model=self.d_model,
                grid_height=self.grid_height,
                kernel_width=collapse_kernel_width,
                depth=collapse_depth
            )

        self.use_decoder = bool(use_decoder)
        self.ctc_loss_weight = ctc_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.decoder_label_smoothing = decoder_label_smoothing
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_dropout = decoder_dropout
        self.decoder_embed_scale = math.sqrt(self.d_model)

        if self.use_decoder:
            self.decoder_embed = nn.Embedding(
                self.num_classes, self.d_model, padding_idx=self.tokenizer.pad_token_id
            )
            self.decoder_pos = PositionalEncoding(d_model=self.d_model, dropout=0.1, max_len=self.max_seq_len)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.decoder_heads,
                dim_feedforward=self.decoder_dim_feedforward,
                dropout=self.decoder_dropout,
                activation='gelu',
                batch_first=False,
                norm_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=self.decoder_layers, norm=nn.LayerNorm(self.d_model)
            )
            self.decoder_output = nn.Linear(self.d_model, self.num_classes)

            ce_class_weight = torch.ones(self.num_classes)
            ce_class_weight[self.tokenizer.eos_token_id] = decoder_eos_weight
            self.register_buffer('decoder_ce_weight', ce_class_weight)

        # LM warm-start first, so a checkpoint's own decoder overrides it; a checkpoint without a
        # decoder leaves the warm-start intact because the weights load with strict=False
        if self.use_decoder and decoder_lm_path is not None:
            self._load_pretrained_decoder_lm(Path(decoder_lm_path))

        if self.weight_path is not None:
            self._load_pretrained_weights()

    def _update_router_biases(self):
        for layer in self.transformer_layers:
            layer.moe_block.update_router_bias()

    def _generate_span_mask(self, B, NumTokens, device):
        mask = torch.zeros((B, NumTokens), dtype=torch.bool, device=device)

        num_masked_tokens = int(NumTokens * self.mask_ratio)
        num_spans = max(1, num_masked_tokens // max(1, self.span_length))

        if num_spans == 0:
            return mask

        max_start = max(1, NumTokens - self.span_length)
        start_indices = torch.randint(0, max_start, (B, num_spans), device=device)

        offsets = torch.arange(self.span_length, device=device)

        all_indices = start_indices.unsqueeze(-1) + offsets    # [B, num_spans, span_length]
        all_indices = torch.clamp(all_indices, max=NumTokens - 1)

        flat_indices = all_indices.view(B, -1)
        mask.scatter_(1, flat_indices, True)

        return mask

    def set_routing_prob(self, prob: float):
        for module in self.modules():
            if isinstance(module, MoEBlock):
                module.routing_prob = prob

    def set_mask_ratio(self, ratio: float):
        self.mask_ratio = ratio

    def print_parameter_summary(self):
        groups = {'backbone': 0, 'encoder': 0, 'ctc_head': 0, 'decoder': 0, 'other': 0}
        total = 0
        trainable = 0
        for name, param in self.named_parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
            if name.startswith('backbone.'):
                groups['backbone'] += count
            elif name.startswith('decoder'):
                groups['decoder'] += count
            elif name.startswith('transformer_layers.'):
                groups['encoder'] += count
            elif name.startswith('output_layer.') or name.startswith('column_collapse.'):
                groups['ctc_head'] += count
            else:
                groups['other'] += count

        def fmt(value):
            return f"{value / 1e6:8.2f}M"

        print("Parameter summary:")
        print(f"  Backbone ({self.backbone_name}):".ljust(25) + fmt(groups['backbone']))
        print(f"  MoE encoder:           {fmt(groups['encoder'])}")
        print(f"  CTC head:              {fmt(groups['ctc_head'])}")
        if self.use_decoder:
            print(f"  Decoder head:          {fmt(groups['decoder'])}")
        print(f"  Other (proj/pos/norm): {fmt(groups['other'])}")
        print(f"  Total:                 {fmt(total)} ({total:,})")
        print(f"  Trainable:             {fmt(trainable)} ({trainable:,})")

    def convention_vector(self, dataset_labels: Sequence[str], device: torch.device) -> Optional[torch.Tensor]:
        """Flag vector per sample from its dataset label; unknown labels get the normalized
        convention, all zeros."""
        if self.convention_proj is None:
            return None
        default = [0.0] * len(self.convention_names)
        rows = [self.convention_by_dataset.get(label, default) for label in dataset_labels]
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _encode(self,
                x: torch.Tensor,
                attention_masks: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                conventions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor], Dict]:

        B = x.size(0)

        with torch.set_grad_enabled(not self.freeze_backbone):
            feature_grid = self.backbone(x)    # [B, grid_height, grid_width, feature_dim]

        grid_width = feature_grid.size(2)

        if self.use_2d_structure:
            # [B, H, W, D] -> [B, W, H, D]
            feature_grid = feature_grid.transpose(1, 2)
            sequence = feature_grid.reshape(B, grid_width * self.grid_height, self.feature_dim)
            sequence = self.feature_projection(sequence)
            sequence = self.pos_encoder(sequence, height=self.grid_height, width=grid_width)
            projected_sequence = sequence.transpose(0, 1)    # [Seq_Len, B, d_model]
            seq_len = grid_width * self.grid_height
        else:
            sequence = feature_grid.permute(2, 0, 1, 3)    # [W, B, H, D]
            sequence_features = sequence.reshape(grid_width, B, self.grid_height * self.feature_dim)
            projected_sequence = self.feature_projection(sequence_features)
            projected_sequence = self.pos_encoder(projected_sequence)    # [Seq_Len, B, d_model]
            seq_len = grid_width

        if attention_masks is not None:
            if attention_masks.shape != (B, seq_len):
                raise ValueError(
                    f"Attention mask shape mismatch! Expected [{B}, {seq_len}], got {attention_masks.shape}."
                )

        if self.convention_proj is not None and conventions is not None:
            # one vector added to every column, so both heads are conditioned
            projected_sequence = projected_sequence + self.convention_proj(
                conventions.to(projected_sequence.dtype)).unsqueeze(0)

        active_span_mask = None

        if getattr(self, 'use_span_mask', False):
            generated_mask = None

            if self.training or return_attention:
                generated_mask = self._generate_span_mask(B, seq_len, projected_sequence.device)
                active_span_mask = generated_mask

            if self.training and generated_mask is not None:
                x_t = projected_sequence.transpose(0, 1)
                mask_token_expanded = self.mask_token.expand(B, seq_len, self.d_model)
                x_masked = torch.where(generated_mask.unsqueeze(-1), mask_token_expanded, x_t)
                projected_sequence = x_masked.transpose(0, 1)

        encoded_sequence = projected_sequence

        aux_losses = {}
        total_aux_loss = torch.tensor(0.0, device=projected_sequence.device)
        total_z_loss = torch.tensor(0.0, device=projected_sequence.device)
        expert_assignments = []

        for i, layer in enumerate(self.transformer_layers):
            encoded_sequence, balance_loss, z_loss, topk_indices = layer(encoded_sequence, src_key_padding_mask=attention_masks)

            total_aux_loss += balance_loss
            total_z_loss += z_loss
            aux_losses[f'layer_{i}'] = balance_loss

            if return_attention:
                primary_experts = topk_indices[:, 0].view(seq_len, B).transpose(0, 1)
                expert_assignments.append(primary_experts)

        aux_losses['total'] = total_aux_loss
        aux_losses['z_total'] = total_z_loss

        encoded_sequence = self.final_norm(encoded_sequence)

        if self.collapse_height_for_ctc:
            memory = self.column_collapse(encoded_sequence, B, self.grid_height, grid_width)
            if attention_masks is not None:
                memory_key_padding_mask = attention_masks.view(B, grid_width, self.grid_height).all(dim=2)
            else:
                memory_key_padding_mask = None
        else:
            memory = encoded_sequence
            memory_key_padding_mask = attention_masks

        enc_state = {
            'B': B,
            'grid_width': grid_width,
            'seq_len': seq_len,
            'projected_sequence': projected_sequence,
            'encoded_sequence': encoded_sequence,
            'active_span_mask': active_span_mask,
            'expert_assignments': expert_assignments,
        }

        return memory, memory_key_padding_mask, aux_losses, enc_state

    def forward(self,
                x: torch.Tensor,
                attention_masks: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                decoder_targets: Optional[torch.Tensor] = None,
                decoder_tgt_key_padding_mask: Optional[torch.Tensor] = None,
                conventions: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[torch.Tensor]]:

        memory, memory_key_padding_mask, aux_losses, enc_state = self._encode(
            x, attention_masks, return_attention, conventions)

        B = enc_state['B']
        grid_width = enc_state['grid_width']

        logits = self.output_layer(memory)
        log_probs = nn.functional.log_softmax(logits, dim=2)

        viz_outputs = {}
        if return_attention:
            projected_sequence = enc_state['projected_sequence']
            encoded_sequence = enc_state['encoded_sequence']
            active_span_mask = enc_state['active_span_mask']
            expert_assignments = enc_state['expert_assignments']
            projected_b_first = projected_sequence.permute(1, 0, 2)    # [B, seq_len, d_model]
            encoded_b_first = encoded_sequence.permute(1, 0, 2)    # [B, seq_len, d_model]
            log_probs_b_first = log_probs.permute(1, 0, 2)    # [B, seq_len, num_classes]

            def seq_to_grid(tensor_1d):
                if self.use_2d_structure:
                    if tensor_1d.dim() == 2:
                        return tensor_1d.view(B, 1, grid_width, self.grid_height).transpose(-1, -2)
                    elif tensor_1d.dim() == 3:
                        return tensor_1d.view(B, -1, grid_width, self.grid_height).transpose(-1, -2)
                else:
                    return tensor_1d.view(B, 1, 1, grid_width)

            def logit_seq_to_grid(tensor_1d):
                if self.collapse_height_for_ctc:
                    if tensor_1d.dim() == 2:
                        return tensor_1d.view(B, 1, 1, grid_width)
                    return tensor_1d.view(B, -1, 1, grid_width)
                return seq_to_grid(tensor_1d)

            viz_outputs['norm_before_encoder'] = torch.linalg.norm(projected_b_first, dim=-1).mean()
            viz_outputs['norm_after_encoder'] = torch.linalg.norm(encoded_b_first, dim=-1).mean()

            feature_norms_before = torch.linalg.norm(projected_b_first, dim=-1)    # [B, seq_len]
            feature_norms_after = torch.linalg.norm(encoded_b_first, dim=-1)    # [B, seq_len]

            viz_outputs['feature_norm_map_before'] = seq_to_grid(feature_norms_before)
            viz_outputs['feature_norm_map_after'] = seq_to_grid(feature_norms_after)

            norm_delta = (feature_norms_after - feature_norms_before).abs()
            viz_outputs['feature_norm_delta_map'] = seq_to_grid(norm_delta)

            probs_b_first = torch.exp(log_probs_b_first)    # [B, seq_len, C]
            confidence, predicted_tokens = probs_b_first.max(dim=2)    # [B, seq_len]

            viz_outputs['confidence_map'] = logit_seq_to_grid(confidence)
            viz_outputs['predicted_token_map'] = logit_seq_to_grid(predicted_tokens.float())

            entropy = -(probs_b_first * log_probs_b_first).sum(dim=2)    # [B, seq_len]
            viz_outputs['entropy_map'] = logit_seq_to_grid(entropy)

            feature_mean_before = projected_b_first.mean(dim=-1)    # [B, seq_len]
            feature_mean_after = encoded_b_first.mean(dim=-1)    # [B, seq_len]

            viz_outputs['feature_mean_map_before'] = seq_to_grid(feature_mean_before)
            viz_outputs['feature_mean_map_after'] = seq_to_grid(feature_mean_after)

            if probs_b_first.shape[-1] > 1:
                top2_probs, _ = torch.topk(probs_b_first, k=2, dim=2)    # [B, seq_len, 2]
                confidence_margin = top2_probs[:, :, 0] - top2_probs[:, :, 1]    # [B, seq_len]
                viz_outputs['confidence_margin_map'] = logit_seq_to_grid(confidence_margin)

            # in eval the mask is generated but never applied, so plotting it would mislead
            if active_span_mask is not None and self.training:
                viz_outputs['span_mask'] = seq_to_grid(active_span_mask.float())

            for i, assignment in enumerate(expert_assignments):
                viz_outputs[f'expert_routing_layer_{i}'] = seq_to_grid(assignment.float())

        decoder_logits = None
        if self.use_decoder and decoder_targets is not None:
            decoder_logits = self._decode_teacher_forced(
                memory, memory_key_padding_mask, decoder_targets, decoder_tgt_key_padding_mask
            )

        return log_probs, aux_losses, viz_outputs, decoder_logits

    def _build_decoder_io(
        self, targets_concat: torch.Tensor, target_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splits the CTC-style concatenated targets into padded decoder input/target tensors.
        decoder_input is BOS-prefixed; decoder_target is EOS-suffixed; both are padded with pad_id."""
        sequences = torch.split(targets_concat, target_lengths.tolist())
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        batch_size = len(sequences)
        max_len = (int(target_lengths.max().item()) + 1) if batch_size > 0 else 1
        device = targets_concat.device

        decoder_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        decoder_target = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        tgt_key_padding_mask = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)

        for i, seq in enumerate(sequences):
            length = seq.numel()
            decoder_input[i, 0] = bos_id
            decoder_input[i, 1:1 + length] = seq
            decoder_target[i, :length] = seq
            decoder_target[i, length] = eos_id
            tgt_key_padding_mask[i, :length + 1] = False

        return decoder_input, decoder_target, tgt_key_padding_mask

    def _decode_teacher_forced(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        decoder_input: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        seq_len = decoder_input.size(1)
        embedded = self.decoder_embed(decoder_input) * self.decoder_embed_scale
        embedded = self.decoder_pos(embedded.transpose(0, 1))
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=memory.device), diagonal=1
        )
        decoded = self.decoder(
            tgt=embedded,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.decoder_output(decoded)

    @torch.no_grad()
    def decode_autoregressive(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        max_len: int = 256
    ) -> torch.Tensor:
        """Greedy autoregressive decode. Returns generated token ids [B, T] including the leading BOS."""
        batch_size = memory.size(1)
        device = memory.device
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            seq_len = generated.size(1)
            embedded = self.decoder_embed(generated) * self.decoder_embed_scale
            embedded = self.decoder_pos(embedded.transpose(0, 1))
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
            )
            decoded = self.decoder(
                tgt=embedded,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            step_logits = self.decoder_output(decoded[-1])
            next_token = step_logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, pad_id), next_token)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if bool(finished.all()):
                break

        return generated

    @torch.no_grad()
    def rescore_hypotheses(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor],
        nbest_lists: List[List[Tuple[str, float]]],
        alpha: float = 0.3,
        beta: float = 0.0,
    ) -> List[str]:
        """Reranks CTC beam n-best lists with the AR decoder. All hypotheses of the batch are
        scored in one teacher-forced pass (memory columns repeated per hypothesis); the final
        score is ctc_score + alpha * decoder_log_prob + beta * text_length."""
        device = memory.device

        flat_texts = []
        flat_ctc_scores = []
        flat_sample_idx = []
        for sample_idx, hypos in enumerate(nbest_lists):
            for text, ctc_score in hypos:
                flat_texts.append(text)
                flat_ctc_scores.append(ctc_score)
                flat_sample_idx.append(sample_idx)

        if not flat_texts:
            return ["" for _ in nbest_lists]

        encoded = [self.tokenizer.encode(text) for text in flat_texts]
        target_lengths = torch.tensor([len(seq) for seq in encoded], dtype=torch.long, device=device)
        targets_concat = torch.cat([
            torch.tensor(seq, dtype=torch.long, device=device) if seq else torch.empty(0, dtype=torch.long, device=device)
            for seq in encoded
        ])

        decoder_input, decoder_target, tgt_key_padding_mask = self._build_decoder_io(targets_concat, target_lengths)

        sample_idx_tensor = torch.tensor(flat_sample_idx, dtype=torch.long, device=device)
        memory_rep = memory[:, sample_idx_tensor, :]
        mask_rep = memory_key_padding_mask[sample_idx_tensor] if memory_key_padding_mask is not None else None

        decoder_logits = self._decode_teacher_forced(memory_rep, mask_rep, decoder_input, tgt_key_padding_mask)
        log_probs = F.log_softmax(decoder_logits.float(), dim=-1).permute(1, 0, 2)    # [N, L, V]

        token_log_probs = log_probs.gather(2, decoder_target.unsqueeze(-1)).squeeze(-1)    # [N, L]
        valid_positions = ~tgt_key_padding_mask
        decoder_scores = (token_log_probs * valid_positions).sum(dim=1)

        ctc_scores = torch.tensor(flat_ctc_scores, dtype=torch.float32, device=device)
        combined = ctc_scores + alpha * decoder_scores + beta * target_lengths.float()

        best_texts = ["" for _ in nbest_lists]
        best_scores = [float('-inf')] * len(nbest_lists)
        for i, sample_idx in enumerate(flat_sample_idx):
            score = combined[i].item()
            if score > best_scores[sample_idx]:
                best_scores[sample_idx] = score
                best_texts[sample_idx] = flat_texts[i]
        return best_texts

    def _decoder_ids_to_text(self, generated: torch.Tensor) -> List[str]:
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        texts = []
        for row in generated.tolist():
            if row and row[0] == bos_id:
                row = row[1:]
            if eos_id in row:
                row = row[:row.index(eos_id)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True))
        return texts

    def _load_pretrained_weights(self):
        try:
            if not self.weight_path or not self.weight_path.exists():
                print(f"Pretrained weight path does not exist: {self.weight_path}")
                return

            checkpoint = torch.load(self.weight_path, map_location='cpu', weights_only=False)
            pretrained_state = checkpoint['model_state_dict']
            model_state = self.state_dict()

            # A different backbone architecture shares some tensor names and shapes by
            # coincidence (the 512-channel squeeze-excite layers, for one); loading those would
            # seed the new backbone with gates fitted to another net's statistics.
            pretrained_backbone = checkpoint.get('backbone_name')
            if pretrained_backbone is not None and pretrained_backbone != self.backbone_name:
                for key in [k for k in pretrained_state if k.startswith('backbone.')]:
                    pretrained_state.pop(key)
                print(f"Pretrained backbone is '{pretrained_backbone}', this model uses "
                      f"'{self.backbone_name}'; backbone weights skipped, the rest loads.")

            for old_key in [k for k in pretrained_state if k.startswith('backbone.')]:
                if old_key in model_state:
                    continue
                if old_key.startswith('backbone.model.'):
                    candidate = 'backbone.' + old_key[len('backbone.model.'):]
                else:
                    candidate = 'backbone.model.' + old_key[len('backbone.'):]
                if candidate in model_state:
                    pretrained_state[candidate] = pretrained_state.pop(old_key)

            old_tokens = checkpoint.get('tokenizer_tokens')
            new_tokens = self.tokenizer.tokens_list
            vocab_changed = old_tokens is not None and old_tokens != new_tokens

            if old_tokens is None and 'output_layer.weight' in pretrained_state:
                if pretrained_state['output_layer.weight'].shape != model_state['output_layer.weight'].shape:
                    print(f"Legacy checkpoint without tokenizer_tokens and output_layer shape mismatch "
                          f"({pretrained_state['output_layer.weight'].shape} vs "
                          f"{model_state['output_layer.weight'].shape}); dropping output_layer weights.")
                    pretrained_state.pop('output_layer.weight', None)
                    pretrained_state.pop('output_layer.bias', None)

            if vocab_changed:
                old_idx = {t: i for i, t in enumerate(old_tokens)}
                row_src = [old_idx.get(t) for t in new_tokens]
                n_matched = sum(1 for src in row_src if src is not None)

                # rebuilt from the tokenizer at construction, so it already matches the new vocab
                pretrained_state.pop('decoder_ce_weight', None)

                # Vocab-indexed tensors are found by shape, not name (CTC head and decoder alike):
                # one missed tensor makes load_state_dict raise even with strict=False, which
                # abandons the whole warm start and silently trains from scratch.
                remapped, dropped = [], []
                for key in list(pretrained_state):
                    if key not in model_state:
                        continue
                    pre, tgt = pretrained_state[key], model_state[key]
                    if pre.dim() == 0 or tgt.dim() == 0:
                        continue    # scalar buffers (step counters etc.) have no vocab axis
                    if pre.shape[0] != len(old_tokens) or tgt.shape[0] != len(new_tokens):
                        continue
                    if pre.shape[1:] != tgt.shape[1:]:
                        # d_model changed, so the rows are not reusable; keep the fresh init
                        pretrained_state.pop(key)
                        dropped.append(key)
                        continue

                    if pre.dim() > 1:
                        # Unmatched rows start from random pretrained rows, which DTLR
                        # (arXiv:2409.17095 sec 3.2) found converges better than xavier noise.
                        # Random on purpose: seeding 'a' from Greek alpha biases the model toward
                        # confusing that pair (Calamari #319).
                        seed_rows = torch.randint(0, pre.shape[0], (tgt.shape[0],))
                        new_t = pre[seed_rows].to(dtype=tgt.dtype, device=tgt.device).clone()
                    else:
                        new_t = torch.zeros_like(tgt)

                    for new_i, old_i in enumerate(row_src):
                        if old_i is not None:
                            new_t[new_i] = pre[old_i]
                    pretrained_state[key] = new_t
                    remapped.append(key)

                print(f"Vocab remap: copied {n_matched}/{len(new_tokens)} token rows "
                      f"(old={len(old_tokens)}, new={len(new_tokens)}) across {len(remapped)} "
                      f"tensor(s): {', '.join(sorted(remapped))}")
                if dropped:
                    print(f"  dropped (trailing dims changed): {', '.join(sorted(dropped))}")

            # load_state_dict copies every compatible tensor and then raises on size mismatches,
            # so they are dropped first; vocab-sized tensors were remapped above
            mismatched = [key for key in pretrained_state
                          if key in model_state and pretrained_state[key].shape != model_state[key].shape]
            for key in mismatched:
                pretrained_state.pop(key)
            missing_keys, unexpected_keys = self.load_state_dict(pretrained_state, strict=False)

            if 'output_layer.weight' in missing_keys:
                nn.init.xavier_uniform_(self.output_layer.weight)
                nn.init.constant_(self.output_layer.bias, 0)

            summary = [f"{len(pretrained_state)} tensors loaded"]
            if mismatched:
                summary.append(f"{len(mismatched)} skipped for shape (e.g. {mismatched[0]})")
            if missing_keys:
                summary.append(f"{len(missing_keys)} left at init (e.g. {missing_keys[0]})")
            if unexpected_keys:
                summary.append(f"{len(unexpected_keys)} not in this model (e.g. {unexpected_keys[0]})")
            print("Pretrained weights: " + "; ".join(summary))

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")

    def _load_pretrained_decoder_lm(self, lm_path: Path):
        """Warm-starts the autoregressive decoder from a separately pretrained character LM.
        Transfers the LM self-attention, FFN, embedding and output head into the decoder; the
        cross-attention block stays randomly initialized and is learned during OCR training."""
        try:
            if not lm_path.exists():
                print(f"Decoder LM path does not exist: {lm_path}")
                return

            checkpoint = torch.load(lm_path, map_location='cpu', weights_only=False)
            lm_state = checkpoint['lm_state_dict']

            lm_tokens = checkpoint.get('tokenizer_tokens')
            if lm_tokens is not None and lm_tokens != self.tokenizer.tokens_list:
                print("Decoder LM vocab differs from the current tokenizer; skipping LM warm-start.")
                return

            num_decoder_layers = len(self.decoder.layers)
            lm_layer_indices = sorted({
                int(key.split('.')[2]) for key in lm_state if key.startswith('encoder.layers.')
            })
            num_transfer = min(num_decoder_layers, len(lm_layer_indices))

            transfer = {}
            transfer['decoder_embed.weight'] = lm_state['embed.weight']
            transfer['decoder_output.weight'] = lm_state['output.weight']
            transfer['decoder_output.bias'] = lm_state['output.bias']
            transfer['decoder.norm.weight'] = lm_state['encoder.norm.weight']
            transfer['decoder.norm.bias'] = lm_state['encoder.norm.bias']

            for i in range(num_transfer):
                src = f'encoder.layers.{i}'
                dst = f'decoder.layers.{i}'
                transfer[f'{dst}.self_attn.in_proj_weight'] = lm_state[f'{src}.self_attn.in_proj_weight']
                transfer[f'{dst}.self_attn.in_proj_bias'] = lm_state[f'{src}.self_attn.in_proj_bias']
                transfer[f'{dst}.self_attn.out_proj.weight'] = lm_state[f'{src}.self_attn.out_proj.weight']
                transfer[f'{dst}.self_attn.out_proj.bias'] = lm_state[f'{src}.self_attn.out_proj.bias']
                transfer[f'{dst}.linear1.weight'] = lm_state[f'{src}.linear1.weight']
                transfer[f'{dst}.linear1.bias'] = lm_state[f'{src}.linear1.bias']
                transfer[f'{dst}.linear2.weight'] = lm_state[f'{src}.linear2.weight']
                transfer[f'{dst}.linear2.bias'] = lm_state[f'{src}.linear2.bias']
                transfer[f'{dst}.norm1.weight'] = lm_state[f'{src}.norm1.weight']
                transfer[f'{dst}.norm1.bias'] = lm_state[f'{src}.norm1.bias']
                transfer[f'{dst}.norm3.weight'] = lm_state[f'{src}.norm2.weight']
                transfer[f'{dst}.norm3.bias'] = lm_state[f'{src}.norm2.bias']

            model_state = self.state_dict()
            filtered = {k: v for k, v in transfer.items() if k in model_state and model_state[k].shape == v.shape}
            skipped = [k for k in transfer if k not in filtered]

            self.load_state_dict(filtered, strict=False)
            print(f"Decoder LM warm-start: transferred {len(filtered)} tensors across {num_transfer} layers "
                  f"(cross-attention left randomly initialized).")
            if skipped:
                print(f"Decoder LM warm-start skipped {len(skipped)} tensors due to shape mismatch: {skipped}")

        except Exception as e:
            print(f"Error loading pretrained decoder LM: {e}")

    def train_model(
        self,
        train_dataloader: "DataLoader",
        valid_dataloader: "DataLoader",
        optimizer,
        criterion,
        device: torch.device = torch.device('cpu'),
        epochs: int = 10,
        log_dir: Optional[Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        beam_width: int = 10,
        val_frequency: int = 1,
        beam_val_frequency: int = 10,
        log_samples_frequency: int = 1,
        router_warmup_epochs: int = 20,
        debug: bool = False,
        profile_mode: bool = False,
        ddp_model=None,
        train_sampler=None,
        is_master: bool = True,
        world_size: int = 1,
        disable_aug_after_epoch: Optional[int] = None,
        decoder_warmup_epochs: int = 5,
        decoder_val_frequency: int = 1,
        decoder_max_len: int = 256,
        select_on_decoder: bool = False,
        save_best_of_heads: bool = False,
        ema_decay: float = 0.999,
        mask_ratio_final: Optional[float] = None,
        rescore_alpha: float = 0.3,
        rescore_beta: float = 0.0,
        rescore_nbest: int = 8,
        resume_checkpoint: Optional[Path] = None,
        scalar_log_interval: int = 50
    ):
        self.to(device)

        writer = SummaryWriter(log_dir) if is_master and log_dir else None
        if writer:
            _log_tensorboard_descriptions(writer)

        best_cer = float('inf')
        best_val_loss = float('inf')
        patience = int(os.environ['TRAIN_PATIENCE']) if os.environ.get('TRAIN_PATIENCE') else None
        rounds_since_best = 0
        global_step = 0

        # EMA of weights and buffers (BN stats included); validation and checkpoints use the
        # EMA copy. Updated on all ranks so distributed validation sees identical weights.
        ema_model = AveragedModel(self, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
        eval_module = ema_model.module

        base_mask_ratio = self.mask_ratio

        # Crash-safe resume: restores everything a requeued job needs to continue mid-schedule.
        # Loaded identically on every rank; load_state_dict copies in place, so the DDP wrapper
        # and optimizer built around this module stay valid.
        start_epoch = 0
        if resume_checkpoint is not None:
            resume_state = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
            self.load_state_dict(resume_state['model_state_dict'])
            ema_model.load_state_dict(resume_state['ema_state_dict'])
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            if scheduler is not None and resume_state.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            start_epoch = resume_state['epoch']
            global_step = resume_state.get('global_step', start_epoch * len(train_dataloader))
            best_cer = resume_state.get('best_cer', float('inf'))
            best_val_loss = resume_state.get('best_val_loss', float('inf'))
            rounds_since_best = resume_state.get('rounds_since_best', 0)
            del resume_state
            if is_master:
                print(f"Resumed from {resume_checkpoint}: continuing at epoch "
                      f"{start_epoch + 1}/{epochs}, best CER {best_cer:.4f}")

        # the DDP wrapper's forward is what triggers gradient reduction
        forward_caller = ddp_model if ddp_model is not None else self

        if profile_mode and is_master:
            print("PROFILER MODE ACTIVE: Training will capture 5 batches and exit.")
            profiler_context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir / 'profiler_logs')),
                record_shapes=True,
                with_stack=True
            )
        else:
            profiler_context = nullcontext()

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if disable_aug_after_epoch is not None and epoch >= disable_aug_after_epoch:
                dataset = getattr(train_dataloader, 'dataset', None)
                if dataset is not None and hasattr(dataset, 'set_augmentation_enabled'):
                    dataset.set_augmentation_enabled(False)
                    if is_master and epoch == disable_aug_after_epoch:
                        print(f"Disabling training augmentation from epoch {epoch + 1} onward.")

            if epoch < router_warmup_epochs:
                current_routing_prob = epoch / router_warmup_epochs
            else:
                current_routing_prob = 1.0

            self.set_routing_prob(current_routing_prob)

            if mask_ratio_final is not None:
                mask_progress = epoch / max(1, epochs - 1)
                self.set_mask_ratio(base_mask_ratio + (mask_ratio_final - base_mask_ratio) * mask_progress)
                if is_master and writer:
                    writer.add_scalar('Schedule/mask_ratio', self.mask_ratio, epoch)

            if self.use_decoder and decoder_warmup_epochs > 0:
                current_decoder_weight = self.decoder_loss_weight * min(1.0, (epoch + 1) / decoder_warmup_epochs)
            else:
                current_decoder_weight = self.decoder_loss_weight

            if is_master and writer:
                writer.add_scalar('Schedule/learned_routing_prob', current_routing_prob, epoch)
                if self.use_decoder:
                    writer.add_scalar('Schedule/decoder_loss_weight', current_decoder_weight, epoch)

            self.train()
            total_loss = 0
            num_batches = 0

            loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not is_master)

            with profiler_context as prof:
                for batch_idx, (images, attention_masks, targets, input_lengths, target_lengths, dataset_labels) in enumerate(loop):

                    if is_master and writer and batch_idx == 0:
                        # the only place a transform that erases strokes or washes out the page shows up
                        writer.add_image('Images/train_augmented',
                                         torchvision.utils.make_grid(to_display_image(images[:4]), nrow=1),
                                         epoch)

                    conventions = self.convention_vector(dataset_labels, device)
                    images = images.to(device, non_blocking=True)
                    attention_masks = attention_masks.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    input_lengths = input_lengths.to(device, non_blocking=True)
                    target_lengths = target_lengths.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    if self.use_decoder:
                        decoder_input, decoder_target, tgt_key_padding_mask = self._build_decoder_io(
                            targets, target_lengths
                        )
                    else:
                        decoder_input = decoder_target = tgt_key_padding_mask = None

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        log_probs, aux_losses, _, decoder_logits = forward_caller(
                            images,
                            attention_masks=attention_masks,
                            decoder_targets=decoder_input,
                            decoder_tgt_key_padding_mask=tgt_key_padding_mask,
                            conventions=conventions
                        )

                    total_aux_loss = aux_losses['total']
                    total_z_loss = aux_losses['z_total']
                    log_probs_fp32 = log_probs.to(torch.float32)
                    ctc_loss = criterion(log_probs_fp32, targets, input_lengths, target_lengths)

                    loss = (self.ctc_loss_weight * ctc_loss) + (self.moe_penalty * total_aux_loss) + (self.router_z_coef * total_z_loss)

                    ce_loss_value = 0.0
                    if self.use_decoder and decoder_logits is not None:
                        vocab_size = decoder_logits.size(-1)
                        ce_weight = getattr(self, 'decoder_ce_weight', None)
                        ce_loss = nn.functional.cross_entropy(
                            decoder_logits.to(torch.float32).permute(1, 0, 2).reshape(-1, vocab_size),
                            decoder_target.reshape(-1),
                            weight=ce_weight.to(torch.float32) if ce_weight is not None else None,
                            ignore_index=self.tokenizer.pad_token_id,
                            label_smoothing=self.decoder_label_smoothing
                        )
                        loss = loss + (current_decoder_weight * ce_loss)
                        ce_loss_value = ce_loss.detach().item()

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                    optimizer.step()
                    self._update_router_biases()
                    if scheduler is not None:
                        scheduler.step()
                    ema_model.update_parameters(self)

                    batch_loss_float = loss.detach().item()

                    if math.isnan(batch_loss_float) or math.isinf(batch_loss_float):
                        if is_master:
                            print(f"Warning: Invalid loss at epoch {epoch+1}, batch {batch_idx}")
                        continue

                    total_loss += batch_loss_float
                    num_batches += 1

                    # full-rate curves are unreadable and dominate the event file
                    if is_master and writer and global_step % scalar_log_interval == 0:
                        writer.add_scalar('Loss/train_batch_total', batch_loss_float, global_step)
                        writer.add_scalar('Loss/train_batch_ctc', ctc_loss.detach().item(), global_step)
                        if self.use_decoder:
                            writer.add_scalar('Loss/train_batch_ce', ce_loss_value, global_step)

                        balance_is_live = self.transformer_layers[0].moe_block.router_balance == 'aux_loss'
                        if balance_is_live:
                            writer.add_scalar('Loss/moe_balance_total', total_aux_loss.detach().item(), global_step)
                            writer.add_scalar('Loss/router_z_total', total_z_loss.detach().item(), global_step)
                        else:
                            for i, layer in enumerate(self.transformer_layers):
                                block = layer.moe_block
                                writer.add_scalar(f'Router/bias_range_layer_{i}',
                                                  (block.router_bias.max() - block.router_bias.min()).item(),
                                                  global_step)
                                # MaxVio from the loss-free paper: whether the balancer is winning,
                                # where bias_range shows how hard it pushes
                                if block._last_max_violation is not None:
                                    writer.add_scalar(f'Router/max_violation_layer_{i}',
                                                      block._last_max_violation.item(), global_step)

                        # backbone, encoder and decoder share a base LR and the same sqrt(batch)
                        # scaling, so one curve covers all three unless a per-group TRAIN_*_BASE_LR
                        # override splits them.
                        writer.add_scalar('Schedule/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                        # pre-clip norm, so it keeps reporting once clipping starts biting
                        writer.add_scalar('Grad/norm', grad_norm.item(), global_step)

                    if is_master and writer:
                        loop.set_postfix(loss=batch_loss_float, avg_loss=total_loss / num_batches)

                    global_step += 1

                    if profile_mode and is_master:
                        prof.step()
                        if batch_idx >= 5:
                            print(f"\nTrace saved to {log_dir / 'profiler_logs'}. Exiting...")
                            if writer:
                                writer.close()
                            return

            avg_train_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            if writer:
                writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
            if is_master and os.environ.get('TRAIN_OUTPUT'):
                with open(Path(os.environ['TRAIN_OUTPUT']) / 'progress.jsonl', 'a', encoding='utf-8') as _progress:
                    _progress.write(json.dumps({'epoch': epoch + 1, 'loss': avg_train_loss,
                                                'ts': time.time()}) + '\n')

            # Reserved memory in nvidia-smi is a high-water mark of the caching allocator and
            # says little about headroom; this is the number that decides whether the batch fits.
            if torch.cuda.is_available():
                peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
                peak_resv = torch.cuda.max_memory_reserved() / 2 ** 30
                total_mem = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                print(f"[rank {rank}] epoch {epoch} peak VRAM: {peak_alloc:.1f}GiB allocated, "
                      f"{peak_resv:.1f}GiB reserved of {total_mem:.1f}GiB "
                      f"({peak_resv / total_mem * 100:.0f}% used)", flush=True)
                if writer:
                    writer.add_scalar('VRAM/peak_allocated_gib', peak_alloc, epoch)
                    writer.add_scalar('VRAM/peak_reserved_gib', peak_resv, epoch)
                torch.cuda.reset_peak_memory_stats()

            run_val = ((epoch + 1) % val_frequency == 0)

            if valid_dataloader is not None and run_val:
                run_beam_val = ((epoch + 1) % beam_val_frequency == 0)
                run_decoder_val = self.use_decoder and ((epoch + 1) % decoder_val_frequency == 0)

                decoder_cer = float('inf')
                rescored_cer = float('inf')
                val_metrics = None

                if run_beam_val:
                    if is_master:
                        print(f"  Running beam + greedy validation (every {beam_val_frequency} epochs)...")
                    combined_metrics = eval_module.validate(
                        criterion=criterion,
                        dataloader=valid_dataloader,
                        device=device,
                        writer=writer,
                        epoch=epoch,
                        log_attention=True,
                        decoding_mode='both',
                        beam_width=beam_width,
                        length_penalty=0.1,
                        blank_penalty=0.0,
                        debug=debug,
                        run_decoder=run_decoder_val,
                        decoder_max_len=decoder_max_len,
                        world_size=world_size,
                        run_rescore=self.use_decoder,
                        rescore_alpha=rescore_alpha,
                        rescore_beta=rescore_beta,
                        rescore_nbest=rescore_nbest
                    )

                    greedy_cer = combined_metrics['greedy']['cer']
                    greedy_wer = combined_metrics['greedy']['wer']

                    beam_cer = combined_metrics['beam']['cer']
                    beam_wer = combined_metrics['beam']['wer']

                    val_loss = combined_metrics['val_loss']
                    val_metrics = combined_metrics

                    self._last_beam_cer = beam_cer
                    self._last_beam_wer = beam_wer

                else:
                    greedy_metrics = eval_module.validate(
                        criterion=criterion,
                        dataloader=valid_dataloader,
                        device=device,
                        writer=writer,
                        epoch=epoch,
                        log_attention=True,
                        decoding_mode='greedy',
                        debug=debug,
                        run_decoder=run_decoder_val,
                        decoder_max_len=decoder_max_len,
                        world_size=world_size
                    )

                    greedy_cer = greedy_metrics['cer']
                    greedy_wer = greedy_metrics['wer']
                    val_loss = greedy_metrics['val_loss']
                    val_metrics = greedy_metrics

                    beam_cer = getattr(self, '_last_beam_cer', float('inf'))
                    beam_wer = getattr(self, '_last_beam_wer', float('inf'))

                if run_decoder_val and val_metrics is not None and 'decoder' in val_metrics:
                    decoder_cer = val_metrics['decoder']['cer']

                if val_metrics is not None and 'rescored' in val_metrics:
                    rescored_cer = val_metrics['rescored']['cer']

                # validate() already wrote Loss/validation and every Metrics/{CER,WER}_* series for
                # the modes it ran, so only the cross-mode deltas are left to write here.
                if writer:
                    if run_beam_val:
                        cer_improvement = ((greedy_cer - beam_cer) / greedy_cer * 100) if greedy_cer > 0 else 0
                        wer_improvement = ((greedy_wer - beam_wer) / greedy_wer * 100) if greedy_wer > 0 else 0
                        writer.add_scalar('Metrics/beam_vs_greedy_cer_improvement', cer_improvement, epoch)
                        writer.add_scalar('Metrics/beam_vs_greedy_wer_improvement', wer_improvement, epoch)

                    if run_decoder_val and decoder_cer != float('inf'):
                        dec_vs_greedy = ((greedy_cer - decoder_cer) / greedy_cer * 100) if greedy_cer > 0 else 0
                        writer.add_scalar('Metrics/decoder_vs_greedy_cer_improvement', dec_vs_greedy, epoch)

                if save_best_of_heads and run_decoder_val and decoder_cer != float('inf'):
                    if decoder_cer <= greedy_cer:
                        current_cer, current_mode = decoder_cer, 'decoder'
                    else:
                        current_cer, current_mode = greedy_cer, 'greedy'
                elif select_on_decoder and run_decoder_val and decoder_cer != float('inf'):
                    current_cer = decoder_cer
                    current_mode = 'decoder'
                else:
                    # best decoding path run this epoch; rescore is usually the winner
                    candidates = {'greedy': greedy_cer}
                    if run_beam_val:
                        candidates['beam'] = beam_cer
                    if run_decoder_val and decoder_cer != float('inf'):
                        candidates['decoder'] = decoder_cer
                    if rescored_cer != float('inf'):
                        candidates['rescored'] = rescored_cer
                    current_mode = min(candidates, key=candidates.get)
                    current_cer = candidates[current_mode]

                # a CER tie only replaces the checkpoint if validation loss also improved
                improved = current_cer < best_cer or (
                    current_cer == best_cer and val_loss < best_val_loss)
                if improved:
                    improvement = ((best_cer - current_cer) / best_cer * 100) if best_cer != float('inf') else 0
                    best_cer = current_cer
                    best_val_loss = val_loss
                    rounds_since_best = 0
                    if is_master and log_dir:
                        eval_module.save(log_dir / 'best_model.pt', extra_state={'raw_model_state_dict': self.state_dict()})
                        print(
                            f"  Saved best model ({current_mode} CER: {best_cer:.4f}, improvement: {improvement:.2f}%)"
                        )
                else:
                    rounds_since_best += 1

                if writer:
                    writer.add_scalar('Metrics/best_cer', best_cer, epoch)

                if (epoch + 1) % log_samples_frequency == 0:

                    if run_beam_val:
                        gts = combined_metrics['ground_truths'][:5]
                        greedy_preds = combined_metrics['greedy']['predictions'][:5]
                        beam_preds = combined_metrics['beam']['predictions'][:5]

                        comparison_text = "\n\n".join([
                            f"**Sample {i+1}:**\n\n"
                            f"Ground Truth: `{gt}`\n\n"
                            f"Greedy: `{gpred}` {'(exact)' if gt == gpred else ''} (CER: {jiwer.cer([gt], [gpred]):.2%})\n\n"
                            f"Beam:   `{bpred}` {'(exact)' if gt == bpred else ''} (CER: {jiwer.cer([gt], [bpred]):.2%})"
                            for i, (gt, gpred, bpred) in enumerate(zip(gts, greedy_preds, beam_preds))
                        ])
                        if writer:
                            writer.add_text('Predictions/beam_vs_greedy', comparison_text, epoch)
                    else:
                        gts = greedy_metrics['ground_truths'][:5]
                        greedy_preds = greedy_metrics['predictions'][:5]

                        greedy_text = "\n\n".join([
                            f"**Sample {i+1}:**\n\n"
                            f"Ground Truth: `{gt}`\n\n"
                            f"Prediction: `{pred}` {'(exact)' if gt == pred else ''} (CER: {jiwer.cer([gt], [pred]):.2%})"
                            for i, (gt, pred) in enumerate(zip(gts, greedy_preds))
                        ])
                        if writer:
                            writer.add_text('Predictions/greedy_samples', greedy_text, epoch)

                    if run_decoder_val and writer and val_metrics is not None and 'decoder' in val_metrics:
                        gts = val_metrics['ground_truths'][:5]
                        decoder_preds = val_metrics['decoder']['predictions'][:5]
                        decoder_text = "\n\n".join([
                            f"**Sample {i+1}:**\n\n"
                            f"Ground Truth: `{gt}`\n\n"
                            f"Decoder: `{pred}` {'(exact)' if gt == pred else ''} (CER: {jiwer.cer([gt], [pred]):.2%})"
                            for i, (gt, pred) in enumerate(zip(gts, decoder_preds))
                        ])
                        writer.add_text('Predictions/decoder_samples', decoder_text, epoch)

                if is_master:
                    print(f"Best CER so far: {best_cer:.4f}")
                    if os.environ.get('TRAIN_OUTPUT'):
                        point = {'epoch': epoch + 1, 'cer': current_cer, 'best_cer': best_cer,
                                 'ts': time.time()}
                        if val_metrics is not None and val_metrics.get('cer_by_dataset'):
                            point['cer_by_dataset'] = val_metrics['cer_by_dataset']
                        with open(Path(os.environ['TRAIN_OUTPUT']) / 'progress.jsonl', 'a', encoding='utf-8') as _progress:
                            _progress.write(json.dumps(point) + '\n')

                if patience is not None and rounds_since_best >= patience:
                    if is_master:
                        print(f"Early stop: no improvement in {rounds_since_best} validation rounds (patience {patience})")
                    break

            if is_master and log_dir:
                # Rolling resume checkpoint (see resume_checkpoint above). Written via a temp
                # file + rename so a crash mid-save cannot corrupt the previous one.
                checkpoint_tmp = log_dir / 'last_checkpoint.pt.tmp'
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_cer': best_cer,
                    'best_val_loss': best_val_loss,
                    'rounds_since_best': rounds_since_best,
                    'model_state_dict': self.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'tokenizer_tokens': self.tokenizer.tokens_list,
                }, checkpoint_tmp)
                checkpoint_tmp.replace(log_dir / 'last_checkpoint.pt')

        if writer:
            writer.close()

        return best_cer

    def _log_per_dataset_metrics(
        self,
        writer: SummaryWriter,
        epoch: int,
        ground_truths: List[str],
        labels: List[str],
        mode_preds: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Logs CER and WER per source dataset under 'per_dataset/<label>/' on the run's own
        writer; a child SummaryWriter per dataset spawned ~125 sibling runs in the TensorBoard
        sidebar. Returns the per-dataset greedy CER for the progress file."""
        label_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_indices[label].append(i)

        greedy_cer_by_label: Dict[str, float] = {}
        for label, indices in label_indices.items():
            # sample_count is constant across epochs, so once is enough
            if epoch == 0:
                writer.add_scalar(f'per_dataset/{label}/sample_count', len(indices), epoch)
            gts = [ground_truths[i] for i in indices]

            for mode_name, preds in mode_preds.items():
                subset_preds = [preds[i] for i in indices]
                try:
                    cer = jiwer.cer(gts, subset_preds)
                    wer = jiwer.wer(gts, subset_preds)
                except Exception:
                    continue
                # char accuracy is 1 - CER up to length effects; the two extra series per mode
                # multiply out to thousands once the corpus has hundreds of datasets
                writer.add_scalar(f'per_dataset/{label}/CER_{mode_name}', cer, epoch)
                writer.add_scalar(f'per_dataset/{label}/WER_{mode_name}', wer, epoch)
                if mode_name == 'greedy':
                    greedy_cer_by_label[label] = cer
        return greedy_cer_by_label

    def validate(
        self,
        criterion,
        dataloader: DataLoader,
        device: torch.device,
        writer: Optional[SummaryWriter],
        epoch: int,
        log_attention: bool = False,
        decoding_mode: Literal['greedy', 'beam_search', 'both'] = 'greedy',
        beam_width: int = 10,
        length_penalty: float = 0.1,
        blank_penalty: float = 0.0,
        debug: bool = False,
        run_decoder: bool = False,
        decoder_max_len: int = 256,
        world_size: int = 1,
        run_rescore: bool = False,
        rescore_alpha: float = 0.3,
        rescore_beta: float = 0.0,
        rescore_nbest: int = 8
    ) -> dict:
        self.eval()

        all_ground_truths = []
        all_greedy_preds = []
        all_beam_preds = []
        all_decoder_preds = []
        all_rescored_preds = []
        all_labels = []
        total_val_loss = 0.0
        num_val_batches = 0

        run_decoder = run_decoder and self.use_decoder
        run_rescore = run_rescore and self.use_decoder

        random_batch_index = random.randint(0, len(dataloader) - 1)

        with torch.no_grad():
            for batch_idx, (images, attention_masks, targets, input_lengths, target_lengths, dataset_labels) in enumerate(
                tqdm(dataloader, desc=f"Validating ({decoding_mode})", leave=False)
            ):
                images = images.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                attention_masks = attention_masks.to(device)

                decoder_ids = None
                conventions = self.convention_vector(dataset_labels, device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    memory, memory_key_padding_mask, _, _ = self._encode(
                        images, attention_masks=attention_masks, conventions=conventions)
                    logits = self.output_layer(memory)
                    log_probs = nn.functional.log_softmax(logits, dim=2)

                    try:
                        val_loss = criterion(log_probs.float(), targets, input_lengths, target_lengths)
                        if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                            total_val_loss += val_loss.item()
                            num_val_batches += 1
                    except Exception as e:
                        if debug:
                            print(f"  Validation loss calculation failed: {e}")

                    if run_decoder:
                        decoder_ids = self.decode_autoregressive(
                            memory, memory_key_padding_mask, max_len=decoder_max_len
                        )

                    if debug and batch_idx == 0:
                        print("\n Log probs debug:")
                        print(f"   Shape: {log_probs.shape}")
                        print(f"   Min/Max: {log_probs.min():.2f} / {log_probs.max():.2f}")
                        print(f"   Contains NaN: {torch.isnan(log_probs).any()}")
                        print(f"   Contains Inf: {torch.isinf(log_probs).any()}")
                        print(f"   Blank token ID: {self.tokenizer.blank_token_id}")
                        print(f"   Vocab size: {len(self.tokenizer)}")

                if decoder_ids is not None:
                    all_decoder_preds.extend(self._decoder_ids_to_text(decoder_ids))

                gt_sequences = torch.split(targets, target_lengths.tolist())
                batch_ground_truths = self.tokenizer.batch_decode(gt_sequences, skip_special_tokens=True)
                all_ground_truths.extend(batch_ground_truths)
                all_labels.extend(dataset_labels)

                if decoding_mode in ['greedy', 'both']:
                    pred_tokens = log_probs.argmax(dim=2).permute(1, 0)    # [B, T]
                    batch_pred_seqs = []
                    for i in range(pred_tokens.shape[0]):
                        T_i = input_lengths[i].item()
                        pred_seq = pred_tokens[i, :T_i].tolist()
                        batch_pred_seqs.append(pred_seq)

                    batch_greedy_preds = self.tokenizer.batch_ctc_decode(batch_pred_seqs, skip_special_tokens=True)
                    all_greedy_preds.extend(batch_greedy_preds)

                if decoding_mode in ['beam_search', 'both']:
                    beam_preds_batch = self.tokenizer.batch_decode_beam_search(
                        log_probs,
                        input_lengths,
                        beam_width=beam_width,
                        blank_penalty=blank_penalty,
                        length_penalty=length_penalty
                    )
                    all_beam_preds.extend(beam_preds_batch)

                if run_rescore:
                    nbest_lists = self.tokenizer.batch_beam_search_nbest(
                        log_probs, input_lengths, beam_width=beam_width, nbest=rescore_nbest
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        rescored_batch = self.rescore_hypotheses(
                            memory, memory_key_padding_mask, nbest_lists,
                            alpha=rescore_alpha, beta=rescore_beta
                        )
                    all_rescored_preds.extend(rescored_batch)

                if writer and epoch is not None and batch_idx == random_batch_index:
                    # one per row, since lines side by side are too wide to read
                    img_grid = torchvision.utils.make_grid(to_display_image(images[:4]), nrow=1)
                    writer.add_image('Images/validation_samples', img_grid, epoch)
                    if log_attention:
                        self._log_attention_visualization(
                            writer, images[:2], attention_masks[:2], epoch, device, prefix='Validation', debug=debug
                        )

        if world_size > 1 and dist.is_initialized():
            def _gather_concat(local_items):
                gathered = [None] * world_size
                dist.all_gather_object(gathered, local_items)
                merged = []
                for part in gathered:
                    merged.extend(part)
                return merged

            all_ground_truths = _gather_concat(all_ground_truths)
            all_greedy_preds = _gather_concat(all_greedy_preds)
            all_beam_preds = _gather_concat(all_beam_preds)
            all_decoder_preds = _gather_concat(all_decoder_preds)
            all_rescored_preds = _gather_concat(all_rescored_preds)
            all_labels = _gather_concat(all_labels)

            loss_stats = [None] * world_size
            dist.all_gather_object(loss_stats, (total_val_loss, num_val_batches))
            total_val_loss = sum(part[0] for part in loss_stats)
            num_val_batches = sum(part[1] for part in loss_stats)

        if not all_ground_truths:
            return {
                'cer': float('inf'),
                'wer': float('inf'),
                'val_loss': float('inf'),
                'decoding_mode': decoding_mode
            }

        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        results = {'val_loss': avg_val_loss, 'ground_truths': all_ground_truths[:5]}

        if decoding_mode in ['greedy', 'both'] and all_greedy_preds:
            if debug:
                print(
                    f"  [1/{2 if decoding_mode == 'both' else 1}] Calculated greedy metrics for {len(all_greedy_preds)} samples"
                )

            greedy_cer = jiwer.cer(all_ground_truths, all_greedy_preds)
            greedy_wer = jiwer.wer(all_ground_truths, all_greedy_preds)

            results['greedy'] = {
                'cer': greedy_cer,
                'wer': greedy_wer,
                'predictions': all_greedy_preds[:5]
            }

            if decoding_mode == 'greedy':
                results.update({
                    'cer': greedy_cer,
                    'wer': greedy_wer,
                    'predictions': all_greedy_preds[:5],
                    'decoding_mode': 'greedy'
                })

        if decoding_mode in ['beam_search', 'both'] and all_beam_preds:
            if debug:
                print(f"  [2/2] Calculated beam metrics for {len(all_beam_preds)} samples")

            beam_cer = jiwer.cer(all_ground_truths, all_beam_preds)
            beam_wer = jiwer.wer(all_ground_truths, all_beam_preds)

            results['beam'] = {
                'cer': beam_cer,
                'wer': beam_wer,
                'predictions': all_beam_preds[:5]
            }

            if decoding_mode == 'beam_search':
                results.update({
                    'cer': beam_cer,
                    'wer': beam_wer,
                    'predictions': all_beam_preds[:5],
                    'decoding_mode': 'beam_search'
                })

        if decoding_mode == 'both' and debug and 'greedy' in results and 'beam' in results:
            greedy_cer = results['greedy']['cer']
            beam_cer = results['beam']['cer']
            improvement = ((greedy_cer - beam_cer) / greedy_cer * 100) if greedy_cer > 0 else 0

            print("\n Decoding Comparison:")
            print(f"     Greedy CER: {greedy_cer:.4f} | WER: {results['greedy']['wer']:.4f}")
            print(f"     Beam CER:   {beam_cer:.4f} | WER: {results['beam']['wer']:.4f}")
            print(f"     Improvement: {improvement:+.2f}%")

            differences = [(gt, results['greedy']['predictions'][i], results['beam']['predictions'][i])
                           for i, gt in enumerate(all_ground_truths[:10])
                           if i < len(results['greedy']['predictions']) and i < len(results['beam']['predictions'])
                           and results['greedy']['predictions'][i] != results['beam']['predictions'][i]]

            if differences:
                print("\n  Example differences (first 3):")
                for i, (gt, g, b) in enumerate(differences[:3], 1):
                    g_dist = jiwer.cer([gt], [g]) * len(gt)
                    b_dist = jiwer.cer([gt], [b]) * len(gt)
                    print(f"     {i}. GT:     '{gt}'")
                    print(f"        Greedy: '{g}' (dist: {g_dist:.1f})")
                    print(f"        Beam:   '{b}' (dist: {b_dist:.1f})")
                    print()

        if run_decoder and all_decoder_preds:
            decoder_cer = jiwer.cer(all_ground_truths, all_decoder_preds)
            decoder_wer = jiwer.wer(all_ground_truths, all_decoder_preds)

            results['decoder'] = {
                'cer': decoder_cer,
                'wer': decoder_wer,
                'predictions': all_decoder_preds[:5]
            }

        if run_rescore and all_rescored_preds:
            rescored_cer = jiwer.cer(all_ground_truths, all_rescored_preds)
            rescored_wer = jiwer.wer(all_ground_truths, all_rescored_preds)

            results['rescored'] = {
                'cer': rescored_cer,
                'wer': rescored_wer,
                'predictions': all_rescored_preds[:5]
            }

        if writer and epoch is not None:
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)

            if 'greedy' in results:
                writer.add_scalar('Metrics/CER_greedy', results['greedy']['cer'], epoch)
                writer.add_scalar('Metrics/WER_greedy', results['greedy']['wer'], epoch)

            if 'beam' in results:
                writer.add_scalar('Metrics/CER_beam', results['beam']['cer'], epoch)
                writer.add_scalar('Metrics/WER_beam', results['beam']['wer'], epoch)

            if 'decoder' in results:
                writer.add_scalar('Metrics/CER_decoder', results['decoder']['cer'], epoch)
                writer.add_scalar('Metrics/WER_decoder', results['decoder']['wer'], epoch)

            if 'rescored' in results:
                writer.add_scalar('Metrics/CER_rescored', results['rescored']['cer'], epoch)
                writer.add_scalar('Metrics/WER_rescored', results['rescored']['wer'], epoch)

            if all_labels and len(set(all_labels)) > 1:
                mode_preds = {}
                if decoding_mode in ['greedy', 'both'] and all_greedy_preds:
                    mode_preds['greedy'] = all_greedy_preds
                if decoding_mode in ['beam_search', 'both'] and all_beam_preds:
                    mode_preds['beam'] = all_beam_preds
                if run_decoder and all_decoder_preds:
                    mode_preds['decoder'] = all_decoder_preds
                if run_rescore and all_rescored_preds:
                    mode_preds['rescored'] = all_rescored_preds
                results['cer_by_dataset'] = self._log_per_dataset_metrics(
                    writer, epoch, all_ground_truths, all_labels, mode_preds)

            active_mode = 'beam' if 'beam' in results and decoding_mode == 'beam_search' else 'greedy'
            active_preds = results.get(active_mode, {}).get('predictions', [])

            errors = [(gt, pred) for gt, pred in zip(all_ground_truths[:20], active_preds[:20]) if gt != pred][:5]

            if errors:
                error_text = "\n\n".join(
                    f"**Error {i+1}:**\n\nExpected: `{gt}`\n\nGot: `{pred}`" for i, (gt, pred) in enumerate(errors)
                )
                writer.add_text(f'Errors/samples_{active_mode}', error_text, epoch)

            correct = [(gt, pred) for gt, pred in zip(all_ground_truths[:20], active_preds[:20]) if gt == pred][:3]

            if correct:
                correct_text = "\n\n".join(f"**Correct {i+1}:**\n\n`{gt}`" for i, (gt, _) in enumerate(correct))
                writer.add_text(f'Correct/samples_{active_mode}', correct_text, epoch)

        return results

    def predict(
        self,
        images: List[Image.Image],
        device: torch.device,
        batch_size: int = 8,
        use_beam_search: bool = False,
        beam_width: int = 10,
        length_penalty: float = 0.1,
        blank_penalty: float = 0.0,
        fill_value: float = 1.0,
        debug: bool = False,
        use_decoder: bool = False,
        decoder_max_len: int = 256
    ) -> List[str]:
        self.to(device)
        self.eval()
        use_decoder = use_decoder and self.use_decoder

        normalize_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        def _process_pil_image(img: Image.Image) -> Tuple[torch.Tensor, int]:
            """Transparent pixels take the most common opaque colour."""
            img = img.convert('RGBA')
            img_array = np.array(img)
            alpha = img_array[:, :, 3]

            if np.min(alpha) < 255:
                non_transparent_mask = alpha == 255
                non_transparent_pixels = img_array[non_transparent_mask][:, :3]
                if len(non_transparent_pixels) == 0:
                    prominent_color = (255, 255, 255)
                else:
                    colors, counts = np.unique(non_transparent_pixels, axis=0, return_counts=True)
                    prominent_color = tuple(colors[counts.argmax()].tolist())

                background = Image.new('RGB', img.size, prominent_color)
                background.paste(img, (0, 0), mask=img.split()[3])
                img = background
            else:
                img = img.convert('RGB')

            w, h = img.size
            new_width = int(w * (self.target_height / h))
            if self.max_width and new_width > self.max_width:
                new_width = self.max_width

            img = img.resize((new_width, self.target_height), Image.Resampling.LANCZOS)
            tensor = normalize_transform(image=np.array(img))['image']

            return tensor, new_width

        all_predictions = []

        with torch.no_grad():
            for start_idx in range(0, len(images), batch_size):
                batch_images_pil = images[start_idx:start_idx + batch_size]

                processed_tensors = []
                widths = []

                for img in batch_images_pil:
                    try:
                        tensor, w = _process_pil_image(img)
                        processed_tensors.append(tensor)
                        widths.append(w)
                    except Exception as e:
                        if debug:
                            print(f"Skipping invalid image: {e}")
                        # Fallback for failed images to keep batch alignment
                        dummy = torch.ones(3, self.target_height, self.patch_size) * fill_value
                        processed_tensors.append(dummy)
                        widths.append(self.patch_size)

                per_sample_target_W = [math.ceil(w / self.patch_size) * self.patch_size for w in widths]
                batch_target_W = max(per_sample_target_W)

                num_patches_h = self.target_height // self.patch_size
                num_patches_w = batch_target_W // self.patch_size

                padded_images = []
                attention_masks = []
                input_lengths = []

                for img_tensor, w, targ_w in zip(processed_tensors, widths, per_sample_target_W):
                    pad_to_batch = batch_target_W - w
                    if pad_to_batch > 0:
                        img_tensor = F.pad(img_tensor, (0, pad_to_batch, 0, 0), mode="constant", value=fill_value)
                    padded_images.append(img_tensor)

                    real_w_patches = targ_w // self.patch_size

                    if self.use_2d_structure:
                        mask_grid = torch.ones(num_patches_w, num_patches_h, dtype=torch.bool)
                        mask_grid[:real_w_patches, :] = False
                        mask = mask_grid.reshape(-1)
                        input_lengths.append(real_w_patches * num_patches_h)
                    else:
                        mask = torch.ones(num_patches_w, dtype=torch.bool)
                        mask[:real_w_patches] = False
                        input_lengths.append(real_w_patches)

                    attention_masks.append(mask)

                batch_tensor = torch.stack(padded_images).to(device)
                attn_masks = torch.stack(attention_masks).to(device)
                in_lens = torch.as_tensor(input_lengths, dtype=torch.long, device=device)

                decoder_ids = None
                log_probs = None
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        memory, memory_key_padding_mask, _, _ = self._encode(
                            batch_tensor, attention_masks=attn_masks
                        )
                        if use_decoder:
                            decoder_ids = self.decode_autoregressive(
                                memory, memory_key_padding_mask, max_len=decoder_max_len
                            )
                        else:
                            logits = self.output_layer(memory)
                            log_probs = nn.functional.log_softmax(logits, dim=2)
                except Exception as e:
                    if debug:
                        print(f"Forward pass error: {e}")
                    all_predictions.extend([''] * len(batch_images_pil))
                    continue

                if use_decoder:
                    all_predictions.extend(self._decoder_ids_to_text(decoder_ids))
                    continue

                if use_beam_search:
                    try:
                        beam_preds_batch = self.tokenizer.batch_decode_beam_search(
                            log_probs,
                            in_lens,
                            beam_width=beam_width,
                            blank_penalty=blank_penalty,
                            length_penalty=length_penalty
                        )
                        all_predictions.extend(beam_preds_batch)
                    except Exception as e:
                        if debug:
                            print(f"Beam search error, falling back to greedy: {e}")
                        use_beam_search = False

                if not use_beam_search:
                    pred_tokens = log_probs.argmax(dim=2).permute(1, 0).cpu()
                    for i in range(len(batch_images_pil)):
                        T_i = in_lens[i].item()
                        pred_text = self.tokenizer.ctc_decode(pred_tokens[i, :T_i].tolist(), skip_special_tokens=True)
                        all_predictions.append(pred_text)

        return all_predictions

    def _log_attention_visualization(
        self,
        writer: "SummaryWriter",
        images: "torch.Tensor",
        attention_masks: "torch.Tensor",
        epoch: int,
        device: "torch.device",
        prefix: str = 'Validation',
        debug: bool = False
    ) -> None:
        if writer is None:
            return

        try:

            def get_display_img(tensor):
                return to_display_image(tensor)

            def reshape_mask_to_grid(mask_1d, grid_h, grid_w):
                total = mask_1d.numel()

                # collapsed CTC maps have fewer rows than the encoder grid, so fold the mask
                # over its height axis to get per-column validity
                if self.use_2d_structure and grid_w > 0 and total % grid_w == 0 and total // grid_w > grid_h:
                    full_height = total // grid_w
                    col_valid = mask_1d.view(grid_w, full_height).any(dim=1)
                    return col_valid.view(1, grid_w).expand(grid_h, grid_w).contiguous()

                # a longer mask carries leading CLS tokens
                tgt = grid_h * grid_w
                if total > tgt:
                    mask_1d = mask_1d[-tgt:]
                elif total < tgt:
                    mask_1d = torch.cat([
                        mask_1d,
                        torch.zeros(tgt - total, device=mask_1d.device, dtype=mask_1d.dtype)
                    ])

                if self.use_2d_structure:
                    # column-major: the sequence fills height first
                    return mask_1d.view(grid_w, grid_h).t()
                else:
                    # row-major: the sequence fills width first
                    return mask_1d.view(grid_h, grid_w)

            with torch.no_grad():
                images = images.to(device)
                masks = attention_masks.to(device) if attention_masks is not None else None
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _, _, viz_out, _ = self.forward(images, attention_masks=masks, return_attention=True)

            if not viz_out:
                return

            disp_img = get_display_img(images[0])
            _, H, W = disp_img.shape
            writer.add_image(f'input/{prefix}_sample', disp_img, epoch)

            cmaps = {'confidence_map': 'viridis', 'entropy_map': 'plasma', 'span_mask': 'Reds', 'default': 'magma'}

            for k, v in viz_out.items():
                if not (torch.is_tensor(v) and v.ndim >= 3):
                    continue

                hm = v[0, 0] if v.ndim == 4 else v[0]
                hm = hm.detach().float().cpu().nan_to_num()
                grid_h, grid_w = hm.shape

                # rebuilt from the input mask so the grid geometry is exact
                valid2d = None
                if masks is not None:
                    m = masks[0].detach().cpu().view(-1)
                    valid_1d = ~m    # collate_fn marks padding True
                    valid2d = reshape_mask_to_grid(valid_1d, grid_h, grid_w)

                if 'expert_routing' in k:
                    num_experts = self.num_experts
                    hm = hm / max(1, (num_experts - 1))
                    cname = 'tab10'
                else:
                    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                    cname = cmaps.get(k, cmaps['default'])

                if valid2d is not None and valid2d.shape == hm.shape:
                    hm = hm.masked_fill(~valid2d, 0.0)

                hm_up = torch.nn.functional.interpolate(hm[None, None], size=(H, W), mode='nearest')[0, 0]

                hm_rgb = torch.tensor(cm.get_cmap(cname)(hm_up.numpy())[:, :, :3]).permute(2, 0, 1).to(disp_img)

                alpha = torch.zeros(1, H, W).to(disp_img.device)

                if valid2d is not None:
                    valid_up = torch.nn.functional.interpolate(
                        valid2d.float()[None, None], size=(H, W), mode='nearest'
                    )[0]
                    alpha = valid_up * 0.5
                else:
                    alpha[:] = 0.5

                # 1px erosion so the overlay does not bleed past the edge
                if alpha.sum() > 0:
                    alpha[:, -1, :] = 0
                    alpha[:, :, -1] = 0

                out = (1 - alpha) * disp_img + alpha * hm_rgb
                writer.add_image(f'viz/{prefix}_{k}', out.clamp(0, 1), epoch)

            for k, v in viz_out.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    writer.add_scalar(f'features/{prefix}_{k}', v.item(), epoch)

        except Exception as e:
            if debug:
                print(f"Viz error: {e}")
                traceback.print_exc()

    def save(self, file_path: Path, extra_state: Optional[dict] = None):
        payload = {
            'target_height': self.target_height,
            'backbone_name': self.backbone_name,
            'cnn_width_stride': self.patch_size_w,
            'cnn_patch_size': self.patch_size,
            'dino_v3_backbone': self.dinov3_ckpt,
            'use_2d_structure': self.use_2d_structure,
            'collapse_height_for_ctc': self.collapse_height_for_ctc,
            'collapse_kernel_width': self.collapse_kernel_width,
            'collapse_depth': self.collapse_depth,
            'num_encoder_layers': self.num_layers,
            'num_experts': self.transformer_layers[0].moe_block.num_ffn_experts,
            'top_k_experts': self.transformer_layers[0].moe_block.top_k,
            'expert_ffn': self.transformer_layers[0].moe_block.expert_ffn,
            'expert_hidden': self.transformer_layers[0].moe_block.expert_hidden,
            'num_shared_experts': self.transformer_layers[0].moe_block.num_shared_experts,
            'router_balance': self.transformer_layers[0].moe_block.router_balance,
            'router_bias_rate': self.transformer_layers[0].moe_block.router_bias_rate,
            'num_conv_experts': self.transformer_layers[0].moe_block.num_conv_experts,
            'conv_expert_kernel': self.transformer_layers[0].moe_block.conv_expert_kernel,
            'convention_names': self.convention_names,
            'convention_by_dataset': self.convention_by_dataset,
            'tokenizer_normalize_unicode': self.tokenizer.normalize_unicode,
            'tokenizer_unk_token': self.tokenizer.unk_token,
            'tokenizer_blank_token': self.tokenizer.blank_token,
            'tokenizer_pad_token': self.tokenizer.pad_token,
            'tokenizer_eos_token': self.tokenizer.eos_token,
            'tokenizer_bos_token': self.tokenizer.bos_token,
            'tokenizer_tokens': self.tokenizer.tokens_list,
            'max_width': self.max_width,
            'use_decoder': self.use_decoder,
            'decoder_layers': self.decoder_layers,
            'decoder_heads': self.decoder_heads,
            'decoder_dim_feedforward': self.decoder_dim_feedforward,
            'decoder_dropout': self.decoder_dropout,
            'model_state_dict': self.state_dict()
        }
        if extra_state:
            payload.update(extra_state)
        torch.save(payload, file_path)

    @staticmethod
    def _resolve_collapse_config(checkpoint: dict, state_dict: dict) -> Tuple[bool, int, int]:
        """Recovers the ColumnCollapse geometry from a checkpoint. Checkpoints written before
        these keys were saved carry the answer only in the weights: one Conv2d per collapse
        depth, each with a (grid_height, collapse_kernel_width) kernel."""
        conv_weights = [state_dict[key] for key in state_dict
                        if re.fullmatch(r'column_collapse\.convs\.\d+\.weight', key)]
        enabled = checkpoint.get('collapse_height_for_ctc', bool(conv_weights))
        kernel_width = checkpoint.get('collapse_kernel_width')
        depth = checkpoint.get('collapse_depth')
        if kernel_width is None:
            kernel_width = conv_weights[0].shape[-1] if conv_weights else 3
        if depth is None:
            depth = len(conv_weights) if conv_weights else 1
        return bool(enabled), int(kernel_width), int(depth)

    @classmethod
    def load(cls, file_path: str, device: torch.device = torch.device('cpu'),
             kenlm_path: Optional[Path] = None, lm_dict_path: Optional[Path] = None,
             lm_weight: float = 2.0):
        """Loads a model and its tokenizer from a saved checkpoint. Every architecture setting
        comes from the checkpoint, so the rebuilt model always matches the saved weights. Pass
        kenlm_path/lm_dict_path to fuse an n-gram LM into the tokenizer's beam search."""
        checkpoint = torch.load(file_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        # tokenizer_tokens carries the checkpoint's own inventory, so character-vocabulary
        # checkpoints still load alongside byte-level ones
        tokenizer = DeepTokenizer(
            normalize_unicode=checkpoint.get('tokenizer_normalize_unicode', True),
            unk_token=checkpoint.get('tokenizer_unk_token', '[UNK]'),
            blank_token=checkpoint.get('tokenizer_blank_token', '[BLANK]'),
            pad_token=checkpoint.get('tokenizer_pad_token', '[PAD]'),
            eos_token=checkpoint.get('tokenizer_eos_token', '[EOS]'),
            bos_token=checkpoint.get('tokenizer_bos_token', '[BOS]'),
            tokens=checkpoint.get('tokenizer_tokens'),
            kenlm_path=kenlm_path,
            lm_dict_path=lm_dict_path,
            lm_weight=lm_weight,
        )

        # Infer num_experts from checkpoint if not saved explicitly
        num_experts = checkpoint.get('num_experts', 8)
        top_k_experts = checkpoint.get('top_k_experts', 2)
        if num_experts == 8:
            max_expert = 0
            for k in state_dict:
                if 'moe_block.experts.' in k:
                    segment = k.split('moe_block.experts.')[1].split('.')[0]
                    if segment.isdigit():    # stacked SwiGLU weights have no per-expert index
                        max_expert = max(max_expert, int(segment))
            if max_expert > 0:
                num_experts = max_expert + 1

        num_encoder_layers = checkpoint.get('num_encoder_layers')
        if num_encoder_layers is None:
            layer_indices = {int(k.split('.')[1]) for k in state_dict if k.startswith('transformer_layers.')}
            num_encoder_layers = (max(layer_indices) + 1) if layer_indices else 3

        collapse_enabled, collapse_kernel_width, collapse_depth = cls._resolve_collapse_config(
            checkpoint, state_dict
        )

        model = cls(
            tokenizer=tokenizer,
            target_height=checkpoint.get('target_height', 96),
            backbone_name=checkpoint.get('backbone_name', 'dinov3'),
            cnn_width_stride=checkpoint.get('cnn_width_stride', 16),
            cnn_patch_size=checkpoint.get('cnn_patch_size', 16),
            dinov3_ckpt=checkpoint.get('dino_v3_backbone', "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            use_2d_structure=checkpoint.get('use_2d_structure', False),
            num_encoder_layers=num_encoder_layers,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            expert_ffn=checkpoint.get('expert_ffn', 'gelu'),
            expert_hidden=checkpoint.get('expert_hidden', 2048),
            num_shared_experts=checkpoint.get('num_shared_experts', 0),
            router_balance=checkpoint.get('router_balance', 'aux_loss'),
            router_bias_rate=checkpoint.get('router_bias_rate', 1e-3),
            num_conv_experts=checkpoint.get('num_conv_experts', 0),
            conv_expert_kernel=checkpoint.get('conv_expert_kernel', 15),
            convention_names=checkpoint.get('convention_names'),
            convention_by_dataset=checkpoint.get('convention_by_dataset'),
            max_width=checkpoint.get('max_width', 2048),
            collapse_height_for_ctc=collapse_enabled,
            collapse_kernel_width=collapse_kernel_width,
            collapse_depth=collapse_depth,
            use_decoder=checkpoint.get('use_decoder', False),
            decoder_layers=checkpoint.get('decoder_layers', 4),
            decoder_heads=checkpoint.get('decoder_heads', 8),
            decoder_dim_feedforward=checkpoint.get('decoder_dim_feedforward', 2048),
            decoder_dropout=checkpoint.get('decoder_dropout', 0.1),
        )

        model_keys = set(model.state_dict().keys())
        state_keys = set(state_dict.keys())
        missing = model_keys - state_keys
        unexpected = state_keys - model_keys
        remap = {}
        for unexp in unexpected:
            if unexp.startswith('backbone.model.'):
                candidate = 'backbone.' + unexp[len('backbone.model.'):]
            elif unexp.startswith('backbone.'):
                candidate = 'backbone.model.' + unexp[len('backbone.'):]
            else:
                continue
            if candidate in missing:
                remap[unexp] = candidate
        if remap:
            for old_key, new_key in remap.items():
                state_dict[new_key] = state_dict.pop(old_key)

        model.load_state_dict(state_dict)
        model.to(device)
        return model, tokenizer
