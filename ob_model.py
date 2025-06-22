# %%
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging


@dataclass
class OrderbookModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, 2, num_features)`):
    """

    sample: torch.FloatTensor


def modulate2d(x, shift, scale):
    # x should be (bs, num_features)
    # shift and scale should be (bs, num_features)
    return x * (1 + scale) + shift

def modulate3d(x, shift, scale):
    # x should be (bs, length, num_features)
    # shift and scale should be (bs, num_features)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False, eps=1e-6)

    def forward(self, x, shift, scale):
        if x.dim() == 2:
            return modulate2d(self.norm(x), shift, scale)
        elif x.dim() == 3:
            return modulate3d(self.norm(x), shift, scale)
        raise ValueError(f"Expected x to be 2D or 3D tensor, but got {x.dim()} dimensions.")


class SinusoidalPositionEmbeddings(nn.Module):
    # TODO: maybe can do better
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalBlock(nn.Module):
    def __init__(self, data_dim, cond_dim, activation_fn):
        super().__init__()

        self.linear1 = nn.Linear(data_dim, data_dim)
        self.linear2 = nn.Linear(data_dim, data_dim)
        self.norm1 = nn.LayerNorm(data_dim)
        self.norm2 = nn.LayerNorm(data_dim)
        self.activation = get_activation(
            activation_fn
        )  # SiLU (Swish) is common in modern models

        # A small projection for the time embedding to match the data dimension
        self.cond_mlp = nn.Linear(cond_dim, data_dim)

    def forward(self, x, cond_emb):
        """
        x: The data input (e.g., from the previous layer)
        cond_emb: The centrally computed conditioning embedding (could be timestep, class, etc)
        """
        # Store original x for the residual connection
        x_residual = x

        # Project the time embedding and add it to the data
        # This is the core of the conditioning
        cond_projection = self.cond_mlp(self.activation(cond_emb))

        # First part of the main path
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)

        # INJECT THE CONDITIONING
        # TODO: this is using additive conditioning, could also use concat
        # print(x.shape, cond_projection.shape)
        x = x + cond_projection

        # Second part of the main path
        x = self.norm2(x)
        x = self.activation(x)
        x = self.linear2(x)

        # Add the residual connection
        return x + x_residual


class NaiveOrderbookModel(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        # dummy parameters for compatibility
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.sample_size = sample_size
        self.activation_fn = activation_fn
        self.time_emb_dim = time_emb_dim

        self.act_fn = get_activation(activation_fn)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            self.act_fn,
        )

        self.input_proj = nn.Linear(in_channels * sample_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_channels * sample_size)

        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    data_dim=hidden_dim,
                    cond_dim=time_emb_dim,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, sample, timestep, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size, 1), but got {timestep.shape}"

        original_size = sample.size()
        if sample.dim() == 3:
            sample = sample.reshape(original_size[0], -1)

        t_emb = self.time_mlp(timestep.to(sample.device))

        sample = self.input_proj(sample)

        for block in self.blocks:
            sample = block(sample, t_emb)  # Pass the SAME time embedding to each block

        sample = self.output_proj(sample)
        sample = sample.reshape(original_size[0], self.out_channels, -1)
        if not return_dict:
            return (sample,)

        return OrderbookModelOutput(sample=sample)


class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, n_heads=8, activation_fn="silu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = get_activation(activation_fn)

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)

    def forward(self, x, attn_mask=None):
        q, k, v = self.qkv_proj(self.norm1(x)).chunk(3, dim=-1)
        assert (
            q.shape == k.shape == v.shape == x.shape
        ), "Q, K, V must have the same shape"

        if attn_mask is None:
            attn_output, _ = self.attention(q, k, v)
        else:
            attn_output, _ = self.attention(
                q, k, v, attn_mask=attn_mask, is_causal=True
            )

        x = attn_output + x
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x))))
        return x


class AdaLNTransformerBlock(nn.Module):

    def __init__(self, hidden_dim, cond_dim, n_heads=8, activation_fn="silu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = AdaLN(hidden_dim)
        self.norm2 = AdaLN(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.activation = get_activation(activation_fn)

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), # took from https://github.com/facebookresearch/DiT/blob/main/models.py#L101
            nn.Linear(cond_dim, 4 * hidden_dim, bias=True)
        )

    def forward(self, x, cond_emb, attn_mask=None):
        # TODO: maybe also add gate mlp? Because its in https://github.com/facebookresearch/DiT/blob/main/models.py#L101
        shift1, shift2, scale1, scale2 = self.adaln_modulation(cond_emb).chunk(4, dim=-1)

        q, k, v = self.qkv_proj(self.norm1(x, shift1, scale1)).chunk(3, dim=-1)
        assert (
            q.shape == k.shape == v.shape == x.shape
        ), "Q, K, V must have the same shape"

        if attn_mask is None:
            attn_output, _ = self.attention(q, k, v)
        else:
            attn_output, _ = self.attention(
                q, k, v, attn_mask=attn_mask, is_causal=True
            )

        x = attn_output + x
        x = x + self.linear2(self.activation(self.linear1(self.norm2(x, shift2, scale2))))
        return x

class NaiveNonCausalTransformer(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        n_heads: int = 8,
        nonlinear_embed: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_heads = n_heads
        self.sample_size = sample_size
        if not nonlinear_embed:
            self.input_proj = nn.Linear(self.input_channels, hidden_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_channels, 2 * hidden_dim),
                get_activation(activation_fn),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
        self.act_fn = get_activation(activation_fn)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            self.act_fn,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, activation_fn)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, sample, timestep, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size, 1), but got {timestep.shape}"

        t_emb = self.time_mlp(timestep.to(sample.device))

        assert sample.dim() == 3
        bs, cn, length = sample.shape
        assert (
            cn == self.input_channels and length == self.sample_size
        ), f"Expected input shape (batch_size, {self.input_channels}, {self.sample_size}), but got {sample.shape}"
        hidden = self.input_proj(sample.transpose(1, 2))

        hidden += t_emb.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden)

        sample = self.output_proj(hidden).transpose(1, 2)
        if not return_dict:
            return (sample,)
        return OrderbookModelOutput(sample=sample)


class NaiveCausalTransformer(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        n_heads: int = 8,
        nonlinear_embed: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_heads = n_heads
        self.sample_size = sample_size
        if not nonlinear_embed:
            self.input_proj = nn.Linear(self.input_channels, hidden_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_channels, 2 * hidden_dim),
                get_activation(activation_fn),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
        self.act_fn = get_activation(activation_fn)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            self.act_fn,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, activation_fn)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, out_channels)

        self.register_buffer(
            "mask", torch.triu(torch.ones(sample_size, sample_size), diagonal=1).bool()
        )

    def forward(self, sample, timestep, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size, 1), but got {timestep.shape}"

        t_emb = self.time_mlp(timestep.to(sample.device))
        assert sample.dim() == 3
        bs, cn, length = sample.shape
        assert (
            cn == self.input_channels and length == self.sample_size
        ), f"Expected input shape (batch_size, {self.input_channels}, {self.sample_size}), but got {sample.shape}"
        hidden = self.input_proj(sample.transpose(1, 2))

        hidden += t_emb.unsqueeze(1)
        for block in self.blocks:
            hidden = block(hidden, attn_mask=self.mask[:length, :length])

        sample = self.output_proj(hidden).transpose(1, 2)
        if not return_dict:
            return (sample,)
        return OrderbookModelOutput(sample=sample)


class NaiveConditionalResNet(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        num_classes: int = 2,
        class_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.sample_size = sample_size
        self.activation_fn = activation_fn
        self.time_emb_dim = time_emb_dim

        self.act_fn = get_activation(activation_fn)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            self.act_fn,
        )
        self.class_mlp = nn.Embedding(num_classes, class_emb_dim)

        self.input_proj = nn.Linear(in_channels * sample_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_channels * sample_size)

        self.blocks = nn.ModuleList(
            [
                ConditionalBlock(
                    data_dim=hidden_dim,
                    cond_dim=time_emb_dim + class_emb_dim,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, sample, timestep, cls, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size), but got {timestep.shape}"

        if not isinstance(cls, torch.Tensor):
            raise ValueError(f"Expected cls to be a torch.Tensor, but got {type(cls)}")

        if cls.dim() == 0:
            cls = cls.unsqueeze(0)
        if cls.dim() == 1 and cls.size(0) == 1:
            cls = cls.repeat(sample.size(0))

        assert (
            cls.dim() == 1
        ), f"Expected cls to be of shape (batch_size), but got {cls.shape}"

        original_size = sample.size()
        if sample.dim() == 3:
            sample = sample.reshape(original_size[0], -1)

        t_emb = self.time_mlp(timestep.to(sample.device))
        cls_emb = self.class_mlp(cls.to(sample.device))
        cond_emb = torch.cat(
            (t_emb, cls_emb * 0.2), dim=-1
        )  # 0.2 is try to match the std of t_emb from init

        assert cond_emb.shape == (
            original_size[0],
            self.time_emb_dim + self.class_emb_dim,
        )

        sample = self.input_proj(sample)

        for block in self.blocks:
            sample = block(
                sample, cond_emb
            )  # Pass the SAME time embedding to each block

        sample = self.output_proj(sample)
        sample = sample.reshape(original_size[0], self.out_channels, -1)
        if not return_dict:
            return (sample,)

        return OrderbookModelOutput(sample=sample)

class AdaLNConditionalBlock(nn.Module):
    def __init__(self, data_dim, cond_dim, activation_fn):
        super().__init__()

        self.linear1 = nn.Linear(data_dim, data_dim)
        self.linear2 = nn.Linear(data_dim, data_dim)
        self.norm1 = AdaLN(data_dim)
        self.norm2 = AdaLN(data_dim)
        self.activation = get_activation(
            activation_fn
        )  # SiLU (Swish) is common in modern models

        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), # took from https://github.com/facebookresearch/DiT/blob/main/models.py#L101
            nn.Linear(cond_dim, 4 * data_dim, bias=True)
        )

    def forward(self, x, cond_emb):
        """
        x: The data input (e.g., from the previous layer)
        cond_emb: The centrally computed conditioning embedding (could be timestep, class, etc)
        """
        # Store original x for the residual connection

        shift1, shift2, scale1, scale2 = self.adaln_modulation(cond_emb).chunk(4, dim=-1)
        x_residual = x

        # First part of the main path
        x = self.norm1(x, shift1, scale1)
        x = self.activation(x)
        x = self.linear1(x)

        # Second part of the main path
        x = self.norm2(x, shift2, scale2)
        x = self.activation(x)
        x = self.linear2(x)

        # Add the residual connection
        return x + x_residual

class AdaLNConditionalResNet(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        num_classes: int = 2,
        class_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        **kwargs: Any,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim
        self.sample_size = sample_size
        self.activation_fn = activation_fn
        self.time_emb_dim = time_emb_dim

        self.act_fn = get_activation(activation_fn)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            self.act_fn,
        )
        self.class_mlp = nn.Embedding(num_classes, class_emb_dim)

        self.input_proj = nn.Linear(in_channels * sample_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_channels * sample_size)

        self.blocks = nn.ModuleList(
            [
                AdaLNConditionalBlock(
                    data_dim=hidden_dim,
                    cond_dim=time_emb_dim + class_emb_dim,
                    activation_fn=activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, sample, timestep, cls, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size), but got {timestep.shape}"

        if not isinstance(cls, torch.Tensor):
            raise ValueError(f"Expected cls to be a torch.Tensor, but got {type(cls)}")

        if cls.dim() == 0:
            cls = cls.unsqueeze(0)
        if cls.dim() == 1 and cls.size(0) == 1:
            cls = cls.repeat(sample.size(0))

        assert (
            cls.dim() == 1
        ), f"Expected cls to be of shape (batch_size), but got {cls.shape}"

        original_size = sample.size()
        if sample.dim() == 3:
            sample = sample.reshape(original_size[0], -1)

        t_emb = self.time_mlp(timestep.to(sample.device))
        cls_emb = self.class_mlp(cls.to(sample.device))
        cond_emb = torch.cat(
            (t_emb, cls_emb * 0.2), dim=-1
        )  # 0.2 is try to match the std of t_emb from init

        assert cond_emb.shape == (
            original_size[0],
            self.time_emb_dim + self.class_emb_dim,
        )

        sample = self.input_proj(sample)

        for block in self.blocks:
            sample = block(
                sample, cond_emb
            )  # Pass the SAME time embedding to each block

        sample = self.output_proj(sample)
        sample = sample.reshape(original_size[0], self.out_channels, -1)
        if not return_dict:
            return (sample,)

        return OrderbookModelOutput(sample=sample)


class AdaLNConditionalCausalTransformer(ModelMixin, ConfigMixin):
    _support_grad_checkpointing = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        num_layers: int = 1,
        hidden_dim: int = 128,
        time_emb_dim: int = 128,
        sample_size: int = 32,
        activation_fn: str = "silu",
        n_heads: int = 8,
        nonlinear_embed: bool = False,
        num_classes: int = 2,
        class_emb_dim: int = 128,
        **kwargs: Any,
    ):
        super().__init__()
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_heads = n_heads
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.class_emb_dim = class_emb_dim

        if not nonlinear_embed:
            self.input_proj = nn.Linear(self.input_channels, hidden_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(self.input_channels, 2 * hidden_dim),
                get_activation(activation_fn),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
        self.act_fn = get_activation(activation_fn)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            self.act_fn,
        )
        self.class_mlp = nn.Embedding(num_classes, class_emb_dim)
        self.blocks = nn.ModuleList(
            [
                AdaLNTransformerBlock(hidden_dim, class_emb_dim + time_emb_dim, n_heads=n_heads, activation_fn=activation_fn)
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, out_channels)

        self.register_buffer(
            "mask", torch.triu(torch.ones(sample_size, sample_size), diagonal=1).bool()
        )

    def forward(self, sample, timestep, cls, return_dict=True):
        if not isinstance(sample, torch.Tensor):
            raise ValueError(
                f"Expected sample to be a torch.Tensor, but got {type(sample)}"
            )

        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.size(0) == 1:
            timestep = timestep.repeat(sample.size(0))

        assert (
            timestep.dim() == 1
        ), f"Expected timestep to be of shape (batch_size, 1), but got {timestep.shape}"

        if cls.dim() == 0:
            cls = cls.unsqueeze(0)
        if cls.dim() == 1 and cls.size(0) == 1:
            cls = cls.repeat(sample.size(0))

        assert (
            cls.dim() == 1
        ), f"Expected cls to be of shape (batch_size), but got {cls.shape}"

        t_emb = self.time_mlp(timestep.to(sample.device))
        cls_emb = self.class_mlp(cls.to(sample.device))
        cond_emb = torch.cat((t_emb, cls_emb * 0.2), dim = -1)

        bs, cn, length = sample.shape
        assert (
            cn == self.input_channels and length == self.sample_size
        ), f"Expected input shape (batch_size, {self.input_channels}, {self.sample_size}), but got {sample.shape}"

        hidden = self.input_proj(sample.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden, cond_emb=cond_emb, attn_mask=self.mask[:length, :length])

        sample = self.output_proj(hidden).transpose(1, 2)
        if not return_dict:
            return (sample,)
        return OrderbookModelOutput(sample=sample)

# input = torch.randn(4, 2, 32)
# t = torch.randint(0, 1000, (4,))
# cls = torch.randint(0, 2, (4,))

# model = AdaLNConditionalCausalTransformer(
#     in_channels = 2,
#     out_channels=2,
#     num_layers=4,
#     hidden_dim = 128,
#     time_emb_dim = 128,
#     sample_size=32,
#     num_classes = 2,
#     class_emb_dim=128,
# )

# out = model(input, t, cls)

# %%
if __name__ == "__main__":
    model = NaiveOrderbookModel(num_layers=4, hidden_dim=127)
    model.save_config("./test_config.json")
    model = NaiveOrderbookModel.from_config("./test_config.json")