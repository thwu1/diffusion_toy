#%%
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
from diffusers.models.attention_processor import (AttentionProcessor,
                                                  AttnProcessor)
from diffusers.models.embeddings import (GaussianFourierProjection,
                                         ImageHintTimeEmbedding,
                                         ImageProjection, ImageTimeEmbedding,
                                         TextImageProjection,
                                         TextImageTimeEmbedding,
                                         TextTimeEmbedding, TimestepEmbedding,
                                         Timesteps)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging


#%%
@dataclass
class ObModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, 2, num_features)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor

# tembed = SinusoidalPositionEmbeddings(32)
#%%


# You can reuse the sinusoidal embedding class from before
class SinusoidalPositionEmbeddings(nn.Module):
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

# ---------------------------------------------------------------------------------
# 1. DESIGN THE REUSABLE CONDITIONAL BLOCK
# ---------------------------------------------------------------------------------
class ConditionalBlock(nn.Module):
    def __init__(self, data_dim, time_emb_dim):
        super().__init__()
        
        # Main data processing path
        self.linear1 = nn.Linear(data_dim, data_dim)
        self.linear2 = nn.Linear(data_dim, data_dim)
        self.norm1 = nn.LayerNorm(data_dim)
        self.norm2 = nn.LayerNorm(data_dim)
        self.activation = nn.SiLU() # SiLU (Swish) is common in modern models

        # A small projection for the time embedding to match the data dimension
        # This allows each block to adapt the time embedding for its specific purpose
        self.time_mlp = nn.Linear(time_emb_dim, data_dim)

    def forward(self, x, t_emb):
        """
        x: The data input (e.g., from the previous layer)
        t_emb: The centrally computed time embedding
        """
        # Store original x for the residual connection
        x_residual = x
        
        # Project the time embedding and add it to the data
        # This is the core of the conditioning
        time_projection = self.time_mlp(self.activation(t_emb))
        
        # First part of the main path
        x = self.norm1(x)
        x = self.activation(x)
        x = self.linear1(x)
        
        # INJECT THE TIMESTEP
        x = x + time_projection
        
        # Second part of the main path
        x = self.norm2(x)
        x = self.activation(x)
        x = self.linear2(x)
        
        # Add the residual connection
        return x + x_residual


# --- Usage ---
# model = LargeConditionalModel(data_dim=128, time_emb_dim=128, num_blocks=4)

# x_data = torch.randn(32, 128) # Batch of 32, data dimension 128
# timesteps = torch.randint(0, 1000, (32,)) # A timestep for each item in the batch

# output = model(x_data, timesteps)
# print("Final output shape:", output.shape)
#%%

# class ObBlock(nn.Module):
#     def __init__(self, hidden_dim, time_dim, activation_fn="silu"):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.act_fn = get_activation(activation_fn)
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_dim, hidden_dim),
#             self.act_fn,
#         )
    
#     def forward(self, x, t_emb):
#         x_residual = x
#         time_projection = self.time_mlp(t_emb)
        
#         # First part of the main path
#         x = self.norm1(x)
#         x = self.act_fn(x)
#         x = self.linear1(x)
        
#         # INJECT THE TIMESTEP
#         x = x + time_projection
        
#         # Second part of the main path
#         x = self.norm2(x)
#         x = self.act_fn(x)
#         x = self.linear2(x)
        
#         # Add the residual connection
#         return x + x_residual
    
class ObModel(ModelMixin, ConfigMixin):
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
        
        # --- Initial and Final Projections for Data ---
        self.input_proj = nn.Linear(in_channels * sample_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_channels * sample_size)

        # --- Stack of Conditional Blocks ---
        # nn.ModuleList is the correct way to hold a list of sub-modules
        self.blocks = nn.ModuleList(
            [ConditionalBlock(data_dim=hidden_dim, time_emb_dim=time_emb_dim) for _ in range(num_layers)]
        )

    def forward(self, sample, timestep, return_dict = True):
        original_size = sample.size()
        # print(sample.size())
        if sample.dim() == 3:
            sample = sample.reshape(original_size[0], -1)
        # print(sample.size())
        t_emb = self.time_mlp(timestep.to(sample.device))
        
        # 2. Project the input data
        sample = self.input_proj(sample)
        
        # 3. Pass data and time embedding through all blocks
        for block in self.blocks:
            sample = block(sample, t_emb) # Pass the SAME time embedding to each block
            
        # 4. Final projection
        sample = self.output_proj(sample)
        sample = sample.reshape(original_size[0], 2, -1)
        if not return_dict:
            return (sample,)

        return ObModelOutput(sample=sample)

    



        