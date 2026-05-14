# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.insert(0, '/N/slate/tnn3/DucHGA/Foundation/Src_model')


from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

from Utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)

from Utils.parallel_patch_embed import ParallelVarPatchEmbed


class ClimaX(nn.Module):
    """Implements the ClimaX model as described in the paper,
    https://arxiv.org/abs/2301.10343

    Args:
        variable_names (list[str]): names of input variables.
        img_size (tuple[int, int]): image height and width.
        patch_size (int): patch side length.
        embed_dim (int): token embedding dimension.
        depth (int): number of transformer encoder blocks.
        decoder_depth (int): number of prediction head layers.
        num_heads (int): attention head count.
        mlp_ratio (float): MLP hidden dimension ratio.
        drop_path (float): stochastic depth rate.
        drop_rate (float): dropout probability.
    """

    def __init__(
        self,
        variable_names,
        img_size=(32, 64),
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variable_names = list(variable_names)

        # Separate patch embedding layer for each input variable.
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for _ in self.variable_names]
        )
        self.num_patches = self.token_embeds[0].num_patches

        # Variable-specific embeddings and index mapping.
        self.variable_embed, self.variable_index_map = self.create_variable_embeddings(embed_dim)

        # Cross-variable aggregation with a learnable query.
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Positional embedding and lead-time embedding.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # Transformer backbone.
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction head to map transformer output to patch values.
        head_layers = []
        for _ in range(decoder_depth):
            head_layers.append(nn.Linear(embed_dim, embed_dim))
            head_layers.append(nn.GELU())
        head_layers.append(nn.Linear(embed_dim, len(self.variable_names) * patch_size**2))
        self.head = nn.Sequential(*head_layers)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize positional embeddings, variable embeddings, and patch projection weights."""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = get_1d_sincos_pos_embed_from_grid(
            self.variable_embed.shape[-1], np.arange(len(self.variable_names))
        )
        self.variable_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

        for token_embed in self.token_embeds:
            trunc_normal_(token_embed.proj.weight, std=0.02)
            if token_embed.proj.bias is not None:
                nn.init.constant_(token_embed.proj.bias, 0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_variable_embeddings(self, embed_dim):
        """Create a learnable embedding for each input variable."""
        variable_embed = nn.Parameter(torch.zeros(1, len(self.variable_names), embed_dim), requires_grad=True)
        variable_index_map = {name: idx for idx, name in enumerate(self.variable_names)}
        return variable_embed, variable_index_map

    @lru_cache(maxsize=None)
    def get_variable_ids(self, variables, device):
        """Map variable names to integer ids for embedding lookup."""
        variables = tuple(variables)
        ids = [self.variable_index_map[name] for name in variables]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def select_variable_embeddings(self, embedding, variables):
        """Select embeddings for the requested variables."""
        ids = self.get_variable_ids(variables, embedding.device)
        return embedding[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """Convert patch tokens back into image fields."""
        p = self.patch_size
        num_vars = len(self.variable_names)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1], "Patch count does not match expected image grid"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, num_vars))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], num_vars, h * p, w * p))

    def aggregate_variable_tokens(self, x: torch.Tensor):
        """Aggregate variable-specific patch tokens into a single transformer sequence."""
        batch, num_vars, seq_len, dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * seq_len, num_vars, dim)

        query = self.var_query.expand(batch * seq_len, -1, -1)
        aggregated, _ = self.var_agg(query, x, x)
        aggregated = aggregated.squeeze(1)
        return aggregated.view(batch, seq_len, dim)

    def encode_input(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        """Encode input variables into transformer tokens with positional and lead-time conditioning."""
        if isinstance(variables, list):
            variables = tuple(variables)

        variable_ids = self.get_variable_ids(variables, x.device)

        embeds = []
        for idx, var_id in enumerate(variable_ids):
            embeds.append(self.token_embeds[int(var_id)](x[:, idx : idx + 1]))
        x = torch.stack(embeds, dim=1)

        variable_embed = self.select_variable_embeddings(self.variable_embed, variables)
        x = x + variable_embed.unsqueeze(2)

        x = self.aggregate_variable_tokens(x)
        x = x + self.pos_embed

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)).unsqueeze(1)
        x = x + lead_time_emb
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x, lead_times=None, variables=None, out_variables=None):
        """Forward pass through the model.

        Args:
            x: [B, Vi, H, W] input tensor.
            y: optional target tensor for compatibility.
            lead_times: [B] forecast lead times.
            variables: input variable names.
            out_variables: output variable names to select.
            metric: optional list of metric callables.
            lat: optional latitude information for metrics.

        Returns:
            tuple: (loss, preds) where loss is None if metric is None.
        """
        encoded = self.encode_input(x, lead_times, variables)
        preds = self.head(encoded)
        preds = self.unpatchify(preds)

        if out_variables is not None:
            output_ids = self.get_variable_ids(tuple(out_variables), preds.device)
            preds = preds[:, output_ids]

        return preds