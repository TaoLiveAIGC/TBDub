import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    get_2d_sincos_pos_embed,
)
from typing import Any, Dict, Optional, Tuple, Union, List

class AudioProjLayer(nn.Module):
    def __init__(
        self,
        dim=1024,
        use_attn=False,
        attn_head_dim=64,
        attn_num_heads=16,
    ):
        super().__init__()

        assert dim == attn_head_dim * attn_num_heads, "dim must be equal to attn_head_dim * attn_num_heads"
        self.use_attn = use_attn
        self.attn_head_dim = attn_head_dim
        self.attn_num_heads = attn_num_heads

        self.norm1 = nn.LayerNorm(dim) if use_attn else None
        self.attn = Attention(
            query_dim=dim,
            heads=attn_num_heads,
            dim_head=attn_head_dim,

        ) if use_attn else None

        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(
            dim=dim,
        )

        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim=dim, max_seq_length=100)
        
    def forward(self, x):
        # x: b, f, w, c
        b, f, w, c = x.shape
        x = rearrange(x, 'b f w c -> (b f) w c')
        if self.use_attn:
            x = x + self.attn(self.pos_embed(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        x = rearrange(x, '(b f) w c -> b f w c', f=f)
        return x


class AudioProjModule(nn.Module):
    def __init__(
        self,
        audio_feat_layers=(4,),
        audio_feat_channels=(1024,),
        intermediate_dim=1536,
        output_dim=3072, 
        num_layers=4, 
        use_attn=True,
        attn_head_dim=64,
        attn_num_heads=24,
    ):
        super().__init__()

        assert intermediate_dim == attn_head_dim * attn_num_heads, "dim must be equal to attn_head_dim * attn_num_heads"
        self.audio_feat_layers = audio_feat_layers
        self.audio_feat_channels = audio_feat_channels
        self.input_dim = sum([l * c for l, c in zip(audio_feat_layers, audio_feat_channels)])
        
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.proj_in = torch.nn.Linear(self.input_dim, intermediate_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                AudioProjLayer(
                    dim=intermediate_dim,
                    use_attn=use_attn,
                    attn_head_dim=attn_head_dim,
                    attn_num_heads=attn_num_heads,
                )
            )
        self.proj_out = torch.nn.Linear(intermediate_dim, output_dim)


    def forward(self, audio_embeds):
        # audio_embeds: b, f, w, c
        b, f, w, c = audio_embeds.shape
        audio_embeds = self.proj_in(audio_embeds)
        for layer in self.layers:
            audio_embeds = layer(audio_embeds)
        audio_embeds = self.proj_out(audio_embeds)
        return audio_embeds
    
