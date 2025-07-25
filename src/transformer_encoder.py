import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from typing import Optional

from src.attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_key: int = 64,
        d_value: int = 64,
        n_heads: int = 8,
    ):
        self.multihead_attention = MultiHeadAttention(
            d_model=d_model, 
            d_key=d_key, 
            d_value=d_value, 
            n_heads=n_heads,
        )
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)       
        )
        
        self.layer_norm_2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor
    ):
        identity = x.clone()
        
        out = self.multihead_attention(x)
        out = self.layer_norm_1(out + identity)
        
        identity = out.clone()
        
        out = self.feed_forward(out)
        out = self.layer_norm_2(out + identity)
        
        return out
        
        
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_key: int = 64,
        d_value: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    d_key=d_key,
                    d_value=d_value,
                    n_heads=n_heads,
                )
                for _ in range(n_layers)
            ]
        )
    
    def forward(
        self, 
        x: torch.Tensor,
    ): 
        out = x
        
        for layer in self.layers:
            out = layer(out)
        
        return out