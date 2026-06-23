import torch
import torch.nn as nn
import math
from typing import Optional

from src.attention import MultiHeadAttention
from src.positional_encoding import PositionalEncoding

class DecoderBlock(nn.Module):
    """
    Decoder block: Masked self-attention + Cross-attention + Feed-forward with residual connections.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_key: int = 64,
        d_value: int = 64,
        n_heads: int = 8,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.masked_multihead_attention = MultiHeadAttention(
            d_model=d_model, 
            d_key=d_key, 
            d_value=d_value, 
            n_heads=n_heads,
        )
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        
        self.cross_multihead_attention = MultiHeadAttention(
            d_model=d_model, 
            d_key=d_key, 
            d_value=d_value, 
            n_heads=n_heads,
        )
        
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)       
        )
        
        self.layer_norm_3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ):
        identity = x.clone()
        out = self.masked_multihead_attention(x=x, mask=self_attention_mask)
        out = self.dropout(out)
        out = self.layer_norm_1(out + identity)
        
        identity = out.clone()
        out = self.cross_multihead_attention(
            x=encoder_out, query=out, mask=cross_attention_mask
        )
        out = self.dropout(out)
        out = self.layer_norm_2(out + identity)
        
        identity = out.clone()
        out = self.feed_forward(out)
        out = self.dropout(out)
        out = self.layer_norm_3(out + identity)
        
        return out
        
        
class TransformerDecoder(nn.Module):
    """
    Transformer decoder stack with positional encoding.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        d_key: int = 64,
        d_value: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        p_dropout: float = 0.1,
        max_seq_length: int = 5000,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_seq_length=max_seq_length,
            p_dropout=p_dropout
        )
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    d_key=d_key,
                    d_value=d_value,
                    n_heads=n_heads,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ): 
        out = x * math.sqrt(self.d_model)
        out = self.positional_encoding(out)
        
        for layer in self.layers:
            out = layer(
                out,
                encoder_out,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
            )
        
        return out