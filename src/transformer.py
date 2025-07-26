import torch
import torch.nn as nn
from typing import Optional

from src.transformer_encoder import TransformerEncoder
from src.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    """Transformer, a model architecture relying entirely on an attention mechanism 
        to draw global dependencies between input and output.
    """
    
    def __init__(
        self,
        d_context: int,
        d_vocabulary: int,
        d_model: int = 512,
        d_ff: int = 2048,
        d_key: int = 64,
        d_value: int = 64,
        n_layers:int = 6,
        n_heads: int = 8,
        p_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            d_model=d_model,
            d_ff=d_ff,
            d_key=d_key,
            d_value=d_value,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=p_dropout,
        )
        
        self.decoder = TransformerDecoder(
            d_model=d_model,
            d_ff=d_ff,
            d_key=d_key,
            d_value=d_value,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=p_dropout,   
        )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(d_context * d_model, d_vocabulary)
        self.softmax = nn.Softmax(-1)
    
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(x, encoder_out, mask)
        
        out = self.flatten(decoder_out)
        out = self.linear(out)
        out = self.softmax(out)
        
        return out
