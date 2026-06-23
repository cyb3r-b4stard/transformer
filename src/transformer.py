import torch
import torch.nn as nn
from typing import Optional

from src.mask_utils import generate_decoder_self_mask, generate_padding_mask
from src.transformer_encoder import TransformerEncoder
from src.transformer_decoder import TransformerDecoder

class Transformer(nn.Module):
    """
    Transformer, a model architecture relying entirely on the attention mechanism 
    to draw global dependencies between input and output.
    
    Based on "Attention is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(
        self,
        d_vocabulary: int,
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
        self.d_vocabulary = d_vocabulary
        
        self.embedding = nn.Embedding(d_vocabulary, d_model)
        
        self.encoder = TransformerEncoder(
            d_model=d_model,
            d_ff=d_ff,
            d_key=d_key,
            d_value=d_value,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=p_dropout,
            max_seq_length=max_seq_length,
        )
        
        self.decoder = TransformerDecoder(
            d_model=d_model,
            d_ff=d_ff,
            d_key=d_key,
            d_value=d_value,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=p_dropout,
            max_seq_length=max_seq_length,
        )
        
        self.output_projection = nn.Linear(d_model, d_vocabulary, bias=False)
        self.output_projection.weight = self.embedding.weight.t()
    
    
    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        pad_token_id: int = 0,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_self_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_mask is None:
            encoder_mask = generate_padding_mask(encoder_input, pad_token_id)
        if decoder_self_mask is None:
            decoder_self_mask = generate_decoder_self_mask(
                decoder_input, pad_token_id
            )
        if cross_attention_mask is None:
            cross_attention_mask = generate_padding_mask(
                encoder_input, pad_token_id
            )

        encoder_embeddings = self.embedding(encoder_input)
        decoder_embeddings = self.embedding(decoder_input)
        
        encoder_out = self.encoder(encoder_embeddings, mask=encoder_mask)
        
        decoder_out = self.decoder(
            x=decoder_embeddings,
            encoder_out=encoder_out,
            self_attention_mask=decoder_self_mask,
            cross_attention_mask=cross_attention_mask,
        )
        
        logits = self.output_projection(decoder_out)
        
        return logits
