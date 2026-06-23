import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        max_seq_length: int = 5000,
        p_dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        positional_encodings = torch.zeros(max_seq_length, d_model)
        positions = torch.arange(0, max_seq_length).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)
        
        self.dropout = nn.Dropout(p=p_dropout)
        
        self.register_buffer('positional_encodings', positional_encodings.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        """
        
        seq_length = x.size(1)

        out = x + self.positional_encodings[:, :seq_length, :].to(x.device)
        
        return self.dropout(out)
