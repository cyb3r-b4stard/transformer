import torch
import torch.nn as nn
import math

from typing import Optional
        
class MultiHeadAttention(nn.Module):
    """
    Performs several attention computations in parallel.
    
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions. With a single attention head, averaging inhibits this.
    """
    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 64,
        d_value: int = 64,
        n_heads: int = 8,
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_key
        self.d_v = d_value
        
        self.linear_query = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.linear_key = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.linear_value = nn.Linear(d_model, n_heads * d_value, bias=False)
        self.linear_out = nn.Linear(n_heads * d_value, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        batch_size = x.size(0)
        
        Q = self.linear_query(x if query is None else query)
        K = self.linear_key(x)
        V = self.linear_value(x)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        out = self.linear_out(out)
        
        return out
