import torch
import torch.nn as nn
import math

from typing import Optional


class Attention(nn.Module):
    """Scaled Dot-Product Attention
    
    Output is computed as a weighted sum of the values, where each weight,
    associated with the value is computed using dot product of the query 
    with the corresponding key. 
    """
    
    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 64,
        d_value: int = 64,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_key
        self.d_v = d_value
        
        self.linear_key = nn.Linear(d_model, d_key, bias=False)
        self.linear_query = nn.Linear(d_model, d_key, bias=False)
        self.linear_value = nn.Linear(d_model, d_value, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, 
        x: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        K = self.linear_key(x)
        V = self.linear_value(x)
        Q = self.linear_query(x if query is None else query)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = self.softmax(attention_scores)
        
        out = torch.matmul(attention_weights, V)
        
        return out
        
        
class MultiHeadAttention(nn.Module):
    """Performs several attention computations in parallel.
    
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
            
        self.heads = nn.ModuleList(
            [
                Attention(
                    d_model=d_model,
                    d_key=d_key,
                    d_value=d_value,
                )
                for _ in range(n_heads)
            ]
        )
        
        self.linear = nn.Linear(d_value * n_heads, d_model, bias=False)
        

    def forward(
        self,
        x: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        out = []
        
        for head in self.heads:
            out.append(head(x, query, mask))
        
        out = torch.cat(out, dim=-1)
        
        out = self.linear(out)
        
        return out