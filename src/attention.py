import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from typing import Optional

import math

class Attention(nn.Module):
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
        mask: Optional[torch.Tensor] = None
    ):
        Q = self.linear_query(x)
        K = self.linear_key(x)
        V = self.linear_value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = self.softmax(attention_scores)
        
        out = torch.matmul(attention_weights, V)
        
        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 64,
        d_value: int = 64,
        n_heads: int = 8,
    ):  
        self.heads = nn.ModuleList([
            Attention(
                d_model=d_model,
                d_key=d_key,
                d_value=d_value,
            )
            for _ in range(n_heads)
        ])
        
        self.linear = nn.Linear(d_value * n_heads, d_model, bias=False)
        

    def forward(
        self,
        x: torch.Tensor,
    ):
        outputs = []
        
        for head in self.heads:
            outputs.append(head(x))
        
        output_concatenated = torch.cat(outputs, dim=-1)
        
        out = self.linear(output_concatenated)
        
        return out