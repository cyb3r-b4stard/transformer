import torch


def generate_causal_mask(
    seq_length: int, 
) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_length, seq_length))
    return mask.unsqueeze(0).unsqueeze(0)


def generate_padding_mask(
    tokens: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    mask = (tokens != pad_token_id).float()
    return mask.unsqueeze(1).unsqueeze(1)


def generate_decoder_self_mask(
    tokens: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    seq_length = tokens.size(1)
    causal_mask = generate_causal_mask(seq_length).to(tokens.device)
    padding_mask = generate_padding_mask(tokens, pad_token_id).to(tokens.device)
    
    return causal_mask * padding_mask
