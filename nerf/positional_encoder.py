import numpy as np
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_input: int, n_freqs: int, log_space: bool) -> None:
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]
        
        # Define frequencies in either linear or log space
        if self.log_space:
            freq_bands = 2. ** np.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
        
        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(freq * x))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create empty list to store embeddings
        embed = []
        
        # For each embedding function
        for fn in self.embed_fns:
            # Apply embedding function to input
            embed.append(fn(x))
            
        # Concatenate embeddings along last axis
        embed = torch.cat(embed, dim=-1)
        
        # Reshape from (..., d_input, 1 + 2 * n_freqs) to (..., d_output)
        embed = embed.reshape(*embed.shape[:-2], self.d_output)
        
        return embed
    
