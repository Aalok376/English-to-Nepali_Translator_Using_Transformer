"""
The decoder.

Stacks the decoder layers. It writes the Nepali sentence one token at a time,
looking at the Nepali written so far and at the encoder output.
"""

import math
import torch.nn as nn
from layers import positionalEncoding, DecoderLayer


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = positionalEncoding(d_model, max_length, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, target_ids, encoder_output, source_mask, target_mask):
        # (batch, target_len) -> (batch, target_len, d_model)
        x = self.embedding(target_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        return x