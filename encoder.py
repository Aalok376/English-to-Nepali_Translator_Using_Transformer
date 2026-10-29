import math
import torch.nn as nn
from layers import positionalEncoding,EncoderLayer


class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,num_layers,num_heads,d_ff,max_length,dropout):
        super().__init__()

        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.positional_encoding=positionalEncoding(d_model,max_length,dropout)
        self.layers=nn.ModuleList(
            [
                EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)
            ]
        )

    def forward(self,source_ids,source_mask):
        x=self.embedding(source_ids)*math.sqrt(self.d_model)
        x=self.positional_encoding(x)

        for layer in self.layers:
            x=layer(x,source_mask)

        return x
    