import torch
import math
import torch.nn as nn

class positionalEncoding(nn.Module):
    def __init__(self, d_model,max_length,dropout):
        super().__init__()

        self.dropout=nn.Dropout(dropout)

        positions=torch.arange(max_length).unsqueeze(1)
        frequencies=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0) / d_model))

        encoding=torch.zeros[max_length,d_model]
        encoding[:,0::2]=torch.sin(positions*frequencies)
        encoding[:,1::2]=torch.cos(positions*frequencies)

        self.register_buffer("encoding",encoding.unsqueeze(0))

    def forward(self,embeddings):
        seq_length=embeddings.size(1)
        return self.dropout(embeddings+self.encoding[:,:seq_length])   #encoding[:,:seq_lenth,:]