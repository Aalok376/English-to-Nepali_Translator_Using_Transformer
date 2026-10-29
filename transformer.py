import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,source_vocab_size,target_vocab_size,pad_id,d_model,
                 num_layers,num_heads,d_ff,max_length,dropout):
        super().__init__()

        self.pad_id=pad_id
        self.encoder=Encoder(source_vocab_size,d_model,num_layers,num_heads,d_ff,max_length,dropout)
        self.decoder=Decoder(target_vocab_size, d_model, num_layers, num_heads, d_ff, max_length, dropout)

        self.output_layer=nn.Linear(d_model,target_vocab_size)

        for parameter in self.parameters():
            if parameter.dim()>1:
                nn.init.xavier_uniform_(parameter)


    def make_source_mask(self,source_ids):
        return (source_ids!=self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_target_masks(self,target_ids):
        target_len=target_ids.size(1)

        padding_mask=(target_ids!=self.pad_id).unsqueeze(1).unsqueeze(2)

        casual_mask=causal_mask = torch.tril(torch.ones(target_len, target_len, device=target_ids.device)).bool()

        return padding_mask & casual_mask

    def encode(self, source_ids):
        source_mask = self.make_source_mask(source_ids)
        encoder_output = self.encoder(source_ids, source_mask)
        return encoder_output, source_mask
 
    def decode(self, target_ids, encoder_output, source_mask):
        target_mask = self.make_target_masks(target_ids)
        decoder_output = self.decoder(target_ids, encoder_output, source_mask, target_mask)
        return self.output_layer(decoder_output) 

    def forward(self, source_ids, target_ids):
        encoder_output, source_mask = self.encode(source_ids)
        return self.decode(target_ids, encoder_output, source_mask)