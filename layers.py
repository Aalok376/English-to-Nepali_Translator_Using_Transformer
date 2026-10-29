import torch
import math
import torch.nn as nn

class positionalEncoding(nn.Module):
    def __init__(self, d_model,max_length,dropout):
        super().__init__()

        self.dropout=nn.Dropout(dropout)

        positions=torch.arange(max_length).unsqueeze(1)
        frequencies=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0) / d_model))

        encoding = torch.zeros(max_length, d_model)
        encoding[:,0::2]=torch.sin(positions*frequencies)
        encoding[:,1::2]=torch.cos(positions*frequencies)

        self.register_buffer("encoding",encoding.unsqueeze(0))

    def forward(self,embeddings):
        seq_length=embeddings.size(1)
        return self.dropout(embeddings+self.encoding[:,:seq_length])   #encoding[:,:seq_lenth,:]


class maskedMultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout):
        super().__init__()

        assert d_model%num_heads==0,"d_model must be divisible by num_heads"

        self.num_heads=num_heads
        self.head_dim=d_model//num_heads

        self.query_proj=nn.Linear(in_features=d_model,out_features=d_model)
        self.key_proj=nn.Linear(in_features=d_model,out_features=d_model)
        self.value_proj=nn.Linear(in_features=d_model,out_features=d_model)
        self.output_proj=nn.Linear(in_features=d_model,out_features=d_model)
        self.dropout=nn.Dropout(dropout)

    def split_heads(self,x):
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        batch_size,seq_len,_=x.shape
        x=x.view(batch_size,seq_len,self.num_heads,self.head_dim)
        return x.transpose(1,2)

    def forward(self,query,key,value,mask=None):
        batch_size,query_len,_=query.shape

        query=self.split_heads(self.query_proj(query))
        key=self.split_heads(self.key_proj(key))
        value=self.split_heads(self.value_proj(value))

        scores=query@key.transpose(-2,-1)/math.sqrt(self.head_dim)

        if mask is not None:
            scores=scores.masked_fill(mask==0,float("-inf"))#0=not allowed to look

        attention_weights=self.dropout(torch.softmax(scores,dim=1))
        context=attention_weights@value

        context=context.transpose(1,2).contiguous().view(batch_size,query_len,-1)
        return self.output_proj(context)

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super().__init__()

        self.expand=nn.Linear(d_model,d_ff)
        self.shrink=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.shrink(self.dropout(torch.relu(self.expand(x))))


class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()

        self.self_attention=maskedMultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attention=maskedMultiHeadAttention(d_model,num_heads,dropout)
        self.feed_forward=FeedForward(d_model,d_ff,dropout)
        self.self_attention_norm=nn.LayerNorm(d_model)
        self.cross_attention_norm=nn.LayerNorm(d_model)
        self.feed_forward_norm=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,source_mask):
        attention_out=self.self_attention(x,x,x,source_mask)
        x=self.self_attention_norm(x+self.dropout(attention_out))

        feed_forward_out=self.feed_forward(x)
        x=self.feed_forward_norm(x+self.dropout(feed_forward_out))

        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()

        self.self_attention = maskedMultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = maskedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.cross_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,encoder_output,source_mask,target_mask):

        # 1. masked self-attention: each target word only sees earlier target words
        self_attention_out = self.self_attention(x, x, x, target_mask)
        x = self.self_attention_norm(x + self.dropout(self_attention_out))
 
        # 2. cross-attention: queries from decoder, keys/values from encoder
        cross_attention_out = self.cross_attention(x, encoder_output, encoder_output, source_mask)
        x = self.cross_attention_norm(x + self.dropout(cross_attention_out))
 
        # 3. feed forward
        feed_forward_out = self.feed_forward(x)
        x = self.feed_forward_norm(x + self.dropout(feed_forward_out))
        return x

