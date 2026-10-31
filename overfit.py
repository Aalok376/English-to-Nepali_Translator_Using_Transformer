"""
Sanity check for the model and training code.

Trains on a single batch of 32 pairs until the model memorizes it. If it can
reproduce the references exactly, the architecture and training loop work.
"""

from itertools import islice
import torch
import torch.nn as nn

from config import Config
from data import get_dataloaders, load_tokenizer
from transformer import Transformer
from translate import translate

config = Config()
tokenizer = load_tokenizer(config)

# one fixed batch (32 pairs), reused every epoch
batches = list(islice(get_dataloaders(config)["train"], 1))

model = Transformer(
    source_vocab_size=config.vocab_size,
    target_vocab_size=config.vocab_size,
    pad_id=config.pad_id,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.feed_forward_size,
    max_length=config.max_length,
    dropout=0.0,  # no dropout, we want the model to memorize
).to(config.device)

loss_function = nn.CrossEntropyLoss(ignore_index=config.pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(1, 401):
    total_loss = 0
    for source, decoder_input, labels in batches:
        source = source.to(config.device)
        decoder_input = decoder_input.to(config.device)
        labels = labels.to(config.device)

        logits = model(source, decoder_input)
        loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss {total_loss / len(batches):.4f}")

# now translate sentences the model has seen, it should reproduce the references
model.eval()
source, decoder_input, labels = batches[0]
for index in range(3):
    print()
    print("English  :", tokenizer.decode(source[index].tolist()))
    print("Reference:", tokenizer.decode(labels[index].tolist()))
    print("Model    :", translate(tokenizer.decode(source[index].tolist()), model, tokenizer, config))