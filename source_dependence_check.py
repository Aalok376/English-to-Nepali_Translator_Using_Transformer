"""
Checks that the model really uses the English input.

Computes the validation loss twice, once with the correct English sentence and
once with a shuffled one. A large gap between the two means the decoder relies
on the source sentence.
"""

import torch
import torch.nn as nn
from config import Config
from data import get_dataloaders
from translate import load_model

config = Config()
model = load_model(config)
validation_loader = get_dataloaders(config)["validation"]
loss_function = nn.CrossEntropyLoss(ignore_index=config.pad_id)

@torch.no_grad()
def average_loss(shuffle_source):
    total_loss = 0
    for source, decoder_input, labels in validation_loader:
        source = source.to(config.device)
        decoder_input = decoder_input.to(config.device)
        labels = labels.to(config.device)

        if shuffle_source:
            source = source[torch.randperm(source.size(0))]

        logits = model(source, decoder_input)
        total_loss += loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).item()
    return total_loss / len(validation_loader)

print("correct source :", average_loss(False))
print("shuffled source:", average_loss(True))