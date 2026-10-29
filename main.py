import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data import get_dataloaders
from transformer import Transformer

def run_one_epoch(model,data_loader,loss_function,config,optimizer=None,scheduler=None):
    is_training=optimizer is not None
    model.train(is_training)
    total_loss=0

    for source,decoder_input,labels in data_loader:
        source=source.to(config.device)
        decoder_input=decoder_input.to(config.device)
        labels=labels.to(config.device)

        with torch.set_grad_enabled(is_training):
            logits=model(source,decoder_input)
            loss=loss_function(logits.reshape(-1,logits.size(-1)),labels.reshape(-1))

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        
        total_loss += loss.item()
 
    return total_loss / len(data_loader)


def main():
    config=Config()
    print("device:", config.device)
 
    dataloaders = get_dataloaders(config)

    model = Transformer(
        source_vocab_size=config.vocab_size,
        target_vocab_size=config.vocab_size,
        pad_id=config.pad_id,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.feed_forward_size,
        max_length=config.max_length,
        dropout=config.dropout,
    ).to(config.device)


    print("parameters:", sum(parameter.numel() for parameter in model.parameters()))


    loss_function = nn.CrossEntropyLoss(ignore_index=config.pad_id, label_smoothing=config.label_smoothing)
 
    # warmup, then slow decay (formula from the paper). lr=0.5 gives a peak learning rate of about 0.001
    def learning_rate_scale(step):
        step = max(step, 1)
        return config.d_model ** -0.5 * min(step ** -0.5, step * config.warmup_steps ** -1.5)
 
    optimizer = Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, learning_rate_scale)
 
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    best_validation_loss = float("inf")
 
    for epoch in range(1, config.num_epochs + 1):
        train_loss = run_one_epoch(model, dataloaders["train"], loss_function, config, optimizer, scheduler)

        validation_loss = run_one_epoch(model, dataloaders["validation"], loss_function, config)

        print(f"Epoch {epoch}: train loss {train_loss:.4f} | validation loss {validation_loss:.4f}")
 
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(model.state_dict(), f"{config.checkpoint_dir}/best_model.pt")
            print("  saved best model")



if __name__=="__main__":
    main()