
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from config import Config
from data import get_dataloaders
from transformer import Transformer


def run_one_epoch(
    model,
    data_loader,
    loss_function,
    config,
    optimizer=None,
    scheduler=None,
):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    for source, decoder_input, labels in data_loader:
        source = source.to(config.device)
        decoder_input = decoder_input.to(config.device)
        labels = labels.to(config.device)

        with torch.set_grad_enabled(is_training):
            logits = model(source, decoder_input)

            if not torch.isfinite(logits).all():
                print("Non-finite logits detected!")
                print("Source shape:", source.shape)
                print("Decoder shape:", decoder_input.shape)
                print("Logits min/max:", logits.nan_to_num().min().item(),
                    logits.nan_to_num().max().item())
                raise RuntimeError("Model produced NaN or Inf logits")

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            if not torch.isfinite(loss):
                print("Non-finite loss detected!")
                print("Labels min/max:", labels.min().item(), labels.max().item())
                raise RuntimeError("Loss is NaN or Inf")

        if is_training:
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def main():
    config = Config()
    print("Device:", config.device)

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

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=config.pad_id,
        label_smoothing=config.label_smoothing,
    )

    # Transformer learning-rate schedule
    def learning_rate_scale(step):
        step = max(step, 1)
        return config.d_model ** -0.5 * min(
            step ** -0.5,
            step * config.warmup_steps ** -1.5,
        )

    optimizer = Adam(
        model.parameters(),
        lr=0.5,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    scheduler = LambdaLR(optimizer, learning_rate_scale)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = os.path.join(
        config.checkpoint_dir, "latest_checkpoint.pt"
    )
    best_model_path = os.path.join(
        config.checkpoint_dir, "best_model.pt"
    )

    start_epoch = 1
    best_validation_loss = float("inf")

    # Resume from the latest completed epoch if a checkpoint exists
    if os.path.exists(latest_checkpoint_path):
        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location=config.device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_validation_loss = checkpoint["best_validation_loss"]

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best validation loss: {best_validation_loss:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    try:
        for epoch in range(start_epoch, config.num_epochs + 1):
            train_loss = run_one_epoch(
                model,
                dataloaders["train"],
                loss_function,
                config,
                optimizer,
                scheduler,
            )

            validation_loss = run_one_epoch(
                model,
                dataloaders["validation"],
                loss_function,
                config,
            )

            print(
                f"Epoch {epoch}/{config.num_epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Validation loss: {validation_loss:.4f}"
            )

            # Save the best model separately
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                torch.save(
                    model.state_dict(),
                    best_model_path,
                )

                print("  Saved best model.")

            # Save complete training state after every completed epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }

            torch.save(checkpoint, latest_checkpoint_path)

            print("  Saved latest checkpoint.")

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. "
            "The last completed epoch's checkpoint is preserved."
        )
        print("Run this script again to resume training.")

    print("Training script finished.")


if __name__ == "__main__":
    main()
