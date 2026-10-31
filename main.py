"""
Training script.

Runs the training loop with a warmup learning-rate schedule, label smoothing,
gradient clipping and bfloat16 mixed precision. Checks the validation loss
after every epoch, saves the best model and the latest checkpoint, and
resumes automatically if a checkpoint exists.
"""

import os
import csv
import time

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
    total_grad_norm = 0.0
    batch_count = 0

    epoch_start = time.time()

    for source, decoder_input, labels in data_loader:

        source = source.to(config.device)
        decoder_input = decoder_input.to(config.device)
        labels = labels.to(config.device)

        with torch.set_grad_enabled(is_training):

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(config.device=="cuda"),
            ):
                logits=model(source,decoder_input)

            logits=logits.float()

            if not torch.isfinite(logits).all():
                raise RuntimeError("Model produced NaN or Inf logits.")

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Loss became NaN or Inf."
                )

        if is_training:

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            total_grad_norm += grad_norm.item()

            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        batch_count += 1

    average_loss = total_loss / batch_count

    average_grad_norm = (
        total_grad_norm / batch_count
        if is_training else 0.0
    )

    epoch_duration = time.time() - epoch_start

    return average_loss, average_grad_norm, epoch_duration


def save_history(history, history_path):

    if not history:
        return

    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "average_gradient_norm",
        "train_time_seconds",
        "validation_time_seconds",
    ]

    with open(history_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

def main():

    config = Config()

    print("=" * 60)
    print("ENGLISH-NEPALI TRANSFORMER TRAINING")
    print("=" * 60)

    print("Device:", config.device)
    print("Epochs:", config.num_epochs)
    print("Batch size:", config.batch_size)
    print("Learning rate warmup:", config.warmup_steps)

    dataloaders = get_dataloaders(config)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss_function = nn.CrossEntropyLoss(
        ignore_index=config.pad_id,
        label_smoothing=config.label_smoothing,
    )

    # --------------------------------------------------
    # Learning rate schedule
    # --------------------------------------------------

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

    scheduler = LambdaLR(
        optimizer,
        learning_rate_scale,
    )

    # --------------------------------------------------
    # Checkpoint paths
    # --------------------------------------------------

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = os.path.join(
        config.checkpoint_dir,
        "latest_checkpoint.pt",
    )

    best_model_path = os.path.join(
        config.checkpoint_dir,
        "best_model.pt",
    )

    history_path = os.path.join(
        config.checkpoint_dir,
        "training_history.csv",
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    history = []

    # --------------------------------------------------
    # Resume training if checkpoint exists
    # --------------------------------------------------

    if os.path.exists(latest_checkpoint_path):

        print("\nExisting checkpoint found.")

        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location=config.device,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_validation_loss = checkpoint[
            "best_validation_loss"
        ]

        history = checkpoint.get("history", [])

        print("Resuming from epoch:", start_epoch)
        print(
            "Previous best validation loss:",
            f"{best_validation_loss:.4f}",
        )

    else:

        print("\nNo checkpoint found.")
        print("Starting training from scratch.")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    try:

        for epoch in range(
            start_epoch,
            config.num_epochs + 1,
        ):

            print("\n" + "=" * 60)
            print(f"EPOCH {epoch}/{config.num_epochs}")
            print("=" * 60)

            # Training
            train_loss, average_grad_norm, train_time = run_one_epoch(
                model,
                dataloaders["train"],
                loss_function,
                config,
                optimizer,
                scheduler,
            )

            # Validation
            validation_loss, _, validation_time = run_one_epoch(
                model,
                dataloaders["validation"],
                loss_function,
                config,
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Train loss:          {train_loss:.4f}")
            print(f"Validation loss:     {validation_loss:.4f}")
            print(f"Learning rate:       {current_lr:.8f}")
            print(f"Average grad norm:   {average_grad_norm:.4f}")
            print(f"Training time:       {train_time:.2f} seconds")
            print(f"Validation time:    {validation_time:.2f} seconds")

            # --------------------------------------------------
            # Record history
            # --------------------------------------------------

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr,
                "average_gradient_norm": average_grad_norm,
                "train_time_seconds": train_time,
                "validation_time_seconds": validation_time,
            }

            history.append(epoch_record)

            save_history(history, history_path)

            # --------------------------------------------------
            # Save best model
            # --------------------------------------------------

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                torch.save(
                    model.state_dict(),
                    best_model_path,
                )

                print("\nSaved new best model.")

            # --------------------------------------------------
            # Save complete checkpoint
            # --------------------------------------------------

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "history": history,
            }

            torch.save(
                checkpoint,
                latest_checkpoint_path,
            )

            print("Saved latest checkpoint.")

    except KeyboardInterrupt:

        print("\nTraining interrupted.")
        print("The last completed epoch is preserved.")
        print("Run the script again to resume.")

    print("\nTraining script finished.")

    print(
        "Best validation loss:",
        f"{best_validation_loss:.4f}",
    )


if __name__ == "__main__":
    main()
