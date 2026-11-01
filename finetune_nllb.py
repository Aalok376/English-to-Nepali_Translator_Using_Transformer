"""
Fine-tunes the pretrained NLLB model on the cleaned English-Nepali pairs using
LoRA adapters. Only a small number of extra weights are trained, so the model
fits in 6 GB of GPU memory.
"""

import time

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from config import Config

MODEL_NAME = "nllb_model"   # local folder made by download_nllb.py
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "npi_Deva"

NUM_TRAIN_PAIRS = None    # None uses all training pairs
NUM_EPOCHS = 2
BATCH_SIZE = 8            # if you get "CUDA out of memory", use 4 and set ACCUMULATION_STEPS to 32
ACCUMULATION_STEPS = 16   # 8 x 16 = effective batch of 128, same as the from-scratch model
MAX_LENGTH = 64
LEARNING_RATE = 2e-4
WARMUP_STEPS = 50         # counted in weight updates
SAVE_EVERY = 2000         # the adapter is saved every 2000 batches
ADAPTER_DIR = "nllb_lora"
DEVICE = "cuda"


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    config = Config()

    train_pairs = pd.read_csv(f"{config.data_dir}/train.csv")
    if NUM_TRAIN_PAIRS:
        train_pairs = train_pairs.head(NUM_TRAIN_PAIRS)
    pairs = list(zip(train_pairs["english"], train_pairs["nepali"]))
    print("Training pairs:", len(pairs))

    nllb_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, src_lang=SOURCE_LANGUAGE, tgt_lang=TARGET_LANGUAGE
    )

    def make_batch(batch):
        english, nepali = zip(*batch)
        inputs = nllb_tokenizer(
            list(english),
            text_target=list(nepali),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # padding must not count in the loss
        inputs["labels"][inputs["labels"] == nllb_tokenizer.pad_token_id] = -100
        return inputs

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(DEVICE)

    print("Parameters before tying:", count_parameters(model))
    model.tie_weights()
    print("Parameters after tying: ", count_parameters(model))
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_loader = DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=make_batch)
    total_batches = NUM_EPOCHS * len(data_loader)
    total_updates = total_batches // ACCUMULATION_STEPS

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_updates)

    model.train()
    step = 0
    running_loss = 0.0
    start_time = time.time()

    try:
        for epoch in range(NUM_EPOCHS):
            for batch in data_loader:
                batch = batch.to(DEVICE)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss

                # each small batch contributes 1/ACCUMULATION_STEPS of one weight update
                (loss / ACCUMULATION_STEPS).backward()

                step += 1
                running_loss += loss.item()

                if step % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 100 == 0:
                    seconds = time.time() - start_time
                    remaining_hours = seconds / step * (total_batches - step) / 3600
                    print(
                        f"batch {step}/{total_batches} | "
                        f"loss {running_loss / 100:.4f} | "
                        f"{seconds:.0f} seconds so far | "
                        f"about {remaining_hours:.1f} hours left"
                    )
                    running_loss = 0.0

                if step % SAVE_EVERY == 0:
                    model.save_pretrained(ADAPTER_DIR)

    except KeyboardInterrupt:
        print("Training interrupted.")

    model.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()