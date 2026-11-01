# English-to-Nepali Translator

Two English-to-Nepali translation models, built and compared on the same test sentences:

1. **A Transformer written from scratch** in PyTorch (encoder, decoder, attention, masks, training loop and beam search), trained on about 110k cleaned sentence pairs.
2. **A fine-tuned pretrained model**: Meta's NLLB-200 (distilled, 600M) adapted to the same data with LoRA, on a laptop RTX 3050 with 6 GB of memory.

## Results

All scores are on the same 500 held-out test sentences (the first 500 test pairs that fit the from-scratch model's 58-token limit).

| Model | BLEU greedy | BLEU beam 4 | chrF greedy | chrF beam 4 |
|---|---|---|---|---|
| Transformer from scratch (15M parameters) | 8.77 | 9.44 | 29.49 | 29.71 |
| NLLB pretrained, no fine-tuning | 8.99 | 9.23 | 41.20 | 41.71 |
| NLLB fine-tuned on 20k pairs (14 minutes) | 15.60 | 16.40 | 47.20 | 48.17 |
| **NLLB fine-tuned on all data (2 epochs, 2.2 hours)** | **17.02** | **18.16** | **48.81** | **49.77** |

What the numbers show:
- Fine-tuning roughly doubled BLEU compared with the pretrained model, and chrF is about 20 points above the from-scratch model.
- The first 20k training pairs gave most of the gain. Using all the data added about 1.8 BLEU and 1.6 chrF at beam 4.
- The pretrained model alone had good chrF but low BLEU. The dataset has its own style (numbered legal clauses, particular spellings), and fine-tuning teaches the model that style.
- Beam search (width 4) adds about 1 BLEU and 1 chrF over greedy decoding.

How to read the metrics:
- BLEU counts exact word matches, which is harsh for Nepali because of its many word endings. chrF compares characters and is more forgiving.
- A 500-sentence sample can move BLEU by about 1 point, so differences smaller than that mean little.
- This compares a 15M-parameter model trained from scratch with a 615M-parameter model pretrained on huge multilingual data. It shows the value of transfer learning, not that one architecture is better. I also can't rule out that NLLB's pretraining data overlapped with public sources like this dataset.
- Every translation is saved to CSV, so you can read them: `data/test_output_beam4.csv` (from scratch) and `data/nllb_lora_beam4.csv` (fine-tuned).

## Data

- Dataset: `CohleM/english-to-nepali` from Hugging Face (177,334 pairs).
- Cleaning: remove extra spaces, empty rows and duplicates; remove rows without English or Nepali letters; remove Nepali rows that contain English letters (usually broken font encoding); keep only sentences of 2 to 40 words with a length ratio under 3.
- After cleaning: 127,388 pairs, split 90/5/5 into 114,649 train, 6,369 validation and 6,370 test pairs.
- The from-scratch model also drops pairs longer than 58 tokens of its own tokenizer, which leaves 109,876 train, 6,102 validation and 6,087 test pairs. NLLB trains on the full 114,649 training pairs, cut at 64 of its own tokens.
- The from-scratch tokenizer is trained on the training split only.

## Model 1: Transformer from scratch

| Setting | Value |
|---|---|
| Layers (encoder and decoder) | 6 |
| d_model | 256 |
| Attention heads | 4 |
| Feed-forward size | 1024 |
| Dropout | 0.1 |
| Parameters | about 14.7 million |
| Tokenizer | SentencePiece BPE, shared vocabulary of 8000 |
| Max length | 60 tokens |

Training: Adam with a warmup schedule (1000 steps), label smoothing 0.1, gradient clipping at 1.0, batch size 128 and bfloat16 mixed precision. The embedding table is shared between the encoder, decoder and output layer. About 3.5 minutes per epoch on the RTX 3050. Training was stopped after epoch 30 of 40, because validation loss had almost stopped improving.

| Measure | Value |
|---|---|
| Best validation loss (with label smoothing 0.1) | 3.9965 |
| Validation loss without smoothing | 3.09 (perplexity about 22) |
| Validation loss with a shuffled English input | 6.21 |

The shuffled-input loss is about twice the normal loss, which shows the decoder really uses the English sentence.

What it can and cannot do (the training data is mostly software messages, legal text and Bible text):

| English | Model output | Comment |
|---|---|---|
| The file could not be opened. | फाइल खोल्न सकेन । | Correct |
| Could not connect to host . | होस्ट जडान गर्न सकेन । | Almost the same as the reference |
| The Government of Nepal may appoint a secretary. | (२) नेपाल सरकारले रजिष्ट्रारलाई नियुक्त गर्न सक्नेछ । | Right structure, wrong word (registrar instead of secretary), and an extra clause number |
| i eat rice | nonsense | Everyday sentences are almost absent from the training data |

Known limits:
- Long legal sentences start correctly and then drift into unrelated text.
- Everyday conversation fails.
- It likes to start outputs with a clause number like "(२)", because about half of the training data is numbered legal clauses.
- Input sentences should end with a period, because almost every training sentence does.

## Model 2: NLLB fine-tuned with LoRA

| Setting | Value |
|---|---|
| Base model | `facebook/nllb-200-distilled-600M` (about 615M parameters) |
| Method | LoRA adapters (rank 16, alpha 32, dropout 0.05) on the attention and feed-forward layers |
| Trainable parameters | 8.65 million (1.39% of the model) |
| Training data | 114,649 pairs, 2 epochs |
| Batch | 8 sentences x 16 accumulation steps = effective batch of 128 (same as the from-scratch model) |
| Learning rate | 2e-4, linear decay, 50 warmup updates |
| Precision | bfloat16 |
| Time | about 2.2 hours on the laptop RTX 3050 (6 GB) |

How it fits in 6 GB:
- The checkpoint stores the 256k-token embedding table four times. Tying the copies after loading cuts the model from 1.4 billion to 615 million parameters and frees about 1.5 GB.
- LoRA trains only about 1.4% of the weights, and gradient accumulation keeps the real batch small.
- The model is downloaded once into a local folder, so the scripts never contact the Hugging Face Hub while running.

## Files in the repository

- `data/`: the cleaned train, validation and test splits, the tokenizer, and every translation produced by both models (`test_output_beam*.csv` for the from-scratch model, `nllb_lora_beam*.csv` for the fine-tuned model).
- `nllb_lora/`: the final LoRA adapter (about 1.4% of the model's weights). It does not work alone: it needs the base model, which `download_nllb.py` downloads into `nllb_model/`. After that, set `ADAPTER_DIR = "nllb_lora"` in `evaluate_nllb.py` to load the base model and the adapter together.
- The adapter from the 20k-pair test run (`nllb_lora_short/`) is not included.

## Project files

From-scratch model:
- **config.py**: All settings in one place: model size, training settings, dataset settings, special token ids and folder paths.
- **data.py**: Downloads and cleans the dataset, splits it, trains the tokenizer and builds the batches.
- **layers.py**: Positional encoding, multi-head attention, feed-forward layer, encoder layer and decoder layer.
- **encoder.py**: Stacks the encoder layers. Reads the English sentence.
- **decoder.py**: Stacks the decoder layers. Writes the Nepali sentence one token at a time.
- **transformer.py**: Combines encoder, decoder and output layer, creates the masks, and shares the embedding table.
- **main.py**: Training script. Saves the best model and the latest checkpoint, and resumes automatically.
- **translate.py**: Translates sentences with beam search and a repetition filter.
- **evaluate.py**: Computes BLEU and chrF on the test set and saves every translation to CSV.
- **source_dependence_check.py**: Compares the loss with the correct English sentence against a shuffled one.
- **overfit.py**: Trains on one small batch to check that the model can memorize it.

NLLB fine-tuning:
- **download_nllb.py**: Downloads the NLLB model files once into `nllb_model/`.
- **load_nllb.py**: Quick check that the pretrained model loads and translates a sentence.
- **finetune_nllb.py**: Fine-tunes NLLB with LoRA and saves the adapter to `nllb_lora/`.
- **evaluate_nllb.py**: Scores NLLB (pretrained or with an adapter) on the same test sentences as `evaluate.py`.

## How to run

Install the libraries (install PyTorch with CUDA support first, from pytorch.org):

```
pip install pandas numpy datasets sentencepiece sacrebleu transformers peft accelerate
```

The data splits and tokenizer are already in `data/`. To rebuild them from scratch (downloads the dataset, cleans it, writes the CSV files and trains the tokenizer):

```
py data.py
```

**Model 1, from scratch**

```
py main.py          # train; Ctrl+C to stop, run again to resume
py translate.py     # translate your own sentences
py evaluate.py      # BLEU and chrF on 500 test sentences
py source_dependence_check.py
py overfit.py
```

**Model 2, NLLB with LoRA**

```
py download_nllb.py     # once, about 2.5 GB
py load_nllb.py         # optional quick check
py evaluate_nllb.py     # set ADAPTER_DIR = None in the file to score the pretrained model
py finetune_nllb.py     # about 2.2 hours on an RTX 3050
py evaluate_nllb.py     # set ADAPTER_DIR = "nllb_lora" to score the fine-tuned model
```

The base NLLB model (`nllb_model/`, 2.5 GB) and the from-scratch checkpoints (`checkpoints/`) are not in the repository. Run `download_nllb.py` for the NLLB base model, or `main.py` to train the from-scratch model again. With the adapter in `nllb_lora/`, you can skip fine-tuning and go straight to `evaluate_nllb.py` after downloading the base model.

## What I learned

- A first version with noisy data and a smaller batch size learned to write fluent Nepali legal text but ignored the English input. Cleaning the data, using a larger batch and training longer fixed this.
- The small overfitting test (memorizing 32 sentences) and the shuffled-source test are quick ways to tell a code bug from a data or training problem.
- With about 110k noisy pairs, the from-scratch model is limited by the data and not by its size or settings. More epochs did not help after epoch 25.
- Fine-tuning a pretrained model gave a much bigger improvement than any tuning of the small model, and the first 20k pairs gave most of it.
- On a 6 GB GPU, memory (not speed) was the main limit. Tying the duplicated embeddings, using LoRA and accumulating gradients made a 615M-parameter model trainable.
- BLEU and chrF can tell different stories. Looking at both, and reading the saved translations, is more reliable than either number alone.