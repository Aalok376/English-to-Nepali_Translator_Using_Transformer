# English-to-Nepali Translator (Transformer from Scratch)

A small English-to-Nepali translation model. The Transformer (encoder, decoder, attention, masks, training loop and decoding) is written from scratch in PyTorch. It is trained on about 110k cleaned sentence pairs.

This is a mini project. A second version using transfer learning is planned, and its results will be added here.

## Results

Trained on a laptop RTX 3050 (6 GB), about 3.5 minutes per epoch. Training was stopped after epoch 30 of 40, because validation loss had almost stopped improving.

| Measure | Value |
|---|---|
| Best validation loss (with label smoothing 0.1) | 3.9965 |
| Validation loss without smoothing | 3.09 (perplexity about 22) |
| Validation loss with a shuffled English input | 6.21 |
| BLEU, greedy decoding (500 test sentences) | 8.77 |
| BLEU, beam search width 4 (500 test sentences) | 9.44 |
| chrF, greedy decoding | 29.49 |
| chrF, beam search width 4 | 29.71 |

- The test sentences are a random subset of the held-out test set, limited to sentences up to 58 tokens.
- The shuffled-input loss is much higher than the normal loss, which shows the decoder really uses the English sentence.
- BLEU is low in absolute terms for Nepali because it only counts exact word matches, and Nepali has many word endings. chrF compares characters and is more forgiving. These numbers are meant for comparing my own versions, not for comparing with other projects.

## What it can and cannot do

The training data is mostly software messages, legal text and Bible text, so the model works best on those.

| English | Model output | Comment |
|---|---|---|
| The file could not be opened. | फाइल खोल्न सकेन । | Correct |
| Could not connect to host . | होस्ट जडान गर्न सकेन । | Almost the same as the reference |
| The Government of Nepal may appoint a secretary. | (२) नेपाल सरकारले रजिष्ट्रारलाई नियुक्त गर्न सक्नेछ । | Right structure, wrong word (registrar instead of secretary), and an extra clause number |
| i eat rice | nonsense | Everyday sentences are almost absent from the training data |

Known limits:
- Long legal sentences start correctly and then drift into unrelated text.
- Everyday conversation fails.
- The model likes to start outputs with a clause number like "(२)", because about half of the training data is numbered legal clauses.
- Input sentences should end with a period, because almost every training sentence does.

## Model

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

Training details: Adam with a warmup schedule (1000 steps), label smoothing 0.1, gradient clipping at 1.0, batch size 128 and bfloat16 mixed precision. The embedding table is shared between the encoder, decoder and output layer.

## Data

- Dataset: `CohleM/english-to-nepali` from Hugging Face (177,334 pairs).
- Cleaning: remove extra spaces, empty rows and duplicates; remove rows without English or Nepali letters; remove Nepali rows that contain English letters (these are usually broken font encoding); keep only sentences of 2 to 40 words with a length ratio under 3.
- After cleaning: 127,388 pairs. After the 60-token length filter: 109,876 train, 6,102 validation and 6,087 test pairs.
- The tokenizer is trained on the training split only.

## Project files

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

## How to run

Install the libraries:

```
pip install torch pandas numpy datasets sentencepiece sacrebleu
```

1. Prepare the data and tokenizer (run once):
   ```
   py data.py
   ```
   This downloads the dataset, cleans it, writes the CSV files to `data/` and trains the tokenizer.
2. Train:
   ```
   py main.py
   ```
   Checkpoints go to `checkpoints/`. Press Ctrl+C to stop, and run the same command again to resume.
3. Translate:
   ```
   py translate.py
   ```
4. Evaluate on the test set:
   ```
   py evaluate.py
   ```
5. Optional checks:
   ```
   py source_dependence_check.py
   py overfit.py
   ```

## What I learned

- A first version with noisy data and a smaller batch size learned to write fluent Nepali legal text but ignored the English input. Cleaning the data, using a larger batch and training longer fixed this.
- The small overfitting test (memorizing 32 sentences) and the shuffled-source test are quick ways to tell a code bug from a data or training problem.
- With about 110k noisy pairs, the model is limited by the data and not by its size or settings. More epochs did not help after epoch 25.

## Next step

Transfer learning: fine-tuning a pretrained multilingual translation model on the same cleaned data, and scoring it on the same test sentences for a fair comparison with the numbers above.