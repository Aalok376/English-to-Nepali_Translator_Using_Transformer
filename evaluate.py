"""
Evaluation script.

Scores the model on the held-out test set with BLEU and chrF, comparing greedy
decoding with beam search. Saves every translation to a CSV file for reading.
"""

import pandas as pd
import sacrebleu

from config import Config
from data import load_tokenizer
from translate import load_model, translate

NUM_SENTENCES = 500   # use a bigger number for more of the test set (much slower)


def load_test_pairs(config, tokenizer):
    pairs = pd.read_csv(f"{config.data_dir}/test.csv")

    # keep only pairs that fit the model's length limit (+2 leaves room for bos and eos), same rule as training
    english_lengths = [len(ids) for ids in tokenizer.encode(list(pairs["english"]))]
    nepali_lengths = [len(ids) for ids in tokenizer.encode(list(pairs["nepali"]))]
    fits = [
        english + 2 <= config.max_length and nepali + 2 <= config.max_length
        for english, nepali in zip(english_lengths, nepali_lengths)
    ]
    return pairs[fits].head(NUM_SENTENCES)


def main():
    config = Config()
    tokenizer = load_tokenizer(config)
    model = load_model(config)

    test_pairs = load_test_pairs(config, tokenizer)
    english = list(test_pairs["english"])
    references = list(test_pairs["nepali"])
    print(f"Evaluating {len(english)} test sentences")

    for beam_size in [1, 4]:
        outputs = [translate(sentence, model, tokenizer, config, beam_size) for sentence in english]

        bleu = sacrebleu.corpus_bleu(outputs, [references])
        chrf = sacrebleu.corpus_chrf(outputs, [references])
        print(f"beam size {beam_size}: BLEU {bleu.score:.2f}, chrF {chrf.score:.2f}")

        results = pd.DataFrame({"english": english, "reference": references, "model": outputs})
        results.to_csv(f"{config.data_dir}/test_output_beam{beam_size}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()