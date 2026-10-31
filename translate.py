"""
Translation script.

Loads the best saved model and translates English sentences into Nepali using
beam search, with a filter that stops repeated phrases. Prints a few test
examples, then lets you type your own sentences.
"""

import pandas as pd
import torch

from config import Config
from data import load_tokenizer
from transformer import Transformer


def load_model(config):
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

    checkpoint = torch.load(f"{config.checkpoint_dir}/best_model.pt", map_location=config.device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    return model


def block_repeated_tokens(token_scores, target_ids):
    # if the last 3 tokens already appeared earlier, do not allow the token that followed them then
    # this stops real loops but still allows a word piece to repeat inside normal text
    if len(target_ids) < 4:
        return token_scores

    last_three = target_ids[-3:]
    for index in range(len(target_ids) - 3):
        if target_ids[index:index + 3] == last_three:
            token_scores[target_ids[index + 3]] = float("-inf")
    return token_scores


def add_final_period(sentence):
    # almost every training sentence ends with a period, so the model expects one
    sentence = sentence.strip()
    if sentence and sentence[-1] not in ".?!:;":
        sentence += "."
    return sentence


@torch.no_grad()
def translate(sentence, model, tokenizer, config, beam_size=4):
    source_ids = tokenizer.encode(sentence)[:config.max_length - 1] + [config.eos_id]
    source = torch.tensor([source_ids], device=config.device)

    # the encoder reads the English sentence only once
    encoder_output, source_mask = model.encode(source)

    beams = [(0.0, [config.bos_id])]   # each beam is (score, token ids so far)
    finished = []                      # beams that reached the end of the sentence

    for _ in range(config.max_length - 1):
        count = len(beams)
        targets = torch.tensor([ids for _, ids in beams], device=config.device)

        # all beams share the same English sentence
        logits = model.decode(
            targets,
            encoder_output.expand(count, -1, -1),
            source_mask.expand(count, -1, -1, -1),
        )
        log_probs = torch.log_softmax(logits[:, -1], dim=-1)

        # every beam proposes its best next tokens
        candidates = []
        for row, (score, ids) in enumerate(beams):
            token_scores = block_repeated_tokens(log_probs[row], ids)
            top_scores, top_tokens = token_scores.topk(beam_size)
            for token_score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                candidates.append((score + token_score, ids + [token]))

        # keep only the best candidates
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        beams = []
        for score, ids in candidates:
            if ids[-1] == config.eos_id:
                # divide by length so short sentences do not win just for being short
                finished.append((score / (len(ids) - 1), ids))
            else:
                beams.append((score, ids))
            if len(beams) == beam_size:
                break

        if len(finished) >= beam_size or not beams:
            break

    # if nothing reached the end, use the unfinished beams
    if not finished:
        finished = [(score / (len(ids) - 1), ids) for score, ids in beams]

    best_ids = max(finished, key=lambda item: item[0])[1]
    best_ids = [token for token in best_ids[1:] if token != config.eos_id]
    return tokenizer.decode(best_ids)


def main():
    config = Config()
    tokenizer = load_tokenizer(config)
    model = load_model(config)

    # first, a few sentences from the test set (the model never trained on these)
    test_pairs = pd.read_csv(f"{config.data_dir}/test.csv").head(5)
    for english, nepali in zip(test_pairs["english"], test_pairs["nepali"]):
        print("English  :", english)
        print("Reference:", nepali)
        print("Model    :", translate(english, model, tokenizer, config))
        print()

    # then your own sentences
    while True:
        sentence = input("English (or 'quit'): ").strip()
        if sentence.lower() == "quit":
            break
        print("Nepali:", translate(add_final_period(sentence), model, tokenizer, config))


if __name__ == "__main__":
    main()