"""
Scores the NLLB model on the same test sentences that evaluate.py uses, with
greedy decoding and beam search. Set ADAPTER_DIR to score a fine-tuned model,
or None to score the original pretrained model.
"""

import pandas as pd
import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import Config
from data import load_tokenizer
from evaluate import load_test_pairs

MODEL_NAME = "nllb_model"   # local folder made by download_nllb.py
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "npi_Deva"
ADAPTER_DIR = "nllb_lora"   # "nllb_lora_short" for the short run, None for the pretrained model
BATCH_SIZE = 16
MAX_LENGTH = 128


def translate_all(sentences, model, nllb_tokenizer, num_beams):
    translations = []
    target_language_id = nllb_tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)

    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start:start + BATCH_SIZE]
        inputs = nllb_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_language_id,
                num_beams=num_beams,
                max_length=MAX_LENGTH,
            )
        translations += nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)

    return translations


def main():
    config = Config()

    # the same 500 test sentences that evaluate.py uses
    test_pairs = load_test_pairs(config, load_tokenizer(config))
    english = list(test_pairs["english"])
    references = list(test_pairs["nepali"])
    print(f"Evaluating {len(english)} test sentences")

    nllb_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANGUAGE, tgt_lang=TARGET_LANGUAGE)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")

    if ADAPTER_DIR:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        model = model.merge_and_unload()   # fold the adapter into the model for faster translation
    model.eval()

    run_name = ADAPTER_DIR if ADAPTER_DIR else "nllb_zero_shot"

    for num_beams in [1, 4]:
        outputs = translate_all(english, model, nllb_tokenizer, num_beams)

        bleu = sacrebleu.corpus_bleu(outputs, [references])
        chrf = sacrebleu.corpus_chrf(outputs, [references])
        print(f"{run_name} beam size {num_beams}: BLEU {bleu.score:.2f}, chrF {chrf.score:.2f}")

        results = pd.DataFrame({"english": english, "reference": references, "model": outputs})
        results.to_csv(f"{config.data_dir}/{run_name}_beam{num_beams}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()