"""
Quick check that the pretrained NLLB model loads from the local folder and
translates one sentence.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "nllb_model"   # local folder made by download_nllb.py
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "npi_Deva"

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

nllb_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SOURCE_LANGUAGE, tgt_lang=TARGET_LANGUAGE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
model.eval()

print("Model parameters:", model.num_parameters())
print("GPU memory used:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")

text = "I love machine learning."
inputs = nllb_tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE),
        max_length=50,
    )

print("English:", text)
print("Nepali:", nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)[0])