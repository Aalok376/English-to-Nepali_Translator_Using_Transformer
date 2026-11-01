"""
Downloads the NLLB model files once into a local folder, so the other scripts
load from disk and never contact the Hugging Face Hub.
"""

import os

from huggingface_hub import snapshot_download

MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_FOLDER = "nllb_model"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_FOLDER,
    allow_patterns=["*.json", "*.model", "*.bin"],   # config, tokenizer and the weights file
)

print("Files in", MODEL_FOLDER)
for file_name in sorted(os.listdir(MODEL_FOLDER)):
    print(" ", file_name)