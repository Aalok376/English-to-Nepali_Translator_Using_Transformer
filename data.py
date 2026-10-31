"""
Everything about the data.

Downloads the English-Nepali dataset, cleans it (extra spaces, duplicates,
broken text, bad length ratios), splits it into train, validation and test,
trains the SentencePiece tokenizer, and builds the batches the model reads.
"""

import os
import pandas as pd
from datasets import load_dataset
import numpy as np
import sentencepiece as spm
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Config

MIN_WORDS=2
MAX_WORDS=40  #since nepali words require more to represent
MAX_LENGTH_RATIO=3

def show_count(step_name,pairs):
    print(f"{step_name}:{len(pairs)} pairs")

def load_pairs(config):
    dataset=load_dataset(config.dataset_name)["train"]
    pairs=dataset.to_pandas()
    return pairs.rename(columns={"en":"english","ne":"nepali"})

def clean_pairs(pairs):
    show_count("Loaded:",pairs)

    #clean extra spaces
    for column in ["english","nepali"]:
        pairs[column]=pairs[column].str.replace(r"\s+", " ", regex=True).str.strip()

    #remove empty row and duplicates
    pairs=pairs[(pairs["english"]!="") & (pairs["nepali"]!="")]
    pairs=pairs.drop_duplicates(subset="english").drop_duplicates(subset="nepali")
    show_count("after removing duplicate rows and column",pairs)

    #now filter english and nepali atleast one english or nepali character must be there eleminates row containing only numbers or symbols
    #(pandas uses the RE2 regex engine here, which writes unicode as \x{0900} instead of \u0900)
    has_english=pairs["english"].str.contains(r"[A-Za-z]")
    has_nepali = pairs["nepali"].str.contains(r"[\x{0900}-\x{097F}]")
    pairs = pairs[has_english & has_nepali]

    #nepali text containing english letters is almost always broken font encoding
    pairs = pairs[~pairs["nepali"].str.contains(r"[A-Za-z]")]
    show_count("after script check", pairs)

    #check length ratio

    english_words=pairs["english"].str.split().str.len()
    nepali_words=pairs["nepali"].str.split().str.len()
    ratio=english_words/nepali_words

    good_length=(
        (english_words >= MIN_WORDS) & (nepali_words >= MIN_WORDS)
        & (english_words <= MAX_WORDS) & (nepali_words <= MAX_WORDS)
        & (ratio <= MAX_LENGTH_RATIO) & (ratio >= 1 / MAX_LENGTH_RATIO)
    )

    pairs=pairs[good_length]
    show_count("after lenth check",pairs)
    return pairs

def split_pairs(pairs,config):
    pairs=pairs.sample(frac=1,random_state=config.seed)
    pairs=pairs.iloc[:config.num_pairs]

    train_end = int(len(pairs) * config.train_ratio)
    validation_end = train_end + int(len(pairs) * config.validation_ratio)

    train = pairs.iloc[:train_end]
    validation = pairs.iloc[train_end:validation_end]
    test = pairs.iloc[validation_end:]
    return train, validation, test

def save_splits(train,validation,test,config):
    os.makedirs(config.data_dir,exist_ok=True)

    train.to_csv(f"{config.data_dir}/train.csv", index=False, encoding="utf-8")
    validation.to_csv(f"{config.data_dir}/validation.csv", index=False, encoding="utf-8")
    test.to_csv(f"{config.data_dir}/test.csv", index=False, encoding="utf-8")

    # the tokenizer learns from training sentences only
    with open(f"{config.data_dir}/all_sentences.txt", "w", encoding="utf-8") as file:
        for sentence in list(train["english"]) + list(train["nepali"]):
            file.write(sentence + "\n")

def prepare_data(config):
    pairs=load_pairs(config)
    pairs=clean_pairs(pairs)
    train,validation,test=split_pairs(pairs,config)
    save_splits(train,validation,test,config)

    print(f"train: {len(train)}, validation: {len(validation)}, test: {len(test)}")
    print(train.sample(10, random_state=config.seed).to_string())


def train_tokenizer(config):
    spm.SentencePieceTrainer.train(
        input=f"{config.data_dir}/all_sentences.txt",
        model_prefix=config.tokenizer_prefix,
        vocab_size=config.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=config.pad_id,
        unk_id=config.unk_id,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
    )

def load_tokenizer(config):
    return spm.SentencePieceProcessor(model_file=f"{config.tokenizer_prefix}.model")

def check_token_lengths(config, tokenizer):
    train = pd.read_csv(f"{config.data_dir}/train.csv")

    for column in ["english", "nepali"]:
        token_lengths = np.array([len(ids) for ids in tokenizer.encode(list(train[column]))])
        too_long = (token_lengths + 2 > config.max_length).mean() * 100
        print(f"{column}: average {token_lengths.mean():.1f} tokens, "
              f"95th percentile {np.percentile(token_lengths, 95):.0f}, "
              f"max {token_lengths.max()}, "
              f"{too_long:.2f}% too long")


class TranslationDataset(Dataset):
    def __init__(self,csv_path, tokenizer, config):
        pairs = pd.read_csv(csv_path)
        english_ids = tokenizer.encode(list(pairs["english"]))
        nepali_ids = tokenizer.encode(list(pairs["nepali"]))

        self.source_ids = []
        self.target_ids = []
        for english, nepali in zip(english_ids, nepali_ids):
            # drop pairs that are too long (+2 leaves room for bos and eos)
            if len(english) + 2 > config.max_length or len(nepali) + 2 > config.max_length:
                continue
            self.source_ids.append(english + [config.eos_id])
            self.target_ids.append([config.bos_id] + nepali + [config.eos_id])

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        return torch.tensor(self.source_ids[index]), torch.tensor(self.target_ids[index])

def make_batch(batch,pad_id):
    sources,targets=zip(*batch)
    source=pad_sequence(sources,batch_first=True,padding_value=pad_id)
    target=pad_sequence(targets,batch_first=True,padding_value=pad_id)

    decoder_input = target[:, :-1]   # bos, w1, w2, ...
    labels = target[:, 1:]           # w1, w2, ..., eos
    return source, decoder_input, labels

def get_dataloaders(config):
    tokenizer = load_tokenizer(config)
    collate = partial(make_batch, pad_id=config.pad_id)

    dataloaders = {}
    for name in ["train", "validation", "test"]:
        dataset = TranslationDataset(f"{config.data_dir}/{name}.csv", tokenizer, config)
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
        )
        print(f"{name}: {len(dataset)} pairs kept")
    return dataloaders

if __name__=="__main__":
    config=Config()
    # prepare_data(config)

    # train_tokenizer(config)

    tokenizer = load_tokenizer(config)
    # print(tokenizer.encode("The server is unavailable; try again later", out_type=str))
    # print(tokenizer.encode("सर्भर उपलब्ध छैन, पछि फेरि प्रयास गर्नुहोस्", out_type=str))
    # check_token_lengths(config, tokenizer)

    dataloaders = get_dataloaders(config)

    source, decoder_input, labels = next(iter(dataloaders["train"]))
    print("source:", source.shape)
    print("decoder_input:", decoder_input.shape)
    print("labels:", labels.shape)
    print(tokenizer.decode(source[0].tolist()))
    print(tokenizer.decode(labels[0].tolist()))