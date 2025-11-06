# scripts/generate_data.py
# This script's ONLY job is to run the model and save the raw materials:
# activations, tokens, token_ids, and the target start index.

import argparse
import gc
import os
import time
import torch
from tqdm import tqdm

from diagnnose.models import import_model
from diagnnose.tokenizer.create import create_tokenizer
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from init_corpora import init_corpora

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_raw_materials(activations, item, item_idx, save_dir, corpus_name, column_name):
    """ Saves all necessary raw materials for a single sentence to a .pt file. """
    if isinstance(activations, torch.Tensor):
        activations = {(-1, 'hx'): activations}

    tokens = getattr(item, column_name)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Get the start index of the target sentence from the corpus item
    start_idx = getattr(item, COLUMNS[column_name])

    data_to_save = {
        'activations': {f"layer_{name[0]}_{name[1]}": tensor.detach().cpu() for name, tensor in activations.items()},
        'tokens': tokens,
        'token_ids': token_ids,
        'target_start_idx': start_idx
    }

    save_path = os.path.join(save_dir, corpus_name, column_name)
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f"item_{item_idx}.pt")
    torch.save(data_to_save, filename)


def create_activation_reader(corpus, sen_column):
    """ Configures and runs the diagnnose extractor. """
    if model.is_causal:
        def selection_func(w_idx, item):
            sen_len = len(getattr(item, sen_column))
            start_idx = getattr(item, COLUMNS[sen_column])
            return (start_idx - 1) <= w_idx <= (sen_len - 2)
    else:
        def selection_func(w_idx, item):
            sen_len = len(getattr(item, sen_column))
            start_idx = getattr(item, COLUMNS[sen_column])
            return start_idx <= w_idx <= (sen_len - 1)

    corpus.sen_column = sen_column
    extractor = Extractor(model, corpus, selection_func=selection_func, **config_dict["extract"])
    activation_reader = extractor.extract()
    return activation_reader


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser(description="Generate raw activations and input_ids from a model.")
    arg_parser.add_argument("--model", type=str, required=True)
    arg_parser.add_argument("--data", type=str, required=True, help="Path to a SINGLE .csv corpus file.")
    arg_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the raw .pt files")
    args = arg_parser.parse_args()

    mode = "masked_lm" if "bert" in args.model else "causal_lm"
    model_name = args.model

    print(f"--- Starting Data Generation for {model_name} ---")

    config_dict = {
        "model": {"transformer_type": model_name, "mode": mode, "device": DEVICE},
        "tokenizer": {"path": model_name},
        "extract": {"batch_size": 16},
    }

    model = import_model(**config_dict["model"])
    tokenizer = create_tokenizer(**config_dict["tokenizer"])

    print("[INFO] Extracting FINAL layer HIDDEN STATE only.")
    activation_names = [(-1, "hx")]
    config_dict["extract"]["activation_names"] = activation_names

    primed_corpora, COLUMNS = init_corpora(args.data, tokenizer)

    for CORPUS, corpus_path in sorted(list(primed_corpora.items())):
        print(f"[INFO] Starting generation for: {CORPUS}")

        config_dict["corpus"] = {
            "path": corpus_path, "header_from_first_line": True, "sen_column": next(iter(COLUMNS)),
            "tokenize_columns": list(COLUMNS.keys()), "convert_numerical": True, "sep": ",",
        }
        corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])

        for column in tqdm(COLUMNS.keys(), desc=f"Columns for {CORPUS}"):
            activation_reader = create_activation_reader(corpus, column)

            for item_idx, (item, activations_for_item) in enumerate(
                    tqdm(zip(corpus, activation_reader[:]), desc=f"Processing {column}", total=len(corpus),
                         leave=False)):
                save_raw_materials(activations_for_item, item, item_idx, args.output_dir, CORPUS, column)

    end_time = time.time()
    print(f"--- DATA GENERATION FINISHED IN {end_time - start_time:.2f} SECONDS ---")