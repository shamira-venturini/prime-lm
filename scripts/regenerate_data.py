# scripts/regenerate_data.py
# This script's only purpose is to re-create the 'new_corpora' folder.

import os
from init_corpora import init_corpora
from diagnnose.tokenizer.create import create_tokenizer

if __name__ == "__main__":
    print("--- Starting regeneration of 'new_corpora' folder ---")

    # We need to load a tokenizer, as init_corpora requires it.
    # We'll use gpt2-large, as that's our target model.
    model_name = "gpt2-large"
    tokenizer_config = {"path": model_name}
    tokenizer = create_tokenizer(**tokenizer_config)
    print(f"Tokenizer for '{model_name}' loaded.")

    # This is the path to the original, raw CSV files.
    raw_data_directory = "PrimeLM/corpora"

    print(f"Looking for raw data in: {raw_data_directory}")

    # This is the function call that does all the work.
    # It will read the raw files and write the processed files to 'new_corpora'.
    primed_corpora, _ = init_corpora(raw_data_directory, tokenizer)

    if primed_corpora:
        print(f"\n--- SUCCESS! ---")
        print(f"Successfully regenerated {len(primed_corpora)} processed files.")
        print("The 'new_corpora' folder has been created in your 'prime-lm' project directory.")
    else:
        print("\n--- ERROR! ---")
        print("Something went wrong. No files were generated.")