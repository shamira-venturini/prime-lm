# scripts/analyze_scores.py
# This FAST script reads the raw data and calculates s-PE and w-PE on demand.

import argparse
import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from transformers import GPT2LMHeadModel

DEVICE = "cpu"  # Analysis is fast, no GPU needed


def calculate_log_probs(hidden_state, token_ids, start_idx):
    """ A standalone function to calculate log probs from raw materials. """
    with torch.no_grad():
        logits = model.get_output_embeddings()(hidden_state.to(DEVICE))

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_token_ids = token_ids[start_idx:]

    token_log_probs = log_probs[range(len(target_token_ids)), target_token_ids]

    return token_log_probs.cpu().numpy()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Analyze raw data to calculate s-PE and w-PE.")
    arg_parser.add_argument("--model", type=str, required=True,
                            help="Model name (e.g., gpt2) to load the decoder head.")
    arg_parser.add_argument("--raw_data_dir", type=str, required=True,
                            help="Directory containing the raw .pt files from generate_data.py")
    arg_parser.add_argument("--output_dir", type=str, required=True,
                            help="Directory to save the final scores and analysis.")
    arg_parser.add_argument("--corpus_name", type=str, default=None, help="Optional: Analyze only a specific corpus.")
    arg_parser.add_argument("--calculate_spe", action="store_true", help="Calculate and save sentence-level PE (s-PE).")
    arg_parser.add_argument("--calculate_wpe", action="store_true", help="Calculate and save token-level PE (w-PE).")
    args = arg_parser.parse_args()

    if not args.calculate_spe and not args.calculate_wpe:
        print("Error: Please specify at least one calculation: --calculate_spe or --calculate_wpe")
        exit()

    print("Loading model head...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)

    if args.corpus_name:
        corpora_dirs = [os.path.join(args.raw_data_dir, args.corpus_name)]
    else:
        corpora_dirs = [d for d in glob(os.path.join(args.raw_data_dir, '*')) if os.path.isdir(d)]

    for corpus_dir in corpora_dirs:
        corpus_name = os.path.basename(corpus_dir)
        print(f"\n--- Analyzing Corpus: {corpus_name} ---")

        num_items = len(glob(os.path.join(corpus_dir, 'x_px', '*.pt')))
        if num_items == 0:
            print(f"Warning: No items found for corpus {corpus_name}. Skipping.")
            continue

        all_results = []

        for i in tqdm(range(num_items), desc="Calculating scores"):
            try:
                data_x_px = torch.load(os.path.join(corpus_dir, 'x_px', f'item_{i}.pt'))
                data_x_py = torch.load(os.path.join(corpus_dir, 'x_py', f'item_{i}.pt'))
            except FileNotFoundError:
                continue

            # Calculate per-token log probs for both conditions
            wpe_x_px = calculate_log_probs(data_x_px['activations']['layer_-1_hx'], data_x_px['token_ids'],
                                           data_x_px['target_start_idx'])
            wpe_x_py = calculate_log_probs(data_x_py['activations']['layer_-1_hx'], data_x_py['token_ids'],
                                           data_x_py['target_start_idx'])

            result = {'item_id': i}

            if args.calculate_spe:
                result['s_pe_po'] = wpe_x_px.sum() - wpe_x_py.sum()

            if args.calculate_wpe:
                min_len = min(len(wpe_x_px), len(wpe_x_py))
                result['w_pe_po'] = wpe_x_px[:min_len] - wpe_x_py[:min_len]
                result['target_tokens'] = data_x_px['tokens'][
                    data_x_px['target_start_idx']:data_x_px['target_start_idx'] + min_len]

            all_results.append(result)

        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(args.output_dir, f"{corpus_name}_analysis.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Analysis saved to {output_path}")