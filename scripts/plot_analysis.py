# scripts/plot_analysis.py (Validates and Plots BOTH Active and Passive PE)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def parse_numpy_string(s):
    try:
        cleaned_s = s.strip().replace('[', '').replace(']', '').replace('\n', '')
        return np.fromstring(cleaned_s, sep=' ')
    except:
        return np.array([])


def analyze_and_plot(analysis_filepath):
    print(f"\n--- Analyzing File: {os.path.basename(analysis_filepath)} ---")
    try:
        df = pd.read_csv(analysis_filepath)
    except FileNotFoundError:
        print(f"ERROR: File not found.")
        return

    # --- VALIDATION ---
    if 's_pe_active' in df.columns:
        print(f"\n[VALIDATION] Mean s-PE for ACTIVE Target: {df['s_pe_active'].mean():.4f}")
    if 's_pe_passive' in df.columns:
        print(f"[VALIDATION] Mean s-PE for PASSIVE Target: {df['s_pe_passive'].mean():.4f}")

    # --- W-PE PLOTTING ---
    for target_type in ['active', 'passive']:
        wpe_col = f'w_pe_{target_type}'
        token_col = f'target_tokens_{target_type}'
        if wpe_col in df.columns:
            print(f"\n[ANALYSIS] Plotting w-PE for {target_type.upper()} target...")
            df[f'{wpe_col}_list'] = df[wpe_col].apply(parse_numpy_string)
            w_pe_scores = df[f'{wpe_col}_list'].tolist()
            if not any(s.any() for s in w_pe_scores): continue

            max_len = max(len(s) for s in w_pe_scores if s is not None)
            padded_scores = np.array(
                [np.pad(s, (0, max_len - len(s)), 'constant', constant_values=np.nan) for s in w_pe_scores])
            mean_w_pe_per_token = np.nanmean(padded_scores, axis=0)
            token_labels = [f"Token {i}" for i in range(len(mean_w_pe_per_token))]

            plt.figure(figsize=(12, 6))
            plt.bar(range(len(mean_w_pe_per_token)), mean_w_pe_per_token, color='skyblue')
            plt.axhline(0, color='grey', linestyle='--')
            plt.title(
                f'Token-Level Priming Effect (w-PE) for {target_type.upper()} Target\n({os.path.basename(analysis_filepath)})')
            plt.ylabel('Mean Priming Effect (Log-Probability Difference)')
            plt.xlabel('Token Position in Target Sentence')
            plt.xticks(range(len(mean_w_pe_per_token)), token_labels, rotation=45, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plot_filename = os.path.basename(analysis_filepath).replace('.csv', f'_wpe_plot_{target_type}.png')
            plot_save_path = os.path.join(os.path.dirname(analysis_filepath), plot_filename)
            plt.savefig(plot_save_path)
            print(f" -> Plot for {target_type.upper()} target saved to: {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot priming effect results.")
    parser.add_argument("analysis_file", type=str, help="Path to the _analysis.csv file to process.")
    args = parser.parse_args()
    analyze_and_plot(args.analysis_file)