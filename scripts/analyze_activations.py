# scripts/analyze_activations.py
# This script implements the "Accumulative Direct Effect" method to find influential neurons.

import argparse
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cpu"  # Gradients can be memory-intensive, start on CPU


def calculate_direct_effect(activation_tensor, token_ids, start_idx):
    """
    Calculates the direct effect (activation * gradient) for each neuron.
    """
    # This is the crucial step: we tell PyTorch to track this tensor for gradient calculation.
    activation_tensor.requires_grad_(True)

    # Pass the activations through the final layer (LM head) to get word predictions (logits).
    logits = model.get_output_embeddings()(activation_tensor.to(DEVICE))

    # Convert logits to log probabilities.
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Isolate the token IDs of the target sentence.
    target_token_ids = token_ids[start_idx:]

    if len(target_token_ids) == 0:
        return np.zeros(activation_tensor.shape[1])  # Return zeros if no target tokens

    # Get the log probability of the actual tokens that occurred in the target sentence.
    # This is our "score" that we want to measure the influence on.
    target_log_probs = log_probs[range(len(target_token_ids)), target_token_ids]
    sentence_log_prob = target_log_probs.sum()

    # The magic step: PyTorch calculates the gradient of our score with respect to the activations.
    sentence_log_prob.backward()

    # The "direct effect" is the activation multiplied by its gradient.
    # This measures how much each neuron's firing contributed to the final sentence probability.
    direct_effects = activation_tensor * activation_tensor.grad

    # We sum the effects across all tokens in the sentence to get a single score per neuron.
    neuron_effects = direct_effects.sum(dim=0)

    return neuron_effects.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find influential neurons using the Accumulative Direct Effect method.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt2-large).")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory of raw .pt files.")
    parser.add_argument("--corpus_name", type=str, required=True, help="Name of the specific corpus to analyze.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final neuron effect scores.")
    args = parser.parse_args()

    print(f"--- Starting Neuron Analysis for {args.corpus_name} ---")
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)

    corpus_dir = os.path.join(args.raw_data_dir, args.corpus_name)
    num_items = len(glob(os.path.join(corpus_dir, 'x_px', '*.pt')))

    # Initialize a running total for the effects of each neuron.
    # For gpt2-large, hidden_size is 1280.
    hidden_size = model.config.n_embd
    total_effect_px = np.zeros(hidden_size)
    total_effect_py = np.zeros(hidden_size)

    for i in tqdm(range(num_items), desc="Analyzing neuron effects"):
        try:
            # --- Congruent Condition (Active Target | Active Prime) ---
            data_px = torch.load(os.path.join(corpus_dir, 'x_px', f'item_{i}.pt'))
            activations_px = data_px['activations']['layer_-1_hx']
            effect_px = calculate_direct_effect(activations_px, data_px['token_ids'], data_px['target_start_idx'])
            total_effect_px += effect_px

            # --- Incongruent Condition (Active Target | Passive Prime) ---
            data_py = torch.load(os.path.join(corpus_dir, 'x_py', f'item_{i}.pt'))
            activations_py = data_py['activations']['layer_-1_hx']
            effect_py = calculate_direct_effect(activations_py, data_py['token_ids'], data_py['target_start_idx'])
            total_effect_py += effect_py

        except FileNotFoundError:
            continue

    # The "priming effect" at the neuron level is the difference in influence
    # between the congruent and incongruent conditions.
    priming_effect_per_neuron = total_effect_px - total_effect_py

    print("\n--- Analysis Complete ---")
    # Save the final array of scores to a file.
    np.save(args.output_file, priming_effect_per_neuron)
    print(f"Neuron priming effect scores saved to: {args.output_file}")

    # Find and print the top 10 most influential neurons for the priming effect.
    top_10_indices = np.argsort(priming_effect_per_neuron)[-10:][::-1]
    print(f"\nTop 10 most influential neuron indices for priming:")
    print(top_10_indices)
    print(f"Their effect scores:")
    print(priming_effect_per_neuron[top_10_indices])