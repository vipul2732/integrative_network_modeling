#!/usr/bin/env python3
"""
Extract all protein pairs with posterior probabilities from MCMC results.

This script extracts ALL pairs (not just top N) with their posterior probabilities
from merged MCMC results files.

Usage:
    python extract_all_pairs.py \
        --results results/cullin_wt_config_multi_seeds/merged/merged_chains.pkl \
        --output results/cullin_wt_config_multi_seeds/all_pairs.tsv
    
    # For all cullin datasets:
    for dataset in wt vif mock; do
        python extract_all_pairs.py \
            --results results/cullin_${dataset}_config_multi_seeds/merged/merged_chains.pkl \
            --output results/cullin_${dataset}_config_multi_seeds/all_pairs.tsv
    done
"""

import argparse
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

import jax.numpy as jnp
import sys
sys.path.insert(0, str(Path(__file__).parent))
from model_variations import Z2A

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def sigmoid(x):
    """Sigmoid function to convert logits to probabilities."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def load_edge_probabilities_from_mcmc(
    results_file: Path,
    model_data_file: Optional[Path] = None
) -> Tuple[Dict[Tuple[str, str], float], Dict[int, str]]:
    """
    Load edge probabilities from MCMC results.
    
    Args:
        results_file: Path to merged MCMC results pickle file
        model_data_file: Optional path to model_data pickle file (for name mappings)
    
    Returns:
        edge_dict: Dictionary mapping (protein1, protein2) -> probability
        node_idx2name: Dictionary mapping node index -> protein name
    """
    logger.info(f"Loading MCMC results from {results_file}")
    
    # Load results
    with open(results_file, 'rb') as f:
        results = pkl.load(f)
    
    # Handle both 'samples' and 'merged_samples' keys
    if 'merged_samples' in results:
        samples = results['merged_samples']
        logger.info("Found 'merged_samples' key")
    elif 'samples' in results:
        samples = results['samples']
        logger.info("Found 'samples' key")
    else:
        raise ValueError("No 'samples' or 'merged_samples' found in results")
    
    if 'z' not in samples:
        raise ValueError("No 'z' samples found in results")
    
    z_samples = samples['z']
    
    # Convert z to numpy if JAX array
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    logger.info(f"z_samples shape: {z_samples.shape}")
    
    # Handle multiple chains: shape might be (n_chains, n_samples, M) or (n_samples, M)
    if len(z_samples.shape) == 3:
        # Multiple chains: flatten to (n_chains * n_samples, M)
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
        logger.info("Flattened multiple chains")
    elif len(z_samples.shape) == 2:
        # Single chain: (n_samples, M)
        pass
    else:
        # Single sample: (M,)
        z_samples = z_samples.reshape(1, -1)
    
    n_samples, M = z_samples.shape
    logger.info(f"Processing {n_samples} MCMC samples with {M} edges")
    
    # Calculate N from M: M = N*(N-1)/2, so N = (1 + sqrt(1 + 8*M)) / 2
    N = int((1 + np.sqrt(1 + 8 * M)) / 2)
    logger.info(f"Inferred N={N} proteins from M={M} edges")
    
    # Load model data for name mappings
    node_idx2name = {}
    edge_filter_map = None
    
    if model_data_file and Path(model_data_file).exists():
        logger.info(f"Loading model data from {model_data_file}")
        with open(model_data_file, 'rb') as f:
            model_data = pkl.load(f)
        node_idx2name = model_data.get('node_idx2name', {})
        edge_filter_map = model_data.get('edge_filter_map', None)
        logger.info(f"Loaded {len(node_idx2name)} protein name mappings")
    else:
        # Try to find model_data file in same directory
        model_data_pattern = str(results_file).replace('merged_chains.pkl', '') + '*_model_data.pkl'
        import glob
        matches = glob.glob(model_data_pattern)
        if matches:
            model_data_file = Path(matches[0])
            logger.info(f"Auto-detected model data file: {model_data_file}")
            with open(model_data_file, 'rb') as f:
                model_data = pkl.load(f)
            node_idx2name = model_data.get('node_idx2name', {})
            edge_filter_map = model_data.get('edge_filter_map', None)
            logger.info(f"Loaded {len(node_idx2name)} protein name mappings")
    
    # If no name mappings, create default names
    if not node_idx2name:
        logger.warning("No protein name mappings found. Using default names.")
        node_idx2name = {i: f"protein_{i}" for i in range(N)}
    
    # Convert z samples (logits) to probabilities using sigmoid
    logger.info("Converting z samples (logits) to probabilities...")
    prob_samples = sigmoid(z_samples)
    
    # Compute mean probability for each edge
    mean_probs = np.mean(prob_samples, axis=0)
    logger.info(f"Computed mean probabilities: min={mean_probs.min():.6f}, max={mean_probs.max():.6f}, mean={mean_probs.mean():.6f}")
    
    # Reconstruct full adjacency matrix if edge_filter_map exists
    if edge_filter_map is not None:
        logger.info(f"Reconstructing full adjacency matrix from filtered edges (edge_filter_map length: {len(edge_filter_map)})")
        # edge_filter_map maps filtered edge index -> original edge index
        full_M = N * (N - 1) // 2
        full_mean_probs = np.zeros(full_M)
        full_mean_probs[edge_filter_map] = mean_probs
        mean_probs = full_mean_probs
        logger.info(f"Reconstructed full edge list with {full_M} edges")
    
    # Create edge dictionary with all pairs (including zero probabilities)
    edge_dict = {}
    edge_idx = 0
    
    def remove_suffix(name: str) -> str:
        """Remove common suffixes like _HUMAN, _MOUSE, etc. from protein names."""
        suffixes = ['_HUMAN', '_MOUSE', '_RAT', '_CHICKEN', '_DROME', '_CAEEL', '_YEAST']
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)]
        return name
    
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[edge_idx])
            name_i = node_idx2name.get(i, f"protein_{i}")
            name_j = node_idx2name.get(j, f"protein_{j}")
            # Remove suffixes
            name_i = remove_suffix(name_i)
            name_j = remove_suffix(name_j)
            # Store in sorted order for consistency
            pair = tuple(sorted([name_i, name_j]))
            edge_dict[pair] = prob
            edge_idx += 1
    
    logger.info(f"Created edge dictionary with {len(edge_dict)} pairs")
    
    return edge_dict, node_idx2name


def save_all_pairs(edge_dict: Dict[Tuple[str, str], float], output_file: Path):
    """
    Save all pairs to TSV file, sorted by probability (descending).
    
    Args:
        edge_dict: Dictionary mapping (protein1, protein2) -> probability
        output_file: Path to output TSV file
    """
    logger.info(f"Saving all pairs to {output_file}")
    
    edges = []
    for (p1, p2), prob in edge_dict.items():
        edges.append({
            'Protein1': p1,
            'Protein2': p2,
            'Probability': prob
        })
    
    df = pd.DataFrame(edges)
    df = df.sort_values('Probability', ascending=False)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep='\t', index=False)
    
    logger.info(f"Saved {len(df)} pairs to {output_file}")
    logger.info(f"  Probability range: {df['Probability'].min():.6f} - {df['Probability'].max():.6f}")
    logger.info(f"  Mean probability: {df['Probability'].mean():.6f}")
    logger.info(f"  Pairs with prob > 0.5: {(df['Probability'] > 0.5).sum()}")
    logger.info(f"  Pairs with prob > 0.7: {(df['Probability'] > 0.7).sum()}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract all protein pairs with posterior probabilities from MCMC results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--results',
        type=Path,
        required=True,
        help='Path to merged MCMC results pickle file'
    )
    parser.add_argument(
        '--model-data',
        type=Path,
        help='Path to model_data pickle file (auto-detected if not provided)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output TSV file'
    )
    
    args = parser.parse_args()
    
    # Load edge probabilities
    edge_dict, node_idx2name = load_edge_probabilities_from_mcmc(
        args.results,
        args.model_data
    )
    
    # Save all pairs
    save_all_pairs(edge_dict, args.output)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
