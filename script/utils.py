#!/usr/bin/env python3
"""
Utility functions for post-processing and analysis.
"""

import argparse
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_map


def merge_chains(results_dir: Path, output_name: str = "merged"):
    """
    Merge multiple MCMC chains from different random seeds.
    
    Args:
        results_dir: Directory containing chain results (e.g., results/run_rseed_13, results/run_rseed_14)
        output_name: Name for merged output
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Merging chains from {results_dir}")
    
    # Find all chain directories
    chain_dirs = [d for d in results_dir.parent.iterdir() 
                  if d.is_dir() and "rseed" in d.name and results_dir.name.split('_rseed_')[0] in d.name]
    
    if not chain_dirs:
        raise ValueError(f"No chain directories found in {results_dir.parent}")
    
    logger.info(f"Found {len(chain_dirs)} chains to merge")
    
    merged_samples = None
    merged_extra_fields = None
    
    for chain_dir in chain_dirs:
        # Find the results pickle file
        pkl_files = list(chain_dir.glob("*.pkl"))
        if not pkl_files:
            logger.warning(f"No pickle files found in {chain_dir}")
            continue
        
        # Load the first matching pickle file
        results_file = pkl_files[0]
        logger.info(f"Loading {results_file}")
        
        with open(results_file, 'rb') as f:
            data = pkl.load(f)
        
        samples = data.get('samples', {})
        extra_fields = data.get('extra_fields', {})
        
        # Expand dimensions and concatenate
        if merged_samples is None:
            merged_samples = tree_map(lambda x: jnp.expand_dims(x, axis=0), samples)
            merged_extra_fields = tree_map(lambda x: jnp.expand_dims(x, axis=0), extra_fields)
        else:
            merged_samples = tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                merged_samples, samples
            )
            merged_extra_fields = tree_map(
                lambda a, b: jnp.concatenate([a, b], axis=0),
                merged_extra_fields, extra_fields
            )
    
    # Save merged results
    output_dir = results_dir.parent / f"{output_name}"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "merged_results.pkl"
    with open(output_file, 'wb') as f:
        pkl.dump({
            'samples': merged_samples,
            'extra_fields': merged_extra_fields
        }, f)
    
    logger.info(f"Merged results saved to {output_file}")


def convert_jax_to_numpy(pkl_path: Path):
    """Convert JAX arrays to NumPy arrays in a pickle file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Converting {pkl_path} from JAX to NumPy")
    
    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)
    
    def convert(x):
        if isinstance(x, (jax.Array, jax.numpy.ndarray)):
            return np.array(x)
        return x
    
    converted = tree_map(convert, data)
    
    output_path = pkl_path.parent / f"{pkl_path.stem}_numpy.pkl"
    with open(output_path, 'wb') as f:
        pkl.dump(converted, f)
    
    logger.info(f"Converted file saved to {output_path}")


def summarize_results(results_dir: Path):
    """Print summary statistics from results."""
    logger = logging.getLogger(__name__)
    
    pkl_files = list(results_dir.glob("*.pkl"))
    if not pkl_files:
        logger.error(f"No pickle files found in {results_dir}")
        return
    
    results_file = pkl_files[0]
    logger.info(f"Loading {results_file}")
    
    with open(results_file, 'rb') as f:
        data = pkl.load(f)
    
    samples = data.get('samples', {})
    extra_fields = data.get('extra_fields', {})
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"\nSamples keys: {list(samples.keys())}")
    print(f"Extra fields keys: {list(extra_fields.keys())}")
    
    if 'z' in samples:
        z = samples['z']
        print(f"\nz shape: {z.shape}")
        print(f"z mean: {jnp.mean(z):.4f}")
        print(f"z std: {jnp.std(z):.4f}")
    
    if 'potential_energy' in extra_fields:
        pe = extra_fields['potential_energy']
        print(f"\nPotential energy shape: {pe.shape}")
        print(f"Potential energy mean: {jnp.mean(pe):.4f}")
        print(f"Potential energy std: {jnp.std(pe):.4f}")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Utility functions for post-processing')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple MCMC chains')
    merge_parser.add_argument('results_dir', type=Path, help='Results directory')
    merge_parser.add_argument('--output', type=str, default='merged', help='Output name')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert JAX to NumPy')
    convert_parser.add_argument('pkl_file', type=Path, help='Pickle file to convert')
    
    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize results')
    summarize_parser.add_argument('results_dir', type=Path, help='Results directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.command == 'merge':
        merge_chains(args.results_dir, args.output)
    elif args.command == 'convert':
        convert_jax_to_numpy(args.pkl_file)
    elif args.command == 'summarize':
        summarize_results(args.results_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()



