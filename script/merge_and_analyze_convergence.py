#!/usr/bin/env python3
"""
Merge multiple MCMC chains from different seeds and compute convergence diagnostics.

This script:
1. Loads results from multiple seed directories (seed_13, seed_42, etc.)
2. Merges chains for convergence analysis
3. Computes R-hat (Gelman-Rubin statistic) and other diagnostics
4. Saves merged results and diagnostics

Usage:
    python merge_and_analyze_convergence.py /path/to/multi_seeds_directory
    python merge_and_analyze_convergence.py /path/to/multi_seeds_directory --burn-in 100
"""

import argparse
import json
import logging
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import pandas as pd

_MERGE_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_MERGE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_MERGE_SCRIPT_DIR))
from post_sampling_z_binary import effective_z_binary_threshold  # noqa: E402

# Try to import arviz for R-hat computation
try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    warnings.warn("arviz not available. R-hat computation will use basic implementation.")

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def find_seed_directories(multi_seeds_dir: Path) -> List[Tuple[int, Path]]:
    """
    Find all seed_* directories in the multi_seeds directory.
    
    Returns:
        List of (seed, directory_path) tuples, sorted by seed
    """
    seed_dirs = []
    for item in multi_seeds_dir.iterdir():
        if item.is_dir() and item.name.startswith('seed_'):
            try:
                seed = int(item.name.split('_')[1])
                seed_dirs.append((seed, item))
            except (ValueError, IndexError):
                continue
    
    seed_dirs.sort(key=lambda x: x[0])
    return seed_dirs


def find_results_file(seed_dir: Path) -> Optional[Path]:
    """
    Find the main results pickle file in a seed directory.
    
    Looks for files matching pattern: *_model23_n_*.pkl (excluding warmup files)
    """
    # Pattern: 0_model23_n_13.pkl
    pkl_files = [f for f in seed_dir.glob("*.pkl") 
                 if "model23_n" in f.name and "warmup" not in f.name and "model_data" not in f.name]
    
    if not pkl_files:
        return None
    
    # Return the first matching file (should be only one)
    return pkl_files[0]


def find_warmup_file(seed_dir: Path) -> Optional[Path]:
    """
    Find the warmup results pickle file in a seed directory.
    
    Looks for files matching pattern: *_warmup*.pkl
    """
    pkl_files = [f for f in seed_dir.glob("*.pkl") 
                 if "warmup" in f.name and "model_data" not in f.name]
    
    if not pkl_files:
        return None
    
    # Return the first matching file (should be only one)
    return pkl_files[0]


def load_chain(seed_dir: Path, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Load MCMC chain from a seed directory, including warmup if available.
    
    Returns:
        Dictionary with 'samples', 'extra_fields', 'warmup_samples', 'warmup_extra_fields', 'seed', or None if failed
    """
    results_file = find_results_file(seed_dir)
    if results_file is None:
        logger.warning(f"No results file found in {seed_dir}")
        return None
    
    try:
        logger.info(f"Loading {results_file}")
        with open(results_file, 'rb') as f:
            data = pkl.load(f)
        
        samples = data.get('samples', {})
        extra_fields = data.get('extra_fields', {})
        
        # Try to load warmup data
        warmup_samples = {}
        warmup_extra_fields = {}
        warmup_file = find_warmup_file(seed_dir)
        if warmup_file:
            try:
                logger.info(f"Loading warmup data from {warmup_file}")
                with open(warmup_file, 'rb') as f:
                    warmup_data = pkl.load(f)
                warmup_samples = warmup_data.get('samples', {})
                warmup_extra_fields = warmup_data.get('extra_fields', {})
            except Exception as e:
                logger.warning(f"Could not load warmup data: {e}")
        
        # Convert JAX arrays to numpy for easier handling
        def to_numpy(x):
            if isinstance(x, (jnp.ndarray, jax.Array)):
                return np.array(x)
            return x
        
        samples = tree_map(to_numpy, samples)
        extra_fields = tree_map(to_numpy, extra_fields)
        if warmup_samples:
            warmup_samples = tree_map(to_numpy, warmup_samples)
        if warmup_extra_fields:
            warmup_extra_fields = tree_map(to_numpy, warmup_extra_fields)
        
        return {
            'samples': samples,
            'extra_fields': extra_fields,
            'warmup_samples': warmup_samples if warmup_samples else None,
            'warmup_extra_fields': warmup_extra_fields if warmup_extra_fields else None,
            'seed': int(seed_dir.name.split('_')[1])
        }
    except Exception as e:
        logger.error(f"Error loading {results_file}: {e}")
        return None


def compute_rhat_basic(chains: List[np.ndarray]) -> np.ndarray:
    """
    Compute R-hat (Gelman-Rubin statistic) using basic formula.
    
    Args:
        chains: List of arrays, each of shape (n_samples, n_vars)
    
    Returns:
        R-hat values for each variable
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    n_vars = chains[0].shape[1] if len(chains[0].shape) > 1 else 1
    
    # Reshape if needed
    if n_vars == 1:
        chains = [c.reshape(-1, 1) for c in chains]
    
    # Stack chains: (n_chains, n_samples, n_vars)
    all_chains = np.stack(chains, axis=0)
    
    # Compute within-chain variance (W)
    chain_means = np.mean(all_chains, axis=1)  # (n_chains, n_vars)
    chain_vars = np.var(all_chains, axis=1, ddof=1)  # (n_chains, n_vars)
    W = np.mean(chain_vars, axis=0)  # (n_vars,)
    
    # Compute between-chain variance (B)
    overall_mean = np.mean(all_chains)  # Scalar
    chain_means_mean = np.mean(chain_means, axis=0)  # (n_vars,)
    B = (n_samples / (n_chains - 1)) * np.sum((chain_means - chain_means_mean)**2, axis=0)  # (n_vars,)
    
    # Compute pooled variance
    var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
    
    # Compute R-hat
    rhat = np.sqrt(var_hat / W)
    
    return rhat if n_vars > 1 else rhat[0]


def plot_trace(var_chains: List[np.ndarray], var_name: str, seeds: List[int], 
               output_dir: Path, logger: logging.Logger):
    """
    Create trace plots for each chain.
    
    Args:
        var_chains: List of arrays, each of shape (n_samples, n_vars)
        var_name: Name of the variable
        seeds: List of seed values for labeling
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if not HAS_MATPLOTLIB:
        return
    
    n_chains = len(var_chains)
    n_samples = var_chains[0].shape[0]
    n_vars = var_chains[0].shape[1] if len(var_chains[0].shape) > 1 else 1
    
    # For multi-dimensional variables, create a summary plot first
    if n_vars > 10:
        # Create summary plot showing statistics across all dimensions
        fig_summary, axes_summary = plt.subplots(2, 2, figsize=(14, 10))
        fig_summary.suptitle(f'Summary Statistics: {var_name} (n={n_vars} dimensions)', 
                           fontsize=14, fontweight='bold')
        
        # Stack all chains: (n_chains, n_samples, n_vars)
        all_chains = np.stack(var_chains, axis=0)
        
        # Plot 1: Mean across dimensions over iterations
        ax1 = axes_summary[0, 0]
        mean_per_iter = np.mean(all_chains, axis=2)  # (n_chains, n_samples)
        for chain_idx in range(n_chains):
            ax1.plot(mean_per_iter[chain_idx], alpha=0.7, label=f'Seed {seeds[chain_idx]}', linewidth=1)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Mean value (across all dimensions)')
        ax1.set_title('Mean Value Over Iterations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation across dimensions over iterations
        ax2 = axes_summary[0, 1]
        std_per_iter = np.std(all_chains, axis=2)  # (n_chains, n_samples)
        for chain_idx in range(n_chains):
            ax2.plot(std_per_iter[chain_idx], alpha=0.7, label=f'Seed {seeds[chain_idx]}', linewidth=1)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Std dev (across all dimensions)')
        ax2.set_title('Standard Deviation Over Iterations')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of final values across dimensions
        ax3 = axes_summary[1, 0]
        final_values = all_chains[:, -100:, :].mean(axis=1)  # Mean of last 100 samples, (n_chains, n_vars)
        for chain_idx in range(n_chains):
            ax3.hist(final_values[chain_idx], bins=50, alpha=0.5, label=f'Seed {seeds[chain_idx]}')
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Final Values (last 100 samples)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of active dimensions (non-zero) over iterations
        ax4 = axes_summary[1, 1]
        # For binary/edge variables, count non-zero
        active_per_iter = np.sum(all_chains > 0.5, axis=2)  # (n_chains, n_samples)
        for chain_idx in range(n_chains):
            ax4.plot(active_per_iter[chain_idx], alpha=0.7, label=f'Seed {seeds[chain_idx]}', linewidth=1)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Number of active dimensions')
        ax4.set_title('Number of Active Dimensions (value > 0.5)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_file = output_dir / f"trace_{var_name}_summary.png"
        plt.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved summary plot: {summary_file} (showing statistics across all {n_vars} dimensions)")
        
        # Also create detailed trace plots for a few representative dimensions
        plot_indices = np.linspace(0, n_vars - 1, min(10, n_vars), dtype=int)
        n_plots = len(plot_indices)
    else:
        plot_indices = np.arange(n_vars)
        n_plots = n_vars
    
    # Create detailed trace plots for selected dimensions
    if n_vars > 1:
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        fig.suptitle(f'Trace Plots: {var_name} (showing {n_plots} representative dimensions out of {n_vars})', 
                    fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
        
        for plot_idx, var_idx in enumerate(plot_indices):
            ax = axes[plot_idx]
            
            for chain_idx, chain in enumerate(var_chains):
                values = chain[:, var_idx]
                ax.plot(values, alpha=0.7, label=f'Seed {seeds[chain_idx]}', 
                       color=colors[chain_idx], linewidth=0.5)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(f'Value (dim {var_idx})')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"trace_{var_name}_detailed.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved detailed trace plot: {plot_file} (showing {n_plots} individual dimensions)")
    else:
        # Single dimension - just one plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle(f'Trace Plot: {var_name}', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_chains))
        for chain_idx, chain in enumerate(var_chains):
            ax.plot(chain, alpha=0.7, label=f'Seed {seeds[chain_idx]}', 
                   color=colors[chain_idx], linewidth=0.5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"trace_{var_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved trace plot: {plot_file}")


def plot_convergence_analysis(chains_for_plotting: Dict[str, List[np.ndarray]], 
                              diagnostics: Dict[str, Any], seeds: List[int],
                              output_dir: Path, logger: logging.Logger):
    """
    Create convergence analysis multiplot showing mean for r_score, s_score, z, and rhat mean.
    
    Args:
        chains_for_plotting: Dictionary mapping variable names to lists of chains
        diagnostics: Dictionary of diagnostics for each variable
        seeds: List of seed values for labeling
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Focus on key variables: r_score, s_score, z
    key_vars = ['r_score', 's_score', 'z']
    
    # Create figure with subplots: 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Convergence Analysis: Mean Values', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Plot mean for r_score, s_score, z
    plot_positions = [(0, 0), (0, 1), (1, 0)]  # r_score, s_score, z
    
    for plot_idx, var_name in enumerate(key_vars):
        if var_name not in chains_for_plotting:
            continue
        
        row, col = plot_positions[plot_idx]
        ax = axes[row, col]
        
        var_chains = chains_for_plotting[var_name]
        # Stack chains: (n_chains, n_samples, n_vars) or (n_chains, n_samples) for 1D
        all_chains = np.stack(var_chains, axis=0)
        
        # Handle different dimensionalities
        if len(all_chains.shape) == 3:
            # Multi-dimensional (e.g., z with multiple edges)
            # Compute mean across dimensions
            mean_per_iter = np.mean(all_chains, axis=2)  # (n_chains, n_samples)
        elif len(all_chains.shape) == 2:
            # 1D variable (e.g., r_score, s_score)
            # Use values directly
            mean_per_iter = all_chains  # (n_chains, n_samples)
        else:
            logger.warning(f"Unexpected shape for {var_name}: {all_chains.shape}")
            continue
        
        # Plot mean over iterations
        for chain_idx in range(len(seeds)):
            ax.plot(mean_per_iter[chain_idx], alpha=0.7, 
                   label=f'Seed {seeds[chain_idx]}', 
                   color=colors[chain_idx], linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Value')
        ax.set_title(f'{var_name} - Mean Over Iterations')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Plot R-hat mean (bottom right)
    ax_rhat = axes[1, 1]
    rhat_means = {}
    for var_name in key_vars:
        if var_name in diagnostics:
            rhat = diagnostics[var_name].get('rhat')
            if rhat is not None and not np.isnan(rhat).all():
                if isinstance(rhat, np.ndarray):
                    rhat_mean = np.mean(rhat)
                else:
                    rhat_mean = float(rhat)
                rhat_means[var_name] = rhat_mean
    
    if rhat_means:
        var_names = list(rhat_means.keys())
        mean_values = [rhat_means[var] for var in var_names]
        bars = ax_rhat.bar(var_names, mean_values, color='skyblue', alpha=0.7, edgecolor='black')
        ax_rhat.set_ylabel('Mean R-hat')
        ax_rhat.set_title('R-hat Mean')
        ax_rhat.grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for bar, val in zip(bars, mean_values):
            height = bar.get_height()
            ax_rhat.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax_rhat.text(0.5, 0.5, 'No R-hat data', ha='center', va='center', 
                    transform=ax_rhat.transAxes)
        ax_rhat.set_title('R-hat Mean')
    
    plt.tight_layout()
    plot_file = output_dir / "convergence_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved convergence analysis plot: {plot_file}")


def plot_total_score(chains_data: List[Dict[str, Any]], seeds: List[int], 
                    output_dir: Path, logger: logging.Logger):
    """
    Plot total score (potential_energy) vs sampling steps for all chains.
    
    Args:
        chains_data: List of chain data dictionaries (from load_chain)
        seeds: List of seed values for labeling
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Total Score (Potential Energy) vs Sampling Steps', 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    for chain_idx, chain_data in enumerate(chains_data):
        extra_fields = chain_data.get('extra_fields', {})
        potential_energy = extra_fields.get('potential_energy')
        
        if potential_energy is None:
            logger.warning(f"No potential_energy found for seed {seeds[chain_idx]}")
            continue
        
        # Convert to numpy if JAX array
        def to_numpy(x):
            if isinstance(x, (jnp.ndarray, jax.Array)):
                return np.array(x)
            return np.asarray(x)
        
        potential_energy = to_numpy(potential_energy)
        
        # If 2D, flatten or take first dimension
        if len(potential_energy.shape) > 1:
            potential_energy = potential_energy.flatten()
        
        # Create iteration indices
        iterations = np.arange(len(potential_energy))
        
        ax.plot(iterations, potential_energy, alpha=0.7, 
               label=f'Seed {seeds[chain_idx]}', 
               color=colors[chain_idx], linewidth=1)
    
    ax.set_xlabel('Sampling Step')
    ax.set_ylabel('Total Score (Potential Energy)')
    ax.set_title('Total Score Over Sampling Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / "total_score.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved total score plot: {plot_file}")


def plot_cost_vs_steps(chains_data: List[Dict[str, Any]], seeds: List[int], 
                      output_dir: Path, logger: logging.Logger, num_warmup: Optional[int] = None):
    """
    Plot cost vs steps for total score and individual score components.
    Includes warmup steps if available.
    
    Args:
        chains_data: List of chain data dictionaries (from load_chain)
        seeds: List of seed values for labeling
        output_dir: Directory to save plots
        logger: Logger instance
        num_warmup: Expected number of warmup steps (from config). If None, will try to infer from data.
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Helper function to convert to numpy
    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return np.asarray(x)
    
    # Score names to plot (deterministic variables from model)
    score_names = [
        'potential_energy',  # Total cost
        'r_score',          # Profile similarity score
        's_score',          # SAINT score
        'degree_powerlaw_score',  # Degree power-law score
        'n_edges_score',    # Edge count score
        'composite_connectivity_score'  # Composite connectivity score (if available)
    ]
    
    # Check which scores are available
    available_scores = {}
    for score_name in score_names:
        for chain_data in chains_data:
            extra_fields = chain_data.get('extra_fields', {})
            warmup_extra_fields = chain_data.get('warmup_extra_fields', {})
            if score_name in extra_fields or (warmup_extra_fields and score_name in warmup_extra_fields):
                available_scores[score_name] = True
                break
    
    if not available_scores:
        logger.warning("No score data found for plotting")
        return
    
    # Create figure with subplots: one for total, one for individual scores
    n_individual = len([s for s in score_names[1:] if s in available_scores])
    fig = plt.figure(figsize=(16, 4 + 2 * n_individual))
    gs = gridspec.GridSpec(1 + n_individual, 1, figure=fig, hspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # Plot 1: Total score (potential_energy) - SAMPLING STEPS ONLY
    if 'potential_energy' in available_scores:
        ax_total = fig.add_subplot(gs[0, 0])
        
        for chain_idx, chain_data in enumerate(chains_data):
            extra_fields = chain_data.get('extra_fields', {})
            
            # Only use sampling data (no warmup)
            sampling_energy = None
            if 'potential_energy' in extra_fields:
                sampling_energy = to_numpy(extra_fields['potential_energy'])
                if len(sampling_energy.shape) > 1:
                    sampling_energy = sampling_energy.flatten()
            
            if sampling_energy is not None:
                sampling_steps = len(sampling_energy)
                logger.info(f"  Chain {chain_idx + 1}: sampling={sampling_steps} steps")
                steps = np.arange(len(sampling_energy))
                ax_total.plot(steps, sampling_energy, alpha=0.8, linewidth=1.5,
                            color=colors[chain_idx], label=f'Chain {chain_idx + 1}')
        
        ax_total.set_xlabel('Step', fontweight='bold', fontsize=14)
        ax_total.set_ylabel('Total Cost (Potential Energy)', fontweight='bold', fontsize=14)
        ax_total.set_title('Total Cost vs Steps', fontweight='bold', fontsize=16)
        ax_total.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=5)
        # Set Y-axis to have more ticks (closer together)
        from matplotlib.ticker import MaxNLocator
        ax_total.yaxis.set_major_locator(MaxNLocator(nbins=15))  # More ticks for better visibility
        for label in ax_total.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax_total.get_yticklabels():
            label.set_fontweight('bold')
        ax_total.legend(fontsize=11, loc='best', prop={'weight': 'bold'})
        ax_total.grid(True, alpha=0.3)
        ax_total.spines['top'].set_visible(False)
        ax_total.spines['right'].set_visible(False)
        ax_total.spines['left'].set_linewidth(1.5)
        ax_total.spines['bottom'].set_linewidth(1.5)
    
    # Plot individual scores
    plot_idx = 1
    for score_name in score_names[1:]:  # Skip potential_energy (already plotted)
        if score_name not in available_scores:
            continue
        
        ax = fig.add_subplot(gs[plot_idx, 0])
        
        for chain_idx, chain_data in enumerate(chains_data):
            extra_fields = chain_data.get('extra_fields', {})
            
            # Only use sampling data (no warmup)
            sampling_score = None
            if score_name in extra_fields:
                sampling_score = to_numpy(extra_fields[score_name])
                if len(sampling_score.shape) > 1:
                    sampling_score = sampling_score.flatten()
            
            if sampling_score is not None:
                steps = np.arange(len(sampling_score))
                ax.plot(steps, sampling_score, alpha=0.8, linewidth=1.5,
                       color=colors[chain_idx], label=f'Chain {chain_idx + 1}')
        
        # Format score name for display
        display_name = score_name.replace('_', ' ').title()
        ax.set_xlabel('Step', fontweight='bold', fontsize=14)
        ax.set_ylabel(f'{display_name}', fontweight='bold', fontsize=14)
        ax.set_title(f'{display_name} vs Steps', fontweight='bold', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=5)
        # Set Y-axis to have more ticks (closer together)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=15))  # More ticks for better visibility
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.legend(fontsize=11, loc='best', prop={'weight': 'bold'})
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
        plot_idx += 1
    
    # Adjust layout to avoid tight_layout warning
    try:
        plt.tight_layout()
    except:
        pass  # If tight_layout fails, continue anyway
    
    plot_file = output_dir / "cost_vs_steps.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cost vs steps plot: {plot_file}")


def plot_rhat_over_steps(chains_data: List[Dict[str, Any]], seeds: List[int],
                         output_dir: Path, logger: logging.Logger, 
                         window_size: int = 100, num_warmup: Optional[int] = None):
    """
    Plot R-hat over steps to show convergence.
    Computes R-hat in sliding windows.
    
    Args:
        chains_data: List of chain data dictionaries
        seeds: List of seed values
        output_dir: Directory to save plots
        logger: Logger instance
        window_size: Size of sliding window for R-hat computation
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Helper function to convert to numpy
    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return np.asarray(x)
    
    # Key variables to compute R-hat for
    key_vars = ['z', 'r_score', 's_score']
    
    # Check which variables are available
    available_vars = {}
    for var_name in key_vars:
        for chain_data in chains_data:
            samples = chain_data.get('samples', {})
            warmup_samples = chain_data.get('warmup_samples', {})
            if var_name in samples or (warmup_samples and var_name in warmup_samples):
                available_vars[var_name] = True
                break
    
    if not available_vars:
        logger.warning("No variable data found for R-hat computation")
        return
    
    n_vars = len(available_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))
    if n_vars == 1:
        axes = [axes]
    
    fig.suptitle('R-hat Convergence Over Steps (sliding window)', 
                 fontsize=16, fontweight='bold')
    
    for plot_idx, var_name in enumerate(key_vars):
        if var_name not in available_vars:
            continue
        
        ax = axes[plot_idx]
        
        # Collect all chains for this variable
        var_chains = []
        for chain_data in chains_data:
            warmup_samples = chain_data.get('warmup_samples', {})
            samples = chain_data.get('samples', {})
            
            # Combine warmup and sampling
            warmup_data = None
            if warmup_samples and var_name in warmup_samples:
                warmup_data = to_numpy(warmup_samples[var_name])
            
            sampling_data = None
            if var_name in samples:
                sampling_data = to_numpy(samples[var_name])
            
            if warmup_data is not None and sampling_data is not None:
                # Combine: shape should be (n_samples, n_dims)
                if len(warmup_data.shape) == 1:
                    warmup_data = warmup_data.reshape(-1, 1)
                if len(sampling_data.shape) == 1:
                    sampling_data = sampling_data.reshape(-1, 1)
                combined = np.concatenate([warmup_data, sampling_data], axis=0)
                var_chains.append(combined)
            elif sampling_data is not None:
                if len(sampling_data.shape) == 1:
                    sampling_data = sampling_data.reshape(-1, 1)
                var_chains.append(sampling_data)
        
        if len(var_chains) < 2:
            logger.warning(f"Need at least 2 chains for R-hat, found {len(var_chains)} for {var_name}")
            continue
        
        # Compute R-hat in sliding windows
        n_chains = len(var_chains)
        min_length = min(len(chain) for chain in var_chains)
        
        if min_length < window_size:
            logger.warning(f"Chain length {min_length} < window_size {window_size} for {var_name}")
            continue
        
        # Compute mean across dimensions for R-hat
        mean_chains = [np.mean(chain, axis=1) if chain.shape[1] > 1 else chain.flatten() 
                      for chain in var_chains]
        
        rhat_values = []
        rhat_positions = []
        
        for start_idx in range(0, min_length - window_size + 1, max(1, window_size // 10)):
            end_idx = start_idx + window_size
            
            # Extract window from each chain
            window_chains = [chain[start_idx:end_idx] for chain in mean_chains]
            
            # Compute R-hat for this window
            try:
                rhat = compute_rhat_basic(window_chains)
                if isinstance(rhat, np.ndarray):
                    rhat = np.mean(rhat)
                rhat_values.append(rhat)
                rhat_positions.append(start_idx + window_size // 2)
            except Exception as e:
                logger.warning(f"Error computing R-hat at position {start_idx}: {e}")
                continue
        
        if rhat_values:
            ax.plot(rhat_positions, rhat_values, linewidth=2, color='#2E86AB', marker='o', markersize=3)
            ax.axhline(y=1.01, color='red', linestyle='--', linewidth=1.5, 
                      label='R-hat = 1.01 (convergence threshold)')
            ax.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, 
                      label='R-hat = 1.0 (perfect convergence)')
            
            # Find warmup end position (from first chain)
            warmup_samples = chains_data[0].get('warmup_samples', {})
            if warmup_samples and var_name in warmup_samples:
                actual_warmup_len = len(to_numpy(warmup_samples[var_name]))
                if num_warmup is not None and actual_warmup_len > num_warmup * 1.5:
                    warmup_len = num_warmup
                else:
                    warmup_len = actual_warmup_len
                ax.axvline(x=warmup_len, color='gray', linestyle=':', 
                          linewidth=1.5, alpha=0.7, label='Warmup end')
        
        ax.set_xlabel('Step', fontweight='bold', fontsize=14)
        ax.set_ylabel('R-hat', fontweight='bold', fontsize=14)
        ax.set_title(f'R-hat Convergence: {var_name}', fontweight='bold', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.legend(fontsize=11, loc='best', prop={'weight': 'bold'})
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_ylim([0.98, max(1.05, max(rhat_values) * 1.1) if rhat_values else 1.05])
    
    plt.tight_layout()
    plot_file = output_dir / "rhat_convergence.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved R-hat convergence plot: {plot_file}")


def _trace_series_from_z(z_samples: np.ndarray, mode: str) -> np.ndarray:
    """Per sampling step scalar summary over edges: binary M mean or mean sigmoid(z)."""
    if len(z_samples.shape) == 1:
        z_samples = z_samples.reshape(-1, 1)
    if mode == "binary_m":
        thr = float(effective_z_binary_threshold(None))
        return np.mean((z_samples > thr).astype(np.float64), axis=1)
    zc = np.clip(z_samples, -500.0, 500.0)
    z_probs = 1.0 / (1.0 + np.exp(-zc))
    return np.mean(z_probs, axis=1)


def _rhat_figure4_legend_label(var_name: str) -> str:
    """Sentence-style labels for Figure 4 R-hat panel (no internal code names like z, r_score)."""
    mapping = {
        "z": "Edge variable",
        "r_score": "AP-MS profile similarity score",
        "s_score": "SAINT-derived score",
    }
    if var_name in mapping:
        return mapping[var_name]
    parts = str(var_name).replace("_", " ").split()
    if not parts:
        return str(var_name)
    return parts[0].capitalize() + (
        " " + " ".join(p.lower() for p in parts[1:]) if len(parts) > 1 else ""
    )


def plot_figure4_exhaustive_sampling(
    chains_data: List[Dict[str, Any]],
    seeds: List[int],
    output_dir: Path,
    logger: logging.Logger,
    num_warmup: Optional[int] = None,
    trace_mode: str = "binary_m",
):
    """
    Create Figure 4: 4-panel convergence / mixing diagnostics (saved as figure4_exhaustive_sampling.png).

    Note: This is chain-level convergence (score, R-hat, traces, autocorrelation), not the
    IMP-style "sampling exhaustiveness" entropy metrics in compute_post_sampling_imp_metrics.py.

    Panel 1: Score vs Steps (all chains)
    Panel 2: R-hat Convergence (all variables in one plot)
    Panel 3: Trace — per-step mean over edges of M=1{{z>Z_thr}} (default) or mean σ(z)
    Panel 4: Autocorrelation of that same series (chain 1)

    Args:
        trace_mode: "binary_m" (aligned with binary network M) or "sigmoid_mean" (legacy).
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Helper function to convert to numpy
    def to_numpy(x):
        if isinstance(x, (jnp.ndarray, jax.Array)):
            return np.array(x)
        return np.asarray(x)
    
    if trace_mode not in ("binary_m", "sigmoid_mean"):
        trace_mode = "binary_m"

    _thr = float(effective_z_binary_threshold(None))
    trace_ylabel = (
        f"Mean M (frac. edges with z>{_thr:g})"
        if trace_mode == "binary_m"
        else "Mean σ(z) (mean edge prob.)"
    )
    trace_note = (
        f"Traces/autocorr: mean binary M=1{{z>{_thr:g}}} per step"
        if trace_mode == "binary_m"
        else "Traces/autocorr: mean sigmoid(z) per step"
    )

    # Create 2x2 subplot layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Convergence and mixing diagnostics (post-warmup sampling phase)\n"
        f"{trace_note}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    
    # ========================================================================
    # Panel 1: Score vs Steps
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for chain_idx, chain_data in enumerate(chains_data):
        extra_fields = chain_data.get('extra_fields', {})
        
        # Only use sampling data (no warmup)
        sampling_energy = None
        if 'potential_energy' in extra_fields:
            sampling_energy = to_numpy(extra_fields['potential_energy'])
            if len(sampling_energy.shape) > 1:
                sampling_energy = sampling_energy.flatten()
            # Negate to make it positive (cost = -log-posterior, so negate to get positive cost)
            # This makes the plot more intuitive: higher values = higher cost = worse fit
            sampling_energy = -sampling_energy
        
        if sampling_energy is not None:
            # Thin data: plot every 10th step to reduce crowding
            step_interval = 10
            thinned_indices = np.arange(0, len(sampling_energy), step_interval)
            thinned_steps = thinned_indices
            thinned_energy = sampling_energy[thinned_indices]
            ax1.plot(thinned_steps, thinned_energy, alpha=0.8, linewidth=1.5,
                    color=colors[chain_idx], label=f'Chain {chain_idx + 1}')
    
    ax1.set_xlabel('Step', fontweight='bold', fontsize=16)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=16)
    ax1.set_title('Score vs Steps', fontweight='bold', fontsize=18, pad=10)
    # Use scientific notation for Y-axis
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    ax1.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=5)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    # Make scientific notation offset text bold
    ax1.yaxis.get_offset_text().set_fontweight('bold')
    ax1.yaxis.get_offset_text().set_fontsize(16)
    ax1.legend(fontsize=16, loc='upper right', prop={'weight': 'bold', 'size': 16})
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # ========================================================================
    # Panel 2: R-hat Convergence (All Variables in One Plot)
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Key variables to compute R-hat for
    key_vars = ['z', 'r_score', 's_score']
    window_size = 100
    
    # Check which variables are available
    available_vars = {}
    for var_name in key_vars:
        for chain_data in chains_data:
            samples = chain_data.get('samples', {})
            if var_name in samples:
                available_vars[var_name] = True
                break
    
    # Collect all chains for R-hat computation
    var_chains_dict = {}
    for var_name in key_vars:
        if var_name not in available_vars:
            continue
        
        var_chains = []
        for chain_data in chains_data:
            samples = chain_data.get('samples', {})
            if var_name in samples:
                sampling_data = to_numpy(samples[var_name])
                if len(sampling_data.shape) == 1:
                    sampling_data = sampling_data.reshape(-1, 1)
                var_chains.append(sampling_data)
        
        if len(var_chains) >= 2:
            var_chains_dict[var_name] = var_chains
    
    # Plot R-hat for each variable
    plot_styles = ['-', '--', '-.']
    plot_colors = ['#2E86AB', '#E63946', '#F18F01']
    
    for plot_idx, var_name in enumerate(key_vars):
        if var_name not in var_chains_dict:
            continue
        
        var_chains = var_chains_dict[var_name]
        n_chains = len(var_chains)
        min_length = min(len(chain) for chain in var_chains)
        
        if min_length < window_size:
            continue
        
        # Compute mean across dimensions for R-hat
        mean_chains = [np.mean(chain, axis=1) if chain.shape[1] > 1 else chain.flatten() 
                      for chain in var_chains]
        
        rhat_values = []
        rhat_positions = []
        
        for start_idx in range(0, min_length - window_size + 1, max(1, window_size // 10)):
            end_idx = start_idx + window_size
            window_chains = [chain[start_idx:end_idx] for chain in mean_chains]
            
            try:
                rhat = compute_rhat_basic(window_chains)
                if isinstance(rhat, np.ndarray):
                    rhat = np.mean(rhat)
                rhat_values.append(rhat)
                rhat_positions.append(start_idx + window_size // 2)
            except Exception as e:
                continue
        
        if rhat_values:
            style = plot_styles[plot_idx % len(plot_styles)]
            color = plot_colors[plot_idx % len(plot_colors)]
            display_name = _rhat_figure4_legend_label(var_name)
            ax2.plot(rhat_positions, rhat_values, linewidth=2, linestyle=style,
                    color=color, marker='o', markersize=3, label=display_name)
    
    ax2.set_xlabel('Step', fontweight='bold', fontsize=16)
    ax2.set_ylabel('R-hat', fontweight='bold', fontsize=16)
    ax2.set_title('R-hat Convergence', fontweight='bold', fontsize=18, pad=10)
    ax2.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=5)
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    if len(ax2.get_legend_handles_labels()[0]) > 0:
        ax2.legend(fontsize=16, loc='upper right', prop={'weight': 'bold', 'size': 16})
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    ax2.set_ylim([0.98, 1.05])
    
    # ========================================================================
    # Panel 3: Trace Plots (Mixing)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    trace_var = 'z'
    if trace_var in available_vars:
        trace_data = []
        for chain_idx, chain_data in enumerate(chains_data):
            samples = chain_data.get('samples', {})
            if trace_var in samples:
                z_samples = to_numpy(samples[trace_var])
                mean_prob = _trace_series_from_z(z_samples, trace_mode)
                trace_data.append((chain_idx, mean_prob))
        
        # Plot traces - thin data: plot every 10th step to reduce crowding
        step_interval = 10
        for chain_idx, trace_values in trace_data:
            thinned_indices = np.arange(0, len(trace_values), step_interval)
            thinned_steps = thinned_indices
            thinned_values = trace_values[thinned_indices]
            ax3.plot(thinned_steps, thinned_values, alpha=0.7, linewidth=1,
                    color=colors[chain_idx], label=f'Chain {chain_idx + 1}')
    
    ax3.set_xlabel('Step', fontweight='bold', fontsize=16)
    ax3.set_ylabel(trace_ylabel, fontweight='bold', fontsize=16)
    ax3.set_title('Trace plot (mixing)', fontweight='bold', fontsize=18, pad=10)
    ax3.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=5)
    for label in ax3.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax3.get_yticklabels():
        label.set_fontweight('bold')
    ax3.legend(fontsize=16, loc='upper right', prop={'weight': 'bold', 'size': 16})
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1.5)
    ax3.spines['bottom'].set_linewidth(1.5)
    
    # ========================================================================
    # Panel 4: Autocorrelation
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Compute autocorrelation for mean edge probability
    if trace_var in available_vars:
        # Use first chain for autocorrelation
        chain_data = chains_data[0]
        samples = chain_data.get('samples', {})
        if trace_var in samples:
            z_samples = to_numpy(samples[trace_var])
            mean_prob = _trace_series_from_z(z_samples, trace_mode)
            
            # Compute autocorrelation
            def autocorr(x, max_lag=100):
                """Compute autocorrelation function."""
                n = len(x)
                x_mean = np.mean(x)
                x_centered = x - x_mean
                var = np.var(x_centered)
                
                if var == 0:
                    return np.zeros(max_lag + 1)
                
                autocorr_vals = []
                for lag in range(max_lag + 1):
                    if lag >= n:
                        break
                    numerator = np.sum(x_centered[:-lag] * x_centered[lag:]) if lag > 0 else np.sum(x_centered ** 2)
                    autocorr_vals.append(numerator / (var * (n - lag)))
                
                return np.array(autocorr_vals)
            
            autocorr_vals = autocorr(mean_prob, max_lag=100)
            lags = np.arange(len(autocorr_vals))
            
            ax4.plot(lags, autocorr_vals, linewidth=2, color='#2E86AB', marker='o', markersize=3)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax4.set_xlabel('Lag', fontweight='bold', fontsize=16)
    ax4.set_ylabel('Autocorrelation', fontweight='bold', fontsize=16)
    ax4.set_title('Autocorrelation', fontweight='bold', fontsize=18, pad=10)
    ax4.tick_params(axis='both', which='major', labelsize=16, width=1.5, length=5)
    for label in ax4.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax4.get_yticklabels():
        label.set_fontweight('bold')
    # No legend needed for autocorrelation (single line)
    ax4.grid(True, alpha=0.3)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_linewidth(1.5)
    ax4.spines['bottom'].set_linewidth(1.5)
    ax4.set_ylim([-0.2, 1.1])
    
    # Save figure
    plt.tight_layout()
    plot_file = output_dir / "figure4_exhaustive_sampling.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Figure 4 (exhaustive sampling): {plot_file}")


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).strip().upper()
    # Remove common suffixes/prefixes
    name = name.replace('_HUMAN', '').replace('_MOUSE', '')
    return name


def load_ground_truth_from_databases(database_dir: Path, logger: logging.Logger) -> set:
    """Load ground truth pairs from all databases."""
    all_gt_pairs = set()
    
    # Load CORUM
    corum_file = database_dir / "corum_humanComplexes.tsv"
    if corum_file.exists():
        try:
            df = pd.read_csv(corum_file, sep='\t', low_memory=False)
            for _, row in df.iterrows():
                gene_names_str = str(row.get('subunits_gene_name', ''))
                if pd.isna(gene_names_str) or gene_names_str == '':
                    continue
                gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split(';')]
                gene_names = [g for g in gene_names if g and g != 'NAN']
                if len(gene_names) >= 2:
                    for i in range(len(gene_names)):
                        for j in range(i + 1, len(gene_names)):
                            all_gt_pairs.add(tuple(sorted([gene_names[i], gene_names[j]])))
            logger.info(f"  Loaded {len([p for p in all_gt_pairs if any('CORUM' in str(corum_file) for _ in [1])])} CORUM pairs")
        except Exception as e:
            logger.warning(f"Error loading CORUM: {e}")
    
    # Load HuMAP2
    humap2_file = database_dir / "humap2.text"
    if humap2_file.exists():
        try:
            df = pd.read_csv(humap2_file, sep=',', low_memory=False)
            for _, row in df.iterrows():
                gene_names_str = str(row.get('genenames', ''))
                if pd.isna(gene_names_str) or gene_names_str == '':
                    continue
                gene_names = [normalize_protein_name(g.strip()) for g in gene_names_str.split()]
                gene_names = [g for g in gene_names if g and g != 'NAN']
                if len(gene_names) >= 2:
                    for i in range(len(gene_names)):
                        for j in range(i + 1, len(gene_names)):
                            all_gt_pairs.add(tuple(sorted([gene_names[i], gene_names[j]])))
        except Exception as e:
            logger.warning(f"Error loading HuMAP2: {e}")
    
    # Load IntAct
    intact_file = database_dir / "intact_human_interactions.txt"
    if intact_file.exists():
        try:
            import re
            df = pd.read_csv(intact_file, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
            alias_col_a = df.columns[4] if len(df.columns) > 4 else None
            alias_col_b = df.columns[5] if len(df.columns) > 5 else None
            
            def extract_gene_name(alias_str):
                if pd.isna(alias_str) or alias_str == '':
                    return None
                alias_str = str(alias_str)
                matches = re.findall(r'\(gene name\)\|?([^|()]+)', alias_str)
                if matches:
                    return normalize_protein_name(matches[0].split('(')[0].strip())
                matches = re.findall(r'([A-Z0-9_-]+)\(gene name\)', alias_str)
                if matches:
                    return normalize_protein_name(matches[0])
                return None
            
            if alias_col_a and alias_col_b:
                for _, row in df.iterrows():
                    p1 = extract_gene_name(row[alias_col_a])
                    p2 = extract_gene_name(row[alias_col_b])
                    if p1 and p2 and p1 != p2 and p1 != 'NAN' and p2 != 'NAN':
                        all_gt_pairs.add(tuple(sorted([p1, p2])))
        except Exception as e:
            logger.warning(f"Error loading IntAct: {e}")
    
    # Load BioGRID
    biogrid_file = database_dir / "BIOGRID-ALL-5.0.253.mitab.txt"
    if biogrid_file.exists():
        try:
            df = pd.read_csv(biogrid_file, sep='\t', low_memory=False, comment='#', on_bad_lines='skip')
            if 'Taxid Interactor A' in df.columns:
                df = df[(df['Taxid Interactor A'] == 'taxid:9606') & (df['Taxid Interactor B'] == 'taxid:9606')]
            
            alias_col_a = df.columns[4] if len(df.columns) > 4 else None
            alias_col_b = df.columns[5] if len(df.columns) > 5 else None
            
            def extract_gene_name(alias_str):
                if pd.isna(alias_str) or alias_str == '':
                    return None
                alias_str = str(alias_str)
                # BioGRID format: gene name is usually first
                parts = alias_str.split('|')
                if parts:
                    gene_name = parts[0].split('(')[0].strip()
                    return normalize_protein_name(gene_name)
                return None
            
            if alias_col_a and alias_col_b:
                for _, row in df.iterrows():
                    p1 = extract_gene_name(row[alias_col_a])
                    p2 = extract_gene_name(row[alias_col_b])
                    if p1 and p2 and p1 != p2 and p1 != 'NAN' and p2 != 'NAN':
                        all_gt_pairs.add(tuple(sorted([p1, p2])))
        except Exception as e:
            logger.warning(f"Error loading BioGRID: {e}")
    
    logger.info(f"  Total ground truth pairs from databases: {len(all_gt_pairs)}")
    return all_gt_pairs


def plot_correlation_distribution(multi_seeds_dir: Path, output_dir: Path, logger: logging.Logger, database_dir: Path = None):
    """
    Plot distribution of actual vs shuffled cosine similarity (correlation) values.
    
    Args:
        multi_seeds_dir: Directory containing seed_* subdirectories
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Find first available seed directory with model_data
    seed_dirs = find_seed_directories(multi_seeds_dir)
    if not seed_dirs:
        logger.warning("No seed directories found for correlation plot")
        return
    
    model_data_file = None
    for seed, seed_dir in seed_dirs:
        # Look for model_data file
        model_data_files = list(seed_dir.glob("*_model_data.pkl"))
        if model_data_files:
            model_data_file = model_data_files[0]
            break
    
    if model_data_file is None:
        logger.warning("No model_data file found for correlation plot")
        return
    
    try:
        logger.info(f"Loading correlation data from {model_data_file}")
        with open(model_data_file, 'rb') as f:
            model_data = pkl.load(f)
        
        # Extract correlation arrays
        apms_corr_flat = model_data.get('apms_corr_flat')
        apms_shuff_corr_flat = model_data.get('apms_shuff_corr_flat')
        
        if apms_corr_flat is None or apms_shuff_corr_flat is None:
            logger.warning("Correlation data not found in model_data")
            return
        
        # Convert to numpy if JAX arrays
        def to_numpy(x):
            if isinstance(x, (jnp.ndarray, jax.Array)):
                return np.array(x)
            return np.asarray(x)
        
        apms_corr_flat = to_numpy(apms_corr_flat)
        apms_shuff_corr_flat = to_numpy(apms_shuff_corr_flat)
        
        # Load ground truth pairs from databases and map to our data
        ground_truth_corrs = None
        node_idx2name = model_data.get('node_idx2name', {})
        N = model_data.get('N', len(node_idx2name))
        
        if database_dir and Path(database_dir).exists() and node_idx2name:
            logger.info(f"Loading ground truth pairs from databases in {database_dir}")
            all_gt_pairs = load_ground_truth_from_databases(Path(database_dir), logger)
            
            # Reconstruct edge list from upper triangle indices
            # apms_corr_flat corresponds to upper triangle (i < j) of N×N matrix
            triu_indices = np.triu_indices(N, k=1)
            gt_indices = []
            
            for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
                p1 = normalize_protein_name(node_idx2name.get(int(i), f"protein_{int(i)}"))
                p2 = normalize_protein_name(node_idx2name.get(int(j), f"protein_{int(j)}"))
                pair = tuple(sorted([p1, p2]))
                if pair in all_gt_pairs:
                    gt_indices.append(idx)
            
            if len(gt_indices) > 0:
                ground_truth_corrs = apms_corr_flat[gt_indices]
                logger.info(f"  Mapped {len(ground_truth_corrs)} ground truth pairs to our data")
            else:
                logger.warning("  No ground truth pairs matched to our data")
        else:
            if not database_dir:
                logger.warning("  Database directory not provided, skipping ground truth")
            elif not Path(database_dir).exists():
                logger.warning(f"  Database directory not found: {database_dir}")
            else:
                logger.warning("  Missing node_idx2name for mapping")
        
        logger.info(f"  Actual correlations (all {len(apms_corr_flat)} pairs): "
                   f"mean={np.mean(apms_corr_flat):.4f}, median={np.median(apms_corr_flat):.4f}, "
                   f"std={np.std(apms_corr_flat):.4f}")
        logger.info(f"  Shuffled correlations (all {len(apms_shuff_corr_flat)} pairs): "
                   f"mean={np.mean(apms_shuff_corr_flat):.4f}, median={np.median(apms_shuff_corr_flat):.4f}, "
                   f"std={np.std(apms_shuff_corr_flat):.4f}")
        
        # Create figure with KDE plots
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Cosine Similarity Distribution: Actual vs Shuffled Correlations', 
                     fontsize=14, fontweight='bold')
        
        # Use KDE plots with transparency
        try:
            from scipy.stats import gaussian_kde
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
            logger.warning("scipy not available, using histogram instead of KDE")
        
        if HAS_SCIPY:
            # KDE for actual correlations
            kde_actual = gaussian_kde(apms_corr_flat)
            kde_shuffled = gaussian_kde(apms_shuff_corr_flat)
            
            x_range = np.linspace(-1.0, 1.0, 200)
            kde_actual_vals = kde_actual(x_range)
            kde_shuffled_vals = kde_shuffled(x_range)
            
            # Plot KDE with transparency
            ax.plot(x_range, kde_actual_vals, color='#2E86AB', linewidth=2.5, 
                   alpha=0.8, label=f'Actual (n={len(apms_corr_flat)})')
            ax.fill_between(x_range, kde_actual_vals, alpha=0.4, color='#2E86AB')
            
            ax.plot(x_range, kde_shuffled_vals, color='#E63946', linewidth=2.5, 
                   alpha=0.8, label=f'Shuffled (n={len(apms_shuff_corr_flat)})')
            ax.fill_between(x_range, kde_shuffled_vals, alpha=0.4, color='#E63946')
            
            # KDE for ground truth pairs if available
            if ground_truth_corrs is not None and len(ground_truth_corrs) > 0:
                kde_gt = gaussian_kde(ground_truth_corrs)
                kde_gt_vals = kde_gt(x_range)
                ax.plot(x_range, kde_gt_vals, color='#FFD700', linewidth=2.5, 
                       alpha=0.9, label=f'Ground truth (n={len(ground_truth_corrs)})')
                ax.fill_between(x_range, kde_gt_vals, alpha=0.5, color='#FFD700')
        else:
            # Fallback to histogram if scipy not available
            bins = np.linspace(-1.0, 1.0, 50)
            ax.hist(apms_corr_flat, bins=bins, alpha=0.6, label=f'Actual (n={len(apms_corr_flat)})', 
                   color='#2E86AB', edgecolor='#1B5E7A', linewidth=1.5, density=True)
            ax.hist(apms_shuff_corr_flat, bins=bins, alpha=0.6, label=f'Shuffled (n={len(apms_shuff_corr_flat)})', 
                   color='#E63946', edgecolor='#B91C26', linewidth=1.5, density=True)
            if ground_truth_corrs is not None and len(ground_truth_corrs) > 0:
                ax.hist(ground_truth_corrs, bins=bins, alpha=0.6, label=f'Ground truth (n={len(ground_truth_corrs)})',
                       color='#FFD700', edgecolor='#DAA520', linewidth=1.5, density=True)
        
        # Add vertical lines for mean
        ax.axvline(np.mean(apms_corr_flat), color='#2E86AB', linestyle='--', alpha=0.7, linewidth=2.0)
        ax.axvline(np.mean(apms_shuff_corr_flat), color='#E63946', linestyle='--', alpha=0.7, linewidth=2.0)
        if ground_truth_corrs is not None and len(ground_truth_corrs) > 0:
            ax.axvline(np.mean(ground_truth_corrs), color='#FFD700', linestyle='--', alpha=0.7, linewidth=2.0)
        
        ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.0, 1.0)
        
        plt.tight_layout()
        plot_file = output_dir / "correlation_distribution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation distribution plot: {plot_file}")
        
    except Exception as e:
        logger.error(f"Error creating correlation plot: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def plot_direct_vs_unknown(multi_seeds_dir: Path, output_dir: Path, logger: logging.Logger, database_dir: Path = None):
    """
    Plot simplified KDE distribution showing only direct (ground truth) vs unknown (actual data) 
    cosine similarity values. This plot demonstrates that direct protein interactions are 
    enriched towards high AP-MS profile similarity values.
    
    Args:
        multi_seeds_dir: Directory containing seed_* subdirectories
        output_dir: Directory to save plots
        logger: Logger instance
        database_dir: Directory containing ground truth databases
    """
    if not HAS_MATPLOTLIB:
        return
    
    # Find first available seed directory with model_data
    seed_dirs = find_seed_directories(multi_seeds_dir)
    if not seed_dirs:
        logger.warning("No seed directories found for correlation plot")
        return
    
    model_data_file = None
    for seed, seed_dir in seed_dirs:
        # Look for model_data file
        model_data_files = list(seed_dir.glob("*_model_data.pkl"))
        if model_data_files:
            model_data_file = model_data_files[0]
            break
    
    if model_data_file is None:
        logger.warning("No model_data file found for correlation plot")
        return
    
    try:
        logger.info(f"Loading correlation data from {model_data_file}")
        with open(model_data_file, 'rb') as f:
            model_data = pkl.load(f)
        
        # Extract correlation arrays
        apms_corr_flat = model_data.get('apms_corr_flat')
        
        if apms_corr_flat is None:
            logger.warning("Correlation data not found in model_data")
            return
        
        # Convert to numpy if JAX arrays
        def to_numpy(x):
            if isinstance(x, (jnp.ndarray, jax.Array)):
                return np.array(x)
            return np.asarray(x)
        
        apms_corr_flat = to_numpy(apms_corr_flat)
        
        # Load ground truth pairs from databases and map to our data
        ground_truth_corrs = None
        node_idx2name = model_data.get('node_idx2name', {})
        N = model_data.get('N', len(node_idx2name))
        
        if database_dir and Path(database_dir).exists() and node_idx2name:
            logger.info(f"Loading ground truth pairs from databases in {database_dir}")
            all_gt_pairs = load_ground_truth_from_databases(Path(database_dir), logger)
            
            # Reconstruct edge list from upper triangle indices
            # apms_corr_flat corresponds to upper triangle (i < j) of N×N matrix
            triu_indices = np.triu_indices(N, k=1)
            gt_indices = []
            
            for idx, (i, j) in enumerate(zip(triu_indices[0], triu_indices[1])):
                p1 = normalize_protein_name(node_idx2name.get(int(i), f"protein_{int(i)}"))
                p2 = normalize_protein_name(node_idx2name.get(int(j), f"protein_{int(j)}"))
                pair = tuple(sorted([p1, p2]))
                if pair in all_gt_pairs:
                    gt_indices.append(idx)
            
            if len(gt_indices) > 0:
                ground_truth_corrs = apms_corr_flat[gt_indices]
                logger.info(f"  Mapped {len(ground_truth_corrs)} ground truth pairs to our data")
            else:
                logger.warning("  No ground truth pairs matched to our data")
        else:
            if not database_dir:
                logger.warning("  Database directory not provided, skipping ground truth")
            elif not Path(database_dir).exists():
                logger.warning(f"  Database directory not found: {database_dir}")
            else:
                logger.warning("  Missing node_idx2name for mapping")
        
        if ground_truth_corrs is None or len(ground_truth_corrs) == 0:
            logger.warning("No ground truth pairs found, cannot create direct vs unknown plot")
            return
        
        # Create figure with KDE plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Use KDE plots with transparency
        try:
            from scipy.stats import gaussian_kde
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
            logger.warning("scipy not available, using histogram instead of KDE")
        
        if HAS_SCIPY:
            # KDE for unknown (all actual data)
            kde_unknown = gaussian_kde(apms_corr_flat)
            
            # KDE for direct (ground truth)
            kde_direct = gaussian_kde(ground_truth_corrs)
            
            # X-axis range from 0 to 1
            x_range = np.linspace(0.0, 1.0, 200)
            kde_unknown_vals = kde_unknown(x_range)
            kde_direct_vals = kde_direct(x_range)
            
            # Plot KDE with thicker lines for multipanel visibility - unknown first (background)
            # Use light gray for better contrast
            ax.plot(x_range, kde_unknown_vals, color='#6C757D', linewidth=3.5, 
                   alpha=0.6, label='unknown')
            ax.fill_between(x_range, kde_unknown_vals, alpha=0.2, color='#6C757D')
            
            # Plot direct (ground truth) on top with thicker lines
            # Use vibrant red for better contrast with gray
            ax.plot(x_range, kde_direct_vals, color='#DC3545', linewidth=3.5, 
                   alpha=0.7, label='direct')
            ax.fill_between(x_range, kde_direct_vals, alpha=0.35, color='#DC3545')
        else:
            # Fallback to histogram if scipy not available - thicker lines for multipanel visibility
            bins = np.linspace(0.0, 1.0, 50)
            ax.hist(apms_corr_flat, bins=bins, alpha=0.3, label='unknown', 
                   color='#6C757D', edgecolor='#495057', linewidth=2.5, density=True)
            ax.hist(ground_truth_corrs, bins=bins, alpha=0.4, label='direct',
                   color='#DC3545', edgecolor='#C82333', linewidth=2.5, density=True)
        
        # Set x-axis limits to 0 to 1
        ax.set_xlim(0.0, 1.0)
        
        # Set font sizes - much larger and bold for multipanel subplot visibility
        # When scaled down in multipanel figures, these will still be readable
        label_fontsize = 18
        tick_fontsize = 16
        legend_fontsize = 16
        
        # Set axis labels - bold and larger for journal figure
        ax.set_xlabel('AP-MS profile similarity', fontsize=label_fontsize, fontweight='bold', color='black')
        ax.set_ylabel('Density', fontsize=label_fontsize, fontweight='bold', color='black')
        
        # Add tick marks with number labels
        # X-axis: mark at 0, 0.5, 1.0 with labels - bold and larger
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(['0', '0.5', '1'], fontsize=tick_fontsize, fontweight='bold', color='black')
        
        # Y-axis: use whole numbers (0, 1, 2, 3, etc.)
        y_min, y_max = ax.get_ylim()
        # Round to nearest integer for y_max, ensure we have whole number ticks
        y_max_int = int(np.ceil(y_max))
        # Generate whole number ticks from 0 to y_max_int
        if y_max_int <= 3:
            y_ticks = list(range(0, y_max_int + 1))
        elif y_max_int <= 5:
            y_ticks = list(range(0, y_max_int + 1, 1))
        else:
            # For larger ranges, use every 2 or every 5
            step = 2 if y_max_int <= 10 else 5
            y_ticks = list(range(0, y_max_int + step, step))
        
        # Format Y-axis labels as whole numbers
        y_labels = [f'{int(y)}' for y in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=tick_fontsize, fontweight='bold', color='black')
        
        # Set y-axis limits to include all ticks nicely
        if len(y_ticks) > 0:
            ax.set_ylim(0, max(y_ticks))
        
        # Add legend - bold and larger
        ax.legend(fontsize=legend_fontsize, loc='best', framealpha=0.9, prop={'weight': 'bold'})
        
        # Remove grid
        ax.grid(False)
        
        # Show both bottom and left spines with tick marks - thicker for multipanel visibility
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['bottom'].set_color('black')
        
        # Make tick marks visible - larger and bold for multipanel visibility
        ax.tick_params(axis='both', which='major', length=8, width=2.0, 
                      direction='out', color='black', labelsize=tick_fontsize)
        
        plt.tight_layout()
        plot_file = output_dir / "direct_vs_unknown.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved direct vs unknown plot: {plot_file}")
        
    except Exception as e:
        logger.error(f"Error creating direct vs unknown plot: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def plot_arviz_diagnostics(var_chains: List[np.ndarray], var_name: str, seeds: List[int],
                         output_dir: Path, logger: logging.Logger):
    """
    Use arviz to create comprehensive diagnostic plots.
    
    Args:
        var_chains: List of arrays, each of shape (n_samples, n_vars)
        var_name: Name of the variable
        seeds: List of seed values
        output_dir: Directory to save plots
        logger: Logger instance
    """
    if not HAS_ARVIZ or not HAS_MATPLOTLIB:
        return
    
    try:
        # Stack chains: (n_chains, n_samples, n_vars)
        all_chains = np.stack(var_chains, axis=0)
        
        # For multi-dimensional variables, use first dimension or mean
        if len(all_chains.shape) == 3 and all_chains.shape[2] > 1:
            # Use mean across dimensions for visualization
            all_chains = np.mean(all_chains, axis=2, keepdims=True)
        
        # Create InferenceData
        data_dict = {var_name: all_chains}
        idata = az.convert_to_inference_data(data_dict)
        
        # Create comprehensive plots
        fig = az.plot_trace(idata, compact=True, divergences='bottom')
        plot_file = output_dir / f"arviz_trace_{var_name}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved arviz trace plot: {plot_file}")
        
        # Rank plots
        try:
            fig = az.plot_rank(idata)
            plot_file = output_dir / f"arviz_rank_{var_name}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved arviz rank plot: {plot_file}")
        except Exception as e:
            logger.debug(f"Could not create rank plot: {e}")
        
        # Autocorrelation plots
        try:
            fig = az.plot_autocorr(idata, max_lag=100)
            plot_file = output_dir / f"arviz_autocorr_{var_name}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved arviz autocorrelation plot: {plot_file}")
        except Exception as e:
            logger.debug(f"Could not create autocorrelation plot: {e}")
            
    except Exception as e:
        logger.warning(f"Could not create arviz plots for {var_name}: {e}")


def compute_rhat_arviz(chains: List[np.ndarray], var_name: str = "variable") -> Dict[str, Any]:
    """
    Compute R-hat using arviz (more robust implementation).
    
    Args:
        chains: List of arrays, each of shape (n_samples, n_vars)
        var_name: Name of the variable for arviz
    
    Returns:
        Dictionary with R-hat and other diagnostics
    """
    # Stack chains: (n_chains, n_samples, n_vars)
    all_chains = np.stack(chains, axis=0)
    
    # Create InferenceData object
    # arviz expects (n_chains, n_draws, ...) shape
    data_dict = {var_name: all_chains}
    idata = az.convert_to_inference_data(data_dict)
    
    # Compute diagnostics
    summary = az.summary(idata, kind="stats")
    rhat = summary['r_hat'].values
    
    # Also compute effective sample size
    ess = az.ess(idata)
    ess_bulk = ess['ess_bulk'].values if 'ess_bulk' in ess else None
    ess_tail = ess['ess_tail'].values if 'ess_tail' in ess else None
    
    return {
        'rhat': rhat,
        'ess_bulk': ess_bulk,
        'ess_tail': ess_tail,
        'summary': summary
    }


def merge_and_analyze(
    multi_seeds_dir: Path,
    burn_in: int = 0,
    use_arviz: bool = True,
    create_plots: bool = True,
    figure4_trace_mode: str = "binary_m",
):
    """
    Merge multiple chains and compute convergence diagnostics.

    Args:
        multi_seeds_dir: Directory containing seed_* subdirectories
        burn_in: Number of samples to discard from beginning of each chain
        use_arviz: Whether to use arviz for R-hat computation (if available)
        create_plots: Whether to create diagnostic plots
        figure4_trace_mode: Panels 3–4 of figure4_exhaustive_sampling.png: "binary_m" or "sigmoid_mean"
    """
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("MERGING AND CONVERGENCE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Multi-seeds directory: {multi_seeds_dir}")
    logger.info(f"Burn-in: {burn_in} samples")
    
    # Try to read config to get num_warmup
    num_warmup = None
    config_files = list(multi_seeds_dir.glob("**/config.json"))
    if config_files:
        try:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
                num_warmup = config.get('sampling', {}).get('num_warmup', None)
                if num_warmup:
                    logger.info(f"Found num_warmup from config: {num_warmup}")
        except Exception as e:
            logger.warning(f"Could not read config for num_warmup: {e}")
    
    # Find all seed directories
    seed_dirs = find_seed_directories(multi_seeds_dir)
    if not seed_dirs:
        raise ValueError(f"No seed_* directories found in {multi_seeds_dir}")
    
    logger.info(f"Found {len(seed_dirs)} seed directories: {[s for s, _ in seed_dirs]}")
    
    # Load all chains
    chains_data = []
    for seed, seed_dir in seed_dirs:
        chain_data = load_chain(seed_dir, logger)
        if chain_data is not None:
            chains_data.append(chain_data)
    
    if not chains_data:
        raise ValueError("No valid chains found!")
    
    logger.info(f"Successfully loaded {len(chains_data)} chains")

    # ------------------------------------------------------------------
    # Sanity check: ensure all chains have compatible shapes for 'z'
    # In rare cases (e.g., interrupted runs or subtle preprocessing
    # differences), some chains can have a different number of edges
    # (second dimension of z), which prevents concatenation.
    # For robustness, we keep only chains with the most common z-shape
    # and drop outliers, logging a clear warning.
    # ------------------------------------------------------------------
    if 'z' in chains_data[0]['samples']:
        z_shapes = []
        for cd in chains_data:
            z_arr = cd['samples'].get('z', None)
            if z_arr is None:
                z_shapes.append(None)
            else:
                shape = tuple(z_arr.shape[1:]) if z_arr.ndim > 1 else ()
                z_shapes.append(shape)

        shape_counts = {}
        for sh in z_shapes:
            if sh is None:
                continue
            shape_counts[sh] = shape_counts.get(sh, 0) + 1

        if shape_counts:
            target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
            kept_indices = [i for i, sh in enumerate(z_shapes) if sh == target_shape]
            dropped_indices = [i for i in range(len(chains_data)) if i not in kept_indices]

            if dropped_indices:
                dropped_seeds = [chains_data[i]['seed'] for i in dropped_indices]
                logger.warning(
                    "Some chains have incompatible 'z' shapes and will be dropped "
                    f"for merging. Keeping shape {target_shape}, dropping seeds: {dropped_seeds}"
                )
                chains_data = [chains_data[i] for i in kept_indices]

    if len(chains_data) < 2:
        logger.warning(
            "Only one compatible chain available after shape checks; "
            "merged results will effectively come from a single chain."
        )
    
    # Extract samples for each variable
    all_variables = set()
    for chain_data in chains_data:
        all_variables.update(chain_data['samples'].keys())
    
    logger.info(f"Variables found: {sorted(all_variables)}")
    
    # Merge chains and compute diagnostics
    merged_results = {}
    diagnostics = {}
    
    for var_name in sorted(all_variables):
        logger.info(f"\nProcessing variable: {var_name}")
        
        # Extract chains for this variable
        var_chains = []
        for chain_data in chains_data:
            var_samples = chain_data['samples'][var_name]
            
            # Apply burn-in
            if burn_in > 0 and len(var_samples) > burn_in:
                var_samples = var_samples[burn_in:]
            
            # Flatten if needed (handle multi-dimensional variables)
            if len(var_samples.shape) > 1:
                # Flatten all dimensions except the first (sample dimension)
                n_samples = var_samples.shape[0]
                var_samples = var_samples.reshape(n_samples, -1)
            
            var_chains.append(var_samples)
        
        # Check that all chains have the same shape
        n_samples_per_chain = [c.shape[0] for c in var_chains]
        n_vars = var_chains[0].shape[1] if len(var_chains[0].shape) > 1 else 1
        
        logger.info(f"  Shape per chain: {n_samples_per_chain[0]} samples, {n_vars} variables")
        logger.info(f"  Total samples: {sum(n_samples_per_chain)}")
        
        # Merge chains (concatenate along sample dimension)
        merged_chain = np.concatenate(var_chains, axis=0)
        merged_results[var_name] = merged_chain
        
        # Compute R-hat
        logger.info(f"  Computing R-hat...")
        if use_arviz and HAS_ARVIZ:
            try:
                diag = compute_rhat_arviz(var_chains, var_name=var_name)
                diagnostics[var_name] = {
                    'rhat': diag['rhat'],
                    'ess_bulk': diag['ess_bulk'],
                    'ess_tail': diag['ess_tail'],
                    'summary': diag['summary'].to_dict() if hasattr(diag['summary'], 'to_dict') else str(diag['summary'])
                }
                logger.info(f"  R-hat: {np.mean(diag['rhat']):.4f} (mean), {np.max(diag['rhat']):.4f} (max)")
                if diag['ess_bulk'] is not None:
                    logger.info(f"  ESS (bulk): {np.mean(diag['ess_bulk']):.1f} (mean)")
            except Exception as e:
                logger.warning(f"  Arviz computation failed: {e}. Using basic R-hat.")
                rhat = compute_rhat_basic(var_chains)
                diagnostics[var_name] = {'rhat': rhat}
                logger.info(f"  R-hat: {np.mean(rhat):.4f} (mean), {np.max(rhat):.4f} (max)")
        else:
            rhat = compute_rhat_basic(var_chains)
            diagnostics[var_name] = {'rhat': rhat}
            logger.info(f"  R-hat: {np.mean(rhat):.4f} (mean), {np.max(rhat):.4f} (max)")
    
    # Save merged results
    output_dir = multi_seeds_dir / "merged"
    output_dir.mkdir(exist_ok=True)
    
    merged_file = output_dir / "merged_chains.pkl"
    logger.info(f"\nSaving merged chains to {merged_file}")
    with open(merged_file, 'wb') as f:
        pkl.dump({
            'merged_samples': merged_results,
            'n_chains': len(chains_data),
            'seeds': [cd['seed'] for cd in chains_data],
            'burn_in': burn_in
        }, f)
    
    # Store chains for plotting (before merging)
    chains_for_plotting = {}
    seeds_list = [cd['seed'] for cd in chains_data]
    for var_name in sorted(all_variables):
        var_chains = []
        for chain_data in chains_data:
            var_samples = chain_data['samples'][var_name]
            if burn_in > 0 and len(var_samples) > burn_in:
                var_samples = var_samples[burn_in:]
            if len(var_samples.shape) > 1:
                n_samples = var_samples.shape[0]
                var_samples = var_samples.reshape(n_samples, -1)
            var_chains.append(var_samples)
        chains_for_plotting[var_name] = var_chains
    
    # Save diagnostics
    diagnostics_file = output_dir / "convergence_diagnostics.json"
    logger.info(f"Saving diagnostics to {diagnostics_file}")
    
    # Convert numpy arrays to lists for JSON
    diagnostics_json = {}
    for var_name, diag in diagnostics.items():
        diagnostics_json[var_name] = {}
        for key, value in diag.items():
            if isinstance(value, np.ndarray):
                diagnostics_json[var_name][key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                diagnostics_json[var_name][key] = float(value)
            else:
                diagnostics_json[var_name][key] = str(value)
    
    with open(diagnostics_file, 'w') as f:
        json.dump(diagnostics_json, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CONVERGENCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Number of chains: {len(chains_data)}")
    logger.info(f"Seeds: {[cd['seed'] for cd in chains_data]}")
    logger.info(f"Burn-in: {burn_in} samples per chain")
    
    for var_name in sorted(diagnostics.keys()):
        diag = diagnostics[var_name]
        rhat = diag['rhat']
        if isinstance(rhat, np.ndarray):
            rhat_mean = np.mean(rhat)
            rhat_max = np.max(rhat)
            rhat_min = np.min(rhat)
        else:
            rhat_mean = rhat_max = rhat_min = float(rhat)
        
        logger.info(f"\n{var_name}:")
        logger.info(f"  R-hat: mean={rhat_mean:.4f}, min={rhat_min:.4f}, max={rhat_max:.4f}")
        if rhat_max > 1.01:
            logger.warning(f"  ⚠️  R-hat > 1.01 indicates potential convergence issues!")
        else:
            logger.info(f"  ✓ R-hat <= 1.01 indicates good convergence")
        
        if 'ess_bulk' in diag and diag['ess_bulk'] is not None:
            ess = diag['ess_bulk']
            if isinstance(ess, np.ndarray):
                ess_mean = np.mean(ess)
            else:
                ess_mean = float(ess)
            logger.info(f"  ESS (bulk): {ess_mean:.1f}")
    
    # Create plots
    if create_plots:
        logger.info("\n" + "=" * 60)
        logger.info("CREATING DIAGNOSTIC PLOTS")
        logger.info("=" * 60)
        
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available. Skipping plot generation.")
        else:
            # Plot 1: Convergence analysis (mean, std dev for r, s, z, and rhat)
            logger.info("Creating convergence analysis plot...")
            plot_convergence_analysis(chains_for_plotting, diagnostics, seeds_list, output_dir, logger)
            
            # Plot 2: Total score (potential_energy) vs sampling steps
            logger.info("Creating total score plot...")
            plot_total_score(chains_data, seeds_list, output_dir, logger)
            
            # Plot 2b: Cost vs steps for total and individual scores (including warmup)
            logger.info("Creating cost vs steps plot (total + individual scores)...")
            plot_cost_vs_steps(chains_data, seeds_list, output_dir, logger, num_warmup=num_warmup)
            
            # Plot 2c: R-hat over steps to show convergence
            logger.info("Creating R-hat convergence plot...")
            plot_rhat_over_steps(chains_data, seeds_list, output_dir, logger, num_warmup=num_warmup)
            
            # Plot 2d: Figure 4 - Exhaustive Sampling (4-panel figure for journal)
            logger.info("Creating Figure 4 (exhaustive sampling)...")
            plot_figure4_exhaustive_sampling(
                chains_data,
                seeds_list,
                output_dir,
                logger,
                num_warmup=num_warmup,
                trace_mode=figure4_trace_mode,
            )
            
            # Plot 3: Cosine similarity distribution (KDE plot)
            logger.info("Creating correlation distribution plot...")
            # Try to find database directory (relative to project root)
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            database_dir = project_root / "database"
            plot_correlation_distribution(multi_seeds_dir, output_dir, logger, database_dir=database_dir)
            
            # Plot 4: Direct vs Unknown (simplified plot for publication)
            logger.info("Creating direct vs unknown plot...")
            plot_direct_vs_unknown(multi_seeds_dir, output_dir, logger, database_dir=database_dir)
            
            logger.info("Plot generation complete!")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple MCMC chains and compute convergence diagnostics'
    )
    parser.add_argument(
        'multi_seeds_dir',
        type=Path,
        help='Directory containing seed_* subdirectories (e.g., results/..._multi_seeds)'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        default=0,
        help='Number of samples to discard from beginning of each chain (default: 0)'
    )
    parser.add_argument(
        '--no-arviz',
        action='store_true',
        help='Disable arviz (use basic R-hat implementation)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--figure4-trace',
        type=str,
        default='binary_m',
        choices=['binary_m', 'sigmoid_mean'],
        help='figure4_exhaustive_sampling.png panels 3–4: mean M=1{z>0} vs legacy mean sigmoid(z)',
    )

    args = parser.parse_args()
    
    if not args.multi_seeds_dir.exists():
        raise ValueError(f"Directory not found: {args.multi_seeds_dir}")
    
    merge_and_analyze(
        args.multi_seeds_dir,
        burn_in=args.burn_in,
        use_arviz=not args.no_arviz,
        create_plots=not args.no_plots,
        figure4_trace_mode=args.figure4_trace,
    )


if __name__ == '__main__':
    import jax
    main()

