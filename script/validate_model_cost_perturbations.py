#!/usr/bin/env python3
"""
Computational validation via model cost function and perturbations.

This script validates that the original simulated model (true model) has:
1. Lowest total cost (best fit to data/constraints)
2. Highest accuracy (100% - perfect match with itself)

It tests perturbations (node permutation, edge flipping, Gaussian noise) and shows
that perturbed models have higher cost and lower accuracy (measured as similarity
to the true model via Spearman correlation).

Usage:
    python validate_model_cost_perturbations.py \
        --results-dir ../results \
        --dataset cullin_mock_config_multi_seeds \
        --output cost_validation/

Binary edge frequency over full merged posterior (same M=1{Z>Z_thr} as post_sampling_z_binary):

    python validate_model_cost_perturbations.py ... --output cost_validation/DS \\
        --binary-m-mean-full-merged

See script/run_cost_validation_binary_m_full_merged.sh for all three base models.
"""

import argparse
import json
import logging
import pickle as pkl
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats

_VAL_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_VAL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_VAL_SCRIPT_DIR))
from post_sampling_z_binary import binary_m_from_z  # noqa: E402

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plots will not be generated.")

# Try to import JAX
try:
    import jax.numpy as jnp
    import jax.nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn("JAX not available. Some computations may be slower.")


def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "cost_validation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def normalize_protein_name(name: str) -> str:
    """Normalize protein name for comparison."""
    name = str(name).strip().upper()
    if name.endswith("_HUMAN"):
        name = name[:-6]
    return name


def load_predictions_and_labels(
    results_dir: Path,
    dataset_name: str,
    pdb_results_dir: Path,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, dict]:
    """
    Load predictions (probabilities) from the true model.
    
    Note: We don't need PDB labels anymore - the true model itself is the ground truth.
    
    Returns:
        pred: DataFrame with columns: gene_a, gene_b, posterior_prob
        model_data: Model data dictionary
    """
    logger.info(f"Loading predictions and labels for {dataset_name}")
    
    # Load edge probabilities
    dataset_dir = results_dir / dataset_name
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return pd.DataFrame(), {}
    
    # Load merged chains
    with open(merged_file, 'rb') as f:
        merged_data = pkl.load(f)
    
    merged_samples = merged_data['merged_samples']
    seeds = merged_data['seeds']
    model_data_file = dataset_dir / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"
    
    if 'z' not in merged_samples:
        logger.error(f"No 'z' samples in merged data for {dataset_name}")
        return pd.DataFrame(), {}
    
    z_samples = merged_samples['z']
    
    # Load model data
    with open(model_data_file, 'rb') as f:
        model_data = pkl.load(f)
    
    node_idx2name = model_data.get('node_idx2name', {})
    if not node_idx2name:
        logger.error(f"No node_idx2name in model_data for {dataset_name}")
        return pd.DataFrame(), {}
    
    # Convert to numpy if JAX array
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Handle multiple chains
    if len(z_samples.shape) == 3:
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif len(z_samples.shape) == 2:
        pass
    else:
        z_samples = z_samples.reshape(1, -1)
    
    # Calculate mean probability for each edge
    if HAS_JAX:
        mean_probs = jax.nn.sigmoid(jnp.array(z_samples)).mean(axis=0)
        mean_probs = np.array(mean_probs)
    else:
        mean_probs = np.mean(1 / (1 + np.exp(-z_samples)), axis=0)
    
    # Create predictions DataFrame (no PDB labels needed - true model is ground truth)
    N = len(node_idx2name)
    predictions = []
    
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[k])
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            
            predictions.append({
                'gene_a': p1,
                'gene_b': p2,
                'posterior_prob': prob
            })
            k += 1
    
    pred = pd.DataFrame(predictions)
    logger.info(f"  Created {len(pred)} predictions from true model")
    
    return pred, model_data


def load_predictions_from_mean_edges(
    mean_edges_path: Path,
    results_dir: Path,
    dataset_name: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, dict]:
    """
    Build the same pred DataFrame as load_predictions_and_labels, but posterior_prob
    comes from an edge-wise mean vector (e.g. mean_edges_cluster_1.npy), upper-tri order.
    Still loads model_data from the first seed pickle (for N, R, S, constraints).
    """
    logger.info(f"Loading mean-edge reference from {mean_edges_path}")
    dataset_dir = results_dir / dataset_name
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return pd.DataFrame(), {}

    with open(merged_file, "rb") as f:
        merged_data = pkl.load(f)
    seeds = merged_data["seeds"]
    model_data_file = dataset_dir / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"
    if not model_data_file.exists():
        logger.error(f"model_data not found: {model_data_file}")
        return pd.DataFrame(), {}

    with open(model_data_file, "rb") as f:
        model_data = pkl.load(f)

    node_idx2name = model_data.get("node_idx2name", {})
    if not node_idx2name:
        logger.error(f"No node_idx2name in model_data for {dataset_name}")
        return pd.DataFrame(), {}

    vec = np.load(mean_edges_path)
    vec = np.asarray(vec, dtype=float).ravel()
    N = len(node_idx2name)
    m_full = N * (N - 1) // 2
    if vec.size != m_full:
        logger.error(
            f"mean_edges length {vec.size} != N(N-1)/2 = {m_full} for N={N}"
        )
        return pd.DataFrame(), {}
    vec = np.clip(vec, 0.0, 1.0)

    predictions = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(vec[k])
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            predictions.append(
                {"gene_a": p1, "gene_b": p2, "posterior_prob": prob}
            )
            k += 1

    pred = pd.DataFrame(predictions)
    logger.info(f"  Created {len(pred)} predictions from mean-edges vector")
    return pred, model_data


def load_predictions_from_binary_m_mean_merged(
    results_dir: Path,
    dataset_name: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, dict]:
    """
    Edge scores = mean over all merged draws of per-draw binary M (see post_sampling_z_binary), in [0,1].

    Same per-draw rule as the binary ensemble / main-cluster frequency pipeline;
    uses the full merged posterior (no cluster mask), not mean(sigmoid(z)).
    """
    logger.info(
        f"Loading binary-M mean over full merged draws for {dataset_name} (Z threshold from post_sampling_z_binary)"
    )
    dataset_dir = results_dir / dataset_name
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return pd.DataFrame(), {}

    with open(merged_file, "rb") as f:
        merged_data = pkl.load(f)
    merged_samples = merged_data["merged_samples"]
    seeds = merged_data["seeds"]
    model_data_file = dataset_dir / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"

    if "z" not in merged_samples:
        logger.error(f"No 'z' samples in merged data for {dataset_name}")
        return pd.DataFrame(), {}

    z_samples = merged_samples["z"]
    if hasattr(z_samples, "block_until_ready"):
        z_samples = np.array(z_samples)
    else:
        z_samples = np.asarray(z_samples, dtype=np.float64)

    if z_samples.ndim == 3:
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif z_samples.ndim == 2:
        pass
    elif z_samples.ndim == 1:
        z_samples = z_samples.reshape(1, -1)
    else:
        logger.error(f"Unexpected z ndim={z_samples.ndim}")
        return pd.DataFrame(), {}

    with open(model_data_file, "rb") as f:
        model_data = pkl.load(f)

    node_idx2name = model_data.get("node_idx2name", {})
    if not node_idx2name:
        logger.error(f"No node_idx2name in model_data for {dataset_name}")
        return pd.DataFrame(), {}

    M = binary_m_from_z(np.asarray(z_samples, dtype=np.float64)).astype(np.float64)
    mean_probs = M.mean(axis=0)
    mean_probs = np.clip(mean_probs, 0.0, 1.0)

    N = len(node_idx2name)
    predictions = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[k])
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            predictions.append(
                {"gene_a": p1, "gene_b": p2, "posterior_prob": prob}
            )
            k += 1

    pred = pd.DataFrame(predictions)
    logger.info(
        f"  Created {len(pred)} predictions (mean binary M over {z_samples.shape[0]} draws)"
    )
    return pred, model_data


def load_predictions_sigmoid_of_mean_z_main_cluster(
    results_dir: Path,
    dataset_name: str,
    main_cluster_summary_path: Path,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, dict]:
    """
    Edge scores = σ(mean_t z_ij) where t ranges only over merged draws assigned to the
    largest τ* cluster (main_cluster_id in main_cluster_summary.json; draw labels from
    labels_source in that JSON). Differs from load_predictions_and_labels, which uses
    mean_t σ(z_ij) over all merged draws.
    """
    logger.info(
        f"Loading σ(mean z) over main τ* cluster for {dataset_name} "
        f"(summary: {main_cluster_summary_path})"
    )
    if not main_cluster_summary_path.is_file():
        logger.error(f"Main cluster summary not found: {main_cluster_summary_path}")
        return pd.DataFrame(), {}

    with open(main_cluster_summary_path, "r") as f:
        summary = json.load(f)

    labels_path = Path(summary["labels_source"])
    if not labels_path.is_file():
        logger.error(f"Draw labels file not found (labels_source): {labels_path}")
        return pd.DataFrame(), {}

    main_id = int(summary["main_cluster_id"])
    labels = np.load(labels_path)
    labels = np.asarray(labels).ravel()

    dataset_dir = results_dir / dataset_name
    merged_file = dataset_dir / "merged" / "merged_chains.pkl"
    if not merged_file.exists():
        logger.error(f"Merged chains file not found: {merged_file}")
        return pd.DataFrame(), {}

    with open(merged_file, "rb") as f:
        merged_data = pkl.load(f)
    merged_samples = merged_data["merged_samples"]
    seeds = merged_data["seeds"]
    model_data_file = dataset_dir / f"seed_{seeds[0]}" / f"0_model23_n_{seeds[0]}_model_data.pkl"

    if "z" not in merged_samples:
        logger.error(f"No 'z' samples in merged data for {dataset_name}")
        return pd.DataFrame(), {}

    z_samples = merged_samples["z"]
    if hasattr(z_samples, "block_until_ready"):
        z_samples = np.array(z_samples)
    else:
        z_samples = np.asarray(z_samples, dtype=np.float64)

    if z_samples.ndim == 3:
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif z_samples.ndim == 2:
        pass
    elif z_samples.ndim == 1:
        z_samples = z_samples.reshape(1, -1)
    else:
        logger.error(f"Unexpected z ndim={z_samples.ndim}")
        return pd.DataFrame(), {}

    if labels.shape[0] != z_samples.shape[0]:
        logger.error(
            f"labels length {labels.shape[0]} != number of z rows {z_samples.shape[0]}"
        )
        return pd.DataFrame(), {}

    mask = labels == main_id
    n_in = int(np.sum(mask))
    if n_in == 0:
        logger.error(f"No draws with label == main_cluster_id ({main_id})")
        return pd.DataFrame(), {}

    z_sub = z_samples[mask]
    mean_z = np.mean(z_sub, axis=0)
    mean_probs = 1.0 / (1.0 + np.exp(-mean_z))
    mean_probs = np.clip(mean_probs.astype(float), 0.0, 1.0)

    with open(model_data_file, "rb") as f:
        model_data = pkl.load(f)

    node_idx2name = model_data.get("node_idx2name", {})
    if not node_idx2name:
        logger.error(f"No node_idx2name in model_data for {dataset_name}")
        return pd.DataFrame(), {}

    N = len(node_idx2name)
    predictions = []
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            prob = float(mean_probs[k])
            p1 = normalize_protein_name(node_idx2name[i])
            p2 = normalize_protein_name(node_idx2name[j])
            predictions.append(
                {"gene_a": p1, "gene_b": p2, "posterior_prob": prob}
            )
            k += 1

    pred = pd.DataFrame(predictions)
    logger.info(
        f"  Created {len(pred)} predictions (σ(mean z) over {n_in} main-cluster draws / "
        f"{z_samples.shape[0]} merged)"
    )
    return pred, model_data


def triu_idx(N, k=1):
    """Get upper triangle indices."""
    return np.triu_indices(N, k)


def safe_logit(p, eps=1e-6):
    """Safe logit transformation."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rebuild_pairwise_from_flat(flat, N, triu):
    """Rebuild pairwise matrix from flat vector."""
    M = np.zeros((N, N), float)
    M[triu] = flat
    M = M + M.T
    np.fill_diagonal(M, 0.0)
    return M


def get_R_S_from_model_data(md, N, triu):
    """Extract correlation (R) and SAINT (S) matrices from model data."""
    # Get correlation matrix R
    if "corr" in md and isinstance(md["corr"], np.ndarray) and md["corr"].shape == (N, N):
        R = md["corr"].astype(float)
    elif "apms_corr_flat" in md:
        R = rebuild_pairwise_from_flat(np.asarray(md["apms_corr_flat"], float), N, triu)
    else:
        R = np.zeros((N, N), float)
    
    # Get SAINT matrix S
    S = None
    for k in ["saint_max_pair_score_pairwise_matrix", "saint_pairwise_matrix", "saint_pair_scores"]:
        if k in md and isinstance(md[k], np.ndarray) and md[k].shape == (N, N):
            S = md[k].astype(float)
            break
    
    if S is None:
        S = np.zeros((N, N), float)
    
    return R, S


def soft_total_cost_from_probs(
    p_vec,
    model_data,
    degree_mode="powerlaw",
    edges_mu=None,
    edges_sd=100.0,
    gamma=2.5
):
    """
    Compute total cost (negative log-posterior) from probability vector.
    
    Cost components:
    1. Correlation cost (R)
    2. SAINT score cost (S)
    3. Edge count constraint
    4. Degree constraint (power-law or Gaussian)
    """
    N = int(model_data["N"])
    triu = triu_idx(N, 1)
    p_vec = np.asarray(p_vec, float).ravel()
    zvec = safe_logit(p_vec)
    
    # Z prior cost - Model uses: dist.Normal(mu=0.5, sigma=1.0)
    z_prior_mu = 0.5
    z_prior_sigma = 1.0
    z_prior_var = z_prior_sigma ** 2
    z_prior_cost = 0.5 * np.sum(np.log(2 * np.pi * z_prior_var) + (zvec - z_prior_mu) ** 2 / z_prior_var)
    
    # Get R and S matrices
    R, S = get_R_S_from_model_data(model_data, N, triu)
    Rv = R[triu]
    Sv = S[triu]
    
    # Correlation cost (R) - r_z_score: dist.Normal(R_vector, 2)
    # Model stores: deterministic("r_z_score", -r_z_score_), so cost = -log-prob (positive)
    rz_cost = 0.5 * np.sum(np.log(2 * np.pi * (2.0 ** 2)) + (zvec - Rv) ** 2 / (2.0 ** 2))
    
    # R cost with variance - FIXED: removed -0.5 offset to match model
    # Model uses: dist.Normal(R_vector, R_vector ** 2 + 1e-2)
    # Model stores: deterministic("r_score", -r_score_), so cost = -log-prob (positive)
    Rvar = np.maximum(Rv ** 2 + 1e-2, 1e-6)
    r_cost = 0.5 * np.sum(np.log(2 * np.pi * Rvar) + (zvec - Rv) ** 2 / Rvar)
    
    # SAINT score cost (S) - FIXED: removed -0.5 offset and fixed variance calculation
    # Model uses: dist.Normal(saint_vector, saint_std) where saint_std = 1.0 if SAINT < 0.01, else saint_vector ** 2 + 1e-2
    # Model stores: deterministic("s_score", -s_score_), so cost = -log-prob (positive)
    Svar = np.where(Sv < 0.01, 1.0, np.maximum(Sv ** 2 + 1e-2, 1e-6))
    s_cost = 0.5 * np.sum(np.log(2 * np.pi * Svar) + (zvec - Sv) ** 2 / Svar)
    
    # Edge count constraint - FIXED: use model_data expected_edges and expected_edges_std
    # Model uses: dist.Normal(expected_edges_scaled, expected_edges_std_scaled)
    if edges_mu is None:
        # Get from model_data (same logic as model)
        hardcoded_expected_edges = model_data.get('hardcoded_expected_edges', None)
        if hardcoded_expected_edges is not None:
            edges_mu = float(hardcoded_expected_edges)
            edges_sd = float(model_data.get('hardcoded_expected_edges_std', edges_mu * 0.33))
        else:
            edges_mu = float(model_data.get('expected_edges', 300.0))
            edges_sd = float(model_data.get('expected_edges_std', 100.0))
    
    # Cap at M * 0.95 (same as model)
    M = len(p_vec)
    edges_mu = min(edges_mu, M * 0.95)
    # Ensure std is reasonable (same as model)
    edges_sd = max(edges_sd, edges_mu * 0.15)
    edges_sd = min(edges_sd, edges_mu * 0.50)
    
    edges_var = max(edges_sd ** 2, 1e-6)
    n_edges = np.sum(p_vec)  # Sum of probabilities
    # Model stores: deterministic("n_edges_score", -n_edges_score_), so cost = -log-prob (positive)
    n_edges_cost = 0.5 * (np.log(2 * np.pi * edges_var) + (n_edges - edges_mu) ** 2 / edges_var)
    
    # Degree constraint - FIXED: match model's degree_powerlaw_logprob
    # Model uses: -gamma * np.sum(np.log(degree_safe)) where degree_safe = max(degree, 0.0) + kmin
    Aexp = np.zeros((N, N), float)
    upper = Aexp[triu]
    L = min(len(upper), len(p_vec))
    upper[:L] = p_vec[:L]
    Aexp[triu] = upper
    Aexp = Aexp + Aexp.T
    np.fill_diagonal(Aexp, 0.0)
    k = Aexp.sum(axis=1)
    
    if degree_mode == "powerlaw":
        # Model: degree_powerlaw_logprob returns -gamma * sum(log(degree + kmin))
        # Model stores: deterministic("degree_powerlaw_score", -degree_powerlaw_score_), so cost = -log-prob (positive)
        kmin = 1.0
        degree_safe = np.maximum(k, 0.0) + kmin
        degree_cost = gamma * np.sum(np.log(degree_safe))
    else:
        mu, sd = 3.0, 3.0
        var = sd ** 2
        degree_cost = 0.5 * np.sum(np.log(2 * np.pi * var) + (k - mu) ** 2 / var)
    
    total = z_prior_cost + rz_cost + r_cost + s_cost + n_edges_cost + degree_cost
    return float(total)


def permute_nodes_probs(p, rng, frac=0.1):
    """Randomly permute a fraction of nodes and shuffle probabilities accordingly."""
    q = p.copy()
    k = max(1, int(frac * len(p)))
    idx = rng.choice(len(p), size=k, replace=False)
    # Shuffle those entries
    q[idx] = rng.permutation(q[idx])
    return q


def flip_edges_random(p, rng, frac=0.1):
    """Randomly flip edges (swap high and low probabilities)."""
    q = p.copy()
    k = max(1, int(frac * len(p)))
    idx = rng.choice(len(p), size=k, replace=False)
    # Flip: swap with random other values
    swap_idx = rng.choice(len(p), size=k, replace=False)
    q[idx], q[swap_idx] = q[swap_idx], q[idx]
    return q


def add_gaussian_noise(p, rng, sigma=0.1):
    """Add Gaussian noise to probabilities."""
    q = p.copy()
    noise = rng.normal(0, sigma, size=len(q))
    q = q + noise
    q = np.clip(q, 0.0, 1.0)
    return q


def remove_edges(p, rng, frac=0.1):
    """Randomly remove a fraction of edges by setting their probabilities to 0."""
    q = p.copy()
    k = max(1, int(frac * len(p)))
    idx = rng.choice(len(p), size=k, replace=False)
    q[idx] = 0.0
    return q


def compute_accuracy_vs_true_model(p_perturbed, p_true):
    """
    Compute accuracy as similarity to true model.
    
    Uses Spearman rank correlation to measure how well
    perturbed predictions match true model predictions.
    """
    try:
        corr, _ = stats.spearmanr(p_perturbed, p_true)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def run_perturbation_analysis(
    pred,
    model_data,
    output_dir: Path,
    logger: logging.Logger,
    reference_label: str = "true model",
):
    """Run perturbation analysis and compute cost vs accuracy."""
    logger.info("=" * 60)
    logger.info("Running perturbation analysis")
    logger.info("=" * 60)
    
    # Reference vector p_base: posterior mean (merged z) or edge vector from --mean-edges-npy, etc.
    p_base = pred["posterior_prob"].to_numpy()
    
    # Compute base cost (using sum of p_base as expected edges)
    BASE_EDGES_MU = float(np.sum(p_base))
    logger.info(f"Base expected edges: {BASE_EDGES_MU:.1f}")
    
    ref_cost = soft_total_cost_from_probs(p_base, model_data, edges_mu=BASE_EDGES_MU)
    ref_accuracy = 1.0  # Spearman(self, self) = 1
    
    logger.info(
        f"Reference ({reference_label}) - Cost: {ref_cost:.2f}, Accuracy: {ref_accuracy:.4f}"
    )
    
    # Store results
    points = []
    points.append((ref_cost, ref_accuracy, reference_label))
    
    # Initialize RNG
    rng = np.random.default_rng(42)
    
    # ========================================================================
    # 1. Node Perturbation
    # ========================================================================
    logger.info("\n1. Node Perturbation")
    logger.info("-" * 60)
    node_fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    for frac in node_fracs:
        logger.info(f"  Testing node permutation: {frac*100:.0f}%")
        for trial in range(30):
            p_new = permute_nodes_probs(p_base, rng, frac=frac)
            cost = soft_total_cost_from_probs(p_new, model_data, edges_mu=BASE_EDGES_MU)
            accuracy = compute_accuracy_vs_true_model(p_new, p_base)
            points.append((cost, accuracy, f"perm {int(frac*100)}% nodes"))
    
    # ========================================================================
    # 2. Edge Flipping
    # ========================================================================
    logger.info("\n2. Edge Flipping")
    logger.info("-" * 60)
    flip_fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    for frac in flip_fracs:
        logger.info(f"  Testing edge flip: {frac*100:.0f}%")
        for trial in range(30):
            p_new = flip_edges_random(p_base, rng, frac=frac)
            cost = soft_total_cost_from_probs(p_new, model_data, edges_mu=BASE_EDGES_MU)
            accuracy = compute_accuracy_vs_true_model(p_new, p_base)
            points.append((cost, accuracy, f"flip {int(frac*100)}% edges"))
    
    # ========================================================================
    # 3. Gaussian Noise
    # ========================================================================
    logger.info("\n3. Gaussian Noise")
    logger.info("-" * 60)
    noise_levels = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    
    for sigma in noise_levels:
        logger.info(f"  Testing noise level: σ={sigma}")
        for trial in range(30):
            p_new = add_gaussian_noise(p_base, rng, sigma=sigma)
            cost = soft_total_cost_from_probs(p_new, model_data, edges_mu=BASE_EDGES_MU)
            accuracy = compute_accuracy_vs_true_model(p_new, p_base)
            points.append((cost, accuracy, f"noise σ={sigma}"))
    
    # ========================================================================
    # 4. Edge Removal
    # ========================================================================
    logger.info("\n4. Edge Removal")
    logger.info("-" * 60)
    removal_fracs = [0.05, 0.10, 0.20, 0.30, 0.50]
    
    for frac in removal_fracs:
        logger.info(f"  Testing edge removal: {frac*100:.0f}%")
        for trial in range(30):
            p_new = remove_edges(p_base, rng, frac=frac)
            cost = soft_total_cost_from_probs(p_new, model_data, edges_mu=BASE_EDGES_MU)
            accuracy = compute_accuracy_vs_true_model(p_new, p_base)
            points.append((cost, accuracy, f"remove {int(frac*100)}% edges"))
    
    # Save results
    results_df = pd.DataFrame(points, columns=['cost', 'accuracy', 'label'])
    results_file = output_dir / "perturbation_results.tsv"
    results_df.to_csv(results_file, sep='\t', index=False)
    logger.info(f"\nSaved results to {results_file}")
    
    # Compute correlation
    xs = np.array([a for (_, a, _) in points], float)  # Accuracy (X-axis)
    ys = np.array([c for (c, _, _) in points], float)  # Cost (Y-axis)
    rho, pval = stats.spearmanr(xs, ys, nan_policy="omit")
    logger.info(f"\nSpearman(Accuracy, Cost) = {rho:.3f} (p={pval:.2e})")
    
    # Generate plot
    if HAS_MATPLOTLIB:
        plot_cost_vs_accuracy(points, output_dir, logger, rho, pval, reference_label)
    
    return results_df


def plot_cost_vs_accuracy(
    points,
    output_dir: Path,
    logger: logging.Logger,
    rho: float,
    pval: float,
    reference_label: str = "true model",
):
    """Plot cost vs accuracy with separate subplots for each perturbation type."""
    logger.info("Generating plot...")
    
    # Helper function to extract numeric fraction from label for sorting
    def extract_fraction(label):
        """Extract numeric fraction from label like 'perm 5% nodes' -> 5"""
        match = re.search(r'(\d+)%', label)
        return int(match.group(1)) if match else 0
    
    # Separate points by perturbation type
    ref_points = [(c, a) for (c, a, l) in points if l == reference_label]
    perm_points = [(c, a, l) for (c, a, l) in points if "perm" in l]
    flip_points = [(c, a, l) for (c, a, l) in points if "flip" in l]
    noise_points = [(c, a, l) for (c, a, l) in points if "noise" in l]
    remove_points = [(c, a, l) for (c, a, l) in points if "remove" in l]
    
    # Create figure with subplots - 2x2 layout for 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Font sizes for multipanel visibility
    label_fontsize = 18
    tick_fontsize = 16
    legend_fontsize = 16
    title_fontsize = 24  # panel titles: Node Permutation, Edge Flipping, etc.
    
    # Publication-quality colors (accessible, distinct)
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    
    # Plot 1: Node Permutation
    ax = axes[0]
    if ref_points:
        true_x, true_y = ref_points[0][1], ref_points[0][0]
        ax.scatter(true_x, true_y, s=400, marker="*", color="black",
                  edgecolor="black", linewidths=2, zorder=10, label=reference_label)
    
    # Group perm points by fraction
    perm_groups = {}
    for c, a, l in perm_points:
        frac = l.split()[1]  # Extract fraction from label
        if frac not in perm_groups:
            perm_groups[frac] = []
        perm_groups[frac].append((c, a))
    
    # Sort by numeric fraction value (5%, 10%, 20%, 30%, 50%)
    sorted_fracs = sorted(perm_groups.items(), key=lambda x: extract_fraction(x[0]))
    for i, (frac, group_points) in enumerate(sorted_fracs):
        xs = [a for (c, a) in group_points]
        ys = [c for (c, a) in group_points]
        ax.scatter(xs, ys, s=60, alpha=0.7, color=colors[i % len(colors)], 
                  edgecolors='none', label=f"{frac} permuted")
    
    ax.set_xlabel("Accuracy", fontweight="bold", fontsize=label_fontsize)
    ax.set_ylabel("Score", fontweight="bold", fontsize=label_fontsize)
    ax.set_title("Node Permutation", fontweight="bold", fontsize=title_fontsize, pad=10)
    # Format Y-axis to use scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=2, length=6)
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Also make scientific notation bold if present
    ax.yaxis.get_offset_text().set_fontweight('bold')
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    ax.legend(frameon=True, loc='upper right', prop={'weight': 'bold', 'size': legend_fontsize})
    
    # Plot 2: Edge Flipping
    ax = axes[1]
    if ref_points:
        true_x, true_y = ref_points[0][1], ref_points[0][0]
        ax.scatter(true_x, true_y, s=400, marker="*", color="black",
                  edgecolor="black", linewidths=2, zorder=10, label=reference_label)
    
    # Group flip points by fraction
    flip_groups = {}
    for c, a, l in flip_points:
        frac = l.split()[1]  # Extract fraction from label
        if frac not in flip_groups:
            flip_groups[frac] = []
        flip_groups[frac].append((c, a))
    
    # Sort by numeric fraction value (5%, 10%, 20%, 30%, 50%)
    sorted_fracs = sorted(flip_groups.items(), key=lambda x: extract_fraction(x[0]))
    for i, (frac, group_points) in enumerate(sorted_fracs):
        xs = [a for (c, a) in group_points]
        ys = [c for (c, a) in group_points]
        ax.scatter(xs, ys, s=60, alpha=0.7, color=colors[i % len(colors)], 
                  edgecolors='none', label=f"{frac} flipped")
    
    ax.set_xlabel("Accuracy", fontweight="bold", fontsize=label_fontsize)
    ax.set_ylabel("Score", fontweight="bold", fontsize=label_fontsize)
    ax.set_title("Edge Flipping", fontweight="bold", fontsize=title_fontsize, pad=10)
    # Format Y-axis to use scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=2, length=6)
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Also make scientific notation bold if present
    ax.yaxis.get_offset_text().set_fontweight('bold')
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    ax.legend(frameon=True, loc='upper right', prop={'weight': 'bold', 'size': legend_fontsize})
    
    # Plot 3: Gaussian Noise
    ax = axes[2]
    if ref_points:
        true_x, true_y = ref_points[0][1], ref_points[0][0]
        ax.scatter(true_x, true_y, s=400, marker="*", color="black",
                  edgecolor="black", linewidths=2, zorder=10, label=reference_label)
    
    # Group noise points by sigma
    noise_groups = {}
    for c, a, l in noise_points:
        sigma = l.split("σ=")[1] if "σ=" in l else l.split()[-1]  # Extract sigma
        if sigma not in noise_groups:
            noise_groups[sigma] = []
        noise_groups[sigma].append((c, a))
    
    # Sort by numeric sigma value
    sorted_sigmas = sorted(noise_groups.items(), key=lambda x: float(x[0]))
    for i, (sigma, group_points) in enumerate(sorted_sigmas):
        xs = [a for (c, a) in group_points]
        ys = [c for (c, a) in group_points]
        ax.scatter(xs, ys, s=60, alpha=0.7, color=colors[i % len(colors)], 
                  edgecolors='none', label=f"σ={sigma}")
    
    ax.set_xlabel("Accuracy", fontweight="bold", fontsize=label_fontsize)
    ax.set_ylabel("Score", fontweight="bold", fontsize=label_fontsize)
    ax.set_title("Gaussian Noise", fontweight="bold", fontsize=title_fontsize, pad=10)
    # Format Y-axis to use scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=2, length=6)
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Also make scientific notation bold if present
    ax.yaxis.get_offset_text().set_fontweight('bold')
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    ax.legend(frameon=True, loc='upper right', prop={'weight': 'bold', 'size': legend_fontsize})
    
    # Plot 4: Edge Removal
    ax = axes[3]
    if ref_points:
        true_x, true_y = ref_points[0][1], ref_points[0][0]
        ax.scatter(true_x, true_y, s=400, marker="*", color="black",
                  edgecolor="black", linewidths=2, zorder=10, label=reference_label)
    
    # Group remove points by fraction
    remove_groups = {}
    for c, a, l in remove_points:
        frac = l.split()[1]  # Extract fraction from label
        if frac not in remove_groups:
            remove_groups[frac] = []
        remove_groups[frac].append((c, a))
    
    # Sort by numeric fraction value (5%, 10%, 20%, 30%, 50%)
    sorted_fracs = sorted(remove_groups.items(), key=lambda x: extract_fraction(x[0]))
    for i, (frac, group_points) in enumerate(sorted_fracs):
        xs = [a for (c, a) in group_points]
        ys = [c for (c, a) in group_points]
        ax.scatter(xs, ys, s=60, alpha=0.7, color=colors[i % len(colors)], 
                  edgecolors='none', label=f"{frac} removed")
    
    ax.set_xlabel("Accuracy", fontweight="bold", fontsize=label_fontsize)
    ax.set_ylabel("Score", fontweight="bold", fontsize=label_fontsize)
    ax.set_title("Edge Removal", fontweight="bold", fontsize=title_fontsize, pad=10)
    # Format Y-axis to use scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=2, length=6)
    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # Also make scientific notation bold if present
    ax.yaxis.get_offset_text().set_fontweight('bold')
    ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(False)
    ax.legend(frameon=True, loc='upper right', prop={'weight': 'bold', 'size': legend_fontsize})
    
    # No overall title - cleaner for panels
    plt.tight_layout()
    plot_file = output_dir / "cost_vs_accuracy.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved plot to {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Computational validation via model cost and perturbations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=Path, required=True,
                       help='Path to results directory')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., cullin_mock_config_multi_seeds)')
    parser.add_argument('--pdb-results', type=Path, required=False,
                       help='Path to PDB validation results directory (not used, kept for compatibility)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument(
        '--mean-edges-npy',
        type=Path,
        default=None,
        help='Optional path to mean_edges_cluster_*.npy; skips merged-z mean',
    )
    parser.add_argument(
        '--binary-m-mean-full-merged',
        action='store_true',
        help=(
            "Use mean per-draw binary M (Z > Z_thr from post_sampling_z_binary) over all merged draws "
            "as edge score [0,1] instead of mean(sigmoid(Z)). Incompatible with --mean-edges-npy."
        ),
    )
    parser.add_argument(
        '--sigmoid-mean-z-main-cluster',
        action='store_true',
        help=(
            "Use σ(mean z) over merged draws assigned to the largest τ* cluster only (not mean(σ(z))). "
            "Requires --main-cluster-summary-json pointing to main_cluster_summary.json "
            "(labels_source + main_cluster_id in that JSON)."
        ),
    )
    parser.add_argument(
        '--main-cluster-summary-json',
        type=Path,
        default=None,
        help='With --sigmoid-mean-z-main-cluster: path to main_cluster_summary.json.',
    )
    parser.add_argument(
        '--reference-label',
        type=str,
        default=None,
        help='Legend/tag for reference point (default: true model)',
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output)
    
    logger.info("=" * 60)
    logger.info("Model Cost Validation via Perturbations")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Output dir: {args.output}")
    if args.mean_edges_npy:
        logger.info(f"Mean edges npy: {args.mean_edges_npy}")
    if args.binary_m_mean_full_merged:
        logger.info("Reference scores: mean per-draw binary M over full merged draws (post_sampling_z_binary)")
    if args.sigmoid_mean_z_main_cluster:
        logger.info("Reference scores: σ(mean z) over draws in main τ* cluster (from summary JSON)")
    logger.info("")

    ref_modes = [
        args.mean_edges_npy is not None,
        args.binary_m_mean_full_merged,
        args.sigmoid_mean_z_main_cluster,
    ]
    if sum(ref_modes) > 1:
        logger.error(
            "Use at most one of --mean-edges-npy, --binary-m-mean-full-merged, "
            "--sigmoid-mean-z-main-cluster."
        )
        return 1

    if args.mean_edges_npy is not None:
        if not args.mean_edges_npy.is_file():
            logger.error(f"Mean edges file not found: {args.mean_edges_npy}")
            return 1
        pred, model_data = load_predictions_from_mean_edges(
            args.mean_edges_npy.resolve(),
            args.results_dir,
            args.dataset,
            logger,
        )
    elif args.binary_m_mean_full_merged:
        pred, model_data = load_predictions_from_binary_m_mean_merged(
            args.results_dir,
            args.dataset,
            logger,
        )
    elif args.sigmoid_mean_z_main_cluster:
        if args.main_cluster_summary_json is None:
            logger.error(
                "--sigmoid-mean-z-main-cluster requires --main-cluster-summary-json "
                "(path to main_cluster_summary.json)."
            )
            return 1
        if not args.main_cluster_summary_json.is_file():
            logger.error(f"Main cluster summary not found: {args.main_cluster_summary_json}")
            return 1
        pred, model_data = load_predictions_sigmoid_of_mean_z_main_cluster(
            args.results_dir,
            args.dataset,
            args.main_cluster_summary_json.resolve(),
            logger,
        )
    else:
        pred, model_data = load_predictions_and_labels(
            args.results_dir,
            args.dataset,
            args.pdb_results if args.pdb_results else Path("."),
            logger,
        )

    if args.reference_label is not None:
        ref_label = args.reference_label
    else:
        ref_label = "true model"
    
    if pred.empty:
        logger.error("Failed to load predictions. Exiting.")
        return 1
    
    results = run_perturbation_analysis(
        pred, model_data, args.output, logger, reference_label=ref_label
    )
    
    logger.info("=" * 60)
    logger.info("Validation complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
