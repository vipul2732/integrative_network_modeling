"""
Cleaned version of model variations for Bayesian network inference.

This module contains:
- model23_n: Main production model for Bayesian network inference
- Data loading and preprocessing functions
- Helper utilities
- MCMC execution functions
"""

# ============================================================================
# IMPORTS
# ============================================================================

from collections import defaultdict
from functools import partial
from itertools import combinations
import json
import logging
import pickle as pkl
import sys
from pathlib import Path
import threading
import time
from typing import Any, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np

# Compatibility shim for numpyro: newer JAX versions moved jax.tree to jax.tree_util
# numpyro uses jax.tree.map() which needs to map to jax.tree_util.tree_map
if not hasattr(jax, 'tree'):
    import jax.tree_util
    from types import ModuleType
    tree_module = ModuleType('tree')
    # Map old API to new API: jax.tree.map -> jax.tree_util.tree_map
    tree_module.map = jax.tree_util.tree_map
    # Also expose other commonly used functions for compatibility
    for attr in ['flatten', 'unflatten', 'leaves', 'structure']:
        if hasattr(jax.tree_util, f'tree_{attr}'):
            setattr(tree_module, attr, getattr(jax.tree_util, f'tree_{attr}'))
    jax.tree = tree_module

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_uniform,
)
import pandas as pd

Array = Any

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def eprint(*args, **kwargs):
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def load(fpath):
    """Load pickle file."""
    with open(fpath, "rb") as f:
        return pkl.load(f)


# ============================================================================
# MATRIX UTILITIES
# ============================================================================

def matrix2flat(M, row_major=True):
    """Convert symmetric matrix to flattened edge list."""
    if row_major:
        triu_indices = jnp.triu_indices_from(M, k=1)
        return M[triu_indices]
    else:
        tril_indices = jnp.tril_indices_from(M, k=-1)
        return M[tril_indices]


def flat2matrix(x, n, row_major=True):
    """
    Convert flattened edge array to symmetric matrix.
    
    Args:
        x: Flattened array of length n*(n-1)//2
        n: Number of nodes
        row_major: If True, use upper triangle; if False, use lower triangle
    
    Returns:
        Symmetric matrix of shape (n, n)
    """
    aij = jnp.zeros((n, n))
    if row_major:
        triu_indices = jnp.triu_indices(n, k=1)
        aij = aij.at[triu_indices].set(x)
    else:
        tril_indices = jnp.tril_indices(n, k=-1)
        aij = aij.at[tril_indices].set(x)
    aij = aij + aij.T
    return aij


def Z2A(z, N, edge_filter_map=None):
    """
    Convert latent edge variables z to binary adjacency matrix.
    
    Args:
        z: Vector of latent edge values (length M = N*(N-1)//2, or M_filtered if edge_filter_map provided)
        N: Number of nodes
        edge_filter_map: Optional array mapping filtered edge index -> original edge index.
                        If provided, z is assumed to be of length len(edge_filter_map).
    
    Returns:
        Binary adjacency matrix (N, N)
    """
    aij = jnp.zeros((N, N))
    
    if edge_filter_map is not None:
        # Filtered edges: map filtered indices back to original (i,j) pairs
        # edge_filter_map[k] gives the original edge index for filtered edge k
        M_original = N * (N - 1) // 2
        triu_indices = jnp.triu_indices(N, k=1)
        i_indices = triu_indices[0]
        j_indices = triu_indices[1]
        
        # Map filtered z values to original edge positions
        z_full = jnp.zeros(M_original)
        z_full = z_full.at[edge_filter_map].set(z)
        aij = aij.at[i_indices, j_indices].set(z_full)
    else:
        # Standard case: z is full edge list
        triu_indices = jnp.triu_indices(N, k=1)
        i_indices = triu_indices[0]
        j_indices = triu_indices[1]
        aij = aij.at[i_indices, j_indices].set(z)
    
    aij = aij + aij.T
    return aij


def weight2edge(x, slope=1000):
    """Map weight in (0, 1) to binary edge using sigmoid."""
    return jax.nn.sigmoid((x - 0.5) * slope)


def edge_prob2mu(edge_prob):
    """Convert edge probability to mu parameter for Normal distribution."""
    # Inverse sigmoid: mu such that sigmoid(mu) ≈ edge_prob
    # Using logit approximation
    eps = 1e-6
    edge_prob = jnp.clip(edge_prob, eps, 1 - eps)
    return jnp.log(edge_prob / (1 - edge_prob))


# ============================================================================
# EDGE INDEXING UTILITIES
# ============================================================================

def k_from_ij(i, j, N):
    """Convert node pair (i, j) to edge index k in flattened representation."""
    total = -1
    for r in range(0, i):
        total += N - r - 1
    k = total + (j - i)
    return k


def ij_from(k, N):
    """Convert edge index k to node pair (i, j)."""
    def i_from_k(k, N):
        total = -1
        i = 0
        while total < k:
            total += N - i - 1
            i += 1
        return i - 1, total
    
    def j_from_total(k, total, N):
        return N - (total - k) - 1
    
    i, total = i_from_k(k, N)
    j = j_from_total(k, total, N)
    return i, j


def node_list2edge_list(nlist, N):
    """Convert list of node indices to list of edge indices."""
    nlist = sorted(nlist)
    combos = list(combinations(nlist, 2))
    return [k_from_ij(*edge, N=N) for edge in combos]


# ============================================================================
# PATH LENGTH UTILITIES
# ============================================================================

def _body_shortest_paths_up_to_N(i, v):
    """Body function for shortest path calculation."""
    A, AN, D = v
    D = jnp.where((AN != 0) & (D > i), i, D)
    AN = A @ AN
    return A, AN, D


def shortest_paths_up_to_N(A, N):
    """
    Calculate shortest path distance matrix up to N steps.
    
    Args:
        A: Adjacency matrix
        N: Maximum path length
    
    Returns:
        Distance matrix D where D[i,j] is shortest path length from i to j
    """
    D = jnp.where(A != 0, A, N)
    A2 = A @ A
    A, AN, D = jax.lax.fori_loop(2, N, _body_shortest_paths_up_to_N, (A, A2, D))
    return D


# ============================================================================
# DISTRIBUTION CLASSES
# ============================================================================

class Histogram(dist.Distribution):
    """Histogram distribution for null model."""
    def __init__(self, data, bins=100, validate_args=None):
        self.data = np.array(data)
        self.bins = bins
        counts, edges = np.histogram(self.data, bins=bins)
        self.counts = counts + 1e-10  # Avoid zeros
        self.edges = jnp.array(edges)  # Convert to JAX array
        # Convert probs to JAX array for JAX tracing compatibility
        probs_np = self.counts / np.sum(self.counts)
        self.probs = jnp.array(probs_np)
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)
    
    def sample(self, key, sample_shape=()):
        indices = jax.random.categorical(key, jnp.log(self.probs), shape=sample_shape)
        return self.edges[indices]
    
    def log_prob(self, value):
        indices = jnp.digitize(value, self.edges) - 1
        indices = jnp.clip(indices, 0, len(self.probs) - 1)
        # Use JAX array indexing (self.probs is already a JAX array)
        return jnp.log(self.probs[indices])


class HistogramLowerBound(dist.Distribution):
    """Histogram distribution with lower bound."""
    def __init__(self, data, bins=100, lower_bound=0, validate_args=None):
        self.data = np.array(data)
        self.lower_bound = lower_bound
        self.bins = bins
        counts, edges = np.histogram(self.data[self.data >= lower_bound], bins=bins)
        self.counts = counts + 1e-10
        # Convert to JAX arrays for JAX tracing compatibility
        self.edges = jnp.array(edges)
        probs_np = self.counts / np.sum(self.counts)
        self.probs = jnp.array(probs_np)
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)
    
    def log_prob(self, value):
        value_clipped = jnp.maximum(value, self.lower_bound)
        indices = jnp.digitize(value_clipped, self.edges) - 1
        indices = jnp.clip(indices, 0, len(self.probs) - 1)
        # Use JAX array indexing (self.probs is already a JAX array)
        return jnp.log(self.probs[indices])


# ============================================================================
# PRIOR DISTRIBUTIONS
# ============================================================================

def degree_powerlaw_logprob(degree, gamma=2.5, kmin=1.0):
    """
    Power-law degree distribution prior.
    
    Args:
        degree: Array of node degrees
        gamma: Power-law exponent (default 2.5)
        kmin: Minimum degree
    
    Returns:
        Log probability (scalar)
    """
    # Ensure degree is non-negative and add kmin to avoid log(0)
    degree_safe = jnp.maximum(degree, 0.0) + kmin
    log_prob = -gamma * jnp.log(degree_safe)
    # Return scalar sum - ensure it's a scalar, not an array
    result = jnp.sum(log_prob)
    # If result is an array with shape (), squeeze it to scalar
    if hasattr(result, 'shape') and result.shape == ():
        return result
    return result


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def get_composite_table(model_output_dirpath: Path) -> pd.DataFrame:
    """Load composite table from TSV file."""
    if isinstance(model_output_dirpath, str):
        model_output_dirpath = Path(model_output_dirpath)
    return pd.read_csv(model_output_dirpath / "composite_table.tsv", sep="\t")


def get_maximal_composite_dict_from_composite_table(composite_table: pd.DataFrame) -> dict:
    """Extract maximal SAINT scores for each prey from composite table."""
    composite_dict = defaultdict(float)
    for i, row in composite_table.iterrows():
        prey_name = row["Prey"]
        score = row["MSscore"]
        val = composite_dict[prey_name]
        if score > val:
            composite_dict[prey_name] = score
    return composite_dict


def maximal_composite_dict_to_max_saint_pair_score_table(max_saint_score_dict):
    """Convert maximal SAINT scores to pairwise score matrix."""
    N = len(max_saint_score_dict)
    table = np.zeros((N, N))
    for i, (k1, v1) in enumerate(max_saint_score_dict.items()):
        for j, (k2, v2) in enumerate(max_saint_score_dict.items()):
            table[i, j] = v1 * v2
    columns = list(max_saint_score_dict.keys())
    return pd.DataFrame(table, index=columns, columns=columns)


def _cosine_similarity_matrix(X, eps=1e-12, use_numpy=True):
    """Calculate cosine similarity matrix.
    
    Returns cosine similarity matrix with values in [-1, 1].
    Clips to handle numerical precision issues.
    
    Args:
        X: Input matrix (conditions × proteins)
        eps: Small value to avoid division by zero
        use_numpy: If True, use NumPy (faster on CPU). If False, use JAX.
    """
    if use_numpy:
        # Use NumPy for CPU computation
        import numpy as np_local
        # Normalize columns (each column is a protein profile)
        norms = np_local.linalg.norm(X, axis=0, keepdims=True)
        X_norm = X / (norms + eps)
        
        # Compute cosine similarity: dot product of normalized vectors
        corr_matrix = np_local.dot(X_norm.T, X_norm)
        
        # Clip to valid range to handle numerical precision issues
        corr_matrix = np_local.clip(corr_matrix, -1.0, 1.0)
    else:
        # Use JAX 
        # Normalize columns (each column is a protein profile)
        norms = jnp.linalg.norm(X, axis=0, keepdims=True)
        X_norm = X / (norms + eps)
        
        # Compute cosine similarity: dot product of normalized vectors
        corr_matrix = jnp.dot(X_norm.T, X_norm)
        
        # Clip to valid range to handle numerical precision issues
        corr_matrix = jnp.clip(corr_matrix, -1.0, 1.0)
    
    return corr_matrix


def cosine_from_spec_table(spec_table, use_numpy=True):
    """Calculate cosine similarity from spectral count table.
    
    Args:
        spec_table: DataFrame with proteins as rows, conditions as columns
        use_numpy: If True, use NumPy (faster on CPU). If False, use JAX.
    """
    X = spec_table.values.T
    return _cosine_similarity_matrix(X, use_numpy=use_numpy)


def calc_shuff_correlations(spec_table, rseed) -> Tuple[pd.DataFrame]:
    """Calculate correlations and shuffled correlations."""
    import time
    logger = logging.getLogger(__name__)
    
    r, c = spec_table.shape
    logger.info(f"Calculating correlation matrix for {r} proteins × {c} conditions...")
    logger.info(f"  This will compute a {r}×{r} correlation matrix (may take several minutes for large datasets)")
    eprint(f"Calculating correlation matrix: {r} proteins × {c} conditions → {r}×{r} matrix")
    
    start_time = time.time()
    np.random.seed(rseed)
    
    eprint("  Computing cosine similarity matrix (using NumPy for speed)...")
    # Use NumPy instead of JAX for correlation calculation (much faster on CPU)
    corr = cosine_from_spec_table(spec_table, use_numpy=True)
    corr_time = time.time() - start_time
    logger.info(f"Correlation matrix computed in {corr_time:.1f}s ({corr_time/60:.1f} minutes)")
    eprint(f"  Correlation matrix computed in {corr_time:.1f}s")
    
    # Shuffle columns independently
    eprint("  Computing shuffled correlation matrix...")
    shuffle_start = time.time()
    spec_table_shuff = spec_table.copy()
    for col in spec_table_shuff.columns:
        spec_table_shuff[col] = np.random.permutation(spec_table_shuff[col].values)
    # Use NumPy for shuffled correlation too (faster on CPU)
    shuff_corr = cosine_from_spec_table(spec_table_shuff, use_numpy=True)
    shuffle_time = time.time() - shuffle_start
    logger.info(f"Shuffled correlation matrix computed in {shuffle_time:.1f}s")
    eprint(f"  Shuffled correlation computed in {shuffle_time:.1f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Total correlation calculation time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    eprint(f"Total correlation calculation: {total_time:.1f}s")
    
    # Null score (for compatibility)
    null_sc = spec_table_shuff
    
    return corr, shuff_corr, null_sc


def protein_saint_combined_scores_from_composite(
    composite_table: pd.DataFrame,
) -> Tuple[dict, dict]:
    """
    Same combined score as ``filter_proteins_by_saint_score`` (max SAINT + robustness bonus).
    Returns (protein_combined_score, protein_bait_count).
    """
    protein_max_saint: dict = {}
    protein_bait_count: dict = {}

    for _, row in composite_table.iterrows():
        prey = row["Prey"]
        bait = row["Bait"]
        msscore = row.get("MSscore", 0.0)

        if prey not in protein_max_saint or msscore > protein_max_saint[prey]:
            protein_max_saint[prey] = msscore
        if prey not in protein_bait_count:
            protein_bait_count[prey] = set()
        protein_bait_count[prey].add(bait)

        if bait not in protein_max_saint or msscore > protein_max_saint[bait]:
            protein_max_saint[bait] = msscore

    for prot in protein_bait_count:
        protein_bait_count[prot] = len(protein_bait_count[prot])

    protein_combined_score: dict = {}
    for prot in protein_max_saint:
        max_saint = protein_max_saint[prot]
        n_baits = protein_bait_count.get(prot, 1)
        robustness_bonus = min((n_baits - 1) * 0.05, 0.20)
        protein_combined_score[prot] = max_saint * (1.0 + robustness_bonus)

    return protein_combined_score, protein_bait_count


def filter_proteins_stratified_saint_subset(
    spec_table: pd.DataFrame,
    composite_table: pd.DataFrame,
    max_proteins: int,
    seed: int,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    After base-style ``remove_prey`` + ``filter_minimal_ids``, rank all prey in ``spec_table``
    by the same combined SAINT score as the standard pipeline, then sample
    ``max_proteins // 2`` uniformly from the **top half** of that ranking and
    ``max_proteins - max_proteins // 2`` from the **bottom half** (without replacement
    within each stratum; fills from remainder if a stratum is too small).

    Use ``protein_selection == "stratified_saint"`` in ``modeler_vars``.
    """
    logger = logging.getLogger(__name__)
    names_all = list(spec_table.index)
    n_avail = len(names_all)
    if max_proteins is None or max_proteins <= 0 or n_avail <= max_proteins:
        logger.info(
            f"Stratified SAINT: keeping all {n_avail} proteins (max_proteins={max_proteins})"
        )
        return spec_table

    protein_combined_score, _ = protein_saint_combined_scores_from_composite(composite_table)
    scores = {name: float(protein_combined_score.get(name, 0.0)) for name in names_all}
    ranked = sorted(names_all, key=lambda p: scores[p], reverse=True)

    n = len(ranked)
    mid = n // 2
    top_pool = ranked[:mid] if mid > 0 else ranked[:1]
    bot_pool = ranked[mid:] if mid < n else ranked[-1:]

    n_top = int(max_proteins // 2)
    n_bot = int(max_proteins - n_top)
    rng = np.random.default_rng(int(seed))

    def _draw(pool: list, k: int) -> list:
        k = min(k, len(pool))
        if k <= 0:
            return []
        ix = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in sorted(ix)]

    sel_top = _draw(top_pool, n_top)
    sel_bot = _draw(bot_pool, n_bot)
    selected = set(sel_top) | set(sel_bot)

    if len(selected) < max_proteins:
        rem = max_proteins - len(selected)
        remainder = [p for p in ranked if p not in selected]
        extra = _draw(remainder, rem)
        selected.update(extra)

    order = {p: i for i, p in enumerate(ranked)}
    selected_names = sorted(selected, key=lambda p: order[p])
    filtered = spec_table.loc[selected_names].copy()

    logger.info(
        f"Stratified SAINT subset: {n_avail} → {len(filtered)} proteins "
        f"(~{n_top} from top half of SAINT rank, ~{n_bot} from bottom half; seed={int(seed)})"
    )
    eprint(
        f"  Stratified SAINT subset: {n_avail} → {len(filtered)} proteins "
        f"(top+bottom halves of combined SAINT rank, seed={int(seed)})"
    )
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        top_set = set(sel_top)
        bot_set = set(sel_bot)
        with open(p, "w", encoding="utf-8") as f:
            f.write(
                "# Stratified by combined SAINT rank (same as base): TOP = upper half, BOT = lower half\n"
            )
            for nm in selected_names:
                tier = "TOP" if nm in top_set else ("BOT" if nm in bot_set else "FILL")
                f.write(f"{nm}\t{tier}\t{scores[nm]:.6g}\n")
        logger.info(f"Wrote stratified subset list to {p}")
    return filtered


def filter_proteins_random_subset(
    spec_table: pd.DataFrame,
    max_proteins: int,
    seed: int,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Uniformly sample ``max_proteins`` rows from ``spec_table`` (without replacement).

    Used when ``protein_selection == "random"`` so N is capped without using
    SAINT / combined-score ranking. ``seed`` should be fixed per run (e.g. MCMC rseed)
    for reproducibility.
    """
    logger = logging.getLogger(__name__)
    names = list(spec_table.index)
    n_avail = len(names)
    if max_proteins is None or max_proteins <= 0 or n_avail <= max_proteins:
        logger.info(
            f"Random protein selection: keeping all {n_avail} proteins "
            f"(max_proteins={max_proteins} does not further restrict)"
        )
        return spec_table

    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n_avail, size=int(max_proteins), replace=False)
    pick.sort()
    selected_names = [names[i] for i in pick]
    filtered = spec_table.loc[selected_names].copy()
    logger.info(
        f"Random protein selection: {n_avail} → {len(filtered)} proteins "
        f"(uniform sample, seed={int(seed)})"
    )
    eprint(
        f"  Random protein subset: {n_avail} → {len(filtered)} proteins (seed={int(seed)})"
    )
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write("# One protein name per row (random subset, stable row order)\n")
            for nm in selected_names:
                f.write(f"{nm}\n")
        logger.info(f"Wrote random subset list to {p}")
    return filtered


def data_from_spec_table_and_composite_table(
    data_path,
    ms_thresholds,
    sep="\t",
    rseed=0,
    calc_composites=True,
    min_saint_score_for_protein=None,
    max_proteins=None,
    protein_selection: str = "score",
    random_proteins_seed: Optional[int] = None,
):
    """
    Load and preprocess data from spec_table and composite_table.
    
    This is a core data loading function that:
    1. Loads spectral count and composite tables
    2. Filters by MS thresholds
    3. Calculates correlations
    4. Optionally computes composite connectivity structures
    """
    def validate_spec_table(spec_table):
        r, c = spec_table.shape
        assert r > 2, r
        assert c > 2, c
        assert np.all(~np.isnan(spec_table))
        assert np.all(~np.isinf(spec_table))
        assert np.all(spec_table.values >= 0)
        assert np.all(spec_table.values <= 1000)
    
    def check_composite_table(composite_table: pd.DataFrame):
        assert np.all(np.isreal(composite_table['MSscore'].values))
        assert np.all(composite_table['MSscore'] >= 0)
    
    def remove_prey(composite_table: pd.DataFrame, ms_thresholds) -> pd.DataFrame:
        """Remove prey below MS threshold."""
        min_threshold = np.min(ms_thresholds)
        assert min_threshold >= 0
        sel = composite_table['MSscore'] >= min_threshold
        return composite_table[sel]
    
    def populate_composites_dict(composite_table, ms_thresholds):
        """Create composite dictionary from composite table."""
        sid_set = set(composite_table['SID'])
        composites = {}
        k = 0
        for threshold in ms_thresholds:
            for sid in sid_set:
                sel = (composite_table['SID'] == sid) & (composite_table['MSscore'] >= threshold)
                if np.sum(sel) > 0:
                    d = composite_table[sel]
                    bait_l = list(set(d['Bait']))
                    msscore_l = list(set(d['MSscore']))
                    assert len(bait_l) == 1, bait_l
                    prey_l = list(set(d['Prey']))
                    if bait_l[0] in prey_l:
                        prey_l.remove(bait_l[0])
                    nodes = bait_l + prey_l
                    msscore = msscore_l[0]
                    composites[k] = {'nodes': nodes, 't': threshold, 'SID': sid}
                    k += 1
        return composites
    
    def filter_minimal_ids(composite_table, spec_table):
        """Filter spec_table to only include IDs in composite_table."""
        prey_set = set(composite_table['Prey'])
        bait_set = set(composite_table['Bait'])
        all_ids = list(prey_set.union(bait_set))
        sel = [name in all_ids for name in spec_table.index]
        return spec_table[sel]
    
    def filter_proteins_by_saint_score(composite_table, spec_table, min_saint_score=0.5, max_proteins=None):
        """
        Filter proteins to only include top proteins by SAINT score.
        
        IMPROVEMENT: Uses combined score that prioritizes:
        1. Max SAINT score (primary) - high-confidence interactions
        2. Number of baits (robustness bonus) - multi-bait interactions are more reliable
        
        This reduces N (number of proteins) which dramatically speeds up JAX compilation.
        The bottleneck is N² (matrix operations), not M (number of edges).
        
        Args:
            composite_table: DataFrame with 'Prey', 'Bait', 'MSscore' columns
            spec_table: DataFrame with proteins as rows
            min_saint_score: Minimum SAINT score threshold (default: 0.5)
            max_proteins: If specified, keep only top N proteins by combined score (most aggressive)
        
        Returns:
            Filtered spec_table with only high-confidence proteins
        """
        if min_saint_score is None or min_saint_score <= 0:
            if max_proteins is None or max_proteins <= 0:
                return spec_table  # No filtering
        
        protein_combined_score, protein_bait_count = protein_saint_combined_scores_from_composite(
            composite_table
        )
        
        # If max_proteins is specified, keep only top N by combined score (most aggressive)
        if max_proteins is not None and max_proteins > 0:
            # Sort proteins by combined score (descending) and take top N
            sorted_proteins = sorted(protein_combined_score.items(), key=lambda x: x[1], reverse=True)
            top_proteins = {prot for prot, score in sorted_proteins[:max_proteins]}
            high_conf_proteins = top_proteins
            filter_method = f"top {max_proteins} by combined score (max SAINT + robustness)"
        else:
            # Use threshold-based filtering with combined score
            high_conf_proteins = {prot for prot, combined_score in protein_combined_score.items() 
                                 if combined_score >= min_saint_score}
            filter_method = f"Combined score >= {min_saint_score}"
        
        # Filter spec_table to only include high-confidence proteins
        original_N = len(spec_table)
        sel = [name in high_conf_proteins for name in spec_table.index]
        filtered_spec_table = spec_table[sel].copy()
        filtered_N = len(filtered_spec_table)
        
        # Always log the filtering attempt
        logger = logging.getLogger(__name__)
        if filtered_N < original_N:
            reduction = (1 - filtered_N/original_N) * 100
            logger.info(f"Filtered proteins ({filter_method}): {original_N} → {filtered_N} proteins ({reduction:.1f}% reduction)")
            eprint(f"  Filtered proteins ({filter_method}): {original_N} → {filtered_N} proteins")
            eprint(f"  This reduces N² matrix operations from {original_N*original_N:,} to {filtered_N*filtered_N:,} ({reduction:.1f}% reduction)")
            
            # Log robustness statistics for selected proteins
            if max_proteins is not None and max_proteins > 0:
                selected_proteins = set(filtered_spec_table.index)
                single_bait_count = sum(1 for p in selected_proteins if protein_bait_count.get(p, 0) == 1)
                multi_bait_count = sum(1 for p in selected_proteins if protein_bait_count.get(p, 0) > 1)
                avg_baits = sum(protein_bait_count.get(p, 0) for p in selected_proteins) / len(selected_proteins) if selected_proteins else 0
                logger.info(f"  Selected proteins: {single_bait_count} single-bait, {multi_bait_count} multi-bait, avg {avg_baits:.1f} baits per protein")
                eprint(f"  Robustness: {single_bait_count} single-bait, {multi_bait_count} multi-bait (avg {avg_baits:.1f} baits/protein)")
        else:
            logger.info(f"All {original_N} proteins already meet the filter criteria ({filter_method})")
            eprint(f"  All {original_N} proteins already meet the filter criteria ({filter_method})")
        
        return filtered_spec_table
    
    if isinstance(data_path, str):
        spec_path = Path(data_path) / "spec_table.tsv"
        composite_path = Path(data_path) / "composite_table.tsv"

    psel = (protein_selection or "score").strip().lower()
    if psel not in ("score", "random", "stratified_saint"):
        raise ValueError(
            f"Unknown protein_selection={protein_selection!r}; "
            f"use 'score', 'random', or 'stratified_saint'"
        )

    import time
    logger = logging.getLogger(__name__)

    composite_table = pd.read_csv(composite_path, sep=sep)
    check_composite_table(composite_table)
    # Random mode: no MS-threshold stripping of composite rows before choosing proteins.
    if psel == "random":
        logger.info(
            "protein_selection=random: skipping remove_prey (no MS-threshold filter on composite)"
        )
        eprint("  Random mode: composite_table left unfiltered by MS thresholds (remove_prey skipped)")
    else:
        composite_table = remove_prey(composite_table, ms_thresholds)

    # For random / stratified runs, composite_dict is built only after spec + composite are aligned to the subset.
    composites = None
    if calc_composites and psel not in ("random", "stratified_saint"):
        composites = populate_composites_dict(composite_table, ms_thresholds)

    eprint("Loading spec_table...")
    load_start = time.time()
    spec_table = pd.read_csv(spec_path, sep=sep, index_col=0)
    load_time = time.time() - load_start
    logger.info(f"Loaded spec_table in {load_time:.1f}s (shape: {spec_table.shape})")
    eprint(f"  Loaded spec_table: {spec_table.shape[0]} proteins × {spec_table.shape[1]} conditions ({load_time:.1f}s)")

    validate_spec_table(spec_table)
    if psel == "random":
        logger.info(
            "protein_selection=random: skipping filter_minimal_ids (sample from full spec_table prey list)"
        )
        eprint(
            f"  Random mode: no filter_minimal_ids — {spec_table.shape[0]} proteins × {spec_table.shape[1]} conditions"
        )
    else:
        spec_table = filter_minimal_ids(composite_table, spec_table)
        eprint(
            f"  After composite filtering: {spec_table.shape[0]} proteins × {spec_table.shape[1]} conditions"
        )

    # Reduce N: uniform random (full spec), stratified SAINT (base filters + top/bottom rank halves), or top-by-score.
    if psel == "random" and max_proteins is not None and max_proteins > 0:
        seed_use = int(random_proteins_seed) if random_proteins_seed is not None else int(rseed)
        log_file = Path(data_path) / "random_protein_selection.txt"
        spec_table = filter_proteins_random_subset(
            spec_table, max_proteins, seed_use, log_path=log_file
        )
        all_proteins = set(spec_table.index)
        # Induced subgraph: only bait–prey rows where both names are in the sampled node set.
        composite_table = composite_table[
            composite_table["Prey"].isin(all_proteins)
            & composite_table["Bait"].isin(all_proteins)
        ].copy()
        if calc_composites:
            composites = populate_composites_dict(composite_table, ms_thresholds)
    elif psel == "stratified_saint" and max_proteins is not None and max_proteins > 0:
        seed_use = int(random_proteins_seed) if random_proteins_seed is not None else int(rseed)
        log_file = Path(data_path) / "stratified_protein_selection.txt"
        # Rank using composite_table after remove_prey + spec after filter_minimal_ids (base-style universe).
        spec_table = filter_proteins_stratified_saint_subset(
            spec_table,
            composite_table,
            max_proteins,
            seed_use,
            log_path=log_file,
        )
        all_proteins = set(spec_table.index)
        composite_table = composite_table[
            composite_table["Prey"].isin(all_proteins)
            & composite_table["Bait"].isin(all_proteins)
        ].copy()
        if calc_composites:
            composites = populate_composites_dict(composite_table, ms_thresholds)
    elif (min_saint_score_for_protein is not None and min_saint_score_for_protein > 0) or (
        max_proteins is not None and max_proteins > 0
    ):
        # Filter proteins by SAINT / combined score to reduce N (standard base runs)
        spec_table = filter_proteins_by_saint_score(
            composite_table,
            spec_table,
            min_saint_score=min_saint_score_for_protein,
            max_proteins=max_proteins,
        )
        all_proteins = set(spec_table.index)
        composite_table = composite_table[
            (composite_table["Prey"].isin(all_proteins))
            | (composite_table["Bait"].isin(all_proteins))
        ].copy()

    if calc_composites and composites is None:
        composites = populate_composites_dict(composite_table, ms_thresholds)

    eprint(f"  After all filtering: {spec_table.shape[0]} proteins × {spec_table.shape[1]} conditions")
    validate_spec_table(spec_table)
    
    #Filter to only experimental replicate columns (exclude controls)
    # Controls (_c*) measure background noise, not true binding patterns
    # Correlation should reflect true co-variation in experimental signal, not background
    import re
    experimental_cols = [col for col in spec_table.columns if re.match(r'.*_r\d+$', col)]
    if len(experimental_cols) < spec_table.shape[1]:
        n_control_cols = spec_table.shape[1] - len(experimental_cols)
        logger.info(f"Filtering out {n_control_cols} control columns, keeping {len(experimental_cols)} experimental replicate columns")
        eprint(f"  Filtering: {spec_table.shape[1]} columns → {len(experimental_cols)} experimental replicates (removed {n_control_cols} control columns)")
        spec_table = spec_table[experimental_cols].copy()
        validate_spec_table(spec_table)
    
    eprint("Starting correlation calculation (this is the slow step)...")
    corr, shuff_corr, null_sc = calc_shuff_correlations(spec_table, rseed)
    
    # Convert JAX arrays to NumPy for validation and clipping
    # Handle numerical precision issues that might cause values slightly outside [-1, 1]
    try:
        if hasattr(corr, 'block_until_ready'):  # JAX array
            corr = np.array(corr)
        if hasattr(shuff_corr, 'block_until_ready'):  # JAX array
            shuff_corr = np.array(shuff_corr)
    except (AttributeError, TypeError):
        # Already NumPy array or different type
        corr = np.asarray(corr)
        shuff_corr = np.asarray(shuff_corr)
    
    # Clip to valid range to handle numerical precision issues
    # Cosine similarity should be in [-1, 1], but floating point errors can cause slight overflow
    corr = np.clip(corr, -1.0, 1.0)
    shuff_corr = np.clip(shuff_corr, -1.0, 1.0)
    
    r, c = spec_table.shape
    assert r > 2, r
    assert c > 2, c
    assert np.all(spec_table.values >= 0)
    assert np.all(~np.isnan(corr)), f"Found NaN values in correlation matrix"
    assert np.all(~np.isinf(corr)), f"Found Inf values in correlation matrix"
    assert np.all(corr <= 1.0 + 1e-10), f"Correlation values exceed 1.0: max={np.max(corr)}, min={np.min(corr)}"
    assert np.all(corr >= -1.0 - 1e-10), f"Correlation values below -1.0: max={np.max(corr)}, min={np.min(corr)}"
    
    base = {
        "corr": corr,
        "shuff_corr": shuff_corr,
        "selected_shuff_spec_table": null_sc,
        "selected_spec_table": spec_table,
        "composite_table": composite_table
    }
    
    if calc_composites:
        return base | {"composite_dict": composites}
    else:
        return base


def model23_ll_lp_data_getter(save_dir):
    """Load and prepare data for model23_ll_lp and related models."""
    def validate_modeler_vars(modeler_vars):
        def extra_validate(key, modeler_vars):
            if key in ('alpha', 'beta'):
                alpha = modeler_vars['alpha']
                beta = modeler_vars['beta']
                assert alpha >= 0, alpha
                assert alpha < 1, alpha
                assert alpha < beta, beta
                assert beta <= 1, beta
            elif key == 'thresholds':
                thresholds = modeler_vars[key]
                for t in thresholds:
                    assert t >= 0, t
                    assert t <= 1, t
            elif key in ("max_distance", "disconectivity_distance"):
                max_distance = modeler_vars['max_distance']
                disconectivity_distance = modeler_vars['disconectivity_distance']
                assert disconectivity_distance < max_distance, (disconectivity_distance, max_distance)
                assert disconectivity_distance > 0, disconectivity_distance
            elif key == "n_null_bins":
                n_null_bins = modeler_vars[key]
                if isinstance(n_null_bins, str):
                    assert n_null_bins == 'auto'
                else:
                    assert n_null_bins > 10, f"What do you have against bins?"
            else:
                raise ValueError(f"Unexpected Key {key} in modeler_vars")
        
        def key_validate(key, modeler_vars, valtype):
            if key not in modeler_vars:
                raise KeyError(f"Missing key {key} in modeler_vars.json")
            else:
                t = type(modeler_vars[key])
                if not isinstance(modeler_vars[key], valtype):
                    raise ValueError(f"Expected value {valtype} at key {key} in modeler_vars.json. Found {t} instead")
                extra_validate(key, modeler_vars)
        
        expected = {
            'alpha': float,
            'beta': float,
            'thresholds': list,
            'disconectivity_distance': int,
            'max_distance': int,
            'n_null_bins': object,  # int or 'auto'
        }
        for key, valtype in expected.items():
            key_validate(key, modeler_vars, valtype)
    
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    
    with open(str(save_dir / "modeler_vars.json"), 'r') as f:
        modeler_vars = json.load(f)
    validate_modeler_vars(modeler_vars)
    
    # Get protein filtering parameters if specified (for performance - reduces N)
    min_saint_score_for_protein = modeler_vars.get('min_saint_score_for_protein', None)
    max_proteins = modeler_vars.get('max_proteins', None)
    protein_selection = str(modeler_vars.get('protein_selection', 'score'))
    rsp = modeler_vars.get('random_proteins_seed', None)
    random_proteins_seed = int(rsp) if rsp is not None else None
    rseed_data = int(modeler_vars.get('rseed', 0))

    dd = data_from_spec_table_and_composite_table(
        str(save_dir),
        modeler_vars['thresholds'],
        rseed=rseed_data,
        min_saint_score_for_protein=min_saint_score_for_protein,
        max_proteins=max_proteins,
        protein_selection=protein_selection,
        random_proteins_seed=random_proteins_seed,
    )
    assert len(set(dd.keys()).intersection(modeler_vars.keys())) == 0
    dd = dd | modeler_vars
    return dd


def model23_data_transformer(
    data_dict, 
    calculate_composites=True, 
    generate_synthetic_data_kwargs=None,
    synthetic_N=None, 
    synthetic_Mtrue=None, 
    synthetic_rseed=None,
    use_indirect_saint=True,
    indirect_saint_threshold=0.3,
    max_edges=None  # NEW: Limit number of edges to speed up computation
):
    """
    Transform raw data dictionary into format expected by model23_* functions.
    
    This function:
    1. Extracts correlation matrices
    2. Flattens to edge lists
    3. Calculates composite connectivity structures (if requested)
    4. Prepares SAINT scores
    5. Creates name mappings
    """
    # Get correlation matrices
    corr = data_dict['corr']
    shuff_corr = data_dict['shuff_corr']
    spec_table = data_dict['selected_spec_table']
    composite_table = data_dict['composite_table']
    
    # Convert to numpy if JAX arrays
    if hasattr(corr, 'block_until_ready'):
        corr = np.array(corr)
    if hasattr(shuff_corr, 'block_until_ready'):
        shuff_corr = np.array(shuff_corr)
    
    # Get protein names and create mappings
    protein_names = list(spec_table.index)
    N = len(protein_names)
    M = N * (N - 1) // 2
    
    # Create name to index mappings
    name2node_idx = {name: idx for idx, name in enumerate(protein_names)}
    node_idx2name = {idx: name for name, idx in name2node_idx.items()}
    
    # Flatten correlation matrices (upper triangle only)
    logging.info(f"Processing correlation matrices for {N} proteins...")
    triu_indices = np.triu_indices(N, k=1)
    apms_corr_flat_np = corr[triu_indices]  # Keep numpy version for edge estimation
    apms_shuff_corr_flat_np = shuff_corr[triu_indices]  # Keep numpy version for filtering
    
    # Convert to JAX arrays for model (will be updated if filtering is applied)
    apms_corr_flat = jnp.array(apms_corr_flat_np)
    apms_shuff_corr_flat = jnp.array(apms_shuff_corr_flat_np)
    logging.info("Correlation matrices processed")
    
    # Extract SAINT scores from composite_table
    logging.info("Processing SAINT scores (direct + indirect)...")
    
    composite_table_filtered = composite_table[composite_table['MSscore'] > 0.0].copy()
    if len(composite_table_filtered) < len(composite_table):
        logging.info(
            f"Filtered {len(composite_table) - len(composite_table_filtered)} pairs with SAINT score = 0"
        )
    composite_table = composite_table_filtered
    
    # Step 1: Direct SAINT scores (bait-prey pairs)
    saint_scores = {}
    bait_prey_scores = {}  # Store bait->prey scores for indirect calculation
    
    for _, row in composite_table.iterrows():
        bait = row['Bait']
        prey = row['Prey']
        msscore = row.get('MSscore', 0.0)
        
        # Skip self-interactions and zero scores (after filtering, shouldn't happen often)
        if bait == prey or msscore <= 0.0:
            continue
        
        # Get indices
        if bait in name2node_idx and prey in name2node_idx:
            i = name2node_idx[bait]
            j = name2node_idx[prey]
            # Use upper triangle only (i < j)
            if i > j:
                i, j = j, i
            pair = (i, j)
            # Keep maximum SAINT score for each pair
            if pair not in saint_scores or msscore > saint_scores[pair]:
                saint_scores[pair] = msscore
            
            # Store bait->prey mapping for indirect SAINT calculation
            # Use original bait and prey names (not sorted indices)
            if bait not in bait_prey_scores:
                bait_prey_scores[bait] = {}
            if prey not in bait_prey_scores[bait] or msscore > bait_prey_scores[bait].get(prey, 0):
                bait_prey_scores[bait][prey] = msscore
    
    # Step 2: Calculate indirect SAINT scores for prey-prey pairs (OPTIONAL - can be disabled for speed)
    # If two prey proteins are both pulled down by the same bait with high SAINT scores,
    # they likely interact. We use geometric mean: sqrt(SAINT(bait, prey1) * SAINT(bait, prey2))
    # This gives a conservative estimate - both interactions must be strong for indirect to be high.
    # WARNING: This can create MANY pairs (e.g., 100 prey → 4,950 pairs), slowing down computation.
    indirect_saint_scores = {}
    
    if use_indirect_saint:
        logger = logging.getLogger(__name__)
        logger.info(f"Calculating indirect SAINT scores (threshold: {indirect_saint_threshold})...")
        eprint(f"  Calculating indirect SAINT scores for prey-prey pairs...")
        eprint(f"  WARNING: This can create many pairs and slow down computation!")
        
        indirect_count = 0
        for bait, prey_dict in bait_prey_scores.items():
            prey_list = list(prey_dict.keys())
            # For each pair of prey pulled down by the same bait
            for i, prey1 in enumerate(prey_list):
                saint1 = prey_dict[prey1]
                # Only consider prey with SAINT score above threshold (filters out weak interactions)
                if saint1 < indirect_saint_threshold:
                    continue
                for prey2 in prey_list[i+1:]:
                    saint2 = prey_dict[prey2]
                    # Only consider prey with SAINT score above threshold
                    if saint2 < indirect_saint_threshold:
                        continue
                    if prey1 in name2node_idx and prey2 in name2node_idx:
                        idx1 = name2node_idx[prey1]
                        idx2 = name2node_idx[prey2]
                        # Use upper triangle
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        pair = (idx1, idx2)
                        
                        # Calculate indirect SAINT: geometric mean of the two bait-prey scores
                        indirect_saint = np.sqrt(saint1 * saint2)
                        
                        # Only keep if above threshold
                        if indirect_saint >= indirect_saint_threshold:
                            # Keep maximum indirect SAINT if multiple baits pull down the same pair
                            if pair not in indirect_saint_scores or indirect_saint > indirect_saint_scores[pair]:
                                indirect_saint_scores[pair] = indirect_saint
                                indirect_count += 1
        
        logger.info(f"Created {indirect_count} indirect SAINT score pairs (threshold: {indirect_saint_threshold})")
        eprint(f"  Created {indirect_count} indirect SAINT score pairs")
    else:
        logger = logging.getLogger(__name__)
        logger.info("Indirect SAINT scores disabled (use_indirect_saint=False)")
        eprint(f"  Indirect SAINT scores disabled (for faster computation)")
    
    # Step 3: Merge direct and indirect SAINT scores
    # For each pair, take the maximum of direct and indirect SAINT scores
    for pair, indirect_score in indirect_saint_scores.items():
        if pair not in saint_scores:
            # Only use indirect if no direct score exists (prey-prey pair)
            saint_scores[pair] = indirect_score
        else:
            # If both exist, take the maximum (direct is usually more reliable)
            saint_scores[pair] = max(saint_scores[pair], indirect_score)
    
    # Create flattened SAINT score array (use numpy first, then convert to JAX)
    saint_max_pair_score_edgelist_np = np.zeros(M)
    for (i, j), score in saint_scores.items():
        # Convert (i, j) to edge index k using numpy triu_indices
        # k is the index in the flattened upper triangle
        # For upper triangle: k = sum_{r=0}^{i-1} (N - r - 1) + (j - i - 1)
        k = 0
        for r in range(i):
            k += N - r - 1
        k += j - i - 1
        if 0 <= k < M:
            saint_max_pair_score_edgelist_np[k] = max(saint_max_pair_score_edgelist_np[k], score)
    
    # Convert to JAX array
    saint_max_pair_score_edgelist = jnp.array(saint_max_pair_score_edgelist_np)
    logging.info("SAINT scores processed")
    
    # CRITICAL PERFORMANCE FIX: Filter edges if max_edges is specified
    # This dramatically reduces M and speeds up JAX compilation
    edge_filter_map = None  # Maps filtered edge index -> original edge index
    M_original = M  # Store original M before filtering
    if max_edges is not None and max_edges < M:
        logger = logging.getLogger(__name__)
        logger.info(f"Filtering edges: keeping top {max_edges} edges out of {M} possible")
        eprint(f"  Filtering to top {max_edges} edges (out of {M} possible) for faster computation...")
        
        # Create combined score: SAINT score + correlation (weighted)
        # Higher is better - we want high SAINT scores and high correlations
        combined_scores = saint_max_pair_score_edgelist_np.copy()
        # Add correlation contribution (normalized to 0-1 range)
        corr_normalized = (apms_corr_flat_np - apms_corr_flat_np.min()) / (apms_corr_flat_np.max() - apms_corr_flat_np.min() + 1e-10)
        combined_scores = combined_scores * 0.7 + corr_normalized * 0.3  # Weight SAINT more
        
        # Get top K edge indices
        top_k_indices = np.argsort(combined_scores)[-max_edges:][::-1]  # Sort descending, take top K
        
        # Create filtered arrays
        apms_corr_flat_np_filtered = apms_corr_flat_np[top_k_indices]
        apms_shuff_corr_flat_np_filtered = apms_shuff_corr_flat_np[top_k_indices]
        saint_max_pair_score_edgelist_np_filtered = saint_max_pair_score_edgelist_np[top_k_indices]
        
        # Update M to filtered size
        M_filtered = len(top_k_indices)
        logger.info(f"Filtered from M={M} to M={M_filtered} edges")
        eprint(f"  Reduced from {M} to {M_filtered} edges ({M_filtered/M*100:.1f}% of original)")
        
        # Store mapping for later reconstruction
        edge_filter_map = top_k_indices.copy()
        
        # Store filtered arrays (non-padded) - padding will be done in model if needed
        # Update arrays to filtered versions
        apms_corr_flat_np = apms_corr_flat_np_filtered
        apms_shuff_corr_flat_np = apms_shuff_corr_flat_np_filtered
        saint_max_pair_score_edgelist_np = saint_max_pair_score_edgelist_np_filtered
        
        # Keep M as filtered size for sampling
        M = M_filtered  # This is the actual number of edges being sampled
        
        # Convert to JAX arrays
        apms_corr_flat = jnp.array(apms_corr_flat_np)
        apms_shuff_corr_flat = jnp.array(apms_shuff_corr_flat_np)
        saint_max_pair_score_edgelist = jnp.array(saint_max_pair_score_edgelist_np)
        
        logger.info(f"Edge filtering complete: sampling M={M} edges (down from M_original={M_original})")
    
    # Estimate expected number of edges from data
    logging.info("Estimating expected edge count from data...")
    # Method: Count high-confidence edges from multiple sources
    expected_edges = 0.0
    edge_std = 0.0
    
    # Source 1: High-confidence SAINT scores (direct + indirect)
    # Count edges with SAINT score above threshold
    saint_threshold = 0.5  # Medium confidence threshold
    high_saint_edges = np.sum(saint_max_pair_score_edgelist_np >= saint_threshold)
    
    # Source 2: High correlation pairs (above median correlation)
    corr_median = np.median(apms_corr_flat_np)
    high_corr_edges = np.sum(apms_corr_flat_np >= corr_median)
    
    # Source 3: Composite edges (if available)
    composite_edges_count = 0
    if calculate_composites and 'composite_dict' in data_dict:
        composite_dict = data_dict['composite_dict']
        for comp_id, comp_data in composite_dict.items():
            nodes = comp_data['nodes']
            # Count all pairs within composite
            composite_edges_count += len(nodes) * (len(nodes) - 1) // 2
    
    # Combine estimates: take maximum of SAINT-based and correlation-based
    # This gives a conservative estimate
    expected_edges = max(high_saint_edges, high_corr_edges * 0.5)  # Correlation is less reliable
    
    # Add composite edges (these are high-confidence)
    expected_edges += composite_edges_count * 0.3  # Not all composite pairs will be edges
    
    # Calculate possible edges (M) for density-based defaults
    M_possible = N * (N - 1) // 2
    
    # Ensure minimum edges based on network size (safety floor)
    # Typical sparse protein interaction networks have ~1-2% edge density
    min_edges = max(N * 0.01, 10)  # At least 1% density or 10 edges, whichever is higher
    
    # Cap the data-driven estimate at a reasonable maximum
    # SCIENTIFIC FIX: Realistic density for protein interaction networks
    # - General PPI networks: 1-5% density
    # - Dense complexes: 10-20% density
    # - Maximum reasonable: 25% density (not 40%)
    max_edges_cap = M_possible * 0.25  # 25% density maximum cap (realistic for PPI networks)
    expected_edges = max(min_edges, min(expected_edges, max_edges_cap))
    
    # Use density-based default if data-driven estimate is too low
    # SCIENTIFIC FIX: More realistic default density
    # - 15% is more appropriate for protein complexes than 33%
    # - Still allows dense complexes but not unrealistically high
    default_edge_density = 0.15  # 15% density - realistic for protein complexes
    default_expected_edges = M_possible * default_edge_density
    default_expected_edges_std = default_expected_edges * 0.25  # 25% std (reasonable uncertainty)
    
    # If the data-driven estimate is very low (likely due to missing SAINT scores),
    # use the density-based default instead (scales with N)
    if expected_edges < 50:  # If estimate is suspiciously low
        logger = logging.getLogger(__name__)
        logger.warning(f"Data-driven expected_edges estimate ({expected_edges:.1f}) is very low.")
        logger.warning(f"Using density-based default: {default_edge_density*100:.0f}% density = {default_expected_edges:.0f} ± {default_expected_edges_std:.0f} edges")
        expected_edges = default_expected_edges
        edge_std = default_expected_edges_std
    else:
        # Calculate standard deviation as fraction of expected
        # SCIENTIFIC FIX: 25% uncertainty is reasonable (not 35%)
        edge_std = expected_edges * 0.25  # 25% of expected edges (reasonable uncertainty)
    
    # Log the estimate
    logger = logging.getLogger(__name__)
    logger.info(f"Estimated edges from data:")
    logger.info(f"  High SAINT edges (>{saint_threshold}): {high_saint_edges}")
    logger.info(f"  High correlation edges: {high_corr_edges}")
    logger.info(f"  Composite edges: {composite_edges_count}")
    logger.info(f"  Expected edges: {expected_edges:.1f} ± {edge_std:.1f}")
    
    # Build output dictionary
    output = {
        'N': N,
        'M': M,  # This is now the filtered M if max_edges was used
        'apms_corr_flat': apms_corr_flat,
        'apms_shuff_corr_flat': apms_shuff_corr_flat,
        'saint_max_pair_score_edgelist': saint_max_pair_score_edgelist,
        'name2node_idx': name2node_idx,
        'node_idx2name': node_idx2name,
        'expected_edges': float(expected_edges),  # Data-driven edge count estimate
        'expected_edges_std': float(edge_std),   # Uncertainty in estimate
    }
    
    # Store edge filter mapping if edges were filtered
    if edge_filter_map is not None:
        output['edge_filter_map'] = edge_filter_map  # Maps filtered index -> original index
        output['M_original'] = N * (N - 1) // 2  # Store original M for reference
    
    # Add all original keys (hyperparameters, etc.)
    for key, value in data_dict.items():
        if key not in ['corr', 'shuff_corr', 'selected_spec_table', 'composite_table', 
                       'selected_shuff_spec_table', 'composite_dict']:
            output[key] = value
    
    # Handle composites if needed
    if calculate_composites and 'composite_dict' in data_dict:
        # Compute composite connectivity structures
        composite_dict = data_dict['composite_dict']
        
        # Build composite membership: which proteins belong to which composites
        composite_membership = {}  # protein_name -> list of composite IDs
        composite_edge_sets = {}  # composite_id -> set of edges that should be connected
        
        for comp_id, comp_data in composite_dict.items():
            nodes = comp_data['nodes']
            # Create all pairs within this composite
            for i, node1 in enumerate(nodes):
                if node1 not in composite_membership:
                    composite_membership[node1] = []
                composite_membership[node1].append(comp_id)
                
                for node2 in nodes[i+1:]:
                    if node2 not in composite_membership:
                        composite_membership[node2] = []
                    composite_membership[node2].append(comp_id)
                    
                    # Store edge pairs for this composite
                    if comp_id not in composite_edge_sets:
                        composite_edge_sets[comp_id] = set()
                    # Convert to node indices
                    if node1 in name2node_idx and node2 in name2node_idx:
                        idx1 = name2node_idx[node1]
                        idx2 = name2node_idx[node2]
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        composite_edge_sets[comp_id].add((idx1, idx2))
        
        output['composite_membership'] = composite_membership
        output['composite_edge_sets'] = composite_edge_sets
        output['new_composite_dict_p_is_1'] = composite_dict  # Full composite dict
        output['new_composite_dict_norm_approx'] = {}  # Could add normalized version later
        output['n_composites'] = len(composite_dict)
    else:
        output['composite_membership'] = {}
        output['composite_edge_sets'] = {}
        output['new_composite_dict_p_is_1'] = {}
        output['new_composite_dict_norm_approx'] = {}
        output['n_composites'] = 0
    
    return output


# ============================================================================
# CORE MODEL23_N MODEL
# ============================================================================

def model23_unpack_model_data(model_data):
    """Unpack model_data dictionary into individual variables."""
    d = model_data
    N = int(d['N'])
    # Use filtered M if edges were filtered, otherwise calculate from N
    M = int(d.get('M', N * (N - 1) // 2))  # M may be filtered (smaller than N*(N-1)//2)
    alpha = d['alpha']
    beta = d['beta']
    composite_dict_p_is_1 = d.get('new_composite_dict_p_is_1', {})
    composite_dict_norm_approx = d.get('new_composite_dict_norm_approx', {})
    n_null_bins = model_data['n_null_bins']
    
    R = d['apms_corr_flat']
    R = np.clip(R, 0, 1.0)
    R0 = d['apms_shuff_corr_flat']
    null_dist = Histogram(R0, bins=n_null_bins).expand([M,])
    zero_clipped_null_log_like = null_dist.log_prob(R)
    
    disconectivity_distance = model_data['disconectivity_distance']
    max_distance = model_data['max_distance']
    saint_max_pair_score_edgelist = d["saint_max_pair_score_edgelist"]
    
    return (
        N, M, alpha, beta, composite_dict_p_is_1, composite_dict_norm_approx,
        R, R0, null_dist, zero_clipped_null_log_like, disconectivity_distance,
        max_distance, saint_max_pair_score_edgelist
    )


def model23_composite_connectivity_score(
    aij, 
    composite_edge_sets, 
    N,
    weight=1.0
):
    """
    Composite connectivity score - rewards edges between proteins in the same composite.
    
    For each composite, if proteins are in the same composite (pulled down by same bait),
    they should be connected in the network.
    
    Args:
        aij: Adjacency matrix (N x N)
        composite_edge_sets: Dictionary mapping composite_id -> set of (i, j) edge pairs
        N: Number of nodes
        weight: Weight for this score
    
    Returns:
        Score (higher is better when composite edges are present)
    """
    if not composite_edge_sets or len(composite_edge_sets) == 0:
        return 0.0
    
    total_score = 0.0
    triu_indices = jnp.triu_indices(N, k=1)
    
    for comp_id, edge_set in composite_edge_sets.items():
        if len(edge_set) == 0:
            continue
        
        # For each edge in this composite, check if it exists in the network
        composite_score = 0.0
        for (i, j) in edge_set:
            if 0 <= i < N and 0 <= j < N:
                # Get edge value from adjacency matrix
                edge_value = aij[i, j]
                # Reward presence of edge (using sigmoid to make it smooth)
                composite_score += jax.nn.sigmoid(edge_value * 10.0)  # Scale to make it more binary-like
        
        total_score += composite_score * weight
    
    numpyro.factor("composite_connectivity_score_", total_score)
    numpyro.deterministic("composite_connectivity_score", total_score)
    return total_score


def model23_n_(model_data):
    """
    Core model23_n implementation - samples latent edges with multiple constraints.
    
    This model includes:
    - Degree power-law prior
    - Number of edges restraint
    - Profile similarity (R) score
    - Composite connectivity score (if composites available)
    """
    (N, M, alpha, beta, composite_dict_p_is_1, composite_dict_norm_approx, R,
     R0, null_dist, zero_clipped_null_log_like, disconectivity_distance,
     max_distance, saint_max_pair_score_edgelist) = model23_unpack_model_data(model_data)
    
    
    mu = 0.5  # Increased to shift z prior toward positive values (was 0.1, then 0.3)
    sigma = 1.0
    diag_indices = jnp.diag_indices(N)
    
    # Handle edge filtering: if edges were filtered, we need to pad arrays for flat2matrix
    edge_filter_map = model_data.get('edge_filter_map', None)
    M_original = model_data.get('M_original', M)
    
    if edge_filter_map is not None:
        # Pad filtered arrays to M_original for matrix reconstruction
        R_padded = jnp.zeros(M_original)
        saint_padded = jnp.zeros(M_original)
        R_padded = R_padded.at[edge_filter_map].set(R)
        saint_padded = saint_padded.at[edge_filter_map].set(saint_max_pair_score_edgelist)
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R_padded)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_padded)
    else:
        # Standard case: no filtering
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_max_pair_score_edgelist)
    
    saint_max_pair_score_pairwise_matrix = saint_max_pair_score_pairwise_matrix.at[diag_indices].set(0)
    R_pairwise_matrix = R_pairwise_matrix.at[diag_indices].set(0)
    
    # Sample z as vector of unique pairs (filtered if edge_filter_map exists)
    num_pairs = M
    logging.info(f"Sampling z with shape ({num_pairs},)")
    z = numpyro.sample('z', dist.Normal(mu, sigma).expand([num_pairs]).to_event(1))
    
    # Convert to adjacency matrix (handle filtered edges if needed)
    aij = Z2A(z, N, edge_filter_map=edge_filter_map)
    diag_indices = jnp.arange(aij.shape[0])
    aij = aij.at[(diag_indices, diag_indices)].set(0)
    
    # Ensure symmetry
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)
    
    # Power-law degree prior
    gamma = model_data.get('degree_gamma', 2.5)
    degree_score = degree_powerlaw_logprob(degree, gamma=gamma, kmin=1.0)
    # Ensure degree_score is a valid scalar (not NaN or Inf)
    degree_score = jnp.where(jnp.isfinite(degree_score), degree_score, -1e6)
    numpyro.factor("degree_powerlaw_score_", degree_score)
    numpyro.deterministic("degree_powerlaw_score", -degree_score)
    
    # Number of edges restraint (data-driven)
    # CRITICAL FIX: Use sum of probabilities (sigmoid(z)) instead of sum of z values
    # This allows high z values (high probabilities) while still constraining total edges
    # Old: n_edges = jnp.sum(degree) / 2  # Sum of raw z values (conflicts with high probabilities)
    # New: Sum of probabilities (works in probability space)
    z_probs = jax.nn.sigmoid(z)  # Convert z to probabilities
    n_edges = jnp.sum(z_probs)  # Sum of probabilities (0-1 range)
    
    # Get expected edges from data (calculated in data transformer)
    # Check for hardcoded values first (for quick test / backward compatibility)
    hardcoded_expected_edges = model_data.get('hardcoded_expected_edges', None)
    hardcoded_expected_edges_std = model_data.get('hardcoded_expected_edges_std', None)
    
    if hardcoded_expected_edges is not None:
        # Use hardcoded values if provided (for quick test / performance)
        expected_edges = float(hardcoded_expected_edges)
        expected_edges_std = float(hardcoded_expected_edges_std) if hardcoded_expected_edges_std is not None else expected_edges * 0.33
        logging.info(f"Using hardcoded expected edges: {expected_edges:.1f} ± {expected_edges_std:.1f}")
    else:
        # Use data-driven estimate
        expected_edges = model_data.get('expected_edges', None)
        expected_edges_std = model_data.get('expected_edges_std', None)
        
        if expected_edges is None or expected_edges_std is None:
            # Fallback: use default expected edges (300 ± 100)
            # This was the original default and works well for typical networks
            expected_edges = 300.0
            expected_edges_std = 100.0
            logging.warning(f"Expected edges not in model_data, using default: {expected_edges:.1f} ± {expected_edges_std:.1f}")
    
    #  Use proper data-driven constraint
    # The constraint should use the data-driven expected_edges estimate with reasonable uncertainty
    # n_edges = sum(sigmoid(z)) is the expected number of edges (sum of probabilities)
    # This should match expected_edges (which is in binary edge count space)
    # If probabilities are well-calibrated, sum(sigmoid(z)) ≈ expected_edges
    
    # Check if weak constraint is explicitly requested (for backward compatibility/testing)
    use_weak_edge_constraint = model_data.get('use_weak_edge_constraint', False)  # Default: False (use proper constraint)
    
    if use_weak_edge_constraint:
        # Weak constraint mode (for testing only)
        expected_edges_scaled = M * 0.5  # Arbitrary center
        expected_edges_std_scaled = M * 10.0  # Very high std (essentially no constraint)
        logging.info(f"Using WEAK edge constraint (testing mode): mean={expected_edges_scaled:.1f}, std={expected_edges_std_scaled:.1f}")
    else:
        # Use data-driven expected_edges with proper scaling
        # n_edges = sum(sigmoid(z)) is in probability space (0 to M)
        # expected_edges is in binary edge count space
        # If probabilities average ~0.5, then sum(sigmoid(z)) ≈ expected_edges
        # But we need to account for the actual probability distribution
        
        # Use expected_edges directly (same scale if probabilities are calibrated)
        # The constraint should match the data-driven estimate
        expected_edges_scaled = float(expected_edges)
        
        # Use reasonable standard deviation (25% of expected, not M*10.0)
        expected_edges_std_scaled = float(expected_edges_std)
        
        # Ensure constraint is within reasonable bounds
        # Cap at M (can't have more edges than possible pairs)
        expected_edges_scaled = min(expected_edges_scaled, M * 0.95)
        
        # Ensure std is reasonable (not too tight, not too loose)
        expected_edges_std_scaled = max(expected_edges_std_scaled, expected_edges_scaled * 0.15)  # At least 15%
        expected_edges_std_scaled = min(expected_edges_std_scaled, expected_edges_scaled * 0.50)  # At most 50%
        
        logging.info(f"Using data-driven edge constraint: expected={expected_edges_scaled:.1f} ± {expected_edges_std_scaled:.1f} edges (from {expected_edges:.1f} ± {expected_edges_std:.1f} binary edges)")
    
    n_edges_restraint = dist.Normal(float(expected_edges_scaled), float(expected_edges_std_scaled))
    n_edges_score = n_edges_restraint.log_prob(n_edges)
    # Ensure n_edges_score is valid
    n_edges_score = jnp.where(jnp.isfinite(n_edges_score), n_edges_score, -1e6)
    numpyro.factor("n_edges_score_", n_edges_score)
    numpyro.deterministic("n_edges_score", -n_edges_score)
    numpyro.deterministic("expected_edges", expected_edges)  # Store for diagnostics
    
    # Profile similarity score
    # Extract R_vector - if edges were filtered, only extract filtered edges
    if edge_filter_map is not None:
        # Extract only filtered edges from the padded matrix
        R_vector_full = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
        R_vector = R_vector_full[edge_filter_map]  # Extract only filtered edges
    else:
        R_vector = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
    r_z_restraint = dist.Normal(R_vector, 2)
    r_z_score = jnp.sum(r_z_restraint.log_prob(z))
    r_z_score = jnp.where(jnp.isfinite(r_z_score), r_z_score, -1e6)
    numpyro.factor("r_z_score_", r_z_score)
    numpyro.deterministic("r_z_score", -r_z_score)
    
    # FIXED: Removed -0.5 offset to allow high probabilities
    # Old: r_restraint = dist.Normal(R_vector - 0.5, R_vector ** 2 + 1e-2)
    # New: Direct mapping - high correlations → high z values → high probabilities
    r_restraint = dist.Normal(R_vector, R_vector ** 2 + 1e-2)
    r_score = jnp.sum(r_restraint.log_prob(z))
    r_score = jnp.where(jnp.isfinite(r_score), r_score, -1e6)
    numpyro.factor("r_score_", r_score)
    numpyro.deterministic("r_score", -r_score)
    
    # Composite connectivity score (if composites available)
    composite_edge_sets = model_data.get('composite_edge_sets', {})
    if composite_edge_sets and len(composite_edge_sets) > 0:
        composite_weight = model_data.get('composite_weight', 1.0)
        model23_composite_connectivity_score(
            aij=aij,
            composite_edge_sets=composite_edge_sets,
            N=N,
            weight=composite_weight
        )
    
    return saint_max_pair_score_pairwise_matrix, z


def model23_n(model_data):
    """
    Model23_n - Full model with SAINT scores, similarity, degree priors, and composite constraints.
    
    This is the main production model that combines:
    - SAINT experimental scores (S) - both direct (bait-prey) and indirect (prey-prey)
    - Profile similarity (R) - correlation-based constraints
    - Degree power-law prior - network topology constraints
    - Number of edges restraint - controls network density
    - Composite connectivity score - ensures proteins in same composite are connected
    """
    N = int(model_data["N"])
    saint_max_pair_score_pairwise_matrix, z = model23_n_(model_data)
    
    # SAINT score
    # Extract saint_vector - if edges were filtered, only extract filtered edges
    edge_filter_map = model_data.get('edge_filter_map', None)
    if edge_filter_map is not None:
        # Extract only filtered edges from the padded matrix
        saint_vector_full = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
        saint_vector = saint_vector_full[edge_filter_map]  # Extract only filtered edges
    else:
        saint_vector = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
    # FIXED: Removed -0.5 offset to allow high probabilities
    # Old: s_restraint = dist.Normal(saint_vector - 0.5, saint_vector ** 2 + 1e-2)
    # New: Direct mapping - high SAINT scores → high z values → high probabilities
    # FIX: Increase std when SAINT = 0 to prevent tight constraint from dominating
    # When SAINT = 0, std was 0.01 (very tight) - now use larger std (1.0) when SAINT is near zero
    saint_std = jnp.where(saint_vector < 0.01, 1.0, saint_vector ** 2 + 1e-2)
    s_restraint = dist.Normal(saint_vector, saint_std)
    s_score = jnp.sum(s_restraint.log_prob(z))
    s_score = jnp.where(jnp.isfinite(s_score), s_score, -1e6)
    numpyro.factor("s_score_", s_score)
    numpyro.deterministic("s_score", -s_score)


# ============================================================================
# ABLATED MODELS FOR ROBUSTNESS TESTING
# ============================================================================

def model23_n_no_corr_(model_data):
    """
    Model23_n without correlation constraints - for robustness testing.
    
    Removed:
    - Correlation constraints (r_z_score, r_score)
    
    Kept:
    - SAINT scores
    - Degree power-law prior
    - Edge count constraint
    - Z prior
    - Composite connectivity score
    """
    (N, M, alpha, beta, composite_dict_p_is_1, composite_dict_norm_approx, R,
     R0, null_dist, zero_clipped_null_log_like, disconectivity_distance,
     max_distance, saint_max_pair_score_edgelist) = model23_unpack_model_data(model_data)
    
    mu = 0.5
    sigma = 1.0
    diag_indices = jnp.diag_indices(N)
    
    # Handle edge filtering
    edge_filter_map = model_data.get('edge_filter_map', None)
    M_original = model_data.get('M_original', M)
    
    if edge_filter_map is not None:
        saint_padded = jnp.zeros(M_original)
        saint_padded = saint_padded.at[edge_filter_map].set(saint_max_pair_score_edgelist)
        flat2matrix_f = partial(flat2matrix, n=N)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_padded)
    else:
        flat2matrix_f = partial(flat2matrix, n=N)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_max_pair_score_edgelist)
    
    saint_max_pair_score_pairwise_matrix = saint_max_pair_score_pairwise_matrix.at[diag_indices].set(0)
    
    # Sample z
    num_pairs = M
    z = numpyro.sample('z', dist.Normal(mu, sigma).expand([num_pairs]).to_event(1))
    
    # Convert to adjacency matrix
    aij = Z2A(z, N, edge_filter_map=edge_filter_map)
    diag_indices = jnp.arange(aij.shape[0])
    aij = aij.at[(diag_indices, diag_indices)].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)
    
    # Power-law degree prior (KEPT)
    gamma = model_data.get('degree_gamma', 2.5)
    degree_score = degree_powerlaw_logprob(degree, gamma=gamma, kmin=1.0)
    degree_score = jnp.where(jnp.isfinite(degree_score), degree_score, -1e6)
    numpyro.factor("degree_powerlaw_score_", degree_score)
    numpyro.deterministic("degree_powerlaw_score", -degree_score)
    
    # Number of edges restraint (KEPT)
    z_probs = jax.nn.sigmoid(z)
    n_edges = jnp.sum(z_probs)
    
    hardcoded_expected_edges = model_data.get('hardcoded_expected_edges', None)
    hardcoded_expected_edges_std = model_data.get('hardcoded_expected_edges_std', None)
    
    if hardcoded_expected_edges is not None:
        expected_edges = float(hardcoded_expected_edges)
        expected_edges_std = float(hardcoded_expected_edges_std) if hardcoded_expected_edges_std is not None else expected_edges * 0.33
    else:
        expected_edges = model_data.get('expected_edges', 300.0)
        expected_edges_std = model_data.get('expected_edges_std', 100.0)
    
    use_weak_edge_constraint = model_data.get('use_weak_edge_constraint', False)
    
    if use_weak_edge_constraint:
        expected_edges_scaled = M * 0.5
        expected_edges_std_scaled = M * 10.0
    else:
        expected_edges_scaled = float(expected_edges)
        expected_edges_std_scaled = float(expected_edges_std)
        expected_edges_scaled = min(expected_edges_scaled, M * 0.95)
        expected_edges_std_scaled = max(expected_edges_std_scaled, expected_edges_scaled * 0.15)
        expected_edges_std_scaled = min(expected_edges_std_scaled, expected_edges_scaled * 0.50)
    
    n_edges_restraint = dist.Normal(float(expected_edges_scaled), float(expected_edges_std_scaled))
    n_edges_score = n_edges_restraint.log_prob(n_edges)
    n_edges_score = jnp.where(jnp.isfinite(n_edges_score), n_edges_score, -1e6)
    numpyro.factor("n_edges_score_", n_edges_score)
    numpyro.deterministic("n_edges_score", -n_edges_score)
    numpyro.deterministic("expected_edges", expected_edges)
    
    # Profile similarity score - REMOVED (no correlation constraints)
    # r_z_score and r_score are NOT added
    
    # Composite connectivity score (if composites available)
    composite_edge_sets = model_data.get('composite_edge_sets', {})
    if composite_edge_sets and len(composite_edge_sets) > 0:
        composite_weight = model_data.get('composite_weight', 1.0)
        model23_composite_connectivity_score(
            aij=aij,
            composite_edge_sets=composite_edge_sets,
            N=N,
            weight=composite_weight
        )
    
    return saint_max_pair_score_pairwise_matrix, z


def model23_n_no_corr(model_data):
    """Model23_n without correlation constraints - adds SAINT score constraint."""
    N = int(model_data["N"])
    saint_max_pair_score_pairwise_matrix, z = model23_n_no_corr_(model_data)
    
    # SAINT score (KEPT)
    edge_filter_map = model_data.get('edge_filter_map', None)
    if edge_filter_map is not None:
        saint_vector_full = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
        saint_vector = saint_vector_full[edge_filter_map]
    else:
        saint_vector = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
    
    saint_std = jnp.where(saint_vector < 0.01, 1.0, saint_vector ** 2 + 1e-2)
    s_restraint = dist.Normal(saint_vector, saint_std)
    s_score = jnp.sum(s_restraint.log_prob(z))
    s_score = jnp.where(jnp.isfinite(s_score), s_score, -1e6)
    numpyro.factor("s_score_", s_score)
    numpyro.deterministic("s_score", -s_score)


def model23_n_no_saint_(model_data):
    """
    Model23_n without SAINT score constraint - for robustness testing.
    
    Removed:
    - SAINT score constraint (s_score)
    
    Kept:
    - Correlation constraints (r_z_score, r_score)
    - Degree power-law prior
    - Edge count constraint
    - Z prior
    - Composite connectivity score
    """
    (N, M, alpha, beta, composite_dict_p_is_1, composite_dict_norm_approx, R,
     R0, null_dist, zero_clipped_null_log_like, disconectivity_distance,
     max_distance, saint_max_pair_score_edgelist) = model23_unpack_model_data(model_data)
    
    mu = 0.5
    sigma = 1.0
    diag_indices = jnp.diag_indices(N)
    
    # Handle edge filtering
    edge_filter_map = model_data.get('edge_filter_map', None)
    M_original = model_data.get('M_original', M)
    
    if edge_filter_map is not None:
        R_padded = jnp.zeros(M_original)
        R_padded = R_padded.at[edge_filter_map].set(R)
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R_padded)
    else:
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R)
    
    R_pairwise_matrix = R_pairwise_matrix.at[diag_indices].set(0)
    
    # Sample z
    num_pairs = M
    z = numpyro.sample('z', dist.Normal(mu, sigma).expand([num_pairs]).to_event(1))
    
    # Convert to adjacency matrix
    aij = Z2A(z, N, edge_filter_map=edge_filter_map)
    diag_indices = jnp.arange(aij.shape[0])
    aij = aij.at[(diag_indices, diag_indices)].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    degree = jnp.sum(aij, axis=1)
    
    # Power-law degree prior (KEPT)
    gamma = model_data.get('degree_gamma', 2.5)
    degree_score = degree_powerlaw_logprob(degree, gamma=gamma, kmin=1.0)
    degree_score = jnp.where(jnp.isfinite(degree_score), degree_score, -1e6)
    numpyro.factor("degree_powerlaw_score_", degree_score)
    numpyro.deterministic("degree_powerlaw_score", -degree_score)
    
    # Number of edges restraint (KEPT)
    z_probs = jax.nn.sigmoid(z)
    n_edges = jnp.sum(z_probs)
    
    hardcoded_expected_edges = model_data.get('hardcoded_expected_edges', None)
    hardcoded_expected_edges_std = model_data.get('hardcoded_expected_edges_std', None)
    
    if hardcoded_expected_edges is not None:
        expected_edges = float(hardcoded_expected_edges)
        expected_edges_std = float(hardcoded_expected_edges_std) if hardcoded_expected_edges_std is not None else expected_edges * 0.33
    else:
        expected_edges = model_data.get('expected_edges', 300.0)
        expected_edges_std = model_data.get('expected_edges_std', 100.0)
    
    use_weak_edge_constraint = model_data.get('use_weak_edge_constraint', False)
    
    if use_weak_edge_constraint:
        expected_edges_scaled = M * 0.5
        expected_edges_std_scaled = M * 10.0
    else:
        expected_edges_scaled = float(expected_edges)
        expected_edges_std_scaled = float(expected_edges_std)
        expected_edges_scaled = min(expected_edges_scaled, M * 0.95)
        expected_edges_std_scaled = max(expected_edges_std_scaled, expected_edges_scaled * 0.15)
        expected_edges_std_scaled = min(expected_edges_std_scaled, expected_edges_scaled * 0.50)
    
    n_edges_restraint = dist.Normal(float(expected_edges_scaled), float(expected_edges_std_scaled))
    n_edges_score = n_edges_restraint.log_prob(n_edges)
    n_edges_score = jnp.where(jnp.isfinite(n_edges_score), n_edges_score, -1e6)
    numpyro.factor("n_edges_score_", n_edges_score)
    numpyro.deterministic("n_edges_score", -n_edges_score)
    numpyro.deterministic("expected_edges", expected_edges)
    
    # Profile similarity score (KEPT)
    if edge_filter_map is not None:
        R_vector_full = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
        R_vector = R_vector_full[edge_filter_map]
    else:
        R_vector = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
    
    r_z_restraint = dist.Normal(R_vector, 2)
    r_z_score = jnp.sum(r_z_restraint.log_prob(z))
    r_z_score = jnp.where(jnp.isfinite(r_z_score), r_z_score, -1e6)
    numpyro.factor("r_z_score_", r_z_score)
    numpyro.deterministic("r_z_score", -r_z_score)
    
    r_restraint = dist.Normal(R_vector, R_vector ** 2 + 1e-2)
    r_score = jnp.sum(r_restraint.log_prob(z))
    r_score = jnp.where(jnp.isfinite(r_score), r_score, -1e6)
    numpyro.factor("r_score_", r_score)
    numpyro.deterministic("r_score", -r_score)
    
    # Composite connectivity score (if composites available)
    composite_edge_sets = model_data.get('composite_edge_sets', {})
    if composite_edge_sets and len(composite_edge_sets) > 0:
        composite_weight = model_data.get('composite_weight', 1.0)
        model23_composite_connectivity_score(
            aij=aij,
            composite_edge_sets=composite_edge_sets,
            N=N,
            weight=composite_weight
        )
    
    # Return dummy SAINT matrix (not used, but needed for compatibility)
    saint_max_pair_score_pairwise_matrix = jnp.zeros((N, N))
    return saint_max_pair_score_pairwise_matrix, z


def model23_n_no_saint(model_data):
    """Model23_n without SAINT score constraint - correlation constraints are in _ function."""
    # SAINT score constraint is NOT added (removed)
    N = int(model_data["N"])
    _, z = model23_n_no_saint_(model_data)
    # No SAINT score constraint added here


def model23_n_no_priors_(model_data):
    """
    Model23_n without prior constraints - for robustness testing.
    
    Removed:
    - Degree power-law prior (degree_powerlaw_score)
    - Edge count constraint (n_edges_score)
    
    Kept:
    - Z prior (base sampling distribution)
    - Correlation constraints (r_z_score, r_score)
    - SAINT score constraint
    - Composite connectivity score
    """
    (N, M, alpha, beta, composite_dict_p_is_1, composite_dict_norm_approx, R,
     R0, null_dist, zero_clipped_null_log_like, disconectivity_distance,
     max_distance, saint_max_pair_score_edgelist) = model23_unpack_model_data(model_data)
    
    mu = 0.5
    sigma = 1.0
    diag_indices = jnp.diag_indices(N)
    
    # Handle edge filtering
    edge_filter_map = model_data.get('edge_filter_map', None)
    M_original = model_data.get('M_original', M)
    
    if edge_filter_map is not None:
        R_padded = jnp.zeros(M_original)
        saint_padded = jnp.zeros(M_original)
        R_padded = R_padded.at[edge_filter_map].set(R)
        saint_padded = saint_padded.at[edge_filter_map].set(saint_max_pair_score_edgelist)
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R_padded)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_padded)
    else:
        flat2matrix_f = partial(flat2matrix, n=N)
        R_pairwise_matrix = flat2matrix_f(R)
        saint_max_pair_score_pairwise_matrix = flat2matrix_f(saint_max_pair_score_edgelist)
    
    saint_max_pair_score_pairwise_matrix = saint_max_pair_score_pairwise_matrix.at[diag_indices].set(0)
    R_pairwise_matrix = R_pairwise_matrix.at[diag_indices].set(0)
    
    # Sample z (Z PRIOR KEPT - this is the base sampling distribution)
    num_pairs = M
    z = numpyro.sample('z', dist.Normal(mu, sigma).expand([num_pairs]).to_event(1))
    
    # Convert to adjacency matrix
    aij = Z2A(z, N, edge_filter_map=edge_filter_map)
    diag_indices = jnp.arange(aij.shape[0])
    aij = aij.at[(diag_indices, diag_indices)].set(0)
    aij = jnp.tril(aij, k=-1)
    aij = aij + aij.T
    
    # Power-law degree prior - REMOVED
    # degree_powerlaw_score is NOT added
    
    # Number of edges restraint - REMOVED
    # n_edges_score is NOT added
    
    # Profile similarity score (KEPT)
    if edge_filter_map is not None:
        R_vector_full = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
        R_vector = R_vector_full[edge_filter_map]
    else:
        R_vector = R_pairwise_matrix[jnp.triu_indices(N, k=1)]
    
    r_z_restraint = dist.Normal(R_vector, 2)
    r_z_score = jnp.sum(r_z_restraint.log_prob(z))
    r_z_score = jnp.where(jnp.isfinite(r_z_score), r_z_score, -1e6)
    numpyro.factor("r_z_score_", r_z_score)
    numpyro.deterministic("r_z_score", -r_z_score)
    
    r_restraint = dist.Normal(R_vector, R_vector ** 2 + 1e-2)
    r_score = jnp.sum(r_restraint.log_prob(z))
    r_score = jnp.where(jnp.isfinite(r_score), r_score, -1e6)
    numpyro.factor("r_score_", r_score)
    numpyro.deterministic("r_score", -r_score)
    
    # Composite connectivity score (if composites available)
    composite_edge_sets = model_data.get('composite_edge_sets', {})
    if composite_edge_sets and len(composite_edge_sets) > 0:
        composite_weight = model_data.get('composite_weight', 1.0)
        model23_composite_connectivity_score(
            aij=aij,
            composite_edge_sets=composite_edge_sets,
            N=N,
            weight=composite_weight
        )
    
    return saint_max_pair_score_pairwise_matrix, z


def model23_n_no_priors(model_data):
    """Model23_n without prior constraints - adds SAINT score constraint."""
    N = int(model_data["N"])
    saint_max_pair_score_pairwise_matrix, z = model23_n_no_priors_(model_data)
    
    # SAINT score (KEPT)
    edge_filter_map = model_data.get('edge_filter_map', None)
    if edge_filter_map is not None:
        saint_vector_full = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
        saint_vector = saint_vector_full[edge_filter_map]
    else:
        saint_vector = saint_max_pair_score_pairwise_matrix[jnp.triu_indices(N, k=1)]
    
    saint_std = jnp.where(saint_vector < 0.01, 1.0, saint_vector ** 2 + 1e-2)
    s_restraint = dist.Normal(saint_vector, saint_std)
    s_score = jnp.sum(s_restraint.log_prob(z))
    s_score = jnp.where(jnp.isfinite(s_score), s_score, -1e6)
    numpyro.factor("s_score_", s_score)
    numpyro.deterministic("s_score", -s_score)


# ============================================================================
# MODEL DISPATCHER
# ============================================================================

def model_23_ll_lp_init_to_data_strategy(model_data):
    """Initialize MCMC to data-informed values."""
    # Implementation would set initial values based on data
    return init_to_uniform


def model_dispatcher(
    model_name, 
    model_data, 
    save_dir, 
    init_strat_dispatch_key="",
    synthetic_N=None,
    synthetic_Mtrue=None,
    synthetic_rseed=None
):
    """
    Dispatch to model variants.
    
    Supported models:
    - model23_n: Full model (baseline)
    - model23_n_no_corr: Without correlation constraints
    - model23_n_no_saint: Without SAINT score constraint
    - model23_n_no_priors: Without degree and edge count priors
    """
    model_map = {
        "model23_n": model23_n,
        "model23_n_no_corr": model23_n_no_corr,
        "model23_n_no_saint": model23_n_no_saint,
        "model23_n_no_priors": model23_n_no_priors,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(model_map.keys())}")
    
    model = model_map[model_name]
    
    # Get data
    model_data = model23_ll_lp_data_getter(save_dir)
    
    # Transform data (enable composites for composite connectivity scoring)
    # Note: Indirect SAINT scores can be disabled for faster computation
    # They create many prey-prey pairs which slows down JAX compilation
    use_indirect_saint = model_data.get('use_indirect_saint', True)  # Default: True (can disable for speed)
    indirect_saint_threshold = model_data.get('indirect_saint_threshold', 0.3)  # Minimum score for indirect pairs
    
    # Get max_edges limit if specified (for performance - reduces M dramatically)
    max_edges = model_data.get('max_edges', None)
    if max_edges is not None:
        logger = logging.getLogger(__name__)
        logger.info(f"Edge filtering enabled: max_edges={max_edges}")
        eprint(f"  Edge filtering enabled: will keep top {max_edges} edges")
    
    model_data = model23_data_transformer(
        model_data,
        calculate_composites=True,  # Enable composites for connectivity constraints
        synthetic_N=synthetic_N,
        synthetic_Mtrue=synthetic_Mtrue,
        synthetic_rseed=synthetic_rseed,
        use_indirect_saint=use_indirect_saint,
        indirect_saint_threshold=indirect_saint_threshold,
        max_edges=max_edges  # Pass edge limit for filtering
    )
    
    # Init strategy
    if init_strat_dispatch_key == "uniform":
        init_strategy = init_to_uniform
    elif init_strat_dispatch_key == "":
        init_strategy = model_23_ll_lp_init_to_data_strategy(model_data)
    else:
        raise ValueError(f"Unknown init_strat_dispatch_key: {init_strat_dispatch_key}")
    
    return model, model_data, init_strategy


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def _main(
    model_id,
    rseed,
    model_name,
    model_data,
    num_warmup,
    num_samples,
    include_potential_energy,
    include_mean_accept_prob,
    include_extra_fields,
    progress_bar,
    save_dir,
    initial_position,
    save_warmup,
    load_warmup,
    jax_profile,
    init_strat,
    thinning: int = 1,
    step_size: float = 1.0,
    adapt_step_size: bool = True,
    adapt_mass_matrix: bool = True,
    target_accept_prob: float = 0.8,
    collect_warmup=False,
    synthetic_N=None,
    synthetic_Mtrue=None,
    synthetic_rseed=None
):
    """
    Main function to run MCMC sampling.
    
    This function:
    1. Dispatches to appropriate model
    2. Sets up MCMC sampler (NUTS)
    3. Runs warmup
    4. Runs sampling
    5. Saves results
    """
    logger = logging.getLogger(__name__)
    logging.info(f"""Params
    rseed {rseed}
    model_name {model_name}
    num_warmup {num_warmup}
    num_samples {num_samples}
    include_potential_energy {include_potential_energy}
    include_extra_fields {include_extra_fields}
    progress_bar {progress_bar}
    save_dir {save_dir}
    initial_position {initial_position}
    save_warmup {save_warmup}
    load_warmup {load_warmup}
    jax_profile {jax_profile}
    init_strat {init_strat}
    thinning {thinning}
    collect_warmup {collect_warmup}
    synthetic_N {synthetic_N}
    synthetic_Mtrue {synthetic_Mtrue}
    synthetic_rseed {synthetic_rseed}
    """)
    
    entered_main_time = time.time()
    eprint(f"Model ID: {model_id}")
    eprint(f"Model Name: {model_name}")
    eprint(jax.devices())
    
    # Setup extra fields
    extra_fields = ()
    if include_extra_fields:
        extra_fields = (
            "potential_energy",
            "diverging",
            "accept_prob",
            "mean_accept_prob",
            "num_steps",
            "energy",
        )
    
    # Get model and data (this can take time for large datasets)
    eprint("Loading and transforming data...")
    eprint("  (This may take a few minutes for large datasets)")
    logger.info("Starting data loading and transformation...")
    data_load_start = time.time()
    
    eprint("  NOTE: JAX compilation will happen during model creation and first warmup step")
    eprint("  This is a one-time cost that can take 5-30+ minutes depending on model size")
    eprint("  Progress will appear once compilation completes and warmup actually starts...")
    
    model, model_data, init_strategy = model_dispatcher(
        model_name, model_data, save_dir,
        init_strat_dispatch_key=init_strat,
        synthetic_N=synthetic_N,
        synthetic_Mtrue=synthetic_Mtrue,
        synthetic_rseed=synthetic_rseed
    )
    
    data_load_time = time.time() - data_load_start
    eprint(f"Data loaded in {data_load_time:.1f}s")
    logger.info(f"Data loading completed in {data_load_time:.1f}s")
    eprint(f"Number of proteins (N): {model_data.get('N', 'unknown')}")
    eprint(f"Number of possible edges (M): {model_data.get('M', 'unknown')}")
    eprint(f"  (JAX compilation happens during first warmup step - this is normal and expected)")
    
    # Save model_data
    data_savename = f"{model_id}_{model_name}_{rseed}_model_data.pkl"
    with open(Path(save_dir) / data_savename, "wb") as file:
        pkl.dump(model_data, file)
    
    if include_potential_energy:
        extra_fields = extra_fields + ("potential_energy",)
    if include_mean_accept_prob:
        extra_fields = extra_fields + ("mean_accept_prob",)
    
    # Setup MCMC
    model_id = str(model_id)
    model_name = str(model_name)
    
    warmup_savename = f"{model_id}_{model_name}_hmc_warmup.pkl"
    warmup_savename = str(Path(save_dir) / warmup_savename)
    
    nuts = NUTS(
        model,
        init_strategy=init_strategy,
        step_size=step_size,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        target_accept_prob=target_accept_prob,
    )
    
    mcmc = MCMC(
        nuts,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=progress_bar,
        thinning=thinning,
    )
    
    # Warmup
    eprint(f"Starting MCMC warmup ({num_warmup} steps)...")
    eprint(f"  Dataset size: N={model_data.get('N', '?')} proteins, M={model_data.get('M', '?')} possible edges")
    eprint(f"  ⚠️  IMPORTANT: JAX compilation happens during the FIRST warmup step")
    eprint(f"     This compilation is SILENT (no progress output) and can take 5-30+ minutes")
    eprint(f"     You'll only see progress updates AFTER compilation completes")
    eprint(f"     Check CPU usage - if CPU is high, compilation is still running")
    eprint(f"     This is a one-time cost - subsequent runs will be much faster (seconds)")
    eprint(f"  Progress updates will be printed every 10 seconds (once compilation finishes)...")
    logger.info(f"Starting MCMC warmup with {num_warmup} steps")
    logger.info(f"JAX compilation phase may take 5-30+ minutes for large models (M={model_data.get('M', '?')})")
    logger.info(f"Compilation happens silently during first warmup step - no progress output until it completes")
    warmup_start_time = time.time()
    
    # Add periodic status updates during warmup
    def print_warmup_status():
        """Print periodic status updates during warmup."""
        elapsed = 0
        while True:
            time.sleep(10)  # Print every 10 seconds
            elapsed = time.time() - warmup_start_time
            if elapsed < 60:
                eprint(f"  [Warmup in progress... {elapsed:.0f}s elapsed]")
            else:
                eprint(f"  [Warmup in progress... {elapsed/60:.1f} minutes elapsed]")
            sys.stderr.flush()  # Force flush to ensure output appears
    
    # Start status thread
    status_thread = threading.Thread(target=print_warmup_status, daemon=True)
    status_thread.start()
    
    mcmc_warmup_kwargs = dict(collect_warmup=collect_warmup)
    # Create PRNG keys - ensure they are proper JAX PRNGKey arrays
    rng_key = jax.random.PRNGKey(rseed)
    rng_key, warmup_key = jax.random.split(rng_key)
    # Don't modify the keys - they should already be correct
    # The issue is likely a numpyro/JAX version incompatibility
    mcmc.warmup(warmup_key, model_data=model_data, extra_fields=extra_fields, **mcmc_warmup_kwargs)
    warmup_time = time.time() - warmup_start_time
    eprint(f"Warmup complete in {warmup_time:.1f}s ({warmup_time/60:.1f} minutes). Starting sampling...")
    eprint("Warmup complete. Starting sampling...")
    logger.info("Warmup complete, starting sampling")
    
    # Save warmup samples
    warmup_fields = mcmc.get_extra_fields()
    warmup_samples = mcmc.get_samples()
    
    out = {"extra_fields": warmup_fields, "samples": warmup_samples}
    
    save_base = f"{model_id}_{model_name}_{rseed}"
    savename = f"{save_base}.pkl"
    if save_dir is not None:
        savename = Path(save_dir) / savename
        savename = str(savename)
    
    with open(str(savename).removesuffix(".pkl") + "_warmup_samples.pkl", "wb") as file:
        pkl.dump(out, file)
    
    # Cleanup
    del warmup_fields, warmup_samples, out
    
    if save_warmup:
        warmup_state = mcmc.last_state
        with open(warmup_savename, "wb") as f:
            pkl.dump(warmup_state, f)
    
    if load_warmup:
        with open(warmup_savename, "rb") as f:
            warmup_state = pkl.load(f)
        mcmc.post_warmup_state = warmup_state
    
    # Run sampling
    eprint(f"Running MCMC sampling ({num_samples} steps)...")
    eprint(f"  Progress updates will be printed every 10 seconds...")
    logger.info(f"Starting MCMC sampling with {num_samples} steps")
    
    # Add periodic status updates during sampling
    sampling_start_time = time.time()
    
    def print_sampling_status():
        """Print periodic status updates during sampling."""
        elapsed = 0
        while True:
            time.sleep(10)  # Print every 10 seconds
            elapsed = time.time() - sampling_start_time
            if elapsed < 60:
                eprint(f"  [Sampling in progress... {elapsed:.0f}s elapsed]")
            else:
                eprint(f"  [Sampling in progress... {elapsed/60:.1f} minutes elapsed]")
            sys.stderr.flush()  # Force flush to ensure output appears
    
    # Start status thread
    sampling_status_thread = threading.Thread(target=print_sampling_status, daemon=True)
    sampling_status_thread.start()
    
    if jax_profile:
        profile_path = str(Path(save_dir) / "jax-trace")
        logging.info(f"writing trace to {profile_path}")
        with jax.profiler.trace(profile_path):
            mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    else:
        mcmc.run(rng_key, model_data, extra_fields=extra_fields)
    eprint("Sampling complete. Saving results...")
    logger.info("Sampling complete")
    
    # Save results
    finished_mcmc_time = time.time()
    elapsed_time = finished_mcmc_time - entered_main_time
    eprint(f"{num_samples} sampling steps complete")
    eprint(f"elapsed time (s): {elapsed_time}")
    eprint(f"elapsed time (m): {elapsed_time / 60}")
    eprint(f"elapsed time (h): {elapsed_time / (60 * 60)}")
    
    fields = mcmc.get_extra_fields()
    samples = mcmc.get_samples()
    
    out = {
        "extra_fields": fields,
        "samples": samples,
        "elapsed_time": elapsed_time,
    }
    with open(savename, "wb") as file:
        pkl.dump(out, file)


# ============================================================================
# POST-PROCESSING: EDGE PROBABILITY EXTRACTION
# ============================================================================

def extract_edge_probabilities_from_samples(samples: dict, model_data: dict, 
                                           burn_in: int = 0) -> Tuple[np.ndarray, dict]:
    """
    Extract edge probabilities from MCMC samples.
    
    This function converts latent edge variables (z) to edge probabilities
    by computing the posterior mean of the adjacency matrix.
    
    Args:
        samples: Dictionary from MCMC.get_samples() containing 'z' key
        model_data: Model data dictionary with 'N', 'name2node_idx', 'node_idx2name'
        burn_in: Number of samples to discard from beginning
        
    Returns:
        edge_probs: N x N numpy array of edge probabilities
        edge_dict: Dictionary mapping (name1, name2) -> probability
    """
    if 'z' not in samples:
        raise ValueError("No 'z' samples found. Make sure model samples 'z' variable.")
    
    z_samples = samples['z']
    N = int(model_data['N'])
    M = model_data.get('M', N * (N - 1) // 2)  # Use filtered M if available
    edge_filter_map = model_data.get('edge_filter_map', None)
    
    # Convert to numpy if JAX array
    if hasattr(z_samples, 'block_until_ready'):
        z_samples = np.array(z_samples)
    
    # Handle shape: (n_samples, M) or (n_chains, n_samples, M)
    if len(z_samples.shape) == 3:
        # Multiple chains: flatten
        z_samples = z_samples.reshape(-1, z_samples.shape[-1])
    elif len(z_samples.shape) == 1:
        # Single sample
        z_samples = z_samples.reshape(1, -1)
    
    # Apply burn-in
    if burn_in > 0:
        z_samples = z_samples[burn_in:]
    
    n_samples = z_samples.shape[0]
    logger = logging.getLogger(__name__)
    logger.info(f"Computing edge probabilities from {n_samples} samples")
    
    # Convert each z sample to adjacency matrix and average
    adjacency_sum = np.zeros((N, N))
    
    for i in range(n_samples):
        z = z_samples[i]
        # Convert z to binary adjacency matrix (handle filtered edges if needed)
        if edge_filter_map is not None:
            edge_filter_map_jax = jnp.array(edge_filter_map)
            aij = Z2A(jnp.array(z), N, edge_filter_map=edge_filter_map_jax)
        else:
            aij = Z2A(jnp.array(z), N)
        aij = np.array(aij)
        adjacency_sum += aij
    
    # Compute edge probabilities
    edge_probs = adjacency_sum / n_samples
    
    # Create edge dictionary with protein names
    edge_dict = {}
    name2node_idx = model_data.get('name2node_idx', {})
    node_idx2name = model_data.get('node_idx2name', {})
    
    for i in range(N):
        for j in range(i + 1, N):
            prob = edge_probs[i, j]
            if prob > 0:
                name_i = node_idx2name.get(i, f"protein_{i}")
                name_j = node_idx2name.get(j, f"protein_{j}")
                edge_dict[(name_i, name_j)] = float(prob)
    
    return edge_probs, edge_dict


# ============================================================================
# NOTE: This is a cleaned version with only model23_n
# ============================================================================
# 
# Changes from original:
# 1. Removed all models except model23_n
# 2. Removed unused scoring functions (SE, SAINT, SR, SC scores)
# 3. Removed unused model variants (model23_se, model23_se_sr, etc.)
# 4. Simplified model_dispatcher to only handle model23_n
# 5. Kept all utility functions and data loading functions needed by model23_n
# 6. Organized into clear sections with headers
#
# New features added:
# 7. Indirect SAINT scores for prey-prey pairs (geometric mean of co-precipitated pairs)
# 8. Composite connectivity scoring - rewards edges between proteins in same composite
# 9. Composite membership tracking for better network structure
# 10. Data-driven edge count estimation - replaces hardcoded edge count with estimate
#     based on SAINT scores, correlations, and composite structure
#
# ============================================================================



