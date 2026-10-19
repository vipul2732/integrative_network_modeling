#!/usr/bin/env python3
"""
A/B chain mixing tests on hierarchical clusters (used by compute_post_sampling_imp_metrics).

Given an average-linkage Hamming dendrogram Z, scan cophenetic cut heights τ and test whether
each sufficiently large cluster contains the same fraction of draws from sample A (first
⌊K/2⌋ chains) as the full merged posterior, via two-sided binomial tests with Bonferroni
correction across clusters at each τ.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.stats import binomtest


def sample_ab_mask(chain_id: np.ndarray, n_chains: int) -> Tuple[np.ndarray, float]:
    """A = first floor(n_chains/2) chains, B = the rest (works for odd n, e.g. 3→1 vs 2).

    p0 is the global fraction of draws in A (equal samples per chain ⇒ n_A_chains / n_chains).
    """
    n_a_chains = n_chains // 2
    if n_a_chains < 1 or n_a_chains >= n_chains:
        raise ValueError(f"Need n_chains >= 2 for A/B split, got {n_chains}")
    is_a = chain_id < n_a_chains
    p0 = float(is_a.mean())
    return is_a, p0


def mixing_scan_row(
    Z: np.ndarray,
    tau: float,
    is_a: np.ndarray,
    p0: float,
    min_cluster_size: int,
    alpha: float,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    lab = (
        labels.astype(np.int32, copy=False)
        if labels is not None
        else fcluster(Z, tau, criterion="distance").astype(np.int32)
    )
    uniq = np.unique(lab)
    pvals: List[float] = []
    details: List[Dict[str, Any]] = []
    tested = 0
    for c in uniq:
        m = lab == c
        n = int(m.sum())
        n_a = int(is_a[m].sum())
        n_b = n - n_a
        if n < min_cluster_size:
            details.append(
                {
                    "cluster_id": int(c),
                    "n": n,
                    "n_A": n_a,
                    "n_B": n_b,
                    "p_value": None,
                    "tested": False,
                }
            )
            continue
        tested += 1
        res = binomtest(n_a, n, p=p0, alternative="two-sided")
        pv = float(res.pvalue)
        pvals.append(pv)
        details.append(
            {
                "cluster_id": int(c),
                "n": n,
                "n_A": n_a,
                "n_B": n_b,
                "p_value": pv,
                "tested": True,
            }
        )

    bonf = alpha / tested if tested > 0 else alpha
    if tested == 0:
        all_pass = True
        min_p = None
    else:
        min_p = float(np.min(pvals))
        all_pass = min_p >= bonf

    return {
        "tau": float(tau),
        "n_clusters": int(len(uniq)),
        "n_clusters_tested": tested,
        "bonferroni_alpha": float(bonf),
        "min_p_value": min_p,
        "all_mixing_tests_pass": bool(all_pass),
        "clusters": details,
    }


def build_tau_grid(Z: np.ndarray, n_tau: int) -> List[float]:
    """High -> low distance cuts (coarse to fine). n_tau linspace between max and min merge height."""
    h = Z[:, 2].astype(np.float64)
    hmax, hmin = float(h.max()), float(h.min())
    taus = [hmax + 1.0]
    taus.extend(np.linspace(hmax, hmin, max(2, n_tau)).tolist())
    seen: set = set()
    out: List[float] = []
    for t in sorted(taus, reverse=True):
        tf = float(t)
        if tf not in seen:
            seen.add(tf)
            out.append(tf)
    return out
