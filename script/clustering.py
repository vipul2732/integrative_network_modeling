#!/usr/bin/env python3
"""
Hamming + average linkage on binary network draws; cut at τ to get flat clusters.

Goal: find τ where you get several (e.g. 3–4) clusters with enough models each, with A:B
mix close to global ("mixed"). No medoids or majority-vote representatives.

Modes:
  - Single τ: --tau T --out DIR (distance cut; may fragment into many tiny clusters)
  - Scan τ:   --tau-min/--tau-max/--tau-step  OR  --tau-list v1,v2,...
  - Fixed K:  --k-clusters K --out DIR (exactly K clusters, all draws assigned; good for 2–3 large groups)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from binary_ensemble import infer_chain_ids_equal_split, load_binary_ensemble  # noqa: E402

logger = logging.getLogger(__name__)

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_chain_ids(ensemble_dir: Path, n_draws: int, n_chains: Optional[int]) -> np.ndarray:
    p = Path(ensemble_dir) / "chain_id_per_draw.npy"
    if p.exists():
        c = np.load(p)
        if c.shape[0] != n_draws:
            raise ValueError("chain_id_per_draw.npy length mismatch")
        return np.asarray(c, dtype=np.int32)
    if n_chains is None:
        raise FileNotFoundError("Need chain_id_per_draw.npy or --n-chains")
    return infer_chain_ids_equal_split(n_draws, n_chains)


def balance_score(frac_b: float, p_b_global: float) -> float:
    return 1.0 - abs(frac_b - p_b_global)


def mean_pairwise_hamming_subset(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return float("nan")
    d = pdist(X.astype(np.float32, copy=False), metric="cityblock")
    return float(d.mean())


def cluster_table(
    labels: np.ndarray,
    X_uint8: np.ndarray,
    is_A: np.ndarray,
    is_B: np.ndarray,
    p_B: float,
    min_cluster_size: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for lab in np.unique(labels):
        gidx = np.flatnonzero(labels == lab)
        n_tot = int(gidx.size)
        if min_cluster_size > 0 and n_tot < min_cluster_size:
            continue
        n_a = int(is_A[gidx].sum())
        n_b = int(is_B[gidx].sum())
        fb = n_b / n_tot if n_tot else 0.0
        fa = n_a / n_tot if n_tot else 0.0
        bw = balance_score(fb, p_B)
        prec = mean_pairwise_hamming_subset(X_uint8[gidx])
        rows.append(
            {
                "cluster_id": int(lab),
                "n_total": n_tot,
                "n_A": n_a,
                "n_B": n_b,
                "frac_A": fa,
                "frac_B": fb,
                "balance_score": bw,
                "abs_dev_frac_B_vs_global": abs(fb - p_B),
                "cluster_mean_pairwise_hamming": prec,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["balance_score", "n_total"], ascending=[False, False]).reset_index(drop=True)


def summarize_tau_scan_row(
    df: pd.DataFrame,
    top_k: int,
    target_min_clusters: int,
) -> Dict[str, Any]:
    n = len(df)
    top_k = min(top_k, n) if n else 0
    if n == 0:
        return {
            "n_clusters_ge_min": 0,
            "meets_target_n_clusters": False,
            "mean_balance_top_k": float("nan"),
            "mean_within_hamming_top_k": float("nan"),
            "min_balance_top_k": float("nan"),
        }
    sub = df.head(top_k)
    return {
        "n_clusters_ge_min": n,
        "meets_target_n_clusters": n >= target_min_clusters,
        "mean_balance_top_k": float(sub["balance_score"].mean()),
        "mean_within_hamming_top_k": float(sub["cluster_mean_pairwise_hamming"].mean()),
        "min_balance_top_k": float(sub["balance_score"].min()),
    }


def run_single_tau(
    Z_link: np.ndarray,
    X_uint8: np.ndarray,
    labels_cache: Dict[float, np.ndarray],
    tau: float,
    is_A: np.ndarray,
    is_B: np.ndarray,
    p_B: float,
    n_A_tot: int,
    n_B_tot: int,
    min_cluster_size: int,
    top_k_balanced: int,
    output_dir: Path,
    T: int,
    M: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tau_f = float(tau)
    if tau_f not in labels_cache:
        labels_cache[tau_f] = fcluster(Z_link, t=tau_f, criterion="distance").astype(np.int32)
    labels = labels_cache[tau_f]

    df = cluster_table(labels, X_uint8, is_A, is_B, p_B, min_cluster_size)
    if df.empty:
        logger.warning("No clusters with size >= %s at tau=%s", min_cluster_size, tau_f)
        df.to_csv(output_dir / "clusters_at_tau.tsv", sep="\t", index=False)
        return

    df.to_csv(output_dir / "clusters_at_tau.tsv", sep="\t", index=False)
    top = df.head(top_k_balanced)
    top.to_csv(output_dir / "top_balanced_clusters.tsv", sep="\t", index=False)

    summary = {
        "tau": tau_f,
        "n_draws": T,
        "m_edges": M,
        "n_clusters_all_labels": int(np.unique(labels).size),
        "n_clusters_reported_ge_min_size": int(len(df)),
        "min_cluster_size": min_cluster_size,
        "top_k_balanced": int(len(top)),
        "p_B_global": p_B,
        "n_A_total": n_A_tot,
        "n_B_total": n_B_tot,
        "interpretation": {
            "balance_score": "1 - |frac_B(cluster) - p_B|; higher = closer to global A:B mix.",
            "cluster_mean_pairwise_hamming": "Mean pairwise Hamming within cluster; lower = tighter.",
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "tau=%s: %s clusters (size>=%s), wrote top %s mixed → %s",
        tau_f,
        len(df),
        min_cluster_size,
        len(top),
        output_dir,
    )


def run_fixed_k(
    Z_link: np.ndarray,
    X_uint8: np.ndarray,
    k: int,
    is_A: np.ndarray,
    is_B: np.ndarray,
    p_B: float,
    n_A_tot: int,
    n_B_tot: int,
    top_k_balanced: int,
    output_dir: Path,
    T: int,
    M: int,
) -> None:
    """Exactly K clusters via linkage + fcluster(..., criterion='maxclust'); every draw is assigned."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = fcluster(Z_link, t=int(k), criterion="maxclust").astype(np.int32)
    n_lab = int(np.unique(labels).size)
    if n_lab != k:
        logger.warning("Expected %s maxclust groups, got %s unique labels", k, n_lab)

    df = cluster_table(labels, X_uint8, is_A, is_B, p_B, min_cluster_size=0)
    df = df.sort_values(["balance_score", "n_total"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(output_dir / "clusters_maxclust_k.tsv", sep="\t", index=False)
    top = df.head(min(top_k_balanced, len(df)))
    top.to_csv(output_dir / "top_balanced_clusters.tsv", sep="\t", index=False)

    np.save(output_dir / "maxclust_labels.npy", labels)

    sum_n = int(df["n_total"].sum())
    summary = {
        "criterion": "maxclust",
        "k_clusters": int(k),
        "maxclust_labels_npy": "maxclust_labels.npy",
        "n_draws": T,
        "m_edges": M,
        "sum_n_total_in_table": sum_n,
        "all_draws_assigned": sum_n == T,
        "p_B_global": p_B,
        "n_A_total": n_A_tot,
        "n_B_total": n_B_tot,
        "mean_balance_all_clusters": float(df["balance_score"].mean()),
        "min_balance_all_clusters": float(df["balance_score"].min()),
        "mean_within_hamming_all_clusters": float(df["cluster_mean_pairwise_hamming"].mean()),
        "interpretation": {
            "balance_score": "1 - |frac_B(cluster) - p_B|; higher = better chain mix vs global.",
            "largest_mixed": "Sort clusters_maxclust_k.tsv by balance_score then n_total for best mix and size.",
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "maxclust k=%s: %s clusters, sum n_total=%s (dataset T=%s) → %s",
        k,
        len(df),
        sum_n,
        T,
        output_dir,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Binary ensemble: Hamming + average linkage; τ cut, τ scan, or fixed K (maxclust)."
    )
    p.add_argument("--ensemble-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output dir (single tau) or scan summary dir")
    p.add_argument(
        "--k-clusters",
        type=int,
        default=None,
        metavar="K",
        help="Exactly K clusters (maxclust); all draws assigned — use 2 or 3 for a few large groups",
    )
    p.add_argument("--tau", type=float, default=None, help="Single linkage cut height")
    p.add_argument("--tau-min", type=float, default=None)
    p.add_argument("--tau-max", type=float, default=None)
    p.add_argument("--tau-step", type=float, default=None)
    p.add_argument(
        "--tau-list",
        type=str,
        default=None,
        help="Comma-separated τ values for scan (e.g. 100,110,120,130,140,150)",
    )
    p.add_argument("--chains-a", type=int, nargs="+", default=[0])
    p.add_argument("--chains-b", type=int, nargs="+", default=[1, 2])
    p.add_argument("--n-chains", type=int, default=None)
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--top-k-balanced", type=int, default=4, help="Top clusters by mix to save in TSV")
    p.add_argument(
        "--target-min-clusters",
        type=int,
        default=3,
        help="In tau scan: flag rows with at least this many clusters (size>=min-cluster-size)",
    )
    args = p.parse_args()

    if not HAS_SCIPY:
        raise ImportError("scipy required")

    has_scan = (
        args.tau_list is not None
        or (args.tau_min is not None and args.tau_max is not None and args.tau_step is not None)
    )
    has_k = args.k_clusters is not None
    n_modes = sum([args.tau is not None, has_scan, has_k])
    if n_modes == 0:
        p.error(
            "Provide one of: --k-clusters K, --tau, or tau scan (--tau-list / --tau-min+max+step)"
        )
    if n_modes > 1:
        p.error("Use only one of: --k-clusters, --tau, or tau scan")
    if has_k and args.k_clusters is not None and int(args.k_clusters) < 2:
        p.error("--k-clusters must be >= 2")

    ensemble_dir = Path(args.ensemble_dir)
    out = Path(args.out)
    X_uint8, _meta = load_binary_ensemble(ensemble_dir)
    X = np.asarray(X_uint8, dtype=np.float32, order="C")
    T, M = X.shape
    chain_ids = load_chain_ids(ensemble_dir, T, args.n_chains)

    set_a = frozenset(int(c) for c in args.chains_a)
    set_b = frozenset(int(c) for c in args.chains_b)
    is_A = np.array([int(c) in set_a for c in chain_ids], dtype=bool)
    is_B = np.array([int(c) in set_b for c in chain_ids], dtype=bool)
    p_B = float(is_B.mean())
    n_A_tot, n_B_tot = int(is_A.sum()), int(is_B.sum())

    logger.info("pdist/linkage on %s x %s (once) …", T, M)
    Z_link = linkage(pdist(X, metric="cityblock"), method="average")
    labels_cache: Dict[float, np.ndarray] = {}

    if has_k:
        run_fixed_k(
            Z_link,
            X_uint8,
            int(args.k_clusters),
            is_A,
            is_B,
            p_B,
            n_A_tot,
            n_B_tot,
            args.top_k_balanced,
            out,
            T,
            M,
        )
        print(f"Done: {out}")
        return

    if args.tau is not None:
        run_single_tau(
            Z_link,
            X_uint8,
            labels_cache,
            float(args.tau),
            is_A,
            is_B,
            p_B,
            n_A_tot,
            n_B_tot,
            args.min_cluster_size,
            args.top_k_balanced,
            out,
            T,
            M,
        )
        print(f"Done: {out}")
        return

    if args.tau_list is not None:
        taus = [float(x.strip()) for x in args.tau_list.split(",") if x.strip()]
    else:
        lo, hi, st = float(args.tau_min), float(args.tau_max), float(args.tau_step)
        if st <= 0:
            p.error("--tau-step must be positive")
        taus = []
        t = lo
        while t <= hi + 1e-9:
            taus.append(round(t, 10))
            t += st

    out.mkdir(parents=True, exist_ok=True)
    scan_rows: List[Dict[str, Any]] = []
    for tau_f in taus:
        if tau_f not in labels_cache:
            labels_cache[tau_f] = fcluster(Z_link, t=tau_f, criterion="distance").astype(np.int32)
        labels = labels_cache[tau_f]
        df = cluster_table(labels, X_uint8, is_A, is_B, p_B, args.min_cluster_size)
        row: Dict[str, Any] = {"tau": tau_f}
        row.update(
            summarize_tau_scan_row(df, args.top_k_balanced, args.target_min_clusters)
        )
        row["n_labels_total"] = int(np.unique(labels).size)
        scan_rows.append(row)

    scan_df = pd.DataFrame(scan_rows)
    scan_path = out / "tau_scan.tsv"
    scan_df.to_csv(scan_path, sep="\t", index=False)

    preferred = scan_df[scan_df["meets_target_n_clusters"]].copy()
    if not preferred.empty:
        preferred = preferred.sort_values(
            ["mean_balance_top_k", "n_clusters_ge_min"],
            ascending=[False, False],
        )
        preferred.to_csv(out / "tau_scan_meets_target.tsv", sep="\t", index=False)
        logger.info(
            "%s τ values with >=%s clusters (size>=%s); best mean_balance_top_%s preview:\n%s",
            len(preferred),
            args.target_min_clusters,
            args.min_cluster_size,
            args.top_k_balanced,
            preferred.head(5).to_string(index=False),
        )
    else:
        logger.warning(
            "No τ in scan gave >=%s clusters with size>=%s; widen τ range or lower min size.",
            args.target_min_clusters,
            args.min_cluster_size,
        )

    meta_out = {
        "ensemble_dir": str(ensemble_dir),
        "n_draws": T,
        "m_edges": M,
        "min_cluster_size": args.min_cluster_size,
        "target_min_clusters": args.target_min_clusters,
        "top_k_for_summary": args.top_k_balanced,
        "tau_values": taus,
        "columns": {
            "meets_target_n_clusters": f"True if n_clusters_ge_min >= {args.target_min_clusters}",
            "mean_balance_top_k": f"Mean balance_score among top-{args.top_k_balanced} clusters by balance",
            "mean_within_hamming_top_k": f"Mean within-cluster Hamming for those top-{args.top_k_balanced}",
        },
    }
    with open(out / "tau_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    logger.info("Wrote %s (%s τ values)", scan_path, len(scan_df))
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
