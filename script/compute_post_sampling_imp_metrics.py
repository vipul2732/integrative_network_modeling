#!/usr/bin/env python3
"""
Post-sampling metrics for base multi-seed models (IMP-style wording).

Binary sample: M = 1{Z > Z_thr} from merged posterior z (all chains concatenated).
Z_thr defaults from script/post_sampling_z_binary.py (Z_BINARY_THRESHOLD; currently 0.5).
Override with --z-binary-threshold (use 0 for legacy Z>0).

Default: average-linkage Hamming (cityblock pdist) on ALL merged draws (e.g. 40,000);
writes linkage_matrix.npy. Then scans τ from high (coarse) to low (fine). At each τ,
every cluster with n ≥ min size is tested: two-sided binomial on whether the fraction of
draws from chains A vs B matches the global split (first ⌊K/2⌋ chains vs the rest),
with Bonferroni correction across clusters at that τ (see sampling_precision_mixing.py).
The finest τ where all such tests still pass is sampling_precision_tau; labels at that τ are
saved as draw_labels_sampling_precision.npy. Per-cluster counts of draws from sample A vs B
are in cluster_sample_ab_at_sampling_precision.tsv and in tau_balance_scan.json (chosen τ
and each τ row includes clusters_sample_ab when cluster count ≤ --max-clusters-ab-per-tau-row).
draw_labels_full_posterior.npy stays all 1s.

Use --reuse-linkage to reload linkage_matrix.npy and only recompute the τ scan (no pdist).

Run from repo root:
  python script/compute_post_sampling_imp_metrics.py
Long runs / high RAM (~6+ GiB condensed distances for n=40000):
  bash script/run_post_sampling_imp_metrics_nohup.sh

For a quick test with fewer draws: --cluster-max-draws 2000
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sampling_precision_mixing import (  # noqa: E402
    build_tau_grid,
    mixing_scan_row,
    sample_ab_mask,
)
from post_sampling_z_binary import binary_m_from_z, effective_z_binary_threshold  # noqa: E402

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

BASE_DATASETS = [
    "cullin_mock_config_multi_seeds",
    "cullin_wt_config_multi_seeds",
    "cullin_vif_config_multi_seeds",
]

ABLATION_DATASETS = [
    "cullin_mock_no_corr",
    "cullin_mock_no_saint",
    "cullin_mock_no_priors",
    "cullin_wt_no_corr",
    "cullin_wt_no_saint",
    "cullin_wt_no_priors",
    "cullin_vif_no_corr",
    "cullin_vif_no_saint",
    "cullin_vif_no_priors",
]


def _binary_entropy_bits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def load_merged_z(merged_pkl: Path) -> Tuple[np.ndarray, List[int], int]:
    with open(merged_pkl, "rb") as f:
        d = pickle.load(f)
    z = np.asarray(d["merged_samples"]["z"], dtype=np.float32)
    seeds = [int(s) for s in d.get("seeds", [])]
    n_chains = int(d.get("n_chains", len(seeds)))
    return z, seeds, n_chains


def reshape_binary_by_chain(M: np.ndarray, n_chains: int) -> np.ndarray:
    T, Me = M.shape
    if T % n_chains != 0:
        raise ValueError(f"T={T} not divisible by n_chains={n_chains}")
    return M.reshape(n_chains, T // n_chains, Me)


def sampling_precision_metrics(M_c: np.ndarray) -> Dict[str, float]:
    n_c, S, Me = M_c.shape
    freq = M_c.mean(axis=1).astype(np.float64)
    spread = freq.max(axis=0) - freq.min(axis=0)
    corrs: List[float] = []
    for i in range(n_c):
        for j in range(i + 1, n_c):
            a, b = freq[i], freq[j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                corrs.append(1.0)
            else:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "mean_edge_freq_cross_chain_spread": float(spread.mean()),
        "max_edge_freq_cross_chain_spread": float(spread.max()),
        "min_pairwise_chain_freq_pearson_r": float(np.min(corrs)) if corrs else float("nan"),
        "mean_pairwise_chain_freq_pearson_r": float(np.mean(corrs)) if corrs else float("nan"),
    }


def sampling_exhaustiveness_metrics(M: np.ndarray, rng: np.random.Generator) -> Dict[str, float]:
    T, Me = M.shape
    p = M.mean(axis=0).astype(np.float64)
    H = _binary_entropy_bits(p)
    mixed = (p > 0.05) & (p < 0.95)
    sub_n = min(4000, T)
    sub_ix = rng.choice(T, size=sub_n, replace=False)
    sub = M[sub_ix]
    packed = np.packbits(sub, axis=1)
    _, counts = np.unique(packed, axis=0, return_counts=True)
    unique_frac = float(len(counts) / sub_n)
    return {
        "mean_binary_entropy_bits_per_edge": float(H.mean()),
        "median_binary_entropy_bits_per_edge": float(np.median(H)),
        "frac_edges_ambiguous_0p05_0p95": float(mixed.mean()),
        "approx_frac_unique_binary_draws_subsample": unique_frac,
        "subsample_size_for_uniqueness": float(sub_n),
    }


def hamming_average_linkage_only(M: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Average linkage on pairwise cityblock (Hamming) distances; returns Z only + metadata."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy required for hierarchical clustering")
    T, Me = M.shape
    n_pairs = T * (T - 1) // 2
    mem_gb = n_pairs * 8 / (1024**3)
    print(
        f"  pdist: n_draws={T}, n_edges={Me}, condensed pairs={n_pairs} (~{mem_gb:.2f} GiB float64)",
        flush=True,
    )
    Xf = np.ascontiguousarray(M.astype(np.float32, copy=False))
    t0 = time.perf_counter()
    print("  Running pdist (cityblock / Hamming on binary) …", flush=True)
    D = pdist(Xf, metric="cityblock")
    print(f"  pdist done in {time.perf_counter() - t0:.1f}s", flush=True)
    t1 = time.perf_counter()
    Z_link = linkage(D, method="average")
    print(f"  linkage done in {time.perf_counter() - t1:.1f}s", flush=True)
    meta: Dict[str, Any] = {
        "linkage": "average",
        "distance_metric": "cityblock (Hamming sum on 0/1)",
        "n_draws_clustered": int(T),
        "n_edges": int(Me),
    }
    return Z_link, meta


def clusters_sample_ab_from_mixing_details(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten mixing_scan_row 'clusters' entries for JSON/TSV."""
    out: List[Dict[str, Any]] = []
    for d in details:
        row: Dict[str, Any] = {
            "cluster_id": int(d["cluster_id"]),
            "n_draws": int(d["n"]),
            "n_sample_A": int(d["n_A"]),
            "n_sample_B": int(d["n_B"]),
            "frac_sample_A": float(d["n_A"]) / int(d["n"]) if int(d["n"]) > 0 else float("nan"),
            "mixing_tested": bool(d["tested"]),
        }
        pv = d.get("p_value")
        if pv is not None:
            row["mixing_binomial_p_value"] = float(pv)
        out.append(row)
    return sorted(out, key=lambda x: x["cluster_id"])


def attach_clusters_sample_ab_per_tau_row(
    row: Dict[str, Any], mr: Dict[str, Any], max_clusters: int
) -> None:
    """
    Add per-cluster n_sample_A / n_sample_B to one τ-scan row. If too many clusters,
    omit or restrict (see keys clusters_sample_ab_*).
    """
    details: List[Dict[str, Any]] = mr["clusters"]
    n_tot = len(details)
    if n_tot <= max_clusters:
        row["clusters_sample_ab"] = clusters_sample_ab_from_mixing_details(details)
        return
    tested = [d for d in details if d["tested"]]
    row["n_clusters_total"] = n_tot
    if len(tested) <= max_clusters:
        row["clusters_sample_ab_mixing_tested_only"] = True
        row["clusters_sample_ab"] = clusters_sample_ab_from_mixing_details(tested)
        return
    row["clusters_sample_ab_truncated"] = True
    row["n_mixing_tested_clusters"] = len(tested)
    tested_sorted = sorted(tested, key=lambda d: -int(d["n"]))[:max_clusters]
    row["clusters_sample_ab"] = clusters_sample_ab_from_mixing_details(tested_sorted)


def write_cluster_sample_ab_tsv(path: Path, rows: List[Dict[str, Any]]) -> None:
    hdr = (
        "cluster_id\tn_draws\tn_sample_A\tn_sample_B\tfrac_sample_A\t"
        "mixing_tested\tmixing_binomial_p_value\n"
    )
    lines = [hdr]
    for r in rows:
        pv = r.get("mixing_binomial_p_value")
        pv_s = "" if pv is None else f"{float(pv):.10g}"
        lines.append(
            "\t".join(
                [
                    str(r["cluster_id"]),
                    str(r["n_draws"]),
                    str(r["n_sample_A"]),
                    str(r["n_sample_B"]),
                    f"{r['frac_sample_A']:.8f}",
                    str(r["mixing_tested"]),
                    pv_s,
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines))


def chain_id_for_merged(z: np.ndarray, n_chains: int) -> np.ndarray:
    """Merged row order: equal-length chains concatenated → chain index 0..n_chains-1."""
    T = z.shape[0]
    if T % n_chains != 0:
        raise ValueError(f"T={T} not divisible by n_chains={n_chains}")
    per = T // n_chains
    return (np.arange(T, dtype=np.int32) // per).astype(np.int32)


def tau_scan_sampling_precision(
    Z_link: np.ndarray,
    is_a: np.ndarray,
    p0: float,
    min_cluster_size: int,
    alpha: float,
    n_tau: int,
    n_chains: int,
    max_clusters_ab_per_tau_row: int = 2500,
) -> Dict[str, Any]:
    """
    Scan τ from high (coarse) downward. At each τ, A/B chain mixing tests (binomial + Bonferroni).
    Stop at first τ where any tested cluster fails. Last τ that passed is sampling_precision_tau.
    """
    taus_desc = build_tau_grid(Z_link, max(3, int(n_tau)))

    last_tau: Optional[float] = None
    last_lab: Optional[np.ndarray] = None
    last_mr: Optional[Dict[str, Any]] = None
    first_fail_tau: Optional[float] = None
    rows: List[Dict[str, Any]] = []

    for tau in taus_desc:
        lab = fcluster(Z_link, float(tau), criterion="distance").astype(np.int32)
        mr = mixing_scan_row(
            Z_link,
            float(tau),
            is_a,
            p0,
            min_cluster_size,
            alpha,
            labels=lab,
        )
        ok = bool(mr["all_mixing_tests_pass"])
        one_row: Dict[str, Any] = {
            "tau": mr["tau"],
            "n_clusters": mr["n_clusters"],
            "n_clusters_tested": mr["n_clusters_tested"],
            "bonferroni_alpha": mr["bonferroni_alpha"],
            "min_p_value": mr["min_p_value"],
            "all_mixing_tests_pass": ok,
        }
        attach_clusters_sample_ab_per_tau_row(one_row, mr, max_clusters_ab_per_tau_row)
        rows.append(one_row)
        if ok:
            last_tau = float(tau)
            last_lab = lab.copy()
            last_mr = mr
        else:
            first_fail_tau = float(tau)
            break

    if last_tau is None or last_lab is None or last_mr is None:
        raise RuntimeError("τ scan found no partition passing mixing tests")

    _, cnt_final = np.unique(last_lab, return_counts=True)
    clusters_ab_chosen = clusters_sample_ab_from_mixing_details(last_mr["clusters"])
    return {
        "sampling_precision_tau": last_tau,
        "first_tau_failing_mixing_scan_high_to_low": first_fail_tau,
        "n_clusters_at_sampling_precision_tau": int(len(cnt_final)),
        "cluster_sizes_at_sampling_precision_tau": sorted(
            [int(x) for x in cnt_final], reverse=True
        ),
        "labels_at_sampling_precision_tau": last_lab,
        "sample_ab_legend": {
            "sample_A": "draws from first ⌊K/2⌋ MCMC chains (merged order)",
            "sample_B": "draws from remaining chains",
            "chain_indices_sample_A": list(range(n_chains // 2)),
            "chain_indices_sample_B": list(range(n_chains // 2, n_chains)),
            "global_frac_sample_A": float(p0),
        },
        "clusters_sample_ab_at_sampling_precision_tau": clusters_ab_chosen,
        "mixing_test": (
            "Per τ: for each cluster with n >= min_cluster_size, two-sided binomial "
            "H0: P(draw in sample A | cluster) = p0, p0 = global fraction in A (first ⌊K/2⌋ chains); "
            "Bonferroni alpha across clusters tested at that τ."
        ),
        "min_cluster_size_for_mixing_test": int(min_cluster_size),
        "mixing_test_alpha": float(alpha),
        "n_tau_grid_points": len(taus_desc),
        "tau_scan_table": rows,
        "criterion": "scipy.cluster.hierarchy.fcluster(Z, tau, criterion='distance')",
        "scan_order": "tau decreasing (coarse to fine); stop at first τ with a mixing-test failure",
    }


def optional_rhat_from_diagnostics(merged_dir: Path) -> Optional[Dict[str, float]]:
    p = merged_dir / "convergence_diagnostics.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
        z = d.get("z", {})
        rhat = z.get("rhat")
        if rhat is None:
            return None
        arr = np.asarray(rhat, dtype=float)
        return {
            "z_rhat_mean_from_diagnostics": float(np.nanmean(arr)),
            "z_rhat_max_from_diagnostics": float(np.nanmax(arr)),
        }
    except Exception:
        return None


def analyze_one_dataset(
    results_root: Path,
    dataset: str,
    out_dataset_dir: Path,
    cluster_max_draws: Optional[int],
    seed: int,
    *,
    min_cluster_size_mixing: int,
    mixing_test_alpha: float,
    n_tau_balance_scan: int,
    max_clusters_ab_per_tau_row: int,
    reuse_linkage: bool,
    z_binary_threshold: Optional[float],
) -> Dict[str, Any]:
    merged_pkl = results_root / dataset / "merged" / "merged_chains.pkl"
    if not merged_pkl.is_file():
        return {"dataset": dataset, "error": f"missing {merged_pkl}"}

    z, seeds, n_chains = load_merged_z(merged_pkl)
    z_thr = effective_z_binary_threshold(z_binary_threshold)
    M = binary_m_from_z(z, z_threshold=z_binary_threshold)
    M_c = reshape_binary_by_chain(M, n_chains)
    rng = np.random.default_rng(seed)

    if cluster_max_draws is not None and cluster_max_draws < M.shape[0]:
        idx = rng.choice(M.shape[0], size=int(cluster_max_draws), replace=False)
        M_clust = M[idx]
        cluster_note = f"subsampled_indices_n{cluster_max_draws}_seed{seed}"
    else:
        M_clust = M
        idx = None
        cluster_note = "all_merged_draws"

    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    link_dir = out_dataset_dir / "posterior_hamming_linkage"
    link_dir.mkdir(parents=True, exist_ok=True)

    link_path = link_dir / "linkage_matrix.npy"
    n_draw = int(M_clust.shape[0])
    reused = False
    Z_link: Optional[np.ndarray] = None
    link_meta = {}
    if reuse_linkage and idx is None and link_path.is_file():
        Z_try = np.load(link_path)
        if Z_try.ndim == 2 and Z_try.shape[0] == n_draw - 1:
            Z_link = Z_try
            reused = True
            print(f"  Reusing {link_path.name} (skip pdist/linkage), n_draws={n_draw} …", flush=True)
            summ_prev = link_dir / "linkage_summary.json"
            if summ_prev.is_file():
                try:
                    link_meta = json.loads(summ_prev.read_text())
                except Exception:
                    link_meta = {}
            link_meta.setdefault("cluster_input_note", cluster_note)
            link_meta.setdefault("linkage", "average")
            link_meta.setdefault("distance_metric", "cityblock (Hamming sum on 0/1)")
            link_meta["n_draws_clustered"] = n_draw
            link_meta["n_edges"] = int(M_clust.shape[1])
            link_meta["reused_linkage_matrix"] = True
        else:
            print(
                f"  Existing linkage shape {Z_try.shape} != n_draws-1={n_draw - 1}; recomputing …",
                flush=True,
            )

    if not reused:
        print(f"  Hamming linkage ({cluster_note}), shape={M_clust.shape} …", flush=True)
        Z_link, link_meta = hamming_average_linkage_only(M_clust)
        link_meta["cluster_input_note"] = cluster_note
        link_meta.pop("reused_linkage_matrix", None)
        np.save(link_path, Z_link)

    link_meta["partition_note"] = (
        "draw_labels_full_posterior.npy: all draws in one cluster. "
        "draw_labels_sampling_precision.npy: fcluster at sampling_precision_tau "
        "(finest τ where A/B chain mixing tests pass at every cluster; see tau_balance_scan.json)."
    )

    labels_full = np.ones(M.shape[0], dtype=np.int32)
    np.save(link_dir / "draw_labels_full_posterior.npy", labels_full)
    if idx is not None:
        np.save(link_dir / "subsample_row_indices.npy", idx)

    sampling_balance: Dict[str, Any]
    if idx is None and M_clust.shape[0] == M.shape[0]:
        try:
            cid = chain_id_for_merged(z, n_chains)
            is_a, p0 = sample_ab_mask(cid, n_chains)
        except ValueError as e:
            sampling_balance = {"skipped": True, "reason": str(e)}
        else:
            print(
                f"  τ scan (A/B mixing tests: min_cluster≥{min_cluster_size_mixing}, "
                f"α={mixing_test_alpha}, {n_tau_balance_scan} grid points) …",
                flush=True,
            )
            t0 = time.perf_counter()
            spe_full = tau_scan_sampling_precision(
                Z_link,
                is_a=is_a,
                p0=p0,
                min_cluster_size=min_cluster_size_mixing,
                alpha=mixing_test_alpha,
                n_tau=n_tau_balance_scan,
                n_chains=n_chains,
                max_clusters_ab_per_tau_row=max_clusters_ab_per_tau_row,
            )
            print(f"  τ scan done in {time.perf_counter() - t0:.1f}s", flush=True)
            labels_sp = spe_full.pop("labels_at_sampling_precision_tau")
            ab_tsv = link_dir / "cluster_sample_ab_at_sampling_precision.tsv"
            write_cluster_sample_ab_tsv(
                ab_tsv,
                spe_full["clusters_sample_ab_at_sampling_precision_tau"],
            )
            (link_dir / "tau_balance_scan.json").write_text(
                json.dumps(spe_full, indent=2) + "\n"
            )
            sampling_balance = {
                k: v
                for k, v in spe_full.items()
                if k not in ("tau_scan_table", "clusters_sample_ab_at_sampling_precision_tau")
            }
            sampling_balance["cluster_sample_ab_at_sampling_precision_tsv"] = str(
                ab_tsv.resolve()
            )
            np.save(link_dir / "draw_labels_sampling_precision.npy", labels_sp)
    else:
        sampling_balance = {
            "skipped": True,
            "reason": "subsampled linkage (cluster_max_draws); τ labels not aligned to full T",
        }

    link_meta["sampling_precision_balance"] = sampling_balance
    link_meta["z_binary_threshold"] = float(z_thr)

    summary_path = link_dir / "linkage_summary.json"
    summary_path.write_text(json.dumps(link_meta, indent=2) + "\n")

    out_files: Dict[str, str] = {
        "linkage_matrix": str(link_path.resolve()),
        "draw_labels_full_posterior": str((link_dir / "draw_labels_full_posterior.npy").resolve()),
        "linkage_summary_json": str(summary_path.resolve()),
    }
    if not sampling_balance.get("skipped"):
        out_files["draw_labels_sampling_precision"] = str(
            (link_dir / "draw_labels_sampling_precision.npy").resolve()
        )
        out_files["tau_balance_scan_json"] = str((link_dir / "tau_balance_scan.json").resolve())
        tsv_ab = link_dir / "cluster_sample_ab_at_sampling_precision.tsv"
        if tsv_ab.is_file():
            out_files["cluster_sample_ab_at_sampling_precision_tsv"] = str(tsv_ab.resolve())

    out: Dict[str, Any] = {
        "dataset": dataset,
        "merged_chains_pkl": str(merged_pkl.resolve()),
        "binary_rule": f"M = 1 if Z > {z_thr} else 0",
        "z_binary_threshold": float(z_thr),
        "seeds": seeds,
        "n_chains": n_chains,
        "n_samples_total": int(M.shape[0]),
        "n_edges": int(M.shape[1]),
        "samples_per_chain": int(M.shape[0] // n_chains),
        "sampling_precision": sampling_precision_metrics(M_c),
        "sampling_exhaustiveness": sampling_exhaustiveness_metrics(M, rng),
        "diagnostics_z_rhat": optional_rhat_from_diagnostics(merged_pkl.parent),
        "hamming_linkage": link_meta,
        "sampling_precision_balance_tau": sampling_balance,
        "output_files": out_files,
    }
    (out_dataset_dir / "imp_style_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "post_sampling_analysis" / "base_models",
    )
    ap.add_argument(
        "--cluster-max-draws",
        type=int,
        default=None,
        help="If set, random subsample of this many draws for linkage only (faster debug). "
        "Labels for extract_main_cluster remain length T (all draws = cluster 1). "
        "Default: use all merged draws.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reuse-linkage",
        action="store_true",
        help="Load existing posterior_hamming_linkage/linkage_matrix.npy (same n_draws); skip pdist/linkage.",
    )
    ap.add_argument(
        "--min-cluster-size-mixing",
        type=int,
        default=50,
        help="At each τ, only clusters with ≥ this many draws get a chain-mixing binomial test.",
    )
    ap.add_argument(
        "--mixing-test-alpha",
        type=float,
        default=0.05,
        help="Nominal α for two-sided binomial tests (Bonferroni within each τ).",
    )
    ap.add_argument(
        "--n-tau-balance-scan",
        type=int,
        default=400,
        help="Number of τ grid points from max merge height to min (high τ → low τ).",
    )
    ap.add_argument(
        "--max-clusters-ab-per-tau-row",
        type=int,
        default=2500,
        help="Cap per-cluster A/B detail in tau_scan_table when n_clusters is huge.",
    )
    ap.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="NAME",
        help="Run only this results/ subfolder (repeat flag for several). Default: all base models.",
    )
    ap.add_argument(
        "--skip-summary-tsv",
        action="store_true",
        help="Do not write or merge the merged sampling-precision summary TSV.",
    )
    ap.add_argument(
        "--merged-summary-tsv",
        type=Path,
        default=None,
        help=(
            "Merged one-row-per-dataset TSV path (default: <out-dir>/summary_all_base_models.tsv). "
            "For ablations under script/results, pass e.g. script/results/summary_ablations_sampling_precision.tsv"
        ),
    )
    ap.add_argument(
        "--z-binary-threshold",
        type=float,
        default=None,
        help="Per-draw binary M = 1{{Z > thr}}. Default: post_sampling_z_binary.Z_BINARY_THRESHOLD. "
        "Use 0 for legacy Z>0.",
    )
    args = ap.parse_args()

    if not HAS_SCIPY:
        raise SystemExit("scipy is required")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    run_datasets: List[str] = args.datasets if args.datasets else list(BASE_DATASETS)

    for ds in run_datasets:
        print(f"=== {ds} ===", flush=True)
        subdir = args.out_dir / ds
        res = analyze_one_dataset(
            args.results_root,
            ds,
            subdir,
            cluster_max_draws=args.cluster_max_draws,
            seed=args.seed,
            min_cluster_size_mixing=args.min_cluster_size_mixing,
            mixing_test_alpha=args.mixing_test_alpha,
            n_tau_balance_scan=args.n_tau_balance_scan,
            max_clusters_ab_per_tau_row=args.max_clusters_ab_per_tau_row,
            reuse_linkage=args.reuse_linkage,
            z_binary_threshold=args.z_binary_threshold,
        )
        print(f"Wrote {subdir / 'imp_style_metrics.json'}", flush=True)
        summary_rows.append(res)

    tsv_lines: List[str] = []
    for r in summary_rows:
        if "error" in r:
            tsv_lines.append(
                "\t".join(
                    [
                        r["dataset"],
                        "",
                        "",
                        r.get("error", ""),
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "",
                        "",
                    ]
                )
                + "\n"
            )
            continue
        sp = r["sampling_precision"]
        se = r["sampling_exhaustiveness"]
        hl = r.get("hamming_linkage") or {}
        dr = r.get("diagnostics_z_rhat") or {}
        sb = r.get("sampling_precision_balance_tau") or {}
        tau_sp = sb.get("sampling_precision_tau", "")
        ncl_sp = sb.get("n_clusters_at_sampling_precision_tau", "")
        if isinstance(tau_sp, float):
            tau_sp = f"{tau_sp:.8g}"
        tsv_lines.append(
            "\t".join(
                [
                    r["dataset"],
                    str(r["n_chains"]),
                    str(r["n_samples_total"]),
                    hl.get("cluster_input_note", ""),
                    f"{sp['mean_edge_freq_cross_chain_spread']:.6f}",
                    f"{sp['min_pairwise_chain_freq_pearson_r']:.6f}",
                    f"{se['mean_binary_entropy_bits_per_edge']:.6f}",
                    f"{se['frac_edges_ambiguous_0p05_0p95']:.6f}",
                    f"{se['approx_frac_unique_binary_draws_subsample']:.6f}",
                    f"{dr.get('z_rhat_max_from_diagnostics', float('nan')):.6f}",
                    str(tau_sp),
                    str(ncl_sp),
                ]
            )
            + "\n"
        )

    if args.skip_summary_tsv:
        print("Skipped merged sampling-precision summary TSV (--skip-summary-tsv).", flush=True)
    else:
        tsv_path = (
            args.merged_summary_tsv.resolve()
            if args.merged_summary_tsv is not None
            else (args.out_dir / "summary_all_base_models.tsv")
        )
        header = (
            "dataset\tn_chains\tn_samples_total\tlinkage_input\t"
            "mean_cross_chain_freq_spread\tmin_chain_freq_pearson_r\t"
            "mean_edge_binary_entropy_bits\tfrac_edges_ambiguous\tapprox_frac_unique_draws_subsample\t"
            "z_rhat_max\tsampling_precision_tau\tn_clusters_at_tau\n"
        )
        merged: Dict[str, str] = {}
        if args.datasets and tsv_path.is_file():
            for line in tsv_path.read_text().splitlines()[1:]:
                if line.strip() and "\t" in line:
                    merged[line.split("\t", 1)[0]] = line + "\n"
        for line in tsv_lines:
            if line.strip() and "\t" in line:
                merged[line.split("\t", 1)[0]] = line
        order = [d for d in BASE_DATASETS if d in merged]
        order.extend([d for d in ABLATION_DATASETS if d in merged and d not in order])
        order.extend(sorted(set(merged) - set(order)))
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text(header + "".join(merged[d] for d in order))
        print(f"Wrote {tsv_path}", flush=True)


if __name__ == "__main__":
    main()
