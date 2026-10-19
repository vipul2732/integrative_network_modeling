"""
Convert posterior Z samples to binary network ensemble M (per draw).

Rule: M_edge = 1{Z > Z_thr} (Z_thr from post_sampling_z_binary; default 0.5).

Handles:
  - merged_chains.pkl (merged_samples) or single-run pickles (samples)
  - shapes (T, M), (C, S, M) flattened to (C*S, M)
  - edge_filter_map: expands filtered z to full M_full = N(N-1)/2 upper-tri order
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from post_sampling_z_binary import effective_z_binary_threshold

logger = logging.getLogger(__name__)


def load_z_from_results_pkl(path: Path) -> np.ndarray:
    """Load z from merged or single-chain MCMC pickle; return float array (T, M)."""
    z, _ = load_z_from_results_pkl_with_chains(path)
    return z


def load_z_from_results_pkl_with_chains(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load z and optional chain_id per draw (0..C-1) when z was stored as (C, S, M).
    If z is 2D, chain_ids is None (use equal split via n_chains when analyzing).
    """
    with open(path, "rb") as f:
        results = pickle.load(f)

    if "merged_samples" in results:
        samples = results["merged_samples"]
    elif "samples" in results:
        samples = results["samples"]
    else:
        raise ValueError(f"No 'merged_samples' or 'samples' in {path}")

    if "z" not in samples:
        raise ValueError(f"No 'z' in samples: {path}")

    z = samples["z"]
    if hasattr(z, "block_until_ready"):
        z = np.array(z)
    else:
        z = np.asarray(z, dtype=np.float64)

    chain_ids: Optional[np.ndarray] = None
    if z.ndim == 3:
        c, s, m = z.shape
        chain_ids = np.repeat(np.arange(c, dtype=np.int32), s)
        z = z.reshape(-1, m)
        logger.info("Reshaped z from (n_chains=%s, n_samples=%s, M=%s) to (%s, %s)", c, s, m, z.shape[0], m)
    elif z.ndim == 2:
        pass
    elif z.ndim == 1:
        z = z.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected z ndim={z.ndim}, shape={z.shape}")

    return z, chain_ids


def load_model_data(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def find_model_data_pkl(merged_chains_pkl: Path) -> Optional[Path]:
    """
    Heuristic: .../<dataset>/merged/merged_chains.pkl -> .../<dataset>/seed_*/0_*_model_data.pkl
    """
    merged_dir = merged_chains_pkl.parent
    dataset_dir = merged_dir.parent
    if not dataset_dir.is_dir():
        return None
    candidates = sorted(dataset_dir.glob("seed_*/*_model_data.pkl"))
    if not candidates:
        candidates = sorted(dataset_dir.glob("*_model_data.pkl"))
    return candidates[0] if candidates else None


def expand_filtered_z_to_full(
    z_row: np.ndarray,
    n_nodes: int,
    edge_filter_map: Optional[np.ndarray],
) -> np.ndarray:
    """
    z_row: (M_filt,) or (M_full,). Return length M_full upper-tri row-major vector.
    """
    m_full = n_nodes * (n_nodes - 1) // 2
    if edge_filter_map is None:
        if z_row.size != m_full:
            raise ValueError(f"z length {z_row.size} != M_full {m_full} and no edge_filter_map")
        return z_row.astype(np.float64, copy=False)

    ef = np.asarray(edge_filter_map, dtype=np.int64)
    if z_row.size != ef.size:
        raise ValueError(f"z length {z_row.size} != len(edge_filter_map) {ef.size}")
    out = np.zeros(m_full, dtype=np.float64)
    out[ef] = z_row
    return out


def z_row_to_binary_full(
    z_row: np.ndarray,
    n_nodes: int,
    edge_filter_map: Optional[np.ndarray],
    z_binary_threshold: Optional[float] = None,
) -> np.ndarray:
    full = expand_filtered_z_to_full(z_row, n_nodes, edge_filter_map)
    thr = effective_z_binary_threshold(z_binary_threshold)
    return (full > thr).astype(np.uint8)


def flat_upper_to_adj(flat_upper: np.ndarray, n_nodes: int) -> np.ndarray:
    """Upper-tri row-major length M_full -> symmetric (N,N) uint8."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    tri_i, tri_j = np.triu_indices(n_nodes, k=1)
    if flat_upper.size != tri_i.size:
        raise ValueError(f"flat_upper size {flat_upper.size} != {tri_i.size}")
    adj[tri_i, tri_j] = flat_upper
    adj[tri_j, tri_i] = flat_upper
    return adj


def build_binary_ensemble(
    z: np.ndarray,
    n_nodes: int,
    edge_filter_map: Optional[np.ndarray],
    z_binary_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """
    Returns:
        binary_full: (T, M_full) uint8
        m_full: N(N-1)/2
    """
    m_full = n_nodes * (n_nodes - 1) // 2
    t = z.shape[0]
    out = np.zeros((t, m_full), dtype=np.uint8)
    for i in range(t):
        out[i] = z_row_to_binary_full(z[i], n_nodes, edge_filter_map, z_binary_threshold)
    return out, m_full


def protein_names_from_model_data(model_data: Dict[str, Any]) -> List[str]:
    n = int(model_data["N"])
    node_idx2name = model_data.get("node_idx2name") or {}
    return [str(node_idx2name.get(i, f"node_{i}")) for i in range(n)]


def infer_chain_ids_equal_split(n_draws: int, n_chains: int) -> np.ndarray:
    """Assume draws are ordered as chain0 block, chain1 block, … (same as flattening (C,S,M))."""
    if n_draws % n_chains != 0:
        raise ValueError(f"n_draws {n_draws} not divisible by n_chains {n_chains}")
    s = n_draws // n_chains
    return np.repeat(np.arange(n_chains, dtype=np.int32), s)


def load_binary_ensemble(ensemble_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load binary_edges_full.npy + metadata.json from a run_z_to_m output folder."""
    ensemble_dir = Path(ensemble_dir)
    with open(ensemble_dir / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(ensemble_dir / "binary_edges_full.npy")
    return np.asarray(X, dtype=np.uint8), meta


def save_ensemble(
    output_dir: Path,
    binary_full: np.ndarray,
    n_nodes: int,
    protein_names: List[str],
    meta_extra: Optional[Dict[str, Any]] = None,
    save_adjacency: bool = False,
    chain_id_per_draw: Optional[np.ndarray] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "binary_edges_full.npy", binary_full)

    meta: Dict[str, Any] = {
        "n_nodes": n_nodes,
        "m_full": int(n_nodes * (n_nodes - 1) // 2),
        "n_draws": int(binary_full.shape[0]),
        "threshold_rule": "set by caller meta_extra (Z > z_thr; see post_sampling_z_binary.py)",
        "protein_names": protein_names,
    }
    if meta_extra:
        meta.update(meta_extra)

    if chain_id_per_draw is not None:
        meta["has_chain_id_per_draw"] = True

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if chain_id_per_draw is not None:
        np.save(output_dir / "chain_id_per_draw.npy", np.asarray(chain_id_per_draw, dtype=np.int32))

    if save_adjacency:
        t, mf = binary_full.shape
        assert mf == n_nodes * (n_nodes - 1) // 2
        adj = np.zeros((t, n_nodes, n_nodes), dtype=np.uint8)
        for i in range(t):
            adj[i] = flat_upper_to_adj(binary_full[i], n_nodes)
        np.save(output_dir / "binary_adjacency.npy", adj)
        logger.info("Wrote binary_adjacency.npy shape %s", adj.shape)


def run_pipeline(
    merged_or_results_pkl: Path,
    model_data_pkl: Optional[Path],
    output_dir: Path,
    save_adjacency: bool = False,
) -> Path:
    merged_or_results_pkl = Path(merged_or_results_pkl).resolve()
    output_dir = Path(output_dir).resolve()

    z, chain_ids = load_z_from_results_pkl_with_chains(merged_or_results_pkl)

    md_path = model_data_pkl
    if md_path is None or not md_path.exists():
        md_path = find_model_data_pkl(merged_or_results_pkl)
    if md_path is None or not md_path.exists():
        raise FileNotFoundError(
            "model_data pickle required (for N and edge_filter_map). "
            "Pass --model-data or place seed_*/*_model_data.pkl next to merged results."
        )

    model_data = load_model_data(md_path)
    n_nodes = int(model_data["N"])
    edge_filter_map = model_data.get("edge_filter_map")
    if edge_filter_map is not None:
        edge_filter_map = np.asarray(edge_filter_map, dtype=np.int64)

    z_thr = float(effective_z_binary_threshold(None))
    binary_full, m_full = build_binary_ensemble(z, n_nodes, edge_filter_map, z_binary_threshold=None)

    with open(merged_or_results_pkl, "rb") as f:
        results = pickle.load(f)
    meta_extra: Dict[str, Any] = {
        "source_pkl": str(merged_or_results_pkl),
        "model_data_pkl": str(md_path),
        "z_shape_input": list(z.shape),
        "has_edge_filter": edge_filter_map is not None,
        "z_binary_threshold": z_thr,
        "threshold_rule": f"M_edge = 1 if Z_edge > {z_thr} else 0",
    }
    if chain_ids is not None:
        u = np.unique(chain_ids)
        meta_extra["n_chains"] = int(u.size)
        meta_extra["samples_per_chain"] = int(z.shape[0] // u.size) if u.size else 0
    for key in ("seeds", "n_chains", "burn_in"):
        if key in results:
            meta_extra[key] = results[key]

    names = protein_names_from_model_data(model_data)
    save_ensemble(
        output_dir,
        binary_full,
        n_nodes,
        names,
        meta_extra,
        save_adjacency=save_adjacency,
        chain_id_per_draw=chain_ids,
    )
    logger.info("Wrote %s draws, M_full=%s, N=%s -> %s", binary_full.shape[0], m_full, n_nodes, output_dir)
    return output_dir
