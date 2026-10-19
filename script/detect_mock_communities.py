#!/usr/bin/env python3
"""
Detect communities in Mock condition network.

This script performs community detection on the Mock condition network
and generates visualization similar to WT and VIF conditions.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from community_binary_graph import (  # noqa: E402
    build_graph_from_main_cluster_binary,
    default_model_data_pkl,
    load_model_data,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Try to import scipy for convex hull
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Try to import NetworkX community module
try:
    import networkx.algorithms.community as nx_comm
    HAS_NX_COMM = True
except ImportError:
    HAS_NX_COMM = False

# -----------------------------
# Logging / style
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12


# -----------------------------
# I/O + network build
# -----------------------------
def load_interactions(file_path: Path, condition: str, min_prob: float = 0.6) -> Tuple[pd.DataFrame, str]:
    """Load interactions TSV and return filtered df + probability column name."""
    logger.info(f"Loading {condition} interactions from {file_path}")
    df = pd.read_csv(file_path, sep="\t")

    # Replace PEBB with CBFB (if present)
    if "Protein1" in df.columns:
        df["Protein1"] = df["Protein1"].replace("PEBB", "CBFB")
    if "Protein2" in df.columns:
        df["Protein2"] = df["Protein2"].replace("PEBB", "CBFB")

    # Determine probability column
    if "Probability" in df.columns:
        prob_col = "Probability"
    elif "Mock_Probability" in df.columns:
        prob_col = "Mock_Probability"
    else:
        raise ValueError(
            f"No probability column found in {file_path}. Expected 'Probability' or 'Mock_Probability'."
        )

    # Filter & sort
    df = df[df[prob_col] >= min_prob].copy()
    df = df.sort_values(prob_col, ascending=False).copy()

    logger.info(f"  Loaded {len(df)} interactions with {prob_col} >= {min_prob}")
    return df, prob_col


def build_network(df: pd.DataFrame, prob_col: str) -> nx.Graph:
    """Build weighted undirected graph from interactions."""
    G = nx.Graph()
    for _, row in df.iterrows():
        p1 = str(row["Protein1"]).strip()
        p2 = str(row["Protein2"]).strip()
        w = float(row[prob_col])
        if p1 and p2 and p1 != "nan" and p2 != "nan" and p1 != p2:
            G.add_edge(p1, p2, weight=w, probability=w)
    logger.info(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# -----------------------------
# Community detection
# -----------------------------
def _call_with_optional_weight(func, G: nx.Graph, resolution: float, seed: int = 42):
    """NetworkX API differs by version; try weight='weight' then fall back."""
    try:
        return func(G, resolution=resolution, seed=seed, weight="weight")
    except TypeError:
        return func(G, resolution=resolution, seed=seed)


def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0) -> Tuple[Dict[str, int], List[Set[str]]]:
    if not HAS_NX_COMM:
        logger.warning("NetworkX community detection not available")
        return {}, []
    try:
        comms = _call_with_optional_weight(nx_comm.louvain_communities, G, resolution=resolution, seed=42)
        node_to_comm = {}
        comm_list = [set(c) for c in comms]
        for cid, nodes in enumerate(comm_list):
            for n in nodes:
                node_to_comm[n] = cid
        logger.info(f"  Detected {len(comm_list)} communities (Louvain, resolution={resolution})")
        return node_to_comm, comm_list
    except Exception as e:
        logger.warning(f"  Louvain failed: {e}")
        return {}, []


def detect_communities_leiden(G: nx.Graph, resolution: float = 1.0) -> Tuple[Dict[str, int], List[Set[str]]]:
    if not HAS_NX_COMM:
        logger.warning("NetworkX community detection not available")
        return {}, []
    try:
        comms = _call_with_optional_weight(nx_comm.leiden_communities, G, resolution=resolution, seed=42)
        node_to_comm = {}
        comm_list = [set(c) for c in comms]
        for cid, nodes in enumerate(comm_list):
            for n in nodes:
                node_to_comm[n] = cid
        logger.info(f"  Detected {len(comm_list)} communities (Leiden, resolution={resolution})")
        return node_to_comm, comm_list
    except Exception as e:
        logger.warning(f"  Leiden failed: {e}")
        return {}, []


def detect_communities_greedy_modularity(G: nx.Graph) -> Tuple[Dict[str, int], List[Set[str]]]:
    if not HAS_NX_COMM:
        logger.warning("NetworkX community detection not available")
        return {}, []
    try:
        comms = nx_comm.greedy_modularity_communities(G, weight="weight")
    except TypeError:
        comms = nx_comm.greedy_modularity_communities(G)

    comm_list = [set(c) for c in comms]
    node_to_comm = {}
    for cid, nodes in enumerate(comm_list):
        for n in nodes:
            node_to_comm[n] = cid
    logger.info(f"  Detected {len(comm_list)} communities (Greedy modularity)")
    return node_to_comm, comm_list


def calculate_modularity(G: nx.Graph, communities: List[Set[str]]) -> float:
    if not HAS_NX_COMM or not communities:
        return 0.0
    try:
        return float(nx_comm.modularity(G, communities, weight="weight"))
    except TypeError:
        return float(nx_comm.modularity(G, communities))
    except Exception as e:
        logger.warning(f"  Modularity calculation failed: {e}")
        return 0.0


# -----------------------------
# Packed community layout (no overlap, minimal whitespace)
# -----------------------------
def create_community_aware_layout(
    G: nx.Graph,
    communities: List[Set[str]],
    seed: int = 42,
    intra_k: float = 0.9,
    intra_iter: int = 90,
    cluster_scale: float = 1.35,
    gap: float = 0.65,
    pack_iter: int = 600,
) -> Dict[str, Tuple[float, float]]:
    """
    Publication-friendly layout:
    - spring layout within each community
    - estimate a radius for each community
    - pack community centers with collision-avoidance so blobs don't overlap
    - combine local positions + packed centers
    """
    import math

    rng = np.random.default_rng(seed)

    comm_local: List[Dict[str, np.ndarray]] = []
    comm_radii: List[float] = []
    comm_centers: List[np.ndarray] = []

    # A) internal layout per community around origin
    for cid, nodeset in enumerate(communities):
        nodes = list(nodeset)
        if len(nodes) == 0:
            comm_local.append({})
            comm_radii.append(0.0)
            comm_centers.append(np.array([0.0, 0.0]))
            continue

        if len(nodes) == 1:
            local = {nodes[0]: np.array([0.0, 0.0])}
        else:
            sub = G.subgraph(nodes)
            try:
                local_nx = nx.spring_layout(
                    sub, seed=seed + cid, k=intra_k, iterations=intra_iter, weight="weight"
                )
            except TypeError:
                local_nx = nx.spring_layout(sub, seed=seed + cid, k=intra_k, iterations=intra_iter)

            local = {n: np.array(local_nx[n], dtype=float) for n in nodes}

        pts = np.vstack(list(local.values()))
        center = pts.mean(axis=0)
        pts = pts - center
        max_norm = float(np.max(np.linalg.norm(pts, axis=1))) if len(nodes) > 1 else 1.0
        max_norm = max(max_norm, 1e-6)
        pts = (pts / max_norm) * cluster_scale

        local2: Dict[str, np.ndarray] = {}
        for i, n in enumerate(nodes):
            local2[n] = pts[i]
        comm_local.append(local2)

        rad = float(np.max(np.linalg.norm(pts, axis=1))) + 0.35
        comm_radii.append(rad)
        comm_centers.append(np.array([0.0, 0.0]))

    nC = len(communities)
    if nC == 0:
        return {}

    # B) initialize centers on a circle
    total_r = sum(comm_radii) + gap * nC
    circle_R = max(2.0, total_r / (2 * math.pi))
    for i in range(nC):
        ang = 2 * math.pi * i / nC
        comm_centers[i] = np.array([circle_R * math.cos(ang), circle_R * math.sin(ang)])

    # C) collision-avoidance packing
    for _ in range(pack_iter):
        moved = False
        for i in range(nC):
            for j in range(i + 1, nC):
                ci, cj = comm_centers[i], comm_centers[j]
                diff = cj - ci
                dist = float(np.linalg.norm(diff))
                min_dist = comm_radii[i] + comm_radii[j] + gap
                if dist < min_dist:
                    if dist < 1e-6:
                        diff = rng.normal(size=2)
                        dist = float(np.linalg.norm(diff))
                    direction = diff / dist
                    push = (min_dist - dist) * 0.5
                    comm_centers[i] -= direction * push
                    comm_centers[j] += direction * push
                    moved = True
        if not moved:
            break

    # D) combine local + centers
    pos: Dict[str, Tuple[float, float]] = {}
    for cid, local in enumerate(comm_local):
        C = comm_centers[cid]
        for n, p in local.items():
            xy = C + p
            pos[n] = (float(xy[0]), float(xy[1]))

    # E) mild rescale to reduce whitespace variability
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if len(xs) > 0:
        sx = max(xs.max() - xs.min(), 1e-6)
        sy = max(ys.max() - ys.min(), 1e-6)
        mx, my = xs.mean(), ys.mean()
        for n in pos:
            x, y = pos[n]
            x = (x - mx) / sx * 10.0
            y = (y - my) / sy * 10.0
            pos[n] = (x, y)

    return pos


# -----------------------------
# Visualization
# -----------------------------
def visualize_network_with_communities(
    G: nx.Graph,
    node_to_community: Dict[str, int],
    communities: List[Set[str]],
    title: str,
    output_file: Path,
    figsize: tuple = (18, 12),
    label_mode: str = "all_bold",     # "all_bold" or "none"
    dpi: int = 600,
    intra_k: float = 0.9,
    intra_iter: int = 90,
    cluster_scale: float = 1.35,
    node_size: float = 1500.0,
    max_labels_per_community: int | None = None,
    force_labels: List[str] | None = None,
):
    """
    Publication-friendly network visualization.
    label_mode:
      - "all_bold": show all labels bold
      - "none": hide all labels (legend only; often best for journals)
    """
    logger.info(f"Creating publication-quality network visualization: {title}")

    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon, Circle

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")

    pos = create_community_aware_layout(
        G, communities, seed=42,
        intra_k=float(intra_k), intra_iter=int(intra_iter),
        cluster_scale=float(cluster_scale), gap=0.65, pack_iter=600
    )

    nC = len(communities)
    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    colors = [mcolors.to_rgba(base_colors[i % len(base_colors)]) for i in range(nC)]

    # Hulls first (behind nodes)
    if HAS_SCIPY:
        for cid, nodes in enumerate(communities):
            if not nodes:
                continue
            pts = np.array([pos[n] for n in nodes], dtype=float)
            if len(nodes) < 3:
                center = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)) + 0.55) if len(nodes) > 1 else 0.9
                ax.add_patch(Circle(center, radius, facecolor=colors[cid], edgecolor=colors[cid],
                                    alpha=0.10, linewidth=2.0, zorder=0))
                continue
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                ax.add_patch(Polygon(hp, closed=True, facecolor=colors[cid], edgecolor=colors[cid],
                                     alpha=0.10, linewidth=2.0, zorder=0))
            except Exception:
                center = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)) + 0.55)
                ax.add_patch(Circle(center, radius, facecolor=colors[cid], edgecolor=colors[cid],
                                    alpha=0.10, linewidth=2.0, zorder=0))

    # Edges within/between
    within_edges, between_edges = [], []
    for u, v in G.edges():
        cu = node_to_community.get(u, -1)
        cv = node_to_community.get(v, -1)
        if cu == cv and cu != -1:
            within_edges.append((u, v))
        else:
            between_edges.append((u, v))

    if between_edges:
        nx.draw_networkx_edges(G, pos, edgelist=between_edges,
                               alpha=0.10, width=0.4, edge_color="lightgray", ax=ax)
    if within_edges:
        nx.draw_networkx_edges(G, pos, edgelist=within_edges,
                               alpha=0.35, width=1.0, edge_color="gray", ax=ax)

    # Nodes
    for cid, nodes in enumerate(communities):
        nl = list(nodes)
        if not nl:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=nl,
            node_color=[colors[cid]],
            node_size=float(node_size), alpha=0.95,
            edgecolors="white", linewidths=2.5, ax=ax
        )

    # Labels
    if label_mode not in {"all_bold", "none"}:
        label_mode = "all_bold"

    if label_mode == "all_bold":
        if max_labels_per_community is None:
            label_nodes = list(G.nodes())
        else:
            label_nodes = []
            for nodes in communities:
                if not nodes:
                    continue
                sub = G.subgraph(nodes)
                ranked = sorted(sub.degree(), key=lambda x: x[1], reverse=True)
                keep = [n for n, _ in ranked[: int(max_labels_per_community)]]
                label_nodes.extend(keep)
        force_labels = force_labels or []
        label_nodes = sorted(set(label_nodes) | {x for x in force_labels if x in G.nodes()})
        for node in label_nodes:
            x, y = pos[node]
            ax.text(
                x, y, node,
                fontsize=10,
                fontweight="bold",
                color="#1a1a1a",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.72),
                family="sans-serif",
                zorder=3,
            )

    # Legend
    handles = []
    for cid, nodes in enumerate(communities):
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[cid], markersize=14,
                       markeredgecolor="black", markeredgewidth=1.5,
                       label=f"Community {cid} ({len(nodes)})")
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=13, frameon=True, edgecolor="#cccccc",
              facecolor="white", framealpha=0.95)

    ax.axis("off")

    # Tight bounds
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        m = 0.7
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_file, bbox_inches="tight", facecolor="white", dpi=dpi, pad_inches=0.06)
    plt.close()
    logger.info(f"Saved network visualization to {output_file}")


# -----------------------------
# Reporting + outputs
# -----------------------------
def generate_report(
    network_stats: Dict,
    communities: List[Set[str]],
    output_file: Path,
    edge_source: str = "",
):
    """Generate a text report for Mock network."""
    logger.info("Generating report")

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MOCK CONDITION COMMUNITY DETECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        if edge_source:
            f.write(f"Edge source: {edge_source}\n\n")

        f.write("NETWORK STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Nodes: {network_stats['nodes']}\n")
        f.write(f"Edges: {network_stats['edges']}\n")
        f.write(f"Communities: {network_stats['num_communities']}\n")
        f.write(f"Modularity: {network_stats['modularity']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMUNITY DETAILS\n")
        f.write("-" * 80 + "\n")

        for cid, comm in enumerate(communities):
            f.write(f"  Community {cid}: {len(comm)} proteins\n")

    logger.info(f"Saved report to {output_file}")


def save_community_assignments(
    node_to_community: Dict[str, int],
    communities: List[Set[str]],
    output_file: Path,
    condition: str,
):
    """Save community assignments TSV."""
    logger.info(f"Saving community assignments for {condition}")
    rows = []
    for node, cid in node_to_community.items():
        size = len(communities[cid]) if 0 <= cid < len(communities) else 0
        rows.append({"Protein": node, "Community_ID": cid, "Community_Size": size})
    out = pd.DataFrame(rows).sort_values(["Community_ID", "Protein"])
    out.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved community assignments to {output_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Detect communities in Mock condition network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mock-file",
        type=Path,
        default="results/cullin_mock_config_multi_seeds/all_pairs_with_uniprot_mock.tsv",
        help="Mock all pairs TSV (used when --edge-source probability)",
    )
    parser.add_argument(
        "--edge-source",
        type=str,
        default="main_cluster_binary",
        choices=["probability", "main_cluster_binary"],
        help="probability: threshold all-pairs TSV; main_cluster_binary: main-cluster M npy",
    )
    parser.add_argument(
        "--mock-binary-npy",
        type=Path,
        default=(
            "post_sampling_analysis/base_models/cullin_mock_config_multi_seeds/"
            "main_cluster_network/main_cluster_binary_freq_ge_0p6.npy"
        ),
        help="Main-cluster binary edge vector for Mock graph",
    )
    parser.add_argument(
        "--mock-results-root",
        type=Path,
        default="results",
        help="Contains cullin_mock_config_multi_seeds/merged + model_data",
    )
    parser.add_argument(
        "--mock-dataset",
        type=str,
        default="cullin_mock_config_multi_seeds",
        help="Subfolder under --mock-results-root for model_data.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="wt_vif_comparison",
        help="Output directory",
    )
    parser.add_argument("--min-prob", type=float, default=0.6, help="Minimum probability threshold")
    parser.add_argument(
        "--method",
        type=str,
        default="louvain",
        choices=["louvain", "leiden", "greedy"],
        help="Community detection method",
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution for Louvain/Leiden")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="all_bold",
        choices=["all_bold", "none"],
        help="Label mode for plots (all_bold = all node labels bold; none = hide all labels)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures (journal-ready: 600)")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    if args.edge_source == "main_cluster_binary":
        binary_path = root / args.mock_binary_npy if not args.mock_binary_npy.is_absolute() else args.mock_binary_npy
        res_root = root / args.mock_results_root if not args.mock_results_root.is_absolute() else args.mock_results_root
        md_pkl = default_model_data_pkl(res_root, args.mock_dataset)
        logger.info(f"Mock graph from binary M: {binary_path}")
        model_data = load_model_data(md_pkl)
        G_mock = build_graph_from_main_cluster_binary(binary_path, model_data)
        plot_title = "Mock Network Communities (main-cluster M)"
    else:
        mock_path = root / args.mock_file if not args.mock_file.is_absolute() else args.mock_file
        df_mock, prob_mock = load_interactions(mock_path, "Mock", min_prob=args.min_prob)
        G_mock = build_network(df_mock, prob_mock)
        plot_title = "Mock Network Communities"

    # Detect communities
    logger.info(f"\nDetecting communities using {args.method}...")
    if args.method == "louvain":
        node_to_mock, comms_mock = detect_communities_louvain(G_mock, args.resolution)
    elif args.method == "leiden":
        node_to_mock, comms_mock = detect_communities_leiden(G_mock, args.resolution)
    else:
        node_to_mock, comms_mock = detect_communities_greedy_modularity(G_mock)

    if not node_to_mock or not comms_mock:
        logger.error("Community detection failed (empty results). Check NetworkX version / method availability.")
        return

    # Modularity
    mod_mock = calculate_modularity(G_mock, comms_mock)

    results = {
        "nodes": G_mock.number_of_nodes(),
        "edges": G_mock.number_of_edges(),
        "num_communities": len(comms_mock),
        "modularity": mod_mock,
    }

    # Report
    generate_report(
        results,
        comms_mock,
        args.output_dir / "mock_community_report.txt",
        edge_source=args.edge_source,
    )

    # Save assignments
    save_community_assignments(node_to_mock, comms_mock, args.output_dir / "mock_communities.tsv", "Mock")

    # Visualize
    visualize_network_with_communities(
        G_mock, node_to_mock, comms_mock,
        plot_title,
        args.output_dir / "mock_network_communities.png",
        label_mode=args.label_mode,
        dpi=args.dpi,
    )

    logger.info("=" * 80)
    logger.info("Mock community detection complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
