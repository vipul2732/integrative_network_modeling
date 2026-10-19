#!/usr/bin/env python3
"""
Compare community detection between two networks (default: WT vs VIF KO centroid M).

FULL UPDATED SCRIPT (publication-oriented)

Key updates (visual quality + correctness):
- Default (--edge-source main_cluster_binary): both graphs from **updated main-cluster
  centroid hard binary M** only (WT and VIF KO `main_cluster_binary_*.npy`); no TSV lists.
- Legacy TSV∩KO filter: `--edge-source tsv_binary_intersection` (old “VIF-dependent” hybrid).
- `--edge-source probability`: threshold all-pairs TSVs only.
- Partition comparison is label-invariant (ARI/NMI if sklearn available + best-match Jaccard).
- New community layout:
    * spring layout within each community
    * pack community “blobs” with collision-avoidance to prevent overlaps
    * no global spring refinement (prevents overlap/whitespace artifacts)
- Visualization:
    * consistent packed layout
    * convex hulls drawn first (behind nodes)
    * label_mode: "all_bold" or "none" (no more “key proteins only” bolding)
    * higher DPI default for journal figures

Outputs (default paths under --output-dir):
- community_comparison_report.txt
- vif_dependent_communities.tsv  (partition for graph 1; default = WT centroid M)
- vif_ko_communities.tsv         (partition for graph 2; default = VIF KO centroid M)
- vif_dependent_network_communities.png
- vif_ko_network_communities.png
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

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

# Try to import sklearn metrics (optional)
try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
    elif "WT_Probability" in df.columns:
        prob_col = "WT_Probability"
    else:
        raise ValueError(
            f"No probability column found in {file_path}. Expected 'Probability' or 'WT_Probability'."
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


def build_vif_dependent_graph_binary_filtered(
    df_dep: pd.DataFrame,
    prob_col: str,
    G_ko_binary: nx.Graph,
) -> nx.Graph:
    """
    Rows of df_dep should already satisfy the probability cutoff. Keep an edge if the
    pair is ON in the VIF KO main-cluster binary graph, or if either endpoint is
    vifprotein (TSV-only), with weight from prob_col for vif edges and unit weight otherwise.
    """
    G = nx.Graph()
    for _, row in df_dep.iterrows():
        p1 = str(row["Protein1"]).strip()
        p2 = str(row["Protein2"]).strip()
        if not p1 or not p2 or p1 == "nan" or p2 == "nan" or p1 == p2:
            continue
        low = (p1.lower(), p2.lower())
        if "vifprotein" in low:
            w = float(row[prob_col])
            G.add_edge(p1, p2, weight=w, probability=w)
        elif G_ko_binary.has_edge(p1, p2):
            G.add_edge(p1, p2, weight=1.0, probability=1.0)
    logger.info(
        f"  VIF-dependent (binary-filtered): {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
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
# Comparison (label-invariant)
# -----------------------------
def compare_communities(
    comm1: Dict[str, int],
    comm2: Dict[str, int],
    communities1: List[Set[str]],
    communities2: List[Set[str]],
    name1: str,
    name2: str,
) -> Dict:
    """
    Compare partitions across conditions without assuming community IDs match.
    Metrics:
    - shared proteins
    - ARI + NMI on shared proteins (if sklearn available)
    - average best-match Jaccard overlap of communities (on shared proteins)
    """
    logger.info(f"\nComparing communities: {name1} vs {name2}")

    nodes1 = set(comm1.keys())
    nodes2 = set(comm2.keys())
    shared = sorted(nodes1 & nodes2)
    shared_set = set(shared)

    ari: Optional[float] = None
    nmi: Optional[float] = None
    if HAS_SKLEARN and shared:
        labels1 = [comm1[p] for p in shared]
        labels2 = [comm2[p] for p in shared]
        ari = float(adjusted_rand_score(labels1, labels2))
        nmi = float(normalized_mutual_info_score(labels1, labels2))
    elif not HAS_SKLEARN:
        logger.warning("  sklearn not available; ARI/NMI will be omitted.")

    # Best-match Jaccard (community-wise)
    jaccard_scores: List[float] = []
    for c1 in communities1:
        c1s = c1 & shared_set
        if not c1s:
            continue
        best = 0.0
        for c2 in communities2:
            c2s = c2 & shared_set
            if not c2s:
                continue
            inter = len(c1s & c2s)
            union = len(c1s | c2s)
            best = max(best, inter / union if union else 0.0)
        jaccard_scores.append(best)
    avg_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0

    comm1_specific = nodes1 - nodes2
    comm2_specific = nodes2 - nodes1

    out = {
        "total_proteins_1": len(nodes1),
        "total_proteins_2": len(nodes2),
        "shared_proteins": len(shared),
        "avg_jaccard_similarity": avg_jaccard,
        "comm1_specific_proteins": len(comm1_specific),
        "comm2_specific_proteins": len(comm2_specific),
        "num_communities_1": len(communities1),
        "num_communities_2": len(communities2),
        "ARI_shared": ari,
        "NMI_shared": nmi,
    }

    logger.info(f"  Shared proteins: {len(shared)}")
    logger.info(f"  Average best-match Jaccard: {avg_jaccard:.3f}")
    if ari is not None:
        logger.info(f"  ARI (shared): {ari:.3f}")
    if nmi is not None:
        logger.info(f"  NMI (shared): {nmi:.3f}")
    logger.info(f"  {name1}-specific proteins: {len(comm1_specific)}")
    logger.info(f"  {name2}-specific proteins: {len(comm2_specific)}")

    return out


# -----------------------------
# Packed community layout (no overlap, minimal whitespace)
# -----------------------------
def _min_pairwise_distance_xy(pts: np.ndarray) -> float:
    """Minimum Euclidean distance over distinct pairs (pts shape n×2)."""
    if pts.shape[0] < 2:
        return float("inf")
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(np.min(d2)))


def create_community_aware_layout(
    G: nx.Graph,
    communities: List[Set[str]],
    seed: int = 42,
    intra_k: float = 2.0,
    intra_iter: int = 200,
    cluster_scale: float = 1.55,
    gap: float = 0.72,
    pack_iter: int = 800,
) -> Dict[str, Tuple[float, float]]:
    """
    Publication-friendly layout:
    - spring layout within each community (higher k / iterations → less node overlap)
    - scale community diameter up with sqrt(n_nodes) so large communities have more room
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
            nn_sub = sub.number_of_nodes()
            ne_sub = sub.number_of_edges()
            max_edges = nn_sub * (nn_sub - 1) // 2
            edge_density = (ne_sub / max_edges) if max_edges > 0 else 0.0
            # Dense (near-clique) subgraphs: unweighted spring + larger k reduces collapse to a point.
            dense = edge_density > 0.48
            k_use = max(intra_k, 3.2 / math.sqrt(max(nn_sub, 1)))
            try:
                if dense:
                    local_nx = nx.spring_layout(
                        sub, seed=seed + cid, k=k_use, iterations=intra_iter
                    )
                else:
                    local_nx = nx.spring_layout(
                        sub,
                        seed=seed + cid,
                        k=k_use,
                        iterations=intra_iter,
                        weight="weight",
                    )
            except TypeError:
                local_nx = nx.spring_layout(sub, seed=seed + cid, k=k_use, iterations=intra_iter)

            local = {n: np.array(local_nx[n], dtype=float) for n in nodes}

        pts = np.vstack(list(local.values()))
        center = pts.mean(axis=0)
        pts = pts - center
        max_norm = float(np.max(np.linalg.norm(pts, axis=1))) if len(nodes) > 1 else 1.0
        max_norm = max(max_norm, 1e-6)
        nn = len(nodes)
        # Widen dense communities so node disks (drawn on top) overlap less in data space.
        size_boost = 1.0 + 0.32 * math.sqrt(max(nn - 1, 0))
        pts = (pts / max_norm) * cluster_scale * size_boost

        # Hard floor on pairwise separation in layout units (fixes near-clique pile-ups).
        if nn >= 2:
            d_target = min(0.72 + 0.022 * max(nn - 3, 0), 2.8)
            d_m = _min_pairwise_distance_xy(pts)
            if d_m < 1e-9:
                pts = pts + rng.normal(scale=0.02, size=pts.shape)
                d_m = _min_pairwise_distance_xy(pts)
            if d_m < d_target and d_m > 1e-12:
                pts = pts * (d_target / d_m) * 1.08

        local2: Dict[str, np.ndarray] = {}
        for i, n in enumerate(nodes):
            local2[n] = pts[i]
        comm_local.append(local2)

        rad = float(np.max(np.linalg.norm(pts, axis=1))) + 0.48
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
):
    """
    Publication-friendly network visualization.
    label_mode:
      - "all_bold": show all labels bold
      - "none": hide all labels (legend only; often best for journals)
    """
    logger.info(f"Creating publication-quality network visualization: {title}")

    import matplotlib.colors as mcolors
    from matplotlib import patheffects
    from matplotlib.patches import Polygon, Circle

    # Cohesive palette (distinct, readable on light background)
    base_hex = [
        "#0d9488", "#e11d48", "#6366f1", "#d97706", "#0891b2",
        "#7c3aed", "#0f766e", "#be185d", "#4f46e5", "#b45309",
    ]

    n_nodes = max(G.number_of_nodes(), 1)
    max_comm = max((len(c) for c in communities), default=1)
    # Shrink markers when one community is huge so disks do not stack in screen space.
    comm_penalty = 1.0 + 0.55 * max(0, max_comm - 10) ** 0.85
    node_size = float(
        np.clip(300000.0 / (max(n_nodes, 1) * np.sqrt(comm_penalty)), 320.0, 2100.0)
    )
    label_fs = float(np.clip(1250.0 / n_nodes, 7.0, 13.5))
    edge_between_w = float(np.clip(0.35 + 25.0 / n_nodes, 0.3, 0.75))
    edge_within_w = float(np.clip(900.0 / n_nodes, 0.85, 2.2))

    bg = "white"
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    pos = create_community_aware_layout(
        G,
        communities,
        seed=42,
        intra_k=2.1,
        intra_iter=220,
        cluster_scale=1.58,
        gap=0.75,
        pack_iter=900,
    )

    nC = len(communities)
    base_colors = [base_hex[i % len(base_hex)] for i in range(nC)]
    colors = [mcolors.to_rgba(h) for h in base_colors]

    # Hulls first (behind nodes): soft fill + crisp tinted outline
    if HAS_SCIPY:
        for cid, nodes in enumerate(communities):
            if not nodes:
                continue
            pts = np.array([pos[n] for n in nodes], dtype=float)
            ec = mcolors.to_rgba(base_colors[cid], alpha=0.55)
            if len(nodes) < 3:
                center = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)) + 0.55) if len(nodes) > 1 else 0.9
                ax.add_patch(
                    Circle(
                        center,
                        radius,
                        facecolor=colors[cid],
                        edgecolor=ec,
                        alpha=0.16,
                        linewidth=1.8,
                        zorder=0,
                    )
                )
                continue
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                ax.add_patch(
                    Polygon(
                        hp,
                        closed=True,
                        facecolor=colors[cid],
                        edgecolor=ec,
                        alpha=0.16,
                        linewidth=1.8,
                        joinstyle="round",
                        zorder=0,
                    )
                )
            except Exception:
                center = pts.mean(axis=0)
                radius = float(np.max(np.linalg.norm(pts - center, axis=1)) + 0.55)
                ax.add_patch(
                    Circle(
                        center,
                        radius,
                        facecolor=colors[cid],
                        edgecolor=ec,
                        alpha=0.16,
                        linewidth=1.8,
                        zorder=0,
                    )
                )

    # Edges: faint cross-community, stronger within-community (tinted)
    within_by_cid: Dict[int, List[Tuple[str, str]]] = {i: [] for i in range(nC)}
    between_edges: List[Tuple[str, str]] = []
    for u, v in G.edges():
        cu = node_to_community.get(u, -1)
        cv = node_to_community.get(v, -1)
        if cu == cv and cu != -1:
            within_by_cid[cu].append((u, v))
        else:
            between_edges.append((u, v))

    if between_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=between_edges,
            alpha=0.12,
            width=edge_between_w,
            edge_color="#94a3b8",
            style="dashed",
            ax=ax,
        )
    for cid, elist in within_by_cid.items():
        if not elist:
            continue
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=elist,
            alpha=0.5,
            width=edge_within_w,
            edge_color=base_colors[cid],
            ax=ax,
        )

    # Nodes: white ring + community fill
    for cid, nodes in enumerate(communities):
        nl = list(nodes)
        if not nl:
            continue
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nl,
            node_color=[base_colors[cid]],
            node_size=node_size,
            alpha=0.98,
            edgecolors="#ffffff",
            linewidths=max(2.0, node_size ** 0.5 * 0.09),
            ax=ax,
        )

    # Labels
    if label_mode not in {"all_bold", "none"}:
        label_mode = "all_bold"

    if label_mode == "all_bold":
        pe = [
            patheffects.withStroke(linewidth=3.0, foreground="#ffffff", alpha=0.95),
            patheffects.Normal(),
        ]
        for node in G.nodes():
            x, y = pos[node]
            ax.text(
                x,
                y,
                node,
                fontsize=label_fs,
                fontweight="bold",
                color="#0f172a",
                ha="center",
                va="center",
                family="sans-serif",
                zorder=4,
                path_effects=pe,
            )

    # Legend (inside figure — avoids clipping with bbox_inches=tight)
    handles = []
    for cid, nodes in enumerate(communities):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=base_colors[cid],
                markersize=11,
                markeredgecolor="#334155",
                markeredgewidth=1.2,
                linestyle="None",
                label=f"Community {cid}  ·  n = {len(nodes)}",
            )
        )
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        fontsize=11.5,
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
        framealpha=0.94,
        title="Louvain partition",
        title_fontsize=12,
    )
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#334155")
    for t in leg.get_texts():
        t.set_fontweight("normal")
        t.set_color("#1e293b")

    ax.axis("off")

    # Bounds with breathing room for legend
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        pad = 0.55 + 0.06 * max(span_x, span_y)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)
    plt.savefig(
        output_file,
        bbox_inches="tight",
        facecolor=bg,
        dpi=dpi,
        pad_inches=0.12,
    )
    plt.close()
    logger.info(f"Saved network visualization to {output_file}")


# -----------------------------
# Reporting + outputs
# -----------------------------
def generate_comparison_report(
    results: Dict,
    output_file: Path,
    edge_source: str = "",
    *,
    graph1_name: str = "VIF-dependent",
    graph2_name: str = "VIF KO",
    edge_source_note: str = "",
):
    """Generate a text report comparing networks and partitions."""
    logger.info("Generating comparison report")
    comp = results["comparison"]

    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMMUNITY DETECTION COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")
        if edge_source:
            f.write(f"Edge source: {edge_source}\n")
            if edge_source_note:
                f.write(edge_source_note.strip() + "\n")
            f.write("\n")

        f.write("NETWORK STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{graph1_name} network:\n")
        f.write(f"  Nodes: {results['vif_dep']['nodes']}\n")
        f.write(f"  Edges: {results['vif_dep']['edges']}\n")
        f.write(f"  Communities: {results['vif_dep']['num_communities']}\n")
        f.write(f"  Modularity: {results['vif_dep']['modularity']:.4f}\n")

        f.write(f"\n{graph2_name} network:\n")
        f.write(f"  Nodes: {results['vif_ko']['nodes']}\n")
        f.write(f"  Edges: {results['vif_ko']['edges']}\n")
        f.write(f"  Communities: {results['vif_ko']['num_communities']}\n")
        f.write(f"  Modularity: {results['vif_ko']['modularity']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMUNITY PARTITION COMPARISON (LABEL-INVARIANT)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Shared proteins: {comp['shared_proteins']}\n")
        f.write(f"Average best-match Jaccard similarity: {comp['avg_jaccard_similarity']:.4f}\n")
        f.write(f"{graph1_name}-specific proteins: {comp['comm1_specific_proteins']}\n")
        f.write(f"{graph2_name}-specific proteins: {comp['comm2_specific_proteins']}\n")
        if comp.get("ARI_shared") is not None:
            f.write(f"ARI (shared proteins): {comp['ARI_shared']:.4f}\n")
        else:
            f.write("ARI (shared proteins): NA (sklearn not available)\n")
        if comp.get("NMI_shared") is not None:
            f.write(f"NMI (shared proteins): {comp['NMI_shared']:.4f}\n")
        else:
            f.write("NMI (shared proteins): NA (sklearn not available)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMUNITY DETAILS\n")
        f.write("-" * 80 + "\n")

        f.write(f"\n{graph1_name} communities:\n")
        for cid, comm in enumerate(results["vif_dep"]["communities"]):
            f.write(f"  Community {cid}: {len(comm)} proteins\n")

        f.write(f"\n{graph2_name} communities:\n")
        for cid, comm in enumerate(results["vif_ko"]["communities"]):
            f.write(f"  Community {cid}: {len(comm)} proteins\n")

    logger.info(f"Saved comparison report to {output_file}")


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
        description="Compare community detection between VIF-dependent and VIF KO networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--vif-dep-file",
        type=Path,
        default="wt_vif_comparison/vif_dependent_interactions.tsv",
        help="VIF-dependent interactions TSV file",
    )
    parser.add_argument(
        "--vif-ko-file",
        type=Path,
        default="results/cullin_vif_config_multi_seeds/all_pairs_with_uniprot_vif.tsv",
        help="VIF KO all pairs TSV (used when --edge-source probability)",
    )
    parser.add_argument(
        "--edge-source",
        type=str,
        default="main_cluster_binary",
        choices=["probability", "main_cluster_binary", "tsv_binary_intersection"],
        help=(
            "main_cluster_binary: WT + VIF KO graphs from centroid main_cluster_binary npy only; "
            "tsv_binary_intersection: legacy VIF-dep TSV ∩ VIF KO binary (+ vifprotein); "
            "probability: threshold all-pairs TSVs only."
        ),
    )
    parser.add_argument(
        "--wt-binary-npy",
        type=Path,
        default=(
            "post_sampling_analysis/base_models/cullin_wt_config_multi_seeds/"
            "main_cluster_network/main_cluster_binary_freq_ge_0p6.npy"
        ),
        help="Main-cluster binary M for WT graph (main_cluster_binary mode).",
    )
    parser.add_argument(
        "--wt-results-root",
        type=Path,
        default="results",
        help="Contains cullin_wt_config_multi_seeds/merged + seed_*/model_data",
    )
    parser.add_argument(
        "--wt-dataset",
        type=str,
        default="cullin_wt_config_multi_seeds",
        help="Subfolder under --wt-results-root for model_data.pkl (WT).",
    )
    parser.add_argument(
        "--vif-ko-binary-npy",
        type=Path,
        default=(
            "post_sampling_analysis/base_models/cullin_vif_config_multi_seeds/"
            "main_cluster_network/main_cluster_binary_freq_ge_0p6.npy"
        ),
        help="Main-cluster binary M for VIF KO graph",
    )
    parser.add_argument(
        "--vif-ko-results-root",
        type=Path,
        default="results",
        help="Contains cullin_vif_config_multi_seeds/merged + seed_*/model_data",
    )
    parser.add_argument(
        "--vif-ko-dataset",
        type=str,
        default="cullin_vif_config_multi_seeds",
        help="Subfolder under --vif-ko-results-root for model_data.pkl",
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
    vif_dep_path = root / args.vif_dep_file if not args.vif_dep_file.is_absolute() else args.vif_dep_file

    graph1_name = "VIF-dependent"
    graph2_name = "VIF KO"
    edge_note = ""

    if args.edge_source == "main_cluster_binary":
        wt_bin = root / args.wt_binary_npy if not args.wt_binary_npy.is_absolute() else args.wt_binary_npy
        vif_bin = root / args.vif_ko_binary_npy if not args.vif_ko_binary_npy.is_absolute() else args.vif_ko_binary_npy
        wt_res = root / args.wt_results_root if not args.wt_results_root.is_absolute() else args.wt_results_root
        vif_res = root / args.vif_ko_results_root if not args.vif_ko_results_root.is_absolute() else args.vif_ko_results_root
        wt_md = load_model_data(default_model_data_pkl(wt_res, args.wt_dataset))
        vif_md = load_model_data(default_model_data_pkl(vif_res, args.vif_ko_dataset))
        logger.info(f"WT graph from centroid binary M: {wt_bin}")
        logger.info(f"VIF KO graph from centroid binary M: {vif_bin}")
        G_dep = build_graph_from_main_cluster_binary(wt_bin, wt_md)
        G_ko = build_graph_from_main_cluster_binary(vif_bin, vif_md)
        title_dep = "VIF-dependent (WT)"
        title_ko = "VIF KO (VIF)"
        graph1_name = "VIF-dependent (WT)"
        graph2_name = "VIF KO (VIF)"
        edge_note = (
            "Both networks: centroid hard binary M from τ* largest-cluster draws "
            "(mean edge frequency ≥ 0.6; main_cluster_binary_freq_ge_0p6.npy); no TSV edge lists."
        )
    elif args.edge_source == "tsv_binary_intersection":
        binary_path = root / args.vif_ko_binary_npy if not args.vif_ko_binary_npy.is_absolute() else args.vif_ko_binary_npy
        res_root = root / args.vif_ko_results_root if not args.vif_ko_results_root.is_absolute() else args.vif_ko_results_root
        md_pkl = default_model_data_pkl(res_root, args.vif_ko_dataset)
        logger.info(f"VIF KO graph from binary M: {binary_path}")
        model_data = load_model_data(md_pkl)
        G_ko = build_graph_from_main_cluster_binary(binary_path, model_data)
        df_dep, prob_dep = load_interactions(vif_dep_path, "VIF-dependent", min_prob=args.min_prob)
        G_dep = build_vif_dependent_graph_binary_filtered(df_dep, prob_dep, G_ko)
        title_dep = "VIF-dependent (WT) — TSV ∩ VIF KO binary M"
        title_ko = "VIF KO (VIF)"
        graph1_name = "VIF-dependent (WT)"
        graph2_name = "VIF KO (VIF)"
        edge_note = (
            "Graph 1: rows from vif_dependent_interactions.tsv (WT prob ≥ cutoff) kept if edge on "
            "in VIF KO centroid M or either protein is vifprotein. Graph 2: full VIF KO centroid M."
        )
    else:
        df_dep, prob_dep = load_interactions(vif_dep_path, "VIF-dependent", min_prob=args.min_prob)
        vif_ko_path = root / args.vif_ko_file if not args.vif_ko_file.is_absolute() else args.vif_ko_file
        df_ko, prob_ko = load_interactions(vif_ko_path, "VIF KO", min_prob=args.min_prob)
        G_dep = build_network(df_dep, prob_dep)
        G_ko = build_network(df_ko, prob_ko)
        title_dep = "VIF-dependent (WT)"
        title_ko = "VIF KO (VIF)"
        graph1_name = "VIF-dependent (WT)"
        graph2_name = "VIF KO (VIF)"

    # Detect communities
    logger.info(f"\nDetecting communities using {args.method}...")
    if args.method == "louvain":
        node_to_dep, comms_dep = detect_communities_louvain(G_dep, args.resolution)
        node_to_ko, comms_ko = detect_communities_louvain(G_ko, args.resolution)
    elif args.method == "leiden":
        node_to_dep, comms_dep = detect_communities_leiden(G_dep, args.resolution)
        node_to_ko, comms_ko = detect_communities_leiden(G_ko, args.resolution)
    else:
        node_to_dep, comms_dep = detect_communities_greedy_modularity(G_dep)
        node_to_ko, comms_ko = detect_communities_greedy_modularity(G_ko)

    if not node_to_dep or not node_to_ko or not comms_dep or not comms_ko:
        logger.error("Community detection failed (empty results). Check NetworkX version / method availability.")
        return

    # Modularity
    mod_dep = calculate_modularity(G_dep, comms_dep)
    mod_ko = calculate_modularity(G_ko, comms_ko)

    # Compare partitions
    comparison = compare_communities(node_to_dep, node_to_ko, comms_dep, comms_ko, graph1_name, graph2_name)

    results = {
        "vif_dep": {
            "nodes": G_dep.number_of_nodes(),
            "edges": G_dep.number_of_edges(),
            "num_communities": len(comms_dep),
            "modularity": mod_dep,
            "communities": comms_dep,
        },
        "vif_ko": {
            "nodes": G_ko.number_of_nodes(),
            "edges": G_ko.number_of_edges(),
            "num_communities": len(comms_ko),
            "modularity": mod_ko,
            "communities": comms_ko,
        },
        "comparison": comparison,
    }

    # Report
    generate_comparison_report(
        results,
        args.output_dir / "community_comparison_report.txt",
        edge_source=args.edge_source,
        graph1_name=graph1_name,
        graph2_name=graph2_name,
        edge_source_note=edge_note,
    )

    # Save assignments (filenames kept for downstream compatibility)
    save_community_assignments(
        node_to_dep, comms_dep, args.output_dir / "vif_dependent_communities.tsv", graph1_name
    )
    save_community_assignments(
        node_to_ko, comms_ko, args.output_dir / "vif_ko_communities.tsv", graph2_name
    )

    # Visualize
    visualize_network_with_communities(
        G_dep, node_to_dep, comms_dep,
        title_dep,
        args.output_dir / "vif_dependent_network_communities.png",
        label_mode=args.label_mode,
        dpi=args.dpi,
    )
    visualize_network_with_communities(
        G_ko, node_to_ko, comms_ko,
        title_ko,
        args.output_dir / "vif_ko_network_communities.png",
        label_mode=args.label_mode,
        dpi=args.dpi,
    )

    logger.info("=" * 80)
    logger.info("Community detection comparison complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
