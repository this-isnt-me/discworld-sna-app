import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION & COLOUR PALETTE
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Network of Discworld — Literary SNA Explorer",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_RELATIVE_PATH = Path("data") / "discworld_app_bundle.json"

SERIES_COLOURS: Dict[str, str] = {
    "City Watch": "#2B6CB0",
    "Witches": "#B83280",
    "Wizards & Unseen University": "#2C7A7B",
    "Industrial Revolution": "#DD6B20",
    "Death": "#6B46C1",
    "Tiffany Aching": "#E53E3E",
    "Ancient Civilisations": "#D69E2E",
    "Cross-Series": "#718096",
    "Standalone": "#38A169",
    "Unclassified": "#A0AEC0",
}

ROLE_COLOURS: Dict[str, str] = {
    "Connector Hub": "#DD6B20",
    "Provincial Hub": "#2B6CB0",
    "Satellite Broker": "#B83280",
    "Local Member": "#A0AEC0",
}

METRIC_LABELS: Dict[str, str] = {
    "weighted_degree": "Weighted Degree (Interaction Strength)",
    "degree": "Unweighted Degree (Unique Connections)",
    "betweenness": "Betweenness Centrality (Bridge Role)",
    "pagerank": "PageRank (Narrative Influence)",
    "eigenvector": "Eigenvector Centrality (Inner-Circle Proximity)",
    "closeness": "Closeness Centrality (Network Reachability)",
}

METRIC_LAYMAN_DESCRIPTIONS: Dict[str, str] = {
    "weighted_degree": (
        "**Weighted Degree (Screen-Time & Conversation Volume):** Counts every time a character "
        "shares a close moment with someone else across the active books. Series leads who carry "
        "multiple novels—such as **Samuel Vimes**, **Esmerelda Weatherwax**, or **Rincewind**—score "
        "highest because they are constantly in dialogue with their supporting cast."
    ),
    "degree": (
        "**Unweighted Degree (Social Reach):** Counts how many *different* people a character "
        "interacts with, regardless of whether they speak once or a hundred times. Leaders, "
        "merchants, and public figures like **Havelock Vetinari** or **Cut-Me-Own-Throat Dibbler** "
        "score highly because their roles bring them into contact with dozens of distinct faces."
    ),
    "betweenness": (
        "**Betweenness Centrality (The Storyline Bridge):** Measures how often a character sits on "
        "the shortest path linking two otherwise separate groups. If removing someone from the books "
        "would leave the *City Watch* and the *Industrial Revolution* (or Ankh-Morpork and Lancre) "
        "disconnected, they achieve high betweenness."
    ),
    "pagerank": (
        "**PageRank (VIP Influence):** Inspired by Google's search algorithm, PageRank rewards *who* "
        "you know rather than just *how much* you talk. A character who shares scenes with heavyweights "
        "like **Vetinari**, **Death**, or **Granny Weatherwax** earns a higher PageRank than someone "
        "who only chats with minor background guards."
    ),
    "eigenvector": (
        "**Eigenvector Centrality (Inner-Circle Status):** Identifies characters sitting right at the "
        "heart of the Discworld's largest, most densely connected power clique—typically the core "
        "Ankh-Morpork establishment spanning the Watch, the Patrician's Palace, and Unseen University."
    ),
    "closeness": (
        "**Closeness Centrality (Word-of-Mouth Reachability):** Measures how few 'handshakes' it takes "
        "for a character to reach everyone else in the network. Characters with high closeness can pass "
        "a rumour—or a plot thread—to almost any corner of the Discworld through just one or two mutual acquaintances."
    ),
}


# ---------------------------------------------------------------------------
# PURE-NUMPY GRAPH ALGORITHMS (NO SCIPY DEPENDENCY)
# ---------------------------------------------------------------------------
def build_weighted_adjacency(G: nx.Graph) -> Tuple[List[str], np.ndarray]:
    """Builds an ordered node list and symmetric weighted adjacency matrix."""
    nodes = list(G.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=float)

    for u, v, data in G.edges(data=True):
        i, j = idx_map[u], idx_map[v]
        w = float(data.get("weight", 1.0))
        adj[i, j] = w
        adj[j, i] = w

    return nodes, adj


def compute_pagerank_numpy(
    G: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """Computes weighted PageRank using pure NumPy power iteration."""
    nodes, adj = build_weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    if G.number_of_edges() == 0:
        return {node: 1.0 / n for node in nodes}

    out_strength = adj.sum(axis=1)
    dangling = out_strength == 0.0

    M = np.zeros((n, n), dtype=float)
    non_dangling = ~dangling
    M[:, non_dangling] = adj[non_dangling, :].T / out_strength[non_dangling]

    pr = np.full(n, 1.0 / n, dtype=float)
    teleport = (1.0 - alpha) / n

    for _ in range(max_iter):
        dangling_mass = alpha * pr[dangling].sum() / n
        pr_next = alpha * (M @ pr) + dangling_mass + teleport
        if np.linalg.norm(pr_next - pr, ord=1) < tol:
            pr = pr_next
            break
        pr = pr_next

    return {nodes[i]: float(pr[i]) for i in range(n)}


def compute_eigenvector_numpy(
    G: nx.Graph,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> Dict[str, float]:
    """Computes weighted eigenvector centrality using pure NumPy power iteration."""
    nodes, adj = build_weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    if G.number_of_edges() == 0:
        return {node: 0.0 for node in nodes}

    vec = np.full(n, 1.0 / np.sqrt(n), dtype=float)

    for _ in range(max_iter):
        vec_next = adj @ vec
        norm = np.linalg.norm(vec_next)
        if norm == 0.0:
            return {node: 0.0 for node in nodes}
        vec_next /= norm
        if np.linalg.norm(vec_next - vec) < tol:
            vec = vec_next
            break
        vec = vec_next

    return {nodes[i]: float(vec[i]) for i in range(n)}


def compute_spring_layout_numpy(
    G: nx.Graph,
    seed: int = 42,
    iterations: int = 90,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes a deterministic Fruchterman-Reingold force-directed layout in pure NumPy
    without requiring SciPy.
    """
    nodes, adj = build_weighted_adjacency(G)
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))

    max_w = adj.max()
    norm_adj = (adj / max_w) if max_w > 0 else adj
    edge_mask = adj > 0.0

    k = 1.6 / np.sqrt(n)
    temperature = 0.25
    cooling = temperature / max(iterations, 1)

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        safe_dist = np.maximum(dist, 1e-4)

        repulsive = (k * k) / safe_dist
        np.fill_diagonal(repulsive, 0.0)

        attractive = np.where(
            edge_mask, ((safe_dist * safe_dist) / k) * (0.35 + 0.65 * norm_adj), 0.0
        )

        net_force = repulsive - attractive
        unit_vec = delta / safe_dist[:, :, None]
        disp = np.sum(unit_vec * net_force[:, :, None], axis=1)

        disp_len = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-4)
        pos += (disp / disp_len) * np.minimum(disp_len, temperature)
        temperature = max(temperature - cooling, 0.005)

    pos -= pos.mean(axis=0)
    max_coord = np.abs(pos).max()
    if max_coord > 0:
        pos = (pos / max_coord) * 100.0

    return {nodes[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


# ---------------------------------------------------------------------------
# DATA LOADING & EDGE-LEVEL FILTERING HELPERS
# ---------------------------------------------------------------------------
def resolve_default_bundle_path() -> Optional[Path]:
    """Finds data/discworld_app_bundle.json relative to the script or working directory."""
    script_relative = Path(__file__).resolve().parent / BUNDLE_RELATIVE_PATH
    if script_relative.is_file():
        return script_relative

    if BUNDLE_RELATIVE_PATH.is_file():
        return BUNDLE_RELATIVE_PATH

    return None


@st.cache_data(show_spinner=False)
def load_bundle_from_path(path_str: str) -> Dict:
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def filter_raw_edges(
    bundle: Dict,
    series_mode: str,
    selected_series: Tuple[str, ...],
    book_mode: str,
    selected_books: Tuple[str, ...],
) -> List[Dict]:
    """
    Filters novel-level edges directly using the `sub_series` and `book` attributes
    attached to each edge.
    """
    raw_edges = bundle.get("raw_novel_edges", [])
    series_set = set(selected_series)
    books_set = set(selected_books)

    filtered = []
    for edge in raw_edges:
        edge_series = edge.get("sub_series", "Standalone")
        edge_book = edge.get("book", "")

        if series_set:
            if series_mode == "Exclude selected (Filter out)" and edge_series in series_set:
                continue
            if series_mode == "Include only selected" and edge_series not in series_set:
                continue

        if books_set:
            if book_mode == "Exclude selected (Filter out)" and edge_book in books_set:
                continue
            if book_mode == "Include only selected" and edge_book not in books_set:
                continue

        filtered.append(edge)

    return filtered


@st.cache_data(show_spinner=False)
def compute_filtered_character_stats(
    bundle: Dict,
    filtered_edges: List[Dict],
) -> pd.DataFrame:
    """
    Computes distinct novel appearances, sub-series breadth, and interaction totals
    per character from the currently active (filtered) edge list.
    """
    char_books: Dict[str, set] = defaultdict(set)
    char_series_appearances: Dict[str, set] = defaultdict(set)
    char_interaction_sum: Dict[str, int] = defaultdict(int)

    for edge in filtered_edges:
        u, v = edge["source"], edge["target"]
        book = edge["book"]
        sub_series = edge["sub_series"]
        w = int(edge["weight"])

        char_books[u].add(book)
        char_books[v].add(book)
        char_series_appearances[u].add(sub_series)
        char_series_appearances[v].add(sub_series)
        char_interaction_sum[u] += w
        char_interaction_sum[v] += w

    all_nodes = {n["id"]: n for n in bundle["slices"]["All"]["nodes"]}
    primary_map = bundle.get("character_primary_series", {})

    rows = []
    for char in char_books.keys():
        node_meta = all_nodes.get(char, {})
        rows.append(
            {
                "Character": char,
                "Primary Series": node_meta.get(
                    "primary_series", primary_map.get(char, "Unclassified")
                ),
                "Book Appearances": len(char_books[char]),
                "Sub-Series Spanned": len(char_series_appearances[char]),
                "Filtered Interactions": int(char_interaction_sum[char]),
                "Corpus Total Mentions": int(node_meta.get("total_mentions", 0)),
                "Books List": ", ".join(sorted(char_books[char])),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def build_graph_from_filtered_edges(
    bundle: Dict,
    filtered_edges: List[Dict],
    min_edge_weight: int,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Aggregates filtered novel-level edges into a weighted NetworkX Graph,
    computes all six Network of Thrones centrality metrics and Louvain clusters,
    and returns the graph alongside node and edge DataFrames.
    """
    all_nodes_meta = {n["id"]: n for n in bundle["slices"]["All"]["nodes"]}
    primary_series_lookup = bundle.get("character_primary_series", {})

    pair_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_books: Dict[Tuple[str, str], set] = defaultdict(set)
    pair_series: Dict[Tuple[str, str], set] = defaultdict(set)

    for e in filtered_edges:
        pair = tuple(sorted((e["source"], e["target"])))
        pair_weights[pair] += int(e["weight"])
        pair_books[pair].add(e["book"])
        pair_series[pair].add(e["sub_series"])

    G = nx.Graph()
    for (u, v), w in pair_weights.items():
        if w >= min_edge_weight:
            G.add_edge(
                u,
                v,
                weight=w,
                books=sorted(pair_books[(u, v)]),
                sub_series=sorted(pair_series[(u, v)]),
            )

    if G.number_of_nodes() == 0:
        return G, pd.DataFrame(), pd.DataFrame()

    w_deg = dict(G.degree(weight="weight"))
    u_deg = dict(G.degree())

    for _u, _v, d in G.edges(data=True):
        d["distance"] = 1.0 / max(d["weight"], 1)

    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(G, distance="distance")
    pagerank = compute_pagerank_numpy(G)
    eigenvector = compute_eigenvector_numpy(G)

    communities = (
        nx.community.louvain_communities(G, weight="weight", seed=42)
        if G.number_of_edges() > 0
        else [set(G.nodes())]
    )
    communities = sorted(communities, key=len, reverse=True)
    comm_map = {
        member: idx for idx, members in enumerate(communities) for member in members
    }

    pos = compute_spring_layout_numpy(G, seed=42)

    for n_id in G.nodes():
        meta = all_nodes_meta.get(n_id, {})
        x, y = pos[n_id]
        G.nodes[n_id]["primary_series"] = meta.get(
            "primary_series", primary_series_lookup.get(n_id, "Unclassified")
        )
        G.nodes[n_id]["total_mentions"] = int(meta.get("total_mentions", 0))
        G.nodes[n_id]["x"] = x
        G.nodes[n_id]["y"] = y
        G.nodes[n_id]["degree"] = int(u_deg[n_id])
        G.nodes[n_id]["weighted_degree"] = int(w_deg[n_id])
        G.nodes[n_id]["betweenness"] = round(float(betweenness[n_id]), 5)
        G.nodes[n_id]["closeness"] = round(float(closeness[n_id]), 5)
        G.nodes[n_id]["pagerank"] = round(float(pagerank[n_id]), 5)
        G.nodes[n_id]["eigenvector"] = round(float(eigenvector[n_id]), 5)
        G.nodes[n_id]["community"] = int(comm_map.get(n_id, 0))

    node_rows = [{"Character": n_id, **attrs} for n_id, attrs in G.nodes(data=True)]
    edge_rows = [
        {
            "Source": u,
            "Target": v,
            "Weight": int(attrs["weight"]),
            "Books": ", ".join(attrs.get("books", [])),
            "Sub-Series": ", ".join(attrs.get("sub_series", [])),
        }
        for u, v, attrs in G.edges(data=True)
    ]

    return G, pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


@st.cache_data(show_spinner=False)
def compute_community_connectors(
    _G: nx.Graph,
    nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes inter-community connector metrics for every node in the active graph:
      - Participation Coefficient (Louvain communities & canonical sub-series)
      - Within-Community Degree Z-Score
      - Burt's Constraint & Structural Hole Brokerage (1 - constraint)
      - Cross-Community Interaction % and bridged community list
      - Guimerà & Amaral Cartographic Role classification
    """
    if _G.number_of_nodes() == 0 or nodes_df.empty:
        return pd.DataFrame()

    node_comm = {
        row["Character"]: int(row["community"])
        for _, row in nodes_df.iterrows()
    }
    node_series = {
        row["Character"]: str(row["primary_series"])
        for _, row in nodes_df.iterrows()
    }
    betweenness_map = {
        row["Character"]: float(row["betweenness"])
        for _, row in nodes_df.iterrows()
    }

    raw_constraint = nx.constraint(_G, weight="weight") if _G.number_of_edges() > 0 else {}

    internal_strength: Dict[str, float] = {}
    comm_members: Dict[int, List[str]] = defaultdict(list)

    for n in _G.nodes():
        c_home = node_comm[n]
        comm_members[c_home].append(n)
        s_in = sum(
            float(_G[n][nbr]["weight"])
            for nbr in _G.neighbors(n)
            if node_comm[nbr] == c_home
        )
        internal_strength[n] = s_in

    comm_mean_std: Dict[int, Tuple[float, float]] = {}
    for c_id, members in comm_members.items():
        vals = np.array([internal_strength[m] for m in members], dtype=float)
        mean_val = float(vals.mean()) if len(vals) else 0.0
        std_val = float(vals.std()) if len(vals) > 1 else 0.0
        comm_mean_std[c_id] = (mean_val, std_val)

    rows = []
    for n in _G.nodes():
        c_home = node_comm[n]
        total_strength = float(_G.degree(n, weight="weight"))

        strength_by_comm: Dict[int, float] = defaultdict(float)
        strength_by_series: Dict[str, float] = defaultdict(float)

        for nbr in _G.neighbors(n):
            w = float(_G[n][nbr]["weight"])
            strength_by_comm[node_comm[nbr]] += w
            strength_by_series[node_series[nbr]] += w

        if total_strength > 0:
            comm_shares = [(s_m / total_strength) ** 2 for s_m in strength_by_comm.values()]
            participation_comm = 1.0 - sum(comm_shares)

            series_shares = [(s_s / total_strength) ** 2 for s_s in strength_by_series.values()]
            participation_series = 1.0 - sum(series_shares)

            external_weight = total_strength - strength_by_comm.get(c_home, 0.0)
            cross_comm_pct = (external_weight / total_strength) * 100.0
        else:
            participation_comm = 0.0
            participation_series = 0.0
            cross_comm_pct = 0.0

        mean_s, std_s = comm_mean_std[c_home]
        z_score = (internal_strength[n] - mean_s) / std_s if std_s > 0 else 0.0

        c_val = raw_constraint.get(n, 1.0)
        if c_val is None or np.isnan(c_val):
            c_val = 1.0
        brokerage_score = max(0.0, 1.0 - float(c_val))

        if z_score >= 1.25 and participation_comm >= 0.30:
            role = "Connector Hub"
        elif z_score >= 1.25 and participation_comm < 0.30:
            role = "Provincial Hub"
        elif z_score < 1.25 and participation_comm >= 0.45:
            role = "Satellite Broker"
        else:
            role = "Local Member"

        external_comms = sorted(
            [
                (c_id, w)
                for c_id, w in strength_by_comm.items()
                if c_id != c_home and w > 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        bridged_summary = (
            ", ".join(f"C{c_id + 1} ({int(w)})" for c_id, w in external_comms[:4])
            if external_comms
            else "None (Internal only)"
        )

        rows.append(
            {
                "Character": n,
                "Primary Series": node_series[n],
                "Home Community": f"Community {c_home + 1}",
                "Structural Role": role,
                "Participation (Communities)": round(participation_comm, 4),
                "Participation (Sub-Series)": round(participation_series, 4),
                "Cross-Community %": round(cross_comm_pct, 1),
                "Within-Comm Z-Score": round(z_score, 2),
                "Structural Hole Brokerage": round(brokerage_score, 4),
                "Betweenness": round(betweenness_map[n], 5),
                "Weighted Degree": int(total_strength),
                "Communities Bridged": len(strength_by_comm),
                "External Links": bridged_summary,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        max_btw = df["Betweenness"].max() or 1.0
        df["Composite Bridge Index"] = (
            0.40 * df["Participation (Communities)"]
            + 0.30 * df["Structural Hole Brokerage"]
            + 0.30 * (df["Betweenness"] / max_btw)
        ).round(4)

    return df


# ---------------------------------------------------------------------------
# VISUALISATION BUILDERS
# ---------------------------------------------------------------------------
def build_interactive_network_figure(
    G: nx.Graph,
    nodes_df: pd.DataFrame,
    size_metric: str,
    colour_by: str,
    top_n_labels: int,
    highlight_character: Optional[str] = None,
) -> go.Figure:
    """Renders an interactive Plotly network graph with sub-series or Louvain colouring."""
    fig = go.Figure()

    if G.number_of_nodes() == 0:
        return fig

    ego_neighbours = set()
    if highlight_character and highlight_character in G:
        ego_neighbours = set(G.neighbors(highlight_character)) | {highlight_character}

    max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    edge_buckets: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        norm_w = d["weight"] / max_w
        bucket = 1 if norm_w < 0.15 else (2 if norm_w < 0.45 else 3)
        edge_buckets[bucket].append((u, v))

    bucket_widths = {1: 0.6, 2: 1.8, 3: 3.6}
    bucket_alphas = {
        1: "rgba(160, 174, 192, 0.25)",
        2: "rgba(113, 128, 150, 0.45)",
        3: "rgba(74, 85, 104, 0.72)",
    }

    for bucket, pairs in sorted(edge_buckets.items()):
        edge_x, edge_y = [], []
        for u, v in pairs:
            if ego_neighbours and not (u in ego_neighbours and v in ego_neighbours):
                continue
            x0, y0 = G.nodes[u]["x"], G.nodes[u]["y"]
            x1, y1 = G.nodes[v]["x"], G.nodes[v]["y"]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=bucket_widths[bucket], color=bucket_alphas[bucket]),
                hoverinfo="none",
                showlegend=False,
            )
        )

    ranked_chars = (
        nodes_df.sort_values(by=size_metric, ascending=False)["Character"]
        .head(top_n_labels)
        .tolist()
    )
    labelled_set = set(ranked_chars)
    if highlight_character:
        labelled_set.update(ego_neighbours)

    vals = nodes_df[size_metric].astype(float)
    v_min, v_max = vals.min(), vals.max()
    if v_max > v_min:
        scaled_sizes = 10.0 + 32.0 * np.sqrt((vals - v_min) / (v_max - v_min))
    else:
        scaled_sizes = pd.Series([18.0] * len(nodes_df), index=nodes_df.index)
    nodes_df = nodes_df.copy()
    nodes_df["_marker_size"] = scaled_sizes

    if colour_by == "Primary Sub-Series":
        group_col = "primary_series"
        palette_list = list(SERIES_COLOURS.values())
    else:
        nodes_df["_comm_label"] = nodes_df["community"].apply(lambda c: f"Community {c + 1}")
        group_col = "_comm_label"
        palette_list = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

    for idx, (group_val, group_df) in enumerate(nodes_df.groupby(group_col)):
        if colour_by == "Primary Sub-Series":
            marker_colour = SERIES_COLOURS.get(str(group_val), "#718096")
        else:
            marker_colour = palette_list[idx % len(palette_list)]

        opacities = [
            1.0 if (not ego_neighbours or c in ego_neighbours) else 0.15
            for c in group_df["Character"]
        ]
        labels = [c if c in labelled_set else "" for c in group_df["Character"]]

        hover_texts = [
            (
                f"<b>{row['Character']}</b><br>"
                f"Primary Series: {row['primary_series']}<br>"
                f"Louvain Cluster: {row['community'] + 1}<br>"
                f"Weighted Degree: {row['weighted_degree']:,}<br>"
                f"Unique Connections: {row['degree']}<br>"
                f"Betweenness: {row['betweenness']:.5f}<br>"
                f"PageRank: {row['pagerank']:.5f}"
            )
            for _, row in group_df.iterrows()
        ]

        fig.add_trace(
            go.Scatter(
                x=group_df["x"],
                y=group_df["y"],
                mode="markers+text",
                name=str(group_val),
                text=labels,
                textposition="top center",
                textfont=dict(size=10, color="#1A202C"),
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(
                    size=group_df["_marker_size"],
                    color=marker_colour,
                    opacity=opacities,
                    line=dict(width=1.2, color="white"),
                ),
            )
        )

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            title=dict(text=colour_by),
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        hovermode="closest",
    )
    return fig


def build_interaction_heatmap(
    G: nx.Graph,
    characters: List[str],
    colorscale: str = "YlGnBu",
    height: int = 500,
) -> go.Figure:
    """
    Constructs a symmetric NxN heatmap displaying exact co-occurrence weights
    between an ordered list of characters in the active filtered network.
    """
    n = len(characters)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0
            elif G.has_edge(characters[i], characters[j]):
                matrix[i, j] = int(G[characters[i]][characters[j]]["weight"])

    text_labels = [
        ["" if i == j else (str(matrix[i, j]) if matrix[i, j] > 0 else "0") for j in range(n)]
        for i in range(n)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=characters,
            y=characters,
            text=text_labels,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale=colorscale,
            colorbar=dict(title="Interactions"),
            hovertemplate="<b>%{y}</b> & <b>%{x}</b><br>Co-occurrences: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis=dict(tickangle=-35, side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_series_crossover_heatmap(bundle: Dict, filtered_edges: List[Dict]) -> go.Figure:
    """
    Aggregates interaction weights between characters' primary sub-series across
    the currently active filtered edges.
    """
    primary_map = bundle.get("character_primary_series", {})
    if not primary_map:
        primary_map = {
            n["id"]: n.get("primary_series", "Unclassified")
            for n in bundle["slices"]["All"]["nodes"]
        }

    cross_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    series_set = set()

    for edge in filtered_edges:
        s1 = primary_map.get(edge["source"], "Unclassified")
        s2 = primary_map.get(edge["target"], "Unclassified")
        if s1 == "Unclassified" or s2 == "Unclassified":
            continue
        series_set.update([s1, s2])
        w = int(edge["weight"])
        cross_counts[(s1, s2)] += w
        if s1 != s2:
            cross_counts[(s2, s1)] += w

    ordered_series = sorted(series_set)
    m = len(ordered_series)
    z_mat = np.zeros((m, m), dtype=int)
    for i, s_row in enumerate(ordered_series):
        for j, s_col in enumerate(ordered_series):
            z_mat[i, j] = cross_counts[(s_row, s_col)]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_mat,
            x=ordered_series,
            y=ordered_series,
            text=z_mat,
            texttemplate="%{text:,}",
            colorscale="Purples",
            colorbar=dict(title="Total Weight"),
            hovertemplate="<b>%{y}</b> ↔ <b>%{x}</b><br>Interaction Weight: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_role_cartography_figure(connectors_df: pd.DataFrame) -> go.Figure:
    """
    Plots Guimerà & Amaral's Role Cartography:
    Participation Coefficient (X-axis) vs. Within-Community Z-Score (Y-axis).
    """
    top_connectors = set(
        connectors_df.sort_values(by="Composite Bridge Index", ascending=False)
        .head(14)["Character"]
    )
    top_hubs = set(
        connectors_df.sort_values(by="Within-Comm Z-Score", ascending=False)
        .head(8)["Character"]
    )
    label_targets = top_connectors | top_hubs

    plot_df = connectors_df.copy()
    plot_df["_label"] = plot_df["Character"].apply(
        lambda c: c if c in label_targets else ""
    )

    fig = px.scatter(
        plot_df,
        x="Participation (Communities)",
        y="Within-Comm Z-Score",
        size="Weighted Degree",
        color="Structural Role",
        color_discrete_map=ROLE_COLOURS,
        hover_name="Character",
        text="_label",
        hover_data=[
            "Primary Series",
            "Home Community",
            "Cross-Community %",
            "Structural Hole Brokerage",
            "External Links",
        ],
    )

    fig.add_hline(y=1.25, line_dash="dash", line_color="#CBD5E0")
    fig.add_vline(x=0.30, line_dash="dash", line_color="#CBD5E0")
    fig.add_vline(x=0.45, line_dash="dot", line_color="#CBD5E0")

    fig.update_traces(textposition="top center", textfont=dict(size=10))
    fig.update_layout(
        height=490,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis=dict(range=[-0.03, 0.92], title="Participation Coefficient (Inter-Community Spread)"),
        yaxis=dict(title="Within-Community Z-Score (Local Hub Strength)"),
    )
    return fig


# ---------------------------------------------------------------------------
# MAIN APPLICATION UI
# ---------------------------------------------------------------------------
st.title("🐢 Network of Discworld: Social Network & Literary Analysis")
st.markdown(
    "This interactive application maps the social universe of Terry Pratchett's **41 *Discworld* novels** "
    "using **Social Network Analysis (SNA)**—inspired by Beveridge & Shan's *Network of Thrones*. "
    "Instead of relying on manual character lists, the network is built directly from the novel texts to reveal "
    "who drives the plot, which storylines overlap, and which characters stitch the wider Discworld together."
)

with st.expander("📖 Methodology: How the Discworld Social Network Was Built", expanded=False):
    meth_col1, meth_col2 = st.columns(2)
    with meth_col1:
        st.markdown(
            """
            **1. Canonical Name Extraction & Disambiguation**
            * All **41 *Discworld* novels** were parsed in narrative reading order, stripping non-story front/back matter.
            * Named entities were resolved to canonical character identities (for instance, merging *"Granny"*, *"Mistress Weatherwax"*, and *"Esmerelda"* into **Esmerelda Weatherwax**, and resolving scene-level short names such as *"Vimes"*, *"Ridcully"*, and *"Susan"*).
            * Context-aware guards prevent common words (such as sentence-initial *"Death"* or *"Madam"*) from inflating character counts.
            """
        )
    with meth_col2:
        st.markdown(
            """
            **2. The 15-Word Proximity Rule & Edge Tagging**
            * Following the *Network of Thrones* methodology, whenever two characters are mentioned within **15 words** of each other inside the same scene, an interaction is recorded between them (scene breaks act as hard boundaries so unrelated chapters never link).
            * Every connection retains the **novel title** and **sub-series** where it occurred. Filtering out a sub-series or novel in the sidebar removes those exact interactions and recalculates the entire network live.
            """
        )

# Load graph bundle directly from data/discworld_app_bundle.json
bundle_path = resolve_default_bundle_path()
if bundle_path is None:
    st.error(
        "Missing graph file: expected `data/discworld_app_bundle.json`. "
        "Ensure the `data/` folder containing `discworld_app_bundle.json` sits next to `app.py`."
    )
    st.stop()

bundle = load_bundle_from_path(str(bundle_path))

# --- SIDEBAR: EDGE-LEVEL FILTERS & VISUALISATION SETTINGS ---
with st.sidebar:
    st.header("1. Edge Filters (Series & Books)")
    st.caption(
        "Filters edges directly via the `sub_series` and `book` metadata attached to each interaction."
    )

    all_edge_series = sorted({e["sub_series"] for e in bundle.get("raw_novel_edges", [])})
    series_filter_mode = st.radio(
        "Sub-Series Filter Action",
        options=["Exclude selected (Filter out)", "Include only selected"],
        index=0,
        key="series_filter_mode",
    )
    selected_series = st.multiselect(
        "Sub-Series",
        options=all_edge_series,
        default=[],
        placeholder="Choose sub-series...",
    )

    candidate_books = sorted(
        {
            e["book"]
            for e in bundle.get("raw_novel_edges", [])
            if not selected_series
            or (
                series_filter_mode == "Exclude selected (Filter out)"
                and e["sub_series"] not in set(selected_series)
            )
            or (
                series_filter_mode == "Include only selected"
                and e["sub_series"] in set(selected_series)
            )
        }
    )

    book_filter_mode = st.radio(
        "Book Filter Action",
        options=["Exclude selected (Filter out)", "Include only selected"],
        index=0,
        key="book_filter_mode",
    )
    selected_books = st.multiselect(
        "Books",
        options=candidate_books,
        default=[],
        placeholder="Choose novels...",
    )

    st.header("2. Graph Visualisation Settings")
    min_edge_weight = st.slider(
        "Minimum Edge Weight (Prune Casual Mentions)",
        min_value=1,
        max_value=25,
        value=3,
        step=1,
    )
    size_metric = st.selectbox(
        "Importance / Sizing Metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        index=0,
    )
    colour_by = st.radio(
        "Node Colouring",
        options=["Primary Sub-Series", "Louvain Community"],
        index=0,
    )
    top_n_labels = st.slider("Visible Character Labels", min_value=10, max_value=120, value=45, step=5)

# Apply edge-level series & book filters
active_raw_edges = filter_raw_edges(
    bundle=bundle,
    series_mode=series_filter_mode,
    selected_series=tuple(selected_series),
    book_mode=book_filter_mode,
    selected_books=tuple(selected_books),
)

active_books_count = len({e["book"] for e in active_raw_edges})
active_series_count = len({e["sub_series"] for e in active_raw_edges})

if not active_raw_edges:
    st.warning(
        "All edges have been filtered out by the current Sub-Series and Book selections. "
        "Adjust the sidebar filters to restore interactions."
    )
    st.stop()

G_active, nodes_df, edges_df = build_graph_from_filtered_edges(
    bundle=bundle,
    filtered_edges=active_raw_edges,
    min_edge_weight=min_edge_weight,
)
char_book_df = compute_filtered_character_stats(bundle, active_raw_edges)

if G_active.number_of_nodes() == 0:
    st.warning(
        f"Edges remain across {active_books_count} book(s), but none meet the minimum edge weight of {min_edge_weight}. "
        "Lower the Minimum Edge Weight slider in the sidebar."
    )
    st.stop()

# --- TOP KPI METRICS BAR ---
num_nodes = G_active.number_of_nodes()
num_edges = G_active.number_of_edges()
total_weight = int(edges_df["Weight"].sum()) if not edges_df.empty else 0

comm_groups = defaultdict(set)
for _, row in nodes_df.iterrows():
    comm_groups[row["community"]].add(row["Character"])
modularity_q = (
    nx.community.modularity(G_active, list(comm_groups.values()), weight="weight")
    if num_edges > 0
    else 0.0
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Active Novels", f"{active_books_count} / {len(bundle.get('available_books', []))}")
k2.metric("Active Sub-Series", f"{active_series_count}")
k3.metric("Characters (Nodes)", f"{num_nodes:,}")
k4.metric("Relationships (Edges)", f"{num_edges:,}")
k5.metric("Total Co-Occurrences", f"{total_weight:,}")
k6.metric("Louvain Modularity (Q)", f"{modularity_q:.3f}")

st.caption(
    "**Reading the Summary Bar:** Each **Node** is a character and each **Edge** is a recurring relationship "
    "(two characters appearing within 15 words of one another at least as many times as the sidebar threshold). "
    "**Total Co-Occurrences** sums all shared moments across the active books, while **Louvain Modularity ($Q$)** "
    "measures how cleanly the cast divides into separate storylines—scores closer to `1.0` indicate self-contained "
    "casts (such as *Ancient Civilisations*), whereas lower scores reflect dense mingling across Ankh-Morpork."
)

st.divider()

# --- MAIN ANALYTICAL TABS ---
tab_net, tab_cent, tab_bars, tab_comm = st.tabs(
    [
        "🕸️ Interactive Network & Ego-Inspector",
        "📊 Centrality & Bridge Analysis",
        "🔥 Top 10 Bar Charts & Interaction Heatmap",
        "🧩 Community Detection & Bridge Connectors",
    ]
)

# ===========================================================================
# TAB 1: INTERACTIVE NETWORK & CHARACTER EGO-INSPECTOR
# ===========================================================================
with tab_net:
    st.info(
        "**How to Read the Network Map:** Characters who frequently share scenes are pulled together into tight "
        "clusters, while isolated casts (such as *Pyramids* or *Small Gods*) drift toward the outer edges. "
        "Circle size reflects importance under your chosen sidebar metric, thicker lines indicate closer partnerships, "
        "and colours show either a character's home sub-series or their detected social clique. "
        "Currently sizing nodes by: " + METRIC_LAYMAN_DESCRIPTIONS[size_metric]
    )

    col_graph, col_inspector = st.columns([2.4, 1.0])

    sorted_char_names = (
        nodes_df.sort_values(by="weighted_degree", ascending=False)["Character"].tolist()
    )

    with col_inspector:
        st.subheader("Character Ego-Inspector")
        st.caption(
            "Isolate any individual character's personal social circle to see who they talk to most "
            "and which novels bring them together."
        )
        focus_choice = st.selectbox(
            "Highlight Character Ego-Network",
            options=["None (Full Active Network)"] + sorted_char_names,
            index=0,
        )
        highlight_char = None if focus_choice == "None (Full Active Network)" else focus_choice

        if highlight_char and highlight_char in G_active:
            c_meta = nodes_df[nodes_df["Character"] == highlight_char].iloc[0]
            st.markdown(f"**Primary Series:** `{c_meta['primary_series']}`")
            st.markdown(f"**Louvain Cluster:** `Community {int(c_meta['community']) + 1}`")

            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Weighted Degree", f"{int(c_meta['weighted_degree']):,}")
            m_col2.metric("Betweenness", f"{c_meta['betweenness']:.4f}")

            neighbours_data = []
            for nbr in G_active.neighbors(highlight_char):
                edge_attr = G_active[highlight_char][nbr]
                neighbours_data.append(
                    {
                        "Partner": nbr,
                        "Interactions": int(edge_attr["weight"]),
                        "Sub-Series": ", ".join(edge_attr.get("sub_series", [])),
                        "Books": ", ".join(edge_attr.get("books", [])),
                    }
                )
            nbr_df = pd.DataFrame(neighbours_data).sort_values(
                by="Interactions", ascending=False
            )
            st.markdown("**Strongest Connections (Active Filter):**")
            st.dataframe(nbr_df, width='stretch', hide_index=True, height=340)
        else:
            st.info(
                "Select any character above to dim the rest of the Discworld and inspect their "
                "direct companions, interaction counts, and shared novels."
            )

    with col_graph:
        net_fig = build_interactive_network_figure(
            G=G_active,
            nodes_df=nodes_df,
            size_metric=size_metric,
            colour_by=colour_by,
            top_n_labels=top_n_labels,
            highlight_character=highlight_char,
        )
        st.plotly_chart(net_fig, width='stretch')

# ===========================================================================
# TAB 2: CENTRALITY & BRIDGE ANALYSIS
# ===========================================================================
with tab_cent:
    with st.expander("💡 Plain-English Guide to the Six Character Importance Metrics", expanded=False):
        g_col1, g_col2 = st.columns(2)
        metric_keys = list(METRIC_LAYMAN_DESCRIPTIONS.keys())
        for idx, m_key in enumerate(metric_keys):
            target_col = g_col1 if idx < 3 else g_col2
            target_col.markdown(f"* {METRIC_LAYMAN_DESCRIPTIONS[m_key]}")

    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader(f"Top 10 Characters by {METRIC_LABELS[size_metric].split(' (')[0]}")
        st.caption(
            "Ranks the ten highest-scoring characters for your selected sidebar metric, colour-coded by their home sub-series."
        )
        top_10_cent = (
            nodes_df.sort_values(by=size_metric, ascending=False)
            .head(10)
            .sort_values(by=size_metric, ascending=True)
        )
        fig_cent_bar = px.bar(
            top_10_cent,
            x=size_metric,
            y="Character",
            color="primary_series",
            color_discrete_map=SERIES_COLOURS,
            orientation="h",
            text_auto=".4f" if size_metric not in ("weighted_degree", "degree") else True,
            labels={size_metric: METRIC_LABELS[size_metric], "primary_series": "Primary Series"},
        )
        fig_cent_bar.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_cent_bar, width='stretch')

    with c_right:
        st.subheader("Narrative Hubs vs. Cross-Series Bridges")
        st.caption(
            "Separates **protagonists** from **connectors**: characters far to the right carry the dialogue in "
            "their own books (such as **Granny Weatherwax**), while characters high up on the vertical axis "
            "(such as **Havelock Vetinari** and **Death**) act as structural bridges stitching separate storylines together."
        )
        log_x = st.checkbox("Use log scale on X-axis (Weighted Degree)", value=True)
        fig_scatter = px.scatter(
            nodes_df,
            x="weighted_degree",
            y="betweenness",
            size="pagerank",
            color="primary_series",
            color_discrete_map=SERIES_COLOURS,
            hover_name="Character",
            text=nodes_df.apply(
                lambda r: r["Character"]
                if (
                    r["weighted_degree"] >= nodes_df["weighted_degree"].nlargest(12).min()
                    or r["betweenness"] >= nodes_df["betweenness"].nlargest(10).min()
                )
                else "",
                axis=1,
            ),
            log_x=log_x,
            labels={
                "weighted_degree": "Weighted Degree (Interaction Volume)",
                "betweenness": "Betweenness Centrality (Bridge Score)",
                "primary_series": "Primary Series",
            },
        )
        fig_scatter.update_traces(textposition="top center", textfont=dict(size=10))
        fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_scatter, width='stretch')

    st.subheader("Complete Centrality Rankings Table")
    st.caption(
        "Full leaderboard across all six network metrics for every character active under your current filters."
    )
    display_cols = [
        "Character",
        "primary_series",
        "community",
        "weighted_degree",
        "degree",
        "betweenness",
        "pagerank",
        "eigenvector",
        "closeness",
    ]
    table_df = (
        nodes_df[display_cols]
        .sort_values(by=size_metric, ascending=False)
        .rename(
            columns={
                "primary_series": "Primary Series",
                "community": "Community ID",
                "weighted_degree": "Weighted Degree",
                "degree": "Degree",
                "betweenness": "Betweenness",
                "pagerank": "PageRank",
                "eigenvector": "Eigenvector",
                "closeness": "Closeness",
            }
        )
    )
    table_df["Community ID"] = table_df["Community ID"] + 1
    st.dataframe(table_df, width='stretch', hide_index=True)

# ===========================================================================
# TAB 3: TOP 10 BAR CHARTS & INTERACTION HEATMAPS
# ===========================================================================
with tab_bars:
    # Prepare Top 10 rankings once so bar charts and heatmaps stay synchronised
    top_10_books_desc = (
        char_book_df.sort_values(
            by=["Book Appearances", "Filtered Interactions"], ascending=[False, False]
        )
        .head(10)
    )
    top_10_books_bar = top_10_books_desc.sort_values(by="Book Appearances", ascending=True)
    top_10_books_chars = top_10_books_desc["Character"].tolist()

    top_10_inter_desc = (
        char_book_df.sort_values(by="Filtered Interactions", ascending=False)
        .head(10)
    )
    top_10_inter_bar = top_10_inter_desc.sort_values(by="Filtered Interactions", ascending=True)
    top_10_inter_chars = top_10_inter_desc["Character"].tolist()

    # --- ROW 1: BAR CHARTS ---
    bar_col1, bar_col2 = st.columns(2)

    with bar_col1:
        st.subheader("Top 10 Characters by Novel Appearances")
        st.caption(
            "**Series-wide longevity:** Counts how many distinct active novels each character appears in. "
            "Characters who make brief appearances across a dozen books (such as **Death** or **Mustrum Ridcully**) "
            "can outrank protagonists who dominate only three or four."
        )
        fig_books = px.bar(
            top_10_books_bar,
            x="Book Appearances",
            y="Character",
            color="Primary Series",
            color_discrete_map=SERIES_COLOURS,
            orientation="h",
            text="Book Appearances",
            hover_data=["Sub-Series Spanned", "Books List"],
        )
        fig_books.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_books, use_container_width=True)

    with bar_col2:
        st.subheader("Top 10 Most Interactive Characters (Active Filter)")
        st.caption(
            "**Conversational intensity:** Ranks characters by their total co-occurrence weight across the "
            "active sub-series and novels, highlighting the figures who carry the busiest ensemble scenes."
        )
        fig_interactions = px.bar(
            top_10_inter_bar,
            x="Filtered Interactions",
            y="Character",
            color="Primary Series",
            color_discrete_map=SERIES_COLOURS,
            orientation="h",
            text="Filtered Interactions",
            hover_data=["Book Appearances", "Corpus Total Mentions"],
        )
        fig_interactions.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_interactions, use_container_width=True)

    st.divider()

    # --- ROW 2: MATCHING INTERACTION HEATMAPS FOR LONGEVITY & INTENSITY ---
    heat_col1, heat_col2 = st.columns(2)

    with heat_col1:
        st.subheader("Interaction Heatmap: Top 10 by Novel Appearances")
        st.caption(
            "**Do the series' longest-running regulars actually meet?** Because this group spans separate "
            "sub-series (such as **Death**, **Granny Weatherwax**, and **Sam Vimes**), sparse grids and `0` cells "
            "highlight recurring icons who share the canon without ever sharing a room."
        )
        fig_heat_books = build_interaction_heatmap(
            G=G_active,
            characters=top_10_books_chars,
            colorscale="BuGn",
            height=480,
        )
        st.plotly_chart(fig_heat_books, use_container_width=True)

    with heat_col2:
        st.subheader("Interaction Heatmap: Top 10 Most Interactive Characters")
        st.caption(
            "**Core ensemble double-acts:** Focuses on the ten highest-volume conversationalists. Dark clusters "
            "reveal tight double-acts and ensemble cores (such as **Vimes & Carrot** or **Granny & Nanny Ogg**)."
        )
        fig_heat_inter = build_interaction_heatmap(
            G=G_active,
            characters=top_10_inter_chars,
            colorscale="YlOrRd",
            height=480,
        )
        st.plotly_chart(fig_heat_inter, use_container_width=True)

    st.divider()

    # --- ROW 3: METRIC-DRIVEN CENTRALITY HEATMAP ---
    st.subheader("Interaction Heatmap: Top 10 Most Important Characters (By Centrality Metric)")
    st.markdown(
        "**How to Read This Grid:** Choose any centrality metric below to compare the top ten characters under "
        "that definition of importance. Switching from **Weighted Degree** to **Betweenness Centrality** shifts "
        "the grid from local storyline leads to cross-series connectors like **Havelock Vetinari** and **Death**."
    )
    heatmap_rank_metric = st.selectbox(
        "Select Metric Used to Choose Top 10 Characters for Heatmap",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        index=0,
        key="heatmap_metric_select",
    )
    top_10_metric_chars = (
        nodes_df.sort_values(by=heatmap_rank_metric, ascending=False)["Character"]
        .head(10)
        .tolist()
    )
    fig_heat_metric = build_interaction_heatmap(
        G=G_active,
        characters=top_10_metric_chars,
        colorscale="YlGnBu",
        height=540,
    )
    st.plotly_chart(fig_heat_metric, use_container_width=True)

# ===========================================================================
# TAB 4: COMMUNITY DETECTION & BRIDGE CONNECTORS
# ===========================================================================
with tab_comm:
    connectors_df = compute_community_connectors(G_active, nodes_df)

    st.subheader("Key Characters Connecting Discworld Communities")
    st.markdown(
        "Even when all book titles are hidden from the algorithm, **Louvain community detection** naturally rediscovers "
        "the Discworld's storylines purely from who interacts with whom. Comparing how strongly a character anchors their "
        "own cluster against how often they mingle with *other* clusters sorts the cast into **four structural storytelling roles**:"
    )

    role_c1, role_c2, role_c3, role_c4 = st.columns(4)
    role_c1.markdown(
        "**🟠 Connector Hubs**\n\nMajor stars who anchor their own storyline while pulling other casts into their orbit "
        "(e.g., **Havelock Vetinari**, **Samuel Vimes**, **Mustrum Ridcully**)."
    )
    role_c2.markdown(
        "**🔵 Provincial Hubs**\n\nBig fish in their own pond—protagonists like **Brutha** or **Tiffany Aching** who "
        "dominate their local cast with minimal crossover into Ankh-Morpork politics."
    )
    role_c3.markdown(
        "**🟣 Satellite Brokers**\n\nWandering cameos and travelling oddballs who do not headline eight books of their own, "
        "yet bridge separate social circles (e.g., **C.M.O.T. Dibbler**, **Lu-Tze**, **Gaspode**)."
    )
    role_c4.markdown(
        "**⚪ Local Members**\n\nSupporting regulars who keep their home storyline running without crossing over into "
        "distant communities."
    )

    bridge_col1, bridge_col2 = st.columns([1.3, 1.0])

    with bridge_col1:
        st.markdown("**Network Role Cartography (Hubs vs. Community Bridges)**")
        st.caption(
            "Characters further to the **right** split their interactions across multiple distinct communities; "
            "characters higher **up** dominate their home cluster."
        )
        fig_carto = build_role_cartography_figure(connectors_df)
        st.plotly_chart(fig_carto, width='stretch')

    with bridge_col2:
        st.markdown("**Top 12 Community Connectors (Ranked by Bridge Metric)**")
        st.caption(
            "**Participation Coefficient** measures how evenly ties spread across clusters; "
            "**Structural Hole Brokerage** rewards linking groups that otherwise never meet."
        )
        bridge_rank_choice = st.selectbox(
            "Rank Connectors By",
            options=[
                "Composite Bridge Index",
                "Participation (Communities)",
                "Participation (Sub-Series)",
                "Structural Hole Brokerage",
                "Betweenness",
                "Cross-Community %",
            ],
            index=0,
        )
        top_bridges = (
            connectors_df[connectors_df["Weighted Degree"] >= max(min_edge_weight, 5)]
            .sort_values(by=bridge_rank_choice, ascending=False)
            .head(12)
            .sort_values(by=bridge_rank_choice, ascending=True)
        )
        fig_bridge_bar = px.bar(
            top_bridges,
            x=bridge_rank_choice,
            y="Character",
            color="Structural Role",
            color_discrete_map=ROLE_COLOURS,
            orientation="h",
            hover_data=["Primary Series", "Home Community", "External Links"],
        )
        fig_bridge_bar.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bridge_bar, width='stretch')

    st.subheader("Community Connector & Brokerage Breakdown Table")
    role_filter = st.multiselect(
        "Filter by Structural Role",
        options=["Connector Hub", "Satellite Broker", "Provincial Hub", "Local Member"],
        default=["Connector Hub", "Satellite Broker"],
    )
    filtered_connectors = (
        connectors_df[connectors_df["Structural Role"].isin(role_filter)]
        if role_filter
        else connectors_df
    )
    st.dataframe(
        filtered_connectors.sort_values(by="Composite Bridge Index", ascending=False),
        width='stretch',
        hide_index=True,
    )

    st.divider()
    comm_col1, comm_col2 = st.columns([1.2, 1.0])

    with comm_col1:
        st.subheader("Sub-Series Crossover Matrix (Active Filter)")
        st.caption(
            "Shows which storylines share a neighbourhood. High numbers off the main diagonal prove that "
            "**City Watch**, **Industrial Revolution**, and **Wizards & Unseen University** form one overlapping "
            "Ankh-Morpork ecosystem, whereas **Ancient Civilisations** remains largely self-contained."
        )
        fig_crossover = build_series_crossover_heatmap(bundle, active_raw_edges)
        st.plotly_chart(fig_crossover, width='stretch')

    with comm_col2:
        st.subheader("Detected Louvain Communities vs. Canonical Sub-Series")
        st.caption(
            "Compares how readers group the books against how the characters *actually* socialise. When a single "
            "detected cluster blends Watchmen, journalists, and postmen together, it shows that Ankh-Morpork "
            "functions as a single unified social community across multiple sub-series."
        )
        nodes_df_comm = nodes_df.copy()
        nodes_df_comm["Louvain Cluster"] = nodes_df_comm["community"].apply(
            lambda c: f"Cluster {c + 1}"
        )
        comm_breakdown = (
            nodes_df_comm.groupby(["Louvain Cluster", "primary_series"])
            .size()
            .reset_index(name="Characters")
        )
        fig_comm_bar = px.bar(
            comm_breakdown,
            x="Louvain Cluster",
            y="Characters",
            color="primary_series",
            color_discrete_map=SERIES_COLOURS,
            labels={"primary_series": "Primary Sub-Series"},
        )
        fig_comm_bar.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_comm_bar, width='stretch')

    st.subheader("Inspect Members of Each Detected Louvain Community")
    cluster_ids = sorted(nodes_df["community"].unique())
    selected_cluster = st.selectbox(
        "Choose a Louvain Community to Inspect",
        options=cluster_ids,
        format_func=lambda c: f"Community {c + 1} ({len(nodes_df[nodes_df['community'] == c])} characters)",
    )
    cluster_members = (
        nodes_df[nodes_df["community"] == selected_cluster][
            ["Character", "primary_series", "weighted_degree", "betweenness", "pagerank"]
        ]
        .sort_values(by="weighted_degree", ascending=False)
        .rename(
            columns={
                "primary_series": "Primary Series",
                "weighted_degree": "Weighted Degree",
                "betweenness": "Betweenness",
                "pagerank": "PageRank",
            }
        )
    )
    st.dataframe(cluster_members, width='stretch', hide_index=True)