# Data manipulation
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import plotly.graph_objects as go

# Social Network Analysis
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# Widgets and Display
from ipywidgets import Dropdown, Button, Output, VBox, Tab, FloatSlider, HTML
from IPython.display import display, clear_output

# ==============================================
# Define user-level toxicity mapping function
# ==============================================

def build_user_toxicity_map(sna_df, threshold=0.25):
    """
    Builds a dictionary mapping each user to their toxicity status
    based on the proportion of toxic posts.
    """
    # Drop rows with missing author or toxicity label
    df = sna_df.dropna(subset=["author", "final_toxicity_label"]).copy()

    # Normalize author usernames: lowercase and remove '@'
    df["author"] = df["author"].str.lower().str.replace("@", "", regex=False)

    # Group by author and calculate number of toxic posts and total posts
    grouped = df.groupby("author")["final_toxicity_label"].agg(["sum", "count"])

    # Compute the toxicity ratio per user
    grouped["toxicity_ratio"] = grouped["sum"] / grouped["count"]

    # Apply threshold to assign binary toxicity label
    grouped["is_toxic"] = (grouped["toxicity_ratio"] >= threshold).astype(int)

    # Return mapping: author → 1 (toxic), 0 (non-toxic)
    return grouped["is_toxic"].to_dict()


# ==============================================
# Define match-level network visualization function
# ==============================================

# Helper to normalize usernames 
def normalize_handle(x):
    return str(x).lower().replace("@", "").strip()

# Visualize a match-level reply network 
def visualize_match_network(match_id, G, sna_df, match_metadata, tox_threshold=0.25):
    """
    Visualize the reply network for a given match with toxicity labels and club account highlighting.

    Parameters:
        match_id (str): Match identifier (e.g., 'M051')
        G (networkx.DiGraph): Match-level reply network
        sna_df (pd.DataFrame): Main dataset containing authors and toxicity labels
        match_metadata (pd.DataFrame): Metadata with official club handles
        tox_threshold (float): Minimum ratio of toxic posts to consider a user toxic
    """
    if G.number_of_nodes() == 0:
        print(f"Match {match_id} has no nodes to visualize.")
        return

    # Graph layout
    pos = nx.spring_layout(G, k=0.45, iterations=150, seed=42)

    # Build toxicity dictionary based on user threshold
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Get official club accounts for this match
    meta_row = match_metadata[match_metadata["match_id"] == match_id]
    club_accounts = set()
    if not meta_row.empty:
        club_accounts = {
            normalize_handle(meta_row.iloc[0]["home_handle"]),
            normalize_handle(meta_row.iloc[0]["away_handle"])
        }

    # Assign node styles
    node_colors = []
    node_sizes = []
    label_nodes = {}

    for node in G.nodes():
        clean_node = normalize_handle(node)
        label = tox_dict.get(clean_node)

        if clean_node in club_accounts:
            node_colors.append("gold")
            node_sizes.append(400)
            label_nodes[node] = f"@{clean_node}"
        elif label == 1:
            node_colors.append("red")
            node_sizes.append(120)
        elif label == 0:
            node_colors.append("blue")
            node_sizes.append(80)
        else:
            node_colors.append("lightgray")
            node_sizes.append(40)

    # Edge filtering 
    G_filtered = G.copy()
    for u, v in G.edges():
        u_clean = normalize_handle(u)
        v_clean = normalize_handle(v)
        if (
            tox_dict.get(u_clean) is None and
            tox_dict.get(v_clean) is None and
            u_clean not in club_accounts and
            v_clean not in club_accounts
        ):
            G_filtered.remove_edge(u, v)

    # Plotting
    plt.figure(figsize=(15, 12))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.3)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")

    nx.draw_networkx_edges(
        G_filtered,
        pos,
        edge_color="gray",
        alpha=0.2,
        width=0.6,
        arrows=False
    )

    plt.title(f"Match {match_id} — Toxicity Interaction Network", fontsize=16)
    plt.axis("off")

    # Legend 
    legend = [
        mpatches.Patch(color='red', label='Toxic user'),
        mpatches.Patch(color='blue', label='Non-toxic user'),
        mpatches.Patch(color='lightgray', label='Not labeled'),
        mpatches.Patch(color='gold', label='Official club account')
    ]
    plt.legend(handles=legend, loc='lower left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Detect communities and compute toxicity metrics
# ==============================================

def compute_community_toxicity_metrics(G, sna_df, match_id, min_size=5, tox_threshold=0.25):
    """
    Detects communities and computes toxicity-related metrics per community using a toxicity threshold.
    Returns a DataFrame with per-community stats.
    """

    # Community detection on undirected graph
    undirected = G.to_undirected()
    communities = list(greedy_modularity_communities(undirected))

    # Build user-level toxicity labels
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Compute metrics for each sufficiently large community
    results = []
    for idx, community in enumerate(communities):
        if len(community) < min_size:
            continue  # Skip small communities

        toxic, nontoxic, unknown = 0, 0, 0

        # Count user types in the community
        for user in community:
            user_clean = str(user).lower().replace("@", "")
            label = tox_dict.get(user_clean)
            if label == 1:
                toxic += 1
            elif label == 0:
                nontoxic += 1
            else:
                unknown += 1

        total = toxic + nontoxic + unknown
        pct_toxic = round(100 * toxic / total, 2) if total > 0 else 0

        # Store metrics
        results.append({
            "match_id": match_id,
            "community_id": idx,
            "num_users": total,
            "num_toxic": toxic,
            "num_nontoxic": nontoxic,
            "num_unknown": unknown,
            "pct_toxic": pct_toxic
        })

    return pd.DataFrame(results)

# ==============================================
# Visualize top communities in match-level network
# ==============================================

def plot_top_communities(G_full, match_id, sna_df, match_metadata, top_n=10):
    # Community detection
    undirected = G_full.to_undirected()
    communities = list(greedy_modularity_communities(undirected))
    print(f"Total communities detected: {len(communities)}")

    # Keep top N largest communities
    sorted_communities = sorted(communities, key=len, reverse=True)
    top_comms = sorted_communities[:top_n]
    top_nodes = set().union(*top_comms)

    # Build subgraph and layout
    G = G_full.subgraph(top_nodes).copy()
    pos = nx.spring_layout(G, k=0.6, iterations=200, seed=42)

    # Assign each node to its community index
    node_to_comm = {}
    for i, comm in enumerate(top_comms):
        for node in comm:
            node_to_comm[node] = i

    # Assign colors to communities
    cmap = cm.get_cmap('tab10', top_n)
    comm_colors = {i: cmap(i) for i in range(top_n)}
    node_colors = [comm_colors[node_to_comm[n]] for n in G.nodes]

    # Prepare toxicity and club lookups
    tox_map = sna_df.set_index("author")["final_toxicity_label"].to_dict()
    meta_row = match_metadata[match_metadata["match_id"] == match_id]
    club_accounts = set()
    if not meta_row.empty:
        club_accounts = {
            str(meta_row.iloc[0]["home_handle"]).lower().replace("@", ""),
            str(meta_row.iloc[0]["away_handle"]).lower().replace("@", "")
        }

    # Node styling: color edge border by toxicity and size by role
    edge_colors = []
    node_sizes = []
    label_nodes = {}
    for node in G.nodes():
        clean_node = str(node).lower().replace("@", "")
        toxicity = tox_map.get(node, None)

        if clean_node in club_accounts:
            edge_colors.append("black")
            node_sizes.append(380)
            label_nodes[node] = f"@{clean_node}"
        elif toxicity == 1:
            edge_colors.append("red")
            node_sizes.append(110)
        elif toxicity == 0:
            edge_colors.append("blue")
            node_sizes.append(80)
        else:
            edge_colors.append("lightgray")
            node_sizes.append(50)

    # Draw graph
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors=edge_colors, linewidths=1.2, alpha=0.95
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.2, width=0.8)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")

    plt.title(f"{match_id} — Top {top_n} Communities (out of {len(communities)})", fontsize=16)
    plt.axis("off")
    plt.show()

# ==============================================
# Visualize most toxic community in match-level network
# ==============================================

def plot_most_toxic_community_for_match(G_full, match_id, sna_df, match_metadata, min_size=5, tox_threshold=0.25):
    """
    Detect and visualize the most toxic community in a match-level reply network.
    """

    # Detect communities on undirected graph
    undirected = G_full.to_undirected()
    communities = list(greedy_modularity_communities(undirected))

    # Create toxicity label mapping
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Score each community by toxicity %
    toxic_stats = []
    for comm in communities:
        if len(comm) < min_size:
            continue
        tox_count, total = 0, 0
        for node in comm:
            label = tox_dict.get(str(node).lower().replace("@", ""))
            if label in [0, 1]:
                total += 1
                if label == 1:
                    tox_count += 1
        pct_toxic = tox_count / total if total > 0 else 0
        toxic_stats.append((comm, pct_toxic))

    if not toxic_stats:
        print(f"No communities of size ≥ {min_size} found for match {match_id}")
        return

    # Select most toxic community
    most_toxic_comm, toxicity_score = max(toxic_stats, key=lambda x: x[1])
    print(f"Most toxic community has {len(most_toxic_comm)} users and {toxicity_score:.2%} toxicity.")

    # Build subgraph for visualization
    G = G_full.subgraph(most_toxic_comm).copy()
    pos = nx.spring_layout(G, k=0.6, iterations=200, seed=42)

    # Identify official club accounts
    meta_row = match_metadata[match_metadata["match_id"] == match_id]
    club_accounts = set()
    if not meta_row.empty:
        club_accounts = {
            str(meta_row.iloc[0]["home_handle"]).lower().replace("@", ""),
            str(meta_row.iloc[0]["away_handle"]).lower().replace("@", "")
        }

    # Node coloring and sizing
    node_colors = []
    node_sizes = []
    edge_colors = []
    label_nodes = {}
    for node in G.nodes():
        clean_node = str(node).lower().replace("@", "")
        label = tox_dict.get(clean_node)

        if clean_node in club_accounts:
            node_colors.append("gold")
            node_sizes.append(400)
            edge_colors.append("black")
            label_nodes[node] = f"@{clean_node}"
        elif label == 1:
            node_colors.append("red")
            node_sizes.append(110)
            edge_colors.append("red")
        elif label == 0:
            node_colors.append("blue")
            node_sizes.append(80)
            edge_colors.append("blue")
        else:
            node_colors.append("lightgray")
            node_sizes.append(50)
            edge_colors.append("lightgray")

    # Plot network
    plt.figure(figsize=(13, 11))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors=edge_colors, linewidths=1.1, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.25, width=0.8)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")

    plt.title(f"Match {match_id} — Most Toxic Community (toxicity = {toxicity_score:.1%})", fontsize=16)
    plt.axis("off")

    # Legend
    legend = [
        mpatches.Patch(color='red', label='Toxic user'),
        mpatches.Patch(color='blue', label='Non-toxic user'),
        mpatches.Patch(color='lightgray', label='Not labeled'),
        mpatches.Patch(color='gold', label='Official club account')
    ]
    plt.legend(handles=legend, loc='lower left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Plot timeline of toxic vs non-toxic interactions
# ==============================================

def plot_toxicity_timeline_with_yellow_lines(match_id, df, metadata_df, time_bin="H"):
    """
    Plot toxic vs non-toxic interactions with clear yellow dotted lines showing match duration.
    Supports Dutch month names in metadata.
    """

    # Filter data for selected match
    df_match = df[df["match_id"] == match_id].copy()
    meta = metadata_df[metadata_df["match_id"] == match_id]

    if df_match.empty or meta.empty:
        print(f"No data for match {match_id}")
        return

    # Convert timestamps and prepare toxicity label column
    df_match["timestamp"] = pd.to_datetime(df_match["timestamp"], errors="coerce")
    df_match = df_match.dropna(subset=["timestamp", "final_toxicity_label"])
    df_match.set_index("timestamp", inplace=True)
    df_match["tox_label"] = df_match["final_toxicity_label"].map({1.0: "toxic", 0.0: "non-toxic"})

    # Resample interactions by time bin
    toxic = df_match[df_match["tox_label"] == "toxic"].resample(time_bin).size()
    nontoxic = df_match[df_match["tox_label"] == "non-toxic"].resample(time_bin).size()
    timeline_df = pd.DataFrame({'Toxic': toxic, 'Non-toxic': nontoxic}).fillna(0)

    # Parse Dutch date and translate to English month
    date_str = meta.iloc[0]["date"]
    time_str = meta.iloc[0]["time"]
    dutch_to_english = {
        "januari": "January", "februari": "February", "maart": "March", "april": "April",
        "mei": "May", "juni": "June", "juli": "July", "augustus": "August",
        "september": "September", "oktober": "October", "november": "November", "december": "December"
    }
    for dutch, eng in dutch_to_english.items():
        if dutch in date_str.lower():
            date_str = date_str.lower().replace(dutch, eng)
            break

    # Parse match start time and estimate full-time
    match_start = pd.to_datetime(f"{date_str} {time_str}", format="%d %B %Y %H:%M", errors="coerce")
    match_end = match_start + pd.Timedelta(minutes=105)

    # Plot toxic vs. non-toxic timeline
    fig, ax = plt.subplots(figsize=(12, 6))
    timeline_df.plot.area(ax=ax, color=["red", "blue"], alpha=0.4)

    # Add vertical lines for kickoff and full-time
    if pd.notnull(match_start) and pd.notnull(match_end):
        for t, label in zip([match_start, match_end], ["Kick-off", "Full-time"]):
            ax.axvline(t, color="gold", linestyle=":", linewidth=2.5, zorder=10, label=label)
            ax.text(
                t,
                ax.get_ylim()[1] * 0.95,
                label,
                rotation=90,
                color="gold",
                fontsize=9,
                ha="center",
                va="top",
                zorder=11,
                weight="bold"
            )

    # Add titles and labels
    if "home_team" in meta.columns and "away_team" in meta.columns:
        home = meta.iloc[0]["home_team"]
        away = meta.iloc[0]["away_team"]
        ax.set_title(f"Toxic vs Non-Toxic Interactions — {home} vs {away} ({match_id})")
    else:
        ax.set_title(f"Toxic vs Non-Toxic Interactions — Match {match_id}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Interactions")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================
# Visualize club-level reply network
# ==============================================

# Helper to normalize usernames 
def normalize_handle(x):
    return str(x).lower().replace("@", "").strip()

# Visualize a club's reply network 
def visualize_club_network(club_name, G, sna_df, match_metadata, tox_threshold=0.25):
    """
    Visualize the reply network for a given club with toxicity labels and official account highlighting.
    """

    if G.number_of_nodes() == 0:
        print(f"Club {club_name} has no nodes to visualize.")
        return

    # Graph layout
    pos = nx.spring_layout(G, k=0.45, iterations=150, seed=42)

    # Build user toxicity map
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Get all official club handles from metadata
    home_handles = match_metadata["home_handle"].dropna().apply(normalize_handle)
    away_handles = match_metadata["away_handle"].dropna().apply(normalize_handle)
    club_accounts_all = set(home_handles).union(set(away_handles))

    # Node styling setup
    node_colors = []
    node_sizes = []
    label_nodes = {}

    for node in G.nodes():
        clean_node = normalize_handle(node)
        label = tox_dict.get(clean_node)

        if clean_node == normalize_handle(club_name) or clean_node in club_accounts_all:
            node_colors.append("gold")
            node_sizes.append(400)
            label_nodes[node] = f"@{clean_node}"
        elif label == 1:
            node_colors.append("red")
            node_sizes.append(120)
        elif label == 0:
            node_colors.append("blue")
            node_sizes.append(80)
        else:
            node_colors.append("lightgray")
            node_sizes.append(40)

    # Filter out edges between unknown non-club users for clarity
    G_filtered = G.copy()
    for u, v in G.edges():
        u_clean = normalize_handle(u)
        v_clean = normalize_handle(v)
        if (
            tox_dict.get(u_clean) is None and
            tox_dict.get(v_clean) is None and
            u_clean not in club_accounts_all and
            v_clean not in club_accounts_all
        ):
            G_filtered.remove_edge(u, v)

    # Plot the network
    plt.figure(figsize=(15, 12))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.3)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")
    nx.draw_networkx_edges(G_filtered, pos, edge_color="gray", alpha=0.2, width=0.6, arrows=False)

    plt.title(f"Club {club_name} — Toxicity Interaction Network", fontsize=16)
    plt.axis("off")

    # Legend
    legend = [
        mpatches.Patch(color='red', label='Toxic user'),
        mpatches.Patch(color='blue', label='Non-toxic user'),
        mpatches.Patch(color='lightgray', label='Not labeled'),
        mpatches.Patch(color='gold', label='Official club account')
    ]
    plt.legend(handles=legend, loc='lower left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Compute toxicity metrics for communities in a club network
# ==============================================

def compute_club_community_toxicity_metrics(G, sna_df, club_name, min_size=5, tox_threshold=0.25):
    """
    Detects communities and computes toxicity-related metrics per community for a club network.
    """

    # Detect communities on undirected version of the graph
    undirected = G.to_undirected()
    communities = list(greedy_modularity_communities(undirected))

    # Create toxicity label mapping
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Compute metrics for each large enough community
    results = []
    for idx, community in enumerate(communities):
        if len(community) < min_size:
            continue  # Skip small communities

        toxic, nontoxic, unknown = 0, 0, 0
        for user in community:
            user_clean = str(user).lower().replace("@", "")
            label = tox_dict.get(user_clean)
            if label == 1:
                toxic += 1
            elif label == 0:
                nontoxic += 1
            else:
                unknown += 1

        total = toxic + nontoxic + unknown
        pct_toxic = round(100 * toxic / total, 2) if total > 0 else 0

        # Store community stats
        results.append({
            "club_name": club_name,
            "community_id": idx,
            "num_users": total,
            "num_toxic": toxic,
            "num_nontoxic": nontoxic,
            "num_unknown": unknown,
            "pct_toxic": pct_toxic
        })

    return pd.DataFrame(results)

# ==============================================
# Visualize top communities in a club-level network
# ==============================================

def plot_top_communities_for_club(G_full, club_name, sna_df, match_metadata, top_n=10, tox_threshold=0.25):
    """
    Visualize the top N largest communities in a club-level graph with toxicity classification.
    """

    # Community detection
    undirected = G_full.to_undirected()
    communities = list(greedy_modularity_communities(undirected))
    print(f"Total communities detected: {len(communities)}")

    # Sort communities by size and keep top N
    sorted_communities = sorted(communities, key=len, reverse=True)
    top_comms = sorted_communities[:top_n]
    top_nodes = set().union(*top_comms)

    # Extract subgraph and generate layout
    G = G_full.subgraph(top_nodes).copy()
    pos = nx.spring_layout(G, k=0.6, iterations=200, seed=42)

    # Assign nodes to communities
    node_to_comm = {}
    for i, comm in enumerate(top_comms):
        for node in comm:
            node_to_comm[node] = i

    # Assign colors per community
    cmap = cm.get_cmap('tab10', top_n)
    comm_colors = {i: cmap(i) for i in range(top_n)}
    node_colors = [comm_colors[node_to_comm[n]] for n in G.nodes]

    # Build toxicity map
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Collect official club handles
    home_handles = match_metadata["home_handle"].dropna().apply(lambda x: str(x).lower().replace("@", ""))
    away_handles = match_metadata["away_handle"].dropna().apply(lambda x: str(x).lower().replace("@", ""))
    club_accounts = set(home_handles).union(set(away_handles))

    # Node styling
    edge_colors = []
    node_sizes = []
    label_nodes = {}

    for node in G.nodes():
        clean_node = str(node).lower().replace("@", "")
        toxicity = tox_dict.get(clean_node, None)

        if clean_node == club_name.lower() or clean_node in club_accounts:
            edge_colors.append("black")
            node_sizes.append(380)
            label_nodes[node] = f"@{clean_node}"
        elif toxicity == 1:
            edge_colors.append("red")
            node_sizes.append(110)
        elif toxicity == 0:
            edge_colors.append("blue")
            node_sizes.append(80)
        else:
            edge_colors.append("lightgray")
            node_sizes.append(50)

    # Plot
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes,
        edgecolors=edge_colors, linewidths=1.2, alpha=0.95
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.2, width=0.8)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")

    plt.title(f"{club_name} — Top {top_n} Communities (out of {len(communities)})", fontsize=16)
    plt.axis("off")

    # Legend
    legend = [
        mpatches.Patch(color='red', label='Toxic user'),
        mpatches.Patch(color='blue', label='Non-toxic user'),
        mpatches.Patch(color='lightgray', label='Not labeled'),
        mpatches.Patch(color='black', label='Official club account')
    ]
    plt.legend(handles=legend, loc='lower left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Visualize most toxic community in a club network
# ==============================================

def plot_most_toxic_community_for_club(G_full, club_name, sna_df, match_metadata, min_size=5, tox_threshold=0.25):
    """
    Detect and visualize the most toxic community in a club-level network.
    """

    # Detect communities
    undirected = G_full.to_undirected()
    communities = list(greedy_modularity_communities(undirected))

    # Build user toxicity lookup
    tox_dict = build_user_toxicity_map(sna_df, threshold=tox_threshold)

    # Score communities by % toxic users
    toxic_stats = []
    for comm in communities:
        if len(comm) < min_size:
            continue
        tox_count, total = 0, 0
        for node in comm:
            label = tox_dict.get(str(node).lower().replace("@", ""))
            if label in [0, 1]:
                total += 1
                if label == 1:
                    tox_count += 1
        pct_toxic = tox_count / total if total > 0 else 0
        toxic_stats.append((comm, pct_toxic))

    # Abort if no valid communities found
    if not toxic_stats:
        print(f"No communities of size ≥ {min_size} found for {club_name}")
        return

    # Identify most toxic community
    most_toxic_comm, toxicity_score = max(toxic_stats, key=lambda x: x[1])
    print(f"Most toxic community has {len(most_toxic_comm)} users and {toxicity_score:.2%} toxicity.")

    # Build subgraph for visualization
    G = G_full.subgraph(most_toxic_comm).copy()
    pos = nx.spring_layout(G, k=0.6, iterations=200, seed=42)

    # Lookup official club accounts
    home_handles = match_metadata["home_handle"].dropna().apply(lambda x: str(x).lower().replace("@", ""))
    away_handles = match_metadata["away_handle"].dropna().apply(lambda x: str(x).lower().replace("@", ""))
    club_accounts = set(home_handles).union(set(away_handles))

    # Style nodes
    node_colors = []
    node_sizes = []
    edge_colors = []
    label_nodes = {}

    for node in G.nodes():
        clean_node = str(node).lower().replace("@", "")
        label = tox_dict.get(clean_node)

        if clean_node == club_name.lower() or clean_node in club_accounts:
            node_colors.append("gold")
            node_sizes.append(400)
            edge_colors.append("black")
            label_nodes[node] = f"@{clean_node}"
        elif label == 1:
            node_colors.append("red")
            node_sizes.append(110)
            edge_colors.append("red")
        elif label == 0:
            node_colors.append("blue")
            node_sizes.append(80)
            edge_colors.append("blue")
        else:
            node_colors.append("lightgray")
            node_sizes.append(50)
            edge_colors.append("lightgray")

    # Plot
    plt.figure(figsize=(13, 11))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors=edge_colors, linewidths=1.1, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.25, width=0.8)
    nx.draw_networkx_labels(G, pos, labels=label_nodes, font_size=10, font_color="black")

    plt.title(f"{club_name} — Most Toxic Community (toxicity = {toxicity_score:.1%})", fontsize=16)
    plt.axis("off")

    legend = [
        mpatches.Patch(color='red', label='Toxic user'),
        mpatches.Patch(color='blue', label='Non-toxic user'),
        mpatches.Patch(color='lightgray', label='Not labeled'),
        mpatches.Patch(color='gold', label='Official club account')
    ]
    plt.legend(handles=legend, loc='lower left')
    plt.tight_layout()
    plt.show()

# ==============================================
# Interactive network visualization 
# ==============================================

def plot_interactive_network(G, tox_dict=None, title="Interactive Network", node_limit=200):
    """
    Plots an interactive network with Plotly.

    Parameters:
        G (networkx.Graph): The graph to plot (can be directed or undirected)
        tox_dict (dict): Optional user → toxicity label mapping (0, 1, or None)
        title (str): Plot title
        node_limit (int): Max number of nodes to visualize for performance

    Returns:
        Plotly Figure
    """

    # Limit to top-degree nodes for performance
    if len(G.nodes) > node_limit:
        G = G.subgraph(sorted(G.degree, key=lambda x: x[1], reverse=True)[:node_limit])

    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Nodes
    node_x, node_y = [], []
    node_color, node_text = [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        clean_node = str(node).lower().replace("@", "")
        label = tox_dict.get(clean_node) if tox_dict else None

        if label == 1:
            node_color.append("red")
            label_txt = "Toxic"
        elif label == 0:
            node_color.append("blue")
            label_txt = "Non-toxic"
        else:
            node_color.append("gray")
            label_txt = "Unknown"

        node_text.append(f"{node}<br>Status: {label_txt}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=1
        )
    )

    # Assemble figure
    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=dict(text=title, font=dict(size=16)),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False),
                    height=600
                ))
    return fig

# ==============================================
# Network Summary Statistics Function
# ==============================================

def display_network_stats(G, tox_dict=None):
    """
    Print summary statistics for a given networkx graph, including node/edge counts,
    community count, and optionally toxic user percentage.
    """
    from networkx.algorithms.community import greedy_modularity_communities
    from IPython.display import display

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if tox_dict:
        labels = [tox_dict.get(str(n).lower().replace("@", ""), None) for n in G.nodes()]
        num_toxic = labels.count(1)
        pct_toxic = round(100 * num_toxic / len(labels), 2) if labels else 0
    else:
        num_toxic = pct_toxic = "N/A"

    try:
        comms = list(greedy_modularity_communities(G.to_undirected()))
        n_comms = len(comms)
        largest_comm = max(len(c) for c in comms) if comms else 0
    except:
        n_comms = largest_comm = "N/A"

    print(f"Users: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Toxic users: {num_toxic} ({pct_toxic}%)")
    print(f"Communities: {n_comms}")
    print(f"Largest community size: {largest_comm}")

# ==============================================
# Club-Level Network Comparison Function
# ==============================================

def compute_club_comparison_data(club_networks, sna_df, threshold=0.25):
    rows = []
    tox_dict = build_user_toxicity_map(sna_df, threshold=threshold)

    for club_name, G in club_networks.items():
        labels = [
            tox_dict.get(str(n).lower().replace("@", ""), None)
            for n in G.nodes()
        ]
        total = len(labels)
        toxic = labels.count(1)
        nontoxic = labels.count(0)
        unknown = labels.count(None)
        pct_toxic = round(100 * toxic / total, 2) if total else 0

        try:
            comms = list(nx.community.greedy_modularity_communities(G.to_undirected()))
            n_comms = len(comms)
            largest_comm = max(len(c) for c in comms) if comms else 0
        except:
            n_comms = largest_comm = 0

        rows.append({
            "club": club_name,
            "Total_users": total,
            "Toxic_users": toxic,
            "%_toxic": pct_toxic,
            "Communities": n_comms,
            "Largest_community": largest_comm
        })

    return pd.DataFrame(rows)

# ==============================================
# Interactive Dashboard Launcher Function
# ==============================================

def launch_dashboard(match_networks, club_networks, sna_df, match_metadata):
    """
    Launches the interactive dashboard UI for visualizing toxicity by matchday, club, and overall club comparison.
    """

    # Toxicity threshold slider and help 
    toxicity_help = HTML(
    "<b>Toxicity Threshold:</b><br>"
    "Defines how toxic a user must be to be labeled as toxic.<br>"
    "E.g., a threshold of <code>0.55</code> means users with average toxicity ≥ 0.55 across posts are toxic."
   )

    toxicity_slider = FloatSlider(
        value=0.25, min=0.0, max=1.0, step=0.05,
        description='Tox. Threshold:', continuous_update=False
    )

    save_button = Button(description='Save Figure', icon='save')
    stats_output = Output()

    def save_current_figure(filename='network_plot.png'):
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved figure as: {filename}")
        except Exception as e:
            print(f"Could not save figure: {e}")

    save_button.on_click(lambda b: save_current_figure())

    # Matchday Tab 
    matchday_options = {
        f"{row['home_team']} vs {row['away_team']} ({row['match_id']})": row['match_id']
        for _, row in match_metadata.iterrows()
        if row['match_id'] in match_networks
    }

    matchday_selector = Dropdown(
        options=matchday_options,
        description='Matchday:',
        value=list(matchday_options.values())[0]
    )

    matchday_layer_selector = Dropdown(
        options=[
            'Layer 1: Match Network',
            'Layer 2: Top Communities',
            'Layer 3: Community Toxicity Table',
            'Layer 4: Most Toxic Community',
            'Layer 5: Toxicity Timeline',
            'Layer 6: Interactive Network'
        ],
        description='Layer:',
        value='Layer 1: Match Network'
    )

    matchday_button = Button(description='Visualize', button_style='primary')
    matchday_output = Output()

    def on_matchday_click(b):
        with matchday_output:
            clear_output()
            matchday = matchday_selector.value
            layer = matchday_layer_selector.value
            G = match_networks[matchday]
            threshold = toxicity_slider.value

            if layer == 'Layer 1: Match Network':
                visualize_match_network(matchday, G, sna_df, match_metadata, tox_threshold=threshold)
            elif layer == 'Layer 2: Top Communities':
                plot_top_communities(G, matchday, sna_df, match_metadata)
            elif layer == 'Layer 3: Community Toxicity Table':
                df = compute_community_toxicity_metrics(G, sna_df, matchday, tox_threshold=threshold)
                display(df)
            elif layer == 'Layer 4: Most Toxic Community':
                plot_most_toxic_community_for_match(G, matchday, sna_df, match_metadata, tox_threshold=threshold)
            elif layer == 'Layer 5: Toxicity Timeline':
                plot_toxicity_timeline_with_yellow_lines(matchday, sna_df, match_metadata)
            elif layer == 'Layer 6: Interactive Network':
                fig = plot_interactive_network(G, build_user_toxicity_map(sna_df, threshold), title=f"{matchday} — Interactive Network")
                fig.show()

            display_network_stats(G, build_user_toxicity_map(sna_df, threshold))

    matchday_button.on_click(on_matchday_click)

    matchday_ui = VBox([
        matchday_selector,
        matchday_layer_selector,
        toxicity_help,
        toxicity_slider,
        matchday_button,
        save_button,
        matchday_output,
        stats_output
    ])

    # Club Tab 
    club_selector = Dropdown(
        options=sorted(club_networks.keys()),
        description='Club:',
        value=sorted(club_networks.keys())[0]
    )

    club_layer_selector = Dropdown(
        options=[
            'Layer 1: Club Network',
            'Layer 2: Top Communities',
            'Layer 3: Community Toxicity Table',
            'Layer 4: Most Toxic Community'
        ],
        description='Layer:',
        value='Layer 1: Club Network'
    )

    club_button = Button(description='Visualize', button_style='primary')
    club_output = Output()

    def on_club_click(b):
        with club_output:
            clear_output()
            club = club_selector.value
            layer = club_layer_selector.value
            G = club_networks[club]
            threshold = toxicity_slider.value

            if layer == 'Layer 1: Club Network':
                visualize_club_network(club, G, sna_df, match_metadata, tox_threshold=threshold)
            elif layer == 'Layer 2: Top Communities':
                plot_top_communities_for_club(G, club, sna_df, match_metadata, tox_threshold=threshold)
            elif layer == 'Layer 3: Community Toxicity Table':
                df = compute_club_community_toxicity_metrics(G, sna_df, club, tox_threshold=threshold)
                display(df)
            elif layer == 'Layer 4: Most Toxic Community':
                plot_most_toxic_community_for_club(G, club, sna_df, match_metadata, tox_threshold=threshold)

            display_network_stats(G, build_user_toxicity_map(sna_df, threshold))

    club_button.on_click(on_club_click)

    club_ui = VBox([
        club_selector,
        club_layer_selector,
        toxicity_help,
        toxicity_slider,
        club_button,
        save_button,
        club_output,
        stats_output
    ])

    # Club Comparison Tab 
    comparison_metric_selector = Dropdown(
        options=['%_toxic', 'Toxic_users', 'Total_users', 'Communities', 'Largest_community'],
        value='%_toxic',
        description='Compare by:'
    )

    comparison_output = Output()

    def update_comparison_plot(change=None):
        with comparison_output:
            clear_output()
            threshold = toxicity_slider.value
            metric = comparison_metric_selector.value

            df = compute_club_comparison_data(club_networks, sna_df, threshold=threshold)
            top_df = df.sort_values(metric, ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            plt.barh(top_df["club"], top_df[metric], color="steelblue")
            plt.xlabel(metric.replace("_", " ").title())
            plt.title(f"Top 10 Clubs by {metric.replace('_', ' ').title()} (Threshold ≥ {threshold:.2f})")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()

    comparison_metric_selector.observe(update_comparison_plot, names='value')

    comparison_ui = VBox([
        comparison_metric_selector,
        comparison_output
    ])

    tabs = Tab(children=[
    matchday_ui,
    club_ui,
    comparison_ui
    ])
    tabs.set_title(0, 'By Matchday')
    tabs.set_title(1, 'By Club')
    tabs.set_title(2, 'Club Comparison')

    update_comparison_plot()
    display(tabs)
