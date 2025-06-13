import pandas as pd
import numpy as np
import networkx as nx

def visualize_mst(mst_df):
    G = nx.Graph()

    for u, v, w in mst_df.itertuples(index=False):
        w_val = 1e6 if np.isinf(w) else w
        inv_w = 1.0 / w_val if w_val > 0 else 0.001
        label = "∞" if np.isinf(w) else f"{w:.2f}"
        G.add_edge(u, v, weight=w_val, inv_weight=inv_w, label=label)

    # Use inverse weights for layout
    pos = nx.spring_layout(G, weight='inv_weight')

    # Retrieve edge labels from edge attributes
    edge_labels = nx.get_edge_attributes(G, 'label')

    # Plot
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("MST Extended Graph with Edge Weights")
    plt.axis('off')
    plt.show()

def visualize_adjacency_dict(G_dict):
    """
    Visualize an adjacency dictionary as a NetworkX graph.

    Parameters
    ----------
    G_dict : dict
        Output of _build_graph() — {node: set(neighbors)}
    """
    # Convert dict-of-sets into a networkx Graph
    G = nx.Graph()
    for u, neighbors in G_dict.items():
        for v in neighbors:
            G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)

    # Draw graph
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightgreen',
        edge_color='gray',
        node_size=200,
        font_size=10
    )
    plt.title("Connected Component After Edge Removal")
    plt.tight_layout()
    plt.show()