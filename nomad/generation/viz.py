import matplotlib.pyplot as plt
from matplotlib import cm

def plot_sparse_clusters(
    data,
    labels,
    ax,
    full_traj=None,
    buffer=None,              # 0‒1 → pad bbox by (1+buffer); None → no limits
    cmap=cm.tab20c
):
    n_clusters = int(labels[labels >= 0].max() + 1) if (labels >= 0).any() else 0
    for cid in range(n_clusters):
        m = labels == cid
        ax.scatter(data.x[m], data.y[m],
                   s=80, color=cmap(cid / (n_clusters + 1)),
                   zorder=2)
    ax.scatter(data.x, data.y, s=6, color='black', zorder=2)
    if full_traj is not None:
        ax.plot(full_traj.x, full_traj.y, lw=1.2, color='blue', alpha=0.2, zorder=1)
        if buffer is not None:
            x0, x1 = full_traj.x.min(), full_traj.x.max()
            y0, y1 = full_traj.y.min(), full_traj.y.max()
            pad_x = (x1 - x0) * buffer / 2
            pad_y = (y1 - y0) * buffer / 2
            ax.set_xlim(x0 - pad_x, x1 + pad_x)
            ax.set_ylim(y0 - pad_y, y1 + pad_y)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return ax