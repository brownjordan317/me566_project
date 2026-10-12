"""Fast connected-component labels for triangle meshes."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def triangle_components(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label triangles connected through shared edges, as Open3D's
    cluster_connected_triangles does, and return (labels, component sizes)."""
    triangles = np.asarray(triangles, dtype=np.int64)
    count = len(triangles)
    if not count:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    codes = edges[:, 0] * (int(triangles.max()) + 1) + edges[:, 1]
    owners = np.repeat(np.arange(count), 3)
    order = np.argsort(codes)
    codes, owners = codes[order], owners[order]
    # Consecutive equal codes are the same edge; link the triangles that share it.
    shared = codes[1:] == codes[:-1]
    graph = coo_matrix((np.ones(int(shared.sum()), np.int8),
                        (owners[:-1][shared], owners[1:][shared])), shape=(count, count))
    # SciPy numbers components 0..n-1, so labels index the size table directly.
    _, labels = connected_components(graph, directed=False)
    return labels, np.bincount(labels)
