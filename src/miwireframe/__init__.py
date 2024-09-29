from typing import Tuple
import numpy as np

from . import core


def maximal_segments(
    vertex_count: int,
    edges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

    edges = np.ascontiguousarray(edges)
    edge_count, _ = edges.shape

    vertex_indices = np.full((edge_count * 2,), -1, dtype=np.int32)
    segment_indices = np.full((edge_count,), -1, dtype=np.int32)

    core.maximal_segments(vertex_count, edges, vertex_indices, segment_indices)

    vertex_indices = vertex_indices[vertex_indices >= 0]
    segment_indices = segment_indices[segment_indices >= 0]

    return vertex_indices, segment_indices



