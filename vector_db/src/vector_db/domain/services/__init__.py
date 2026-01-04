"""Domain services for the vector database."""

from vector_db.domain.services.distance import (
    batch_cosine_distance,
    batch_euclidean_distance,
    batch_inner_product,
    compute_ground_truth,
    compute_recall,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    get_batch_distance_function,
    get_distance_function,
    inner_product,
    normalize_vectors,
    squared_euclidean_distance,
)
from vector_db.domain.services.hnsw_index import HNSWIndex, HNSWParams
from vector_db.domain.services.ivf_index import IVFIndex, IVFParams
from vector_db.domain.services.pq_quantizer import (
    PQCodebook,
    PQIndex,
    PQParams,
    ProductQuantizer,
)

__all__ = [
    # Distance functions
    "euclidean_distance",
    "squared_euclidean_distance",
    "cosine_distance",
    "cosine_similarity",
    "inner_product",
    "batch_euclidean_distance",
    "batch_cosine_distance",
    "batch_inner_product",
    "get_distance_function",
    "get_batch_distance_function",
    "normalize_vectors",
    "compute_ground_truth",
    "compute_recall",
    # HNSW
    "HNSWIndex",
    "HNSWParams",
    # IVF
    "IVFIndex",
    "IVFParams",
    # PQ
    "ProductQuantizer",
    "PQCodebook",
    "PQIndex",
    "PQParams",
]
