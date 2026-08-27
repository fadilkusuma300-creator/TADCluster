"""Core TADCluster geometry and clustering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence
import math
import random

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

PenaltyName = Literal["exponential", "gaussian", "step"]
FusionName = Literal["rss", "additive"]


@dataclass(frozen=True)
class GeometryConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    umap_components: int = 50
    umap_neighbors: int = 15
    umap_metric: str = "cosine"
    random_seed: int = 42
    embedding_batch_size: int = 64


@dataclass
class SemanticGeometry:
    coordinates: np.ndarray
    distance: np.ndarray


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds used by NumPy, Python and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def encode_and_reduce(texts: Sequence[str], cfg: GeometryConfig) -> np.ndarray:
    """Encode documents with MiniLM and reduce embeddings with UMAP."""
    if len(texts) < 3:
        raise ValueError("At least three documents are required for UMAP geometry.")
    set_global_seed(cfg.random_seed)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required; install requirements.txt first."
        ) from exc
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("umap-learn is required; install requirements.txt first.") from exc

    encoder = SentenceTransformer(cfg.model_name)
    embeddings = encoder.encode(
        list(texts),
        batch_size=cfg.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    n_neighbors = min(cfg.umap_neighbors, max(2, len(texts) - 1))
    n_components = min(cfg.umap_components, max(2, len(texts) - 2))
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=cfg.umap_metric,
        random_state=cfg.random_seed,
        transform_seed=cfg.random_seed,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)
    return np.asarray(reduced, dtype=np.float32)


def normalized_semantic_distance(coordinates: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distance normalized by its global maximum."""
    x = np.asarray(coordinates, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("coordinates must be a 2-D array with at least two rows")
    dist = pairwise_distances(x, metric="euclidean", n_jobs=-1).astype(np.float32, copy=False)
    np.fill_diagonal(dist, 0.0)
    maximum = float(np.max(dist))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("semantic distances are degenerate; cannot normalize")
    dist /= maximum
    return dist


def build_semantic_geometry(texts: Sequence[str], cfg: GeometryConfig) -> SemanticGeometry:
    coordinates = encode_and_reduce(texts, cfg)
    return SemanticGeometry(coordinates=coordinates, distance=normalized_semantic_distance(coordinates))


def timestamps_to_days(timestamps: Sequence) -> np.ndarray:
    """Convert timestamps to floating-point days relative to the earliest record."""
    import pandas as pd

    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"{bad} timestamp(s) could not be parsed")
    values = ts.astype("int64").to_numpy(dtype=np.int64)
    values = values - values.min()
    return values.astype(np.float64) / 86_400_000_000_000.0


def estimate_half_life_global(
    times_days: Sequence[float], percentile: float = 20.0, max_pairs: int = 10_000, seed: int = 42
) -> float:
    """Estimate the corpus-level time scale from absolute pairwise time gaps."""
    t = np.asarray(times_days, dtype=np.float64)
    n = t.size
    if n < 2:
        raise ValueError("at least two timestamps are required")
    if not (0.0 < percentile < 100.0):
        raise ValueError("percentile must lie strictly between 0 and 100")

    if n > 5000:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, n, size=max_pairs * 2)
        j = rng.integers(0, n, size=max_pairs * 2)
        mask = i != j
        gaps = np.abs(t[i[mask]] - t[j[mask]])[:max_pairs]
    else:
        ii, jj = np.triu_indices(n, k=1)
        gaps = np.abs(t[ii] - t[jj])

    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size == 0:
        raise ValueError("all usable time gaps are zero")
    value = float(np.percentile(gaps, percentile))
    return max(value, np.finfo(float).eps)


def estimate_half_life_semantic_neighborhood(
    times_days: Sequence[float], semantic_distance: np.ndarray, min_pts: int = 5
) -> float:
    """Median time gap to each document's MinPts nearest semantic neighbors."""
    t = np.asarray(times_days, dtype=np.float64)
    d = np.asarray(semantic_distance)
    n = len(t)
    if d.shape != (n, n):
        raise ValueError("semantic distance shape does not match timestamps")
    k = min(max(1, min_pts), n - 1)
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    neighbor_idx = np.argpartition(work, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    gaps = np.abs(t[row_idx] - t[neighbor_idx])
    per_document = np.median(gaps, axis=1)
    value = float(np.median(per_document[np.isfinite(per_document)]))
    if value <= 0:
        positive = gaps[gaps > 0]
        if positive.size == 0:
            raise ValueError("semantic-neighborhood time gaps are all zero")
        value = float(np.median(positive))
    return value


def estimate_epsilon(
    semantic_distance: np.ndarray,
    reference_size: int = 500,
    kth_neighbor: int = 5,
    multiplier: float = 1.5,
    seed: int = 42,
) -> float:
    """Set epsilon from the median kth-neighbor distance on a sampled subset."""
    d = np.asarray(semantic_distance)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("semantic_distance must be square")
    n = d.shape[0]
    if n <= kth_neighbor:
        raise ValueError("not enough documents for the requested nearest-neighbor rule")
    rng = np.random.default_rng(seed)
    size = min(reference_size, n)
    idx = np.sort(rng.choice(n, size=size, replace=False))
    sub = d[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, np.inf)
    k = min(kth_neighbor, size - 1)
    kth = np.partition(sub, kth=k - 1, axis=1)[:, k - 1]
    eps = float(multiplier * np.median(kth[np.isfinite(kth)]))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon rule produced an invalid radius")
    return eps


def temporal_penalty(gap_days: np.ndarray, half_life: float, kind: PenaltyName) -> np.ndarray:
    if half_life <= 0 or not np.isfinite(half_life):
        raise ValueError("half_life must be positive and finite")
    x = np.asarray(gap_days, dtype=np.float32)
    if kind == "exponential":
        return 1.0 - np.exp((-math.log(2.0) / half_life) * x)
    if kind == "gaussian":
        return 1.0 - np.exp(-math.log(2.0) * np.square(x / half_life))
    if kind == "step":
        return (x > half_life).astype(np.float32)
    raise ValueError(f"unsupported penalty: {kind}")


def combined_distance(
    semantic_distance: np.ndarray,
    times_days: Sequence[float],
    alpha: float,
    half_life: float,
    penalty: PenaltyName = "exponential",
    fusion: FusionName = "rss",
    block_size: int = 1024,
) -> np.ndarray:
    """Construct the semantic-temporal precomputed distance matrix.

    The implementation avoids materializing a second full time-gap matrix by updating
    the output in row blocks. The returned matrix is float32 and symmetric up to
    floating-point roundoff.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    semantic = np.asarray(semantic_distance, dtype=np.float32)
    n = semantic.shape[0]
    if semantic.shape != (n, n):
        raise ValueError("semantic_distance must be square")
    t = np.asarray(times_days, dtype=np.float32)
    if t.shape != (n,):
        raise ValueError("timestamps do not match the distance matrix")

    if alpha == 0:
        return semantic.copy()

    weight = float(math.sqrt(alpha))
    if fusion == "rss":
        out = np.square(semantic, dtype=np.float32)
    elif fusion == "additive":
        out = semantic.copy()
    else:
        raise ValueError(f"unsupported fusion: {fusion}")

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        gaps = np.abs(t[start:stop, None] - t[None, :])
        temp = temporal_penalty(gaps, half_life=half_life, kind=penalty).astype(np.float32, copy=False)
        if fusion == "rss":
            out[start:stop] += alpha * np.square(temp, dtype=np.float32)
        else:
            out[start:stop] += weight * temp

    if fusion == "rss":
        np.sqrt(out, out=out)
    # Numerical symmetry is important to precomputed-distance clusterers.
    out = ((out + out.T) * 0.5).astype(np.float32, copy=False)
    np.fill_diagonal(out, 0.0)
    return out


def fit_dbscan(distance: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    if eps <= 0:
        raise ValueError("eps must be positive")
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed", n_jobs=-1)
    return model.fit_predict(distance).astype(np.int32)


def fit_hdbscan(
    distance: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = 5,
    cluster_selection_method: str = "eom",
) -> np.ndarray:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is required; install requirements.txt first.") from exc

    # The contrib implementation is most reliable with double-precision precomputed distances.
    d = np.asarray(distance, dtype=np.float64)
    model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_method=cluster_selection_method,
        prediction_data=False,
        core_dist_n_jobs=-1,
    )
    return model.fit_predict(d).astype(np.int32)


def cluster_count(labels: Sequence[int]) -> int:
    labels = np.asarray(labels)
    return int(len(set(labels.tolist()) - {-1}))


def coverage(labels: Sequence[int]) -> float:
    labels = np.asarray(labels)
    return float(np.mean(labels != -1)) if labels.size else 0.0
