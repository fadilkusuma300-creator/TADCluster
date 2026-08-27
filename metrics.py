"""Topic and clustering metrics for TADCluster."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


TOKEN_PATTERN = r"(?u)\b[\w][\w+#.\-]*\b"


@dataclass(frozen=True)
class TopicMetrics:
    clusters: int
    coverage: float
    npmi: float | None
    tdisp_days: float | None


def _ctfidf_top_terms(
    texts: Sequence[str], labels: Sequence[int], top_n: int = 8, stop_words: str | None = None
) -> Dict[int, List[str]]:
    labels_arr = np.asarray(labels)
    cluster_ids = sorted(set(labels_arr.tolist()) - {-1})
    if not cluster_ids:
        return {}

    # c-TF-IDF is fitted over every non-noise class. The >=3-document rule is
    # applied later to NPMI aggregation, not to the class-frequency statistics.
    joined: List[str] = []
    for cluster_id in cluster_ids:
        idx = np.flatnonzero(labels_arr == cluster_id)
        joined.append(" ".join(str(texts[i]) for i in idx))

    vectorizer = CountVectorizer(
        lowercase=False,
        stop_words=stop_words,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
        min_df=1,
    )
    counts = vectorizer.fit_transform(joined).tocsr().astype(np.float64)
    # Same class-based TF-IDF definition used by BERTopic's ClassTfidfTransformer.
    term_freq_across_classes = np.asarray(counts.sum(axis=0)).ravel()
    avg_nr_words = int(np.asarray(counts.sum(axis=1)).mean())
    idf = np.log((avg_nr_words / np.maximum(term_freq_across_classes, 1.0)) + 1.0)
    tf = normalize(counts, axis=1, norm="l1", copy=True)
    ctfidf = tf @ sparse.diags(idf)
    vocab = np.asarray(vectorizer.get_feature_names_out())

    result: Dict[int, List[str]] = {}
    for row, cluster_id in enumerate(cluster_ids):
        values = ctfidf.getrow(row).toarray().ravel()
        nonzero = np.flatnonzero(values > 0)
        if nonzero.size == 0:
            result[cluster_id] = []
            continue
        order = nonzero[np.argsort(values[nonzero], kind="stable")[::-1]]
        result[cluster_id] = vocab[order[:top_n]].tolist()
    return result


def _binary_term_matrix(texts: Sequence[str], vocabulary: Sequence[str], stop_words: str | None = None):
    # A fixed vocabulary keeps co-occurrence probabilities aligned with the selected topic terms.
    vectorizer = CountVectorizer(
        lowercase=False,
        stop_words=stop_words,
        token_pattern=TOKEN_PATTERN,
        vocabulary=list(vocabulary),
        binary=True,
    )
    return vectorizer.transform(texts).astype(np.int8).tocsr()


def mean_topic_npmi(
    texts: Sequence[str], labels: Sequence[int], top_n: int = 8, stop_words: str | None = None
) -> float | None:
    """Unweighted mean cluster NPMI from top c-TF-IDF terms.

    Probabilities are estimated from binary document-level occurrence in the supplied
    dataset/subset. Clusters with fewer than three documents are excluded. A one-cluster
    solution is treated as degenerate and returns None.
    """
    labels_arr = np.asarray(labels)
    cluster_ids = sorted(set(labels_arr.tolist()) - {-1})
    if len(cluster_ids) <= 1:
        return None

    topics = _ctfidf_top_terms(texts, labels_arr, top_n=top_n, stop_words=stop_words)
    vocab = sorted({term for terms in topics.values() for term in terms})
    if len(vocab) < 2:
        return None

    matrix = _binary_term_matrix(texts, vocab, stop_words=stop_words)
    n_docs = matrix.shape[0]
    if n_docs == 0:
        return None
    term_index = {term: i for i, term in enumerate(vocab)}
    doc_freq = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    p_term = doc_freq / n_docs

    cluster_scores: List[float] = []
    eps = 1e-12
    for cluster_id, terms in topics.items():
        if np.count_nonzero(labels_arr == cluster_id) < 3:
            continue
        valid_terms = [t for t in terms if p_term[term_index[t]] > 0]
        if len(valid_terms) < 2:
            continue
        pair_scores = []
        for a, b in combinations(valid_terms, 2):
            ia, ib = term_index[a], term_index[b]
            co = float(matrix[:, ia].multiply(matrix[:, ib]).sum())
            if co <= 0:
                pair_scores.append(-1.0)
                continue
            p_ab = co / n_docs
            pmi = np.log((p_ab + eps) / (p_term[ia] * p_term[ib] + eps))
            denom = -np.log(p_ab + eps)
            if denom <= eps:
                continue
            pair_scores.append(float(np.clip(pmi / denom, -1.0, 1.0)))
        if pair_scores:
            cluster_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(cluster_scores)) if cluster_scores else None


def temporal_dispersion_days(times_days: Sequence[float], labels: Sequence[int]) -> float | None:
    t = np.asarray(times_days, dtype=np.float64)
    labels_arr = np.asarray(labels)
    values: List[float] = []
    for cluster_id in sorted(set(labels_arr.tolist()) - {-1}):
        x = t[labels_arr == cluster_id]
        if x.size >= 2:
            values.append(float(np.std(x, ddof=1)))
    return float(np.mean(values)) if values else None


def summarize_topics(texts: Sequence[str], times_days: Sequence[float], labels: Sequence[int], top_n: int = 8) -> TopicMetrics:
    labels_arr = np.asarray(labels)
    clusters = len(set(labels_arr.tolist()) - {-1})
    cov = float(np.mean(labels_arr != -1)) if labels_arr.size else 0.0
    return TopicMetrics(
        clusters=int(clusters),
        coverage=cov,
        npmi=mean_topic_npmi(texts, labels_arr, top_n=top_n),
        tdisp_days=temporal_dispersion_days(times_days, labels_arr),
    )


def _singletonize_noise(labels: Sequence[int]) -> np.ndarray:
    """Make each noise item a singleton for partition-based B-Cubed scoring."""
    labels_arr = np.asarray(labels, dtype=object).copy()
    noise = np.flatnonzero(np.asarray(labels) == -1)
    for i in noise:
        labels_arr[i] = f"noise_{i}"
    return labels_arr


def bcubed_precision_recall_f1(y_true: Sequence, y_pred: Sequence[int]) -> Tuple[float, float, float]:
    """Compute B-Cubed precision, recall and F1 for a hard partition."""
    truth = np.asarray(y_true, dtype=object)
    pred = _singletonize_noise(y_pred)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    n = len(truth)
    if n == 0:
        raise ValueError("empty label arrays")

    true_members: Dict[object, set[int]] = {}
    pred_members: Dict[object, set[int]] = {}
    for i, (t, p) in enumerate(zip(truth, pred)):
        true_members.setdefault(t, set()).add(i)
        pred_members.setdefault(p, set()).add(i)

    precisions = np.empty(n, dtype=np.float64)
    recalls = np.empty(n, dtype=np.float64)
    for i, (t, p) in enumerate(zip(truth, pred)):
        intersection = len(true_members[t] & pred_members[p])
        precisions[i] = intersection / len(pred_members[p])
        recalls[i] = intersection / len(true_members[t])
    precision = float(np.mean(precisions))
    recall = float(np.mean(recalls))
    f1 = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
    return precision, recall, f1


def bootstrap_paired_ci(differences: Sequence[float], seed: int = 42, n_boot: int = 10_000) -> Tuple[float, float]:
    x = np.asarray(differences, dtype=np.float64)
    if x.size < 2:
        raise ValueError("at least two paired differences are required")
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)
