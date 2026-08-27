"""Run TADCluster experiments and analyses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from metrics import (
    TopicMetrics,
    bcubed_precision_recall_f1,
    bootstrap_paired_ci,
    summarize_topics,
)
from tadcluster import (
    GeometryConfig,
    SemanticGeometry,
    build_semantic_geometry,
    cluster_count,
    combined_distance,
    coverage,
    estimate_epsilon,
    estimate_half_life_global,
    estimate_half_life_semantic_neighborhood,
    fit_dbscan,
    fit_hdbscan,
    set_global_seed,
    timestamps_to_days,
)


DATASETS = ("D2", "D1", "D3")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    return cfg


def load_dataset(path: Path, require_labels: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"id", "text", "timestamp"}
    if require_labels:
        required.add("event_label")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    id_columns = ["source", "id"] if "source" in df.columns else ["id"]
    if df.duplicated(subset=id_columns).any():
        raise ValueError(f"{path} contains duplicate document identifiers within a source")
    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise ValueError(f"{path} contains empty document text")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{path} contains invalid timestamps")
    df = df.copy()
    df["timestamp"] = ts
    return df.sort_values(["timestamp", "id"]).reset_index(drop=True)


def temporal_stratified_indices(
    timestamps: Sequence, fraction: float, seed: int, bins: int = 10
) -> np.ndarray:
    """Sample a fraction from each chronological bin without using topic/event labels."""
    if not (0 < fraction < 1):
        raise ValueError("fraction must be between 0 and 1")
    n = len(timestamps)
    if n < 5:
        raise ValueError("too few documents for a stratified split")
    order = np.argsort(pd.to_datetime(timestamps, utc=True).astype("int64"), kind="stable")
    groups = np.array_split(order, min(bins, n))
    rng = np.random.default_rng(seed)
    chosen: List[int] = []
    for group in groups:
        if group.size == 0:
            continue
        count = int(round(group.size * fraction))
        count = min(group.size - 1 if group.size > 1 else 1, max(1, count))
        chosen.extend(rng.choice(group, size=count, replace=False).tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    # Correct small rounding drift while preserving chronological coverage.
    target = int(round(n * fraction))
    if chosen.size > target:
        chosen = np.sort(rng.choice(chosen, size=target, replace=False))
    elif chosen.size < target:
        remaining = np.setdiff1d(np.arange(n), chosen, assume_unique=False)
        add = rng.choice(remaining, size=target - chosen.size, replace=False)
        chosen = np.sort(np.concatenate([chosen, add]))
    return chosen


def calibration_evaluation_split(df: pd.DataFrame, fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cal_idx = temporal_stratified_indices(df["timestamp"], fraction=fraction, seed=seed)
    mask = np.ones(len(df), dtype=bool)
    mask[cal_idx] = False
    eval_idx = np.flatnonzero(mask)
    if set(cal_idx.tolist()) & set(eval_idx.tolist()):
        raise RuntimeError("calibration/evaluation overlap detected")
    return df.iloc[cal_idx].reset_index(drop=True), df.iloc[eval_idx].reset_index(drop=True)


def geometry_cfg(cfg: Mapping) -> GeometryConfig:
    g = cfg["geometry"]
    return GeometryConfig(
        model_name=str(g["model_name"]),
        umap_components=int(g["umap_components"]),
        umap_neighbors=int(g["umap_neighbors"]),
        umap_metric=str(g.get("umap_metric", "cosine")),
        random_seed=int(cfg["seed"]),
        embedding_batch_size=int(g.get("embedding_batch_size", 64)),
    )


def _texts(df: pd.DataFrame) -> List[str]:
    return df["text"].astype(str).tolist()



def select_tad_alpha(df: pd.DataFrame, cfg: Mapping, half_life_strategy: str = "global") -> Tuple[float, dict]:
    geom = build_semantic_geometry(_texts(df), geometry_cfg(cfg))
    days = timestamps_to_days(df["timestamp"])
    eps_cfg = cfg["dbscan"]
    eps = estimate_epsilon(
        geom.distance,
        reference_size=int(eps_cfg["reference_size"]),
        kth_neighbor=int(eps_cfg["kth_neighbor"]),
        multiplier=float(eps_cfg["epsilon_multiplier"]),
        seed=int(cfg["seed"]),
    )
    half_life = choose_half_life(half_life_strategy, days, geom.distance, cfg)
    rows = []
    for alpha in [float(x) for x in cfg["selection"]["alpha_grid"]]:
        dist = combined_distance(geom.distance, days, alpha=alpha, half_life=half_life)
        labels = fit_dbscan(dist, eps=eps, min_samples=int(cfg["dbscan"]["min_samples"]))
        metrics = summarize_topics(_texts(df), days, labels, top_n=int(cfg["metrics"]["top_terms"]))
        rows.append({"alpha": alpha, **asdict(metrics)})
    valid = [r for r in rows if r["npmi"] is not None and np.isfinite(r["npmi"])]
    if not valid:
        raise RuntimeError("No non-degenerate TADCluster candidate was found on calibration data")
    # NPMI is the sole selection criterion; lower alpha is a deterministic tie-break only.
    best = sorted(valid, key=lambda r: (-float(r["npmi"]), float(r["alpha"])))[0]
    return float(best["alpha"]), {"epsilon": eps, "half_life": half_life, "trajectory": rows}


def choose_half_life(strategy: str, days: np.ndarray, semantic_distance: np.ndarray, cfg: Mapping) -> float:
    if strategy == "global":
        return estimate_half_life_global(
            days,
            percentile=float(cfg["half_life"]["percentile"]),
            max_pairs=int(cfg["half_life"]["sample_pairs"]),
            seed=int(cfg["seed"]),
        )
    if strategy == "fixed30":
        return float(cfg["half_life"]["fixed_days"])
    if strategy == "semantic_neighborhood":
        return estimate_half_life_semantic_neighborhood(
            days, semantic_distance, min_pts=int(cfg["dbscan"]["min_samples"])
        )
    raise ValueError(f"Unknown half-life strategy: {strategy}")


def evaluate_tad(df: pd.DataFrame, alpha: float, cfg: Mapping, half_life_strategy: str = "global") -> Tuple[TopicMetrics, np.ndarray, dict]:
    geom = build_semantic_geometry(_texts(df), geometry_cfg(cfg))
    days = timestamps_to_days(df["timestamp"])
    eps_cfg = cfg["dbscan"]
    eps = estimate_epsilon(
        geom.distance,
        reference_size=int(eps_cfg["reference_size"]),
        kth_neighbor=int(eps_cfg["kth_neighbor"]),
        multiplier=float(eps_cfg["epsilon_multiplier"]),
        seed=int(cfg["seed"]),
    )
    half_life = choose_half_life(half_life_strategy, days, geom.distance, cfg)
    dist = combined_distance(geom.distance, days, alpha=alpha, half_life=half_life)
    labels = fit_dbscan(dist, eps=eps, min_samples=int(cfg["dbscan"]["min_samples"]))
    metrics = summarize_topics(_texts(df), days, labels, top_n=int(cfg["metrics"]["top_terms"]))
    return metrics, labels, {"geometry": geom, "days": days, "epsilon": eps, "half_life": half_life}



def run_bertopic(df: pd.DataFrame, cfg: Mapping, min_cluster_size: int, min_samples: int) -> Tuple[TopicMetrics, np.ndarray]:
    try:
        from bertopic import BERTopic
        import hdbscan
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("BERTopic, hdbscan and umap-learn are required for this baseline") from exc

    seed = int(cfg["seed"])
    set_global_seed(seed)
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_method=str(cfg["hdbscan"].get("cluster_selection_method", "eom")),
        metric="euclidean",
        prediction_data=False,
    )
    # BERTopic's standard low-dimensional clustering geometry with a fixed random seed.
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    model = BERTopic(
        embedding_model=str(cfg["geometry"]["model_name"]),
        umap_model=umap_model,
        hdbscan_model=hdb,
        top_n_words=int(cfg["metrics"]["top_terms"]),
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(_texts(df))
    labels = np.asarray(topics, dtype=np.int32)
    days = timestamps_to_days(df["timestamp"])
    metrics = summarize_topics(_texts(df), days, labels, top_n=int(cfg["metrics"]["top_terms"]))
    return metrics, labels


def select_bertopic_params(df: pd.DataFrame, cfg: Mapping) -> Tuple[int, int, List[dict]]:
    rows = []
    for mcs in cfg["selection"]["bertopic_min_cluster_size"]:
        for ms in cfg["selection"]["bertopic_min_samples"]:
            metrics, _ = run_bertopic(df, cfg, int(mcs), int(ms))
            rows.append({"min_cluster_size": int(mcs), "min_samples": int(ms), **asdict(metrics)})
    valid = [r for r in rows if r["npmi"] is not None]
    if not valid:
        raise RuntimeError("All BERTopic candidates were degenerate")
    best = sorted(valid, key=lambda r: (-float(r["npmi"]), r["min_cluster_size"], r["min_samples"]))[0]
    return int(best["min_cluster_size"]), int(best["min_samples"]), rows


def run_fastopic(df: pd.DataFrame, cfg: Mapping, num_topics: int) -> Tuple[TopicMetrics, np.ndarray]:
    try:
        from fastopic import FASTopic
        from topmost import Preprocess
    except ImportError as exc:
        raise ImportError("fastopic and topmost are required for the FASTopic baseline") from exc

    seed = int(cfg["seed"])
    set_global_seed(seed)
    preprocess = Preprocess()
    model = FASTopic(
        int(num_topics),
        preprocess=preprocess,
        num_top_words=int(cfg["metrics"]["top_terms"]),
        doc_embed_model=str(cfg["geometry"]["model_name"]),
        device=str(cfg["fastopic"].get("device", "cpu")),
        low_memory=bool(cfg["fastopic"].get("low_memory", False)),
        low_memory_batch_size=cfg["fastopic"].get("low_memory_batch_size"),
        verbose=False,
    )
    _, theta = model.fit_transform(
        _texts(df),
        epochs=int(cfg["fastopic"].get("epochs", 200)),
        learning_rate=float(cfg["fastopic"].get("learning_rate", 0.002)),
    )
    labels = np.asarray(theta).argmax(axis=1).astype(np.int32)
    days = timestamps_to_days(df["timestamp"])
    metrics = summarize_topics(_texts(df), days, labels, top_n=int(cfg["metrics"]["top_terms"]))
    return metrics, labels


def select_fastopic_topics(df: pd.DataFrame, cfg: Mapping) -> Tuple[int, List[dict]]:
    rows = []
    for k in cfg["selection"]["fastopic_topics"]:
        metrics, _ = run_fastopic(df, cfg, int(k))
        rows.append({"num_topics": int(k), **asdict(metrics)})
    valid = [r for r in rows if r["npmi"] is not None]
    if not valid:
        raise RuntimeError("All FASTopic candidates were degenerate")
    best = sorted(valid, key=lambda r: (-float(r["npmi"]), r["num_topics"]))[0]
    return int(best["num_topics"]), rows


def main_comparison(data_dir: Path, output_dir: Path, cfg: Mapping) -> dict:
    all_results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        df = load_dataset(data_dir / f"{dataset}.csv")
        cal, eva = calibration_evaluation_split(
            df, fraction=float(cfg["selection"]["calibration_fraction"]), seed=int(cfg["seed"])
        )
        alpha, tad_cal = select_tad_alpha(cal, cfg)
        tad_metrics, tad_labels, tad_state = evaluate_tad(eva, alpha, cfg)

        # DBSCAN, HDBSCAN, and TADCluster share the same SBERT/UMAP geometry
        # on the evaluation subset. Only the clustering distance/backend changes.
        db_labels = fit_dbscan(
            tad_state["geometry"].distance,
            eps=tad_state["epsilon"],
            min_samples=int(cfg["dbscan"]["min_samples"]),
        )
        db_metrics = summarize_topics(
            _texts(eva), tad_state["days"], db_labels, top_n=int(cfg["metrics"]["top_terms"])
        )
        hdb_labels = fit_hdbscan(
            tad_state["geometry"].distance,
            min_cluster_size=5,
            min_samples=5,
            cluster_selection_method=str(cfg["hdbscan"].get("cluster_selection_method", "eom")),
        )
        hdb_metrics = summarize_topics(
            _texts(eva), tad_state["days"], hdb_labels, top_n=int(cfg["metrics"]["top_terms"])
        )

        mcs, ms, bert_cal = select_bertopic_params(cal, cfg)
        bert_metrics, _ = run_bertopic(eva, cfg, mcs, ms)
        k, fast_cal = select_fastopic_topics(cal, cfg)
        fast_metrics, _ = run_fastopic(eva, cfg, k)

        trajectory = []
        for a in [float(x) for x in cfg["selection"]["alpha_grid"]]:
            dist = combined_distance(
                tad_state["geometry"].distance,
                tad_state["days"],
                alpha=a,
                half_life=tad_state["half_life"],
            )
            labels = fit_dbscan(dist, tad_state["epsilon"], int(cfg["dbscan"]["min_samples"]))
            metrics = summarize_topics(_texts(eva), tad_state["days"], labels, int(cfg["metrics"]["top_terms"]))
            trajectory.append({"alpha": a, **asdict(metrics)})

        result = {
            "selected_alpha": alpha,
            "selected_bertopic": {"min_cluster_size": mcs, "min_samples": ms},
            "selected_fastopic_topics": k,
            "evaluation_rows": len(eva),
            "methods": {
                "DBSCAN": asdict(db_metrics),
                "HDBSCAN": asdict(hdb_metrics),
                "BERTopic-HDBSCAN": asdict(bert_metrics),
                "FASTopic": asdict(fast_metrics),
                "TADCluster": asdict(tad_metrics),
            },
            "trajectory": trajectory,
            "calibration": {
                "tadcluster": tad_cal,
                "bertopic": bert_cal,
                "fastopic": fast_cal,
            },
        }
        all_results[dataset] = result
        _write_json(output_dir / f"main_{dataset}.json", result)
    _write_json(output_dir / "main_summary.json", all_results)
    return all_results


def design_ablation(data_dir: Path, output_dir: Path, cfg: Mapping, main_results: Mapping | None = None) -> dict:
    if main_results is None:
        main_results = _read_json(output_dir / "main_summary.json")
    results = {}
    for dataset in DATASETS:
        df = load_dataset(data_dir / f"{dataset}.csv")
        _, eva = calibration_evaluation_split(df, float(cfg["selection"]["calibration_fraction"]), int(cfg["seed"]))
        geom = build_semantic_geometry(_texts(eva), geometry_cfg(cfg))
        days = timestamps_to_days(eva["timestamp"])
        half = choose_half_life("global", days, geom.distance, cfg)
        alpha = float(main_results[dataset]["selected_alpha"])
        variants = [
            ("semantic_only", 0.0, "exponential", "rss"),
            ("step_rss", alpha, "step", "rss"),
            ("gaussian_rss", alpha, "gaussian", "rss"),
            ("exponential_additive", alpha, "exponential", "additive"),
            ("exponential_rss", alpha, "exponential", "rss"),
        ]
        rows = {}
        for name, a, penalty, fusion in variants:
            dist = combined_distance(geom.distance, days, a, half, penalty=penalty, fusion=fusion)
            labels = fit_hdbscan(dist, 5, 5, str(cfg["hdbscan"].get("cluster_selection_method", "eom")))
            rows[name] = asdict(summarize_topics(_texts(eva), days, labels, int(cfg["metrics"]["top_terms"])))
        results[dataset] = rows
    _write_json(output_dir / "design_ablation.json", results)
    return results


def matched_granularity(data_dir: Path, output_dir: Path, cfg: Mapping, main_results: Mapping | None = None) -> dict:
    if main_results is None:
        main_results = _read_json(output_dir / "main_summary.json")
    results = {}
    for dataset in DATASETS:
        df = load_dataset(data_dir / f"{dataset}.csv")
        _, eva = calibration_evaluation_split(df, float(cfg["selection"]["calibration_fraction"]), int(cfg["seed"]))
        alpha = float(main_results[dataset]["selected_alpha"])
        tad_metrics, tad_labels, state = evaluate_tad(eva, alpha, cfg)
        target_k = tad_metrics.clusters
        target_cov = tad_metrics.coverage

        candidates = []
        for mcs in cfg["matched"]["min_cluster_size"]:
            for ms in cfg["matched"]["min_samples"]:
                labels = fit_hdbscan(
                    state["geometry"].distance,
                    int(mcs),
                    int(ms),
                    str(cfg["hdbscan"].get("cluster_selection_method", "eom")),
                )
                k = cluster_count(labels)
                cov = coverage(labels)
                # Only cluster count and coverage are used. Topic quality is calculated after selection.
                match_score = abs(k - target_k) / max(1, target_k) + abs(cov - target_cov) / max(target_cov, 1e-8)
                candidates.append(
                    {
                        "min_cluster_size": int(mcs),
                        "min_samples": int(ms),
                        "clusters": k,
                        "coverage": cov,
                        "match_score": match_score,
                        "labels": labels,
                    }
                )
        best = sorted(candidates, key=lambda r: (r["match_score"], r["min_cluster_size"], r["min_samples"]))[0]
        hdb_metrics = summarize_topics(
            _texts(eva), state["days"], best["labels"], int(cfg["metrics"]["top_terms"])
        )
        results[dataset] = {
            "selected_hdbscan": {
                "min_cluster_size": best["min_cluster_size"],
                "min_samples": best["min_samples"],
                "match_score": best["match_score"],
            },
            "HDBSCAN_matched": asdict(hdb_metrics),
            "TADCluster": asdict(tad_metrics),
        }
    _write_json(output_dir / "matched_granularity.json", results)
    return results


def half_life_analysis(data_dir: Path, output_dir: Path, cfg: Mapping) -> dict:
    results = {}
    for dataset in DATASETS:
        df = load_dataset(data_dir / f"{dataset}.csv")
        cal, eva = calibration_evaluation_split(df, float(cfg["selection"]["calibration_fraction"]), int(cfg["seed"]))
        rows = {}
        for strategy in ("fixed30", "global", "semantic_neighborhood"):
            alpha, _ = select_tad_alpha(cal, cfg, half_life_strategy=strategy)
            metrics, _, state = evaluate_tad(eva, alpha, cfg, half_life_strategy=strategy)
            rows[strategy] = {"selected_alpha": alpha, "half_life_days": state["half_life"], **asdict(metrics)}
        results[dataset] = rows
    _write_json(output_dir / "half_life_analysis.json", results)
    return results


def epsilon_sensitivity(data_dir: Path, output_dir: Path, cfg: Mapping) -> dict:
    df = load_dataset(data_dir / "D1.csv")
    _, eva = calibration_evaluation_split(df, float(cfg["selection"]["calibration_fraction"]), int(cfg["seed"]))
    geom = build_semantic_geometry(_texts(eva), geometry_cfg(cfg))
    days = timestamps_to_days(eva["timestamp"])
    half = choose_half_life("global", days, geom.distance, cfg)
    alpha = float(cfg["sensitivity"]["d1_alpha"])
    tad_dist = combined_distance(geom.distance, days, alpha, half)
    rows = []
    for eps in [float(x) for x in cfg["sensitivity"]["epsilon_grid"]]:
        db_labels = fit_dbscan(geom.distance, eps, int(cfg["dbscan"]["min_samples"]))
        tad_labels = fit_dbscan(tad_dist, eps, int(cfg["dbscan"]["min_samples"]))
        rows.append(
            {
                "epsilon": eps,
                "DBSCAN": asdict(summarize_topics(_texts(eva), days, db_labels, int(cfg["metrics"]["top_terms"]))),
                "TADCluster": asdict(summarize_topics(_texts(eva), days, tad_labels, int(cfg["metrics"]["top_terms"]))),
            }
        )
    result = {"D1": rows}
    _write_json(output_dir / "epsilon_sensitivity.json", result)
    return result


def repeated_sampling(data_dir: Path, output_dir: Path, cfg: Mapping) -> dict:
    """Thirty paired, temporally stratified comparisons used for robustness analysis."""
    per_dataset = {}
    repeats = int(cfg["statistics"]["repeats"])
    for dataset in DATASETS:
        full = load_dataset(data_dir / f"{dataset}.csv")
        comparator = str(cfg["statistics"]["comparators"][dataset])
        tad_values: List[float] = []
        comparator_values: List[float] = []
        runs = []
        for rep in range(repeats):
            resample_idx = temporal_stratified_indices(
                full["timestamp"], fraction=float(cfg["statistics"]["resample_fraction"]), seed=int(cfg["seed"]) + 1000 + rep
            )
            sample = full.iloc[resample_idx].reset_index(drop=True)
            cal, eva = calibration_evaluation_split(
                sample, float(cfg["selection"]["calibration_fraction"]), int(cfg["seed"]) + 2000 + rep
            )
            alpha, _ = select_tad_alpha(cal, cfg)
            tad_metrics, _, tad_state = evaluate_tad(eva, alpha, cfg)
            if tad_metrics.npmi is None:
                raise RuntimeError(f"{dataset} repeat {rep}: degenerate TADCluster result")

            if comparator == "FASTopic":
                k, _ = select_fastopic_topics(cal, cfg)
                cmp_metrics, _ = run_fastopic(eva, cfg, k)
            elif comparator == "HDBSCAN":
                cmp_labels = fit_hdbscan(
                    tad_state["geometry"].distance,
                    min_cluster_size=5,
                    min_samples=5,
                    cluster_selection_method=str(cfg["hdbscan"].get("cluster_selection_method", "eom")),
                )
                cmp_metrics = summarize_topics(
                    _texts(eva), tad_state["days"], cmp_labels, top_n=int(cfg["metrics"]["top_terms"])
                )
                k = None
            else:
                raise ValueError(f"Unsupported comparator: {comparator}")
            if cmp_metrics.npmi is None:
                raise RuntimeError(f"{dataset} repeat {rep}: degenerate comparator result")

            tad_values.append(float(tad_metrics.npmi))
            comparator_values.append(float(cmp_metrics.npmi))
            runs.append(
                {
                    "repeat": rep + 1,
                    "selected_alpha": alpha,
                    "selected_comparator_topics": k,
                    "tad_npmi": tad_metrics.npmi,
                    "comparator_npmi": cmp_metrics.npmi,
                }
            )

        diff = np.asarray(tad_values) - np.asarray(comparator_values)
        lo, hi = bootstrap_paired_ci(diff, seed=int(cfg["seed"]), n_boot=int(cfg["statistics"]["bootstrap_samples"]))
        per_dataset[dataset] = {
            "comparator": comparator,
            "tad_mean": float(np.mean(tad_values)),
            "tad_std": float(np.std(tad_values, ddof=1)),
            "comparator_mean": float(np.mean(comparator_values)),
            "comparator_std": float(np.std(comparator_values, ddof=1)),
            "mean_difference": float(np.mean(diff)),
            "bootstrap_95_interval": [lo, hi],
            "runs": runs,
        }
    _write_json(output_dir / "repeated_sampling.json", per_dataset)
    return per_dataset


def _select_news_hdbscan(train: pd.DataFrame, geom: SemanticGeometry, days: np.ndarray, cfg: Mapping) -> Tuple[int, int, List[dict]]:
    rows = []
    for mcs in cfg["selection"]["bertopic_min_cluster_size"]:
        for ms in cfg["selection"]["bertopic_min_samples"]:
            labels = fit_hdbscan(geom.distance, int(mcs), int(ms), str(cfg["hdbscan"].get("cluster_selection_method", "eom")))
            metrics = summarize_topics(_texts(train), days, labels, int(cfg["metrics"]["top_terms"]))
            rows.append({"min_cluster_size": int(mcs), "min_samples": int(ms), **asdict(metrics)})
    valid = [r for r in rows if r["npmi"] is not None]
    if not valid:
        raise RuntimeError("News2013 calibration produced no non-degenerate HDBSCAN configuration")
    best = sorted(valid, key=lambda r: (-float(r["npmi"]), r["min_cluster_size"], r["min_samples"]))[0]
    return int(best["min_cluster_size"]), int(best["min_samples"]), rows


def news2013_validation(data_dir: Path, output_dir: Path, cfg: Mapping) -> dict:
    train = load_dataset(data_dir / "news2013_train.csv", require_labels=True)
    test = load_dataset(data_dir / "news2013_test.csv", require_labels=True)

    # Parameter selection uses document text and timestamps only.
    train_geom = build_semantic_geometry(_texts(train), geometry_cfg(cfg))
    train_days = timestamps_to_days(train["timestamp"])
    mcs, ms, hdb_grid = _select_news_hdbscan(train, train_geom, train_days, cfg)
    train_half = choose_half_life("global", train_days, train_geom.distance, cfg)

    alpha_rows = []
    for alpha in [float(x) for x in cfg["selection"]["alpha_grid"]]:
        dist = combined_distance(train_geom.distance, train_days, alpha, train_half)
        labels = fit_hdbscan(dist, mcs, ms, str(cfg["hdbscan"].get("cluster_selection_method", "eom")))
        metrics = summarize_topics(_texts(train), train_days, labels, int(cfg["metrics"]["top_terms"]))
        alpha_rows.append({"alpha": alpha, **asdict(metrics)})
    valid = [r for r in alpha_rows if r["npmi"] is not None]
    if not valid:
        raise RuntimeError("News2013 alpha calibration produced only degenerate partitions")
    selected_alpha = float(sorted(valid, key=lambda r: (-float(r["npmi"]), float(r["alpha"])))[0]["alpha"])

    # Event labels are read only for B-Cubed scoring after cluster assignments are fixed.
    test_geom = build_semantic_geometry(_texts(test), geometry_cfg(cfg))
    test_days = timestamps_to_days(test["timestamp"])
    test_half = choose_half_life("global", test_days, test_geom.distance, cfg)
    semantic_labels = fit_hdbscan(
        test_geom.distance, mcs, ms, str(cfg["hdbscan"].get("cluster_selection_method", "eom"))
    )
    temporal_dist = combined_distance(test_geom.distance, test_days, selected_alpha, test_half)
    tad_labels = fit_hdbscan(
        temporal_dist, mcs, ms, str(cfg["hdbscan"].get("cluster_selection_method", "eom"))
    )

    true_labels = test["event_label"].astype(str).to_numpy()
    p0, r0, f0 = bcubed_precision_recall_f1(true_labels, semantic_labels)
    p1, r1, f1 = bcubed_precision_recall_f1(true_labels, tad_labels)
    result = {
        "selection": {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "alpha": selected_alpha,
            "hdbscan_grid": hdb_grid,
            "alpha_grid": alpha_rows,
        },
        "test": {
            "SBERT_HDBSCAN_alpha0": {
                "precision": p0,
                "recall": r0,
                "f1": f0,
                "clusters": cluster_count(semantic_labels),
            },
            "TADCluster_HDBSCAN": {
                "precision": p1,
                "recall": r1,
                "f1": f1,
                "clusters": cluster_count(tad_labels),
            },
        },
    }
    _write_json(output_dir / "news2013.json", result)
    return result


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, allow_nan=False)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--only",
        choices=("main", "ablation", "matched", "half-life", "epsilon", "repeated", "news2013", "all"),
        default="all",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only == "main":
        main_comparison(args.data_dir, args.output_dir, cfg)
    elif args.only == "ablation":
        design_ablation(args.data_dir, args.output_dir, cfg)
    elif args.only == "matched":
        matched_granularity(args.data_dir, args.output_dir, cfg)
    elif args.only == "half-life":
        half_life_analysis(args.data_dir, args.output_dir, cfg)
    elif args.only == "epsilon":
        epsilon_sensitivity(args.data_dir, args.output_dir, cfg)
    elif args.only == "repeated":
        repeated_sampling(args.data_dir, args.output_dir, cfg)
    elif args.only == "news2013":
        news2013_validation(args.data_dir, args.output_dir, cfg)
    else:
        main_results = main_comparison(args.data_dir, args.output_dir, cfg)
        design_ablation(args.data_dir, args.output_dir, cfg, main_results)
        matched_granularity(args.data_dir, args.output_dir, cfg, main_results)
        half_life_analysis(args.data_dir, args.output_dir, cfg)
        epsilon_sensitivity(args.data_dir, args.output_dir, cfg)
        repeated_sampling(args.data_dir, args.output_dir, cfg)
        news2013_validation(args.data_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()
