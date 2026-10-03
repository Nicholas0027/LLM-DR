#!/usr/bin/env python3
"""Reproducible rider persona extraction experiment.

The script derives courier-level behavioral features from the order and routed
wave data, runs k-means clustering without non-standard ML dependencies, and
writes tables/figures for the manuscript.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "avg_daily_work_duration_hr",
    "lunch_ratio",
    "dinner_ratio",
    "late_night_ratio",
    "time_fragmentation",
    "avg_wave_distance_m",
    "avg_wave_duration_s",
    "avg_dist_per_action_m",
    "avg_dur_per_action_s",
    "stacked_order_ratio",
    "activity_area_km2",
    "area_aspect_ratio",
    "rider_level",
    "rider_speed",
    "rider_max_load",
]

SKEWED_FEATURES = {
    "avg_daily_work_duration_hr",
    "time_fragmentation",
    "avg_wave_distance_m",
    "avg_wave_duration_s",
    "avg_dist_per_action_m",
    "avg_dur_per_action_s",
    "activity_area_km2",
    "area_aspect_ratio",
}

PERSONA_ORDER = [
    "Full-time Workhorse",
    "Lunch-Peak Specialist",
    "Super Stacker",
    "Dinner-Peak Core",
]


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave-data",
        default="0_persona/data/amap_routed_wave_data_20200215.csv",
        help="Routed wave CSV.",
    )
    parser.add_argument(
        "--order-data",
        default="0_persona/data/order_20200215_filter.txt",
        help="Filtered order file encoded as UTF-16 TSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/persona",
        help="Directory for generated persona experiment outputs.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of final clusters.")
    parser.add_argument(
        "--min-waves",
        type=int,
        default=1,
        help="Minimum observed waves per courier retained for clustering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--n-init", type=int, default=80)
    parser.add_argument("--stability-runs", type=int, default=40)
    return parser.parse_args()


def literal_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def load_wave_data(path: Path) -> pd.DataFrame:
    wave = pd.read_csv(path)
    wave = wave[wave["wave_processing_status"].isin(["complete_success", "complete_partial_failure"])].copy()

    for col in [
        "tracking_ids_chronological",
        "action_types_chronological",
        "expect_times_chronological",
        "real_distances_segments",
    ]:
        wave[col] = wave[col].apply(literal_list)

    wave["n_orders_in_wave"] = wave["tracking_ids_chronological"].apply(lambda xs: len(set(map(str, xs))))
    wave[["max_concurrent_load", "stacked_state_ratio"]] = wave.apply(
        lambda row: pd.Series(load_profile(row["action_types_chronological"], row["tracking_ids_chronological"])),
        axis=1,
    )
    wave["wave_start_ts"] = wave["expect_times_chronological"].apply(lambda xs: min(xs) if xs else np.nan)
    wave["wave_end_ts"] = wave["expect_times_chronological"].apply(lambda xs: max(xs) if xs else np.nan)
    wave["segment_count"] = wave["real_distances_segments"].apply(len)
    wave["segment_distance_sum_m"] = wave["real_distances_segments"].apply(lambda xs: float(np.nansum(xs)))
    wave["distance_per_segment_m"] = wave["wave_total_nav_distance_m"] / wave["num_original_segments"].clip(lower=1)
    wave["duration_per_segment_s"] = wave["wave_total_nav_duration_s"] / wave["num_original_segments"].clip(lower=1)
    return wave


def load_profile(action_types: list, tracking_ids: list) -> tuple[int, float]:
    """Return max carried orders and share of states carrying more than one.

    The wave file records one ASSIGN event followed by pickup/drop-off events.
    The tracking-id list corresponds to the post-ASSIGN events.
    """
    active: set[str] = set()
    loads: list[int] = []
    ids = [str(x) for x in tracking_ids]
    actions = [str(x).upper() for x in action_types]

    event_actions = actions[1:] if actions and actions[0] == "ASSIGN" else actions
    for action, tracking_id in zip(event_actions, ids):
        if action == "PICKUP":
            active.add(tracking_id)
        elif action == "DELIVERY":
            active.discard(tracking_id)
        loads.append(len(active))

    if not loads:
        return 0, 0.0
    return max(loads), float(np.mean(np.asarray(loads) > 1))


def load_order_data(path: Path) -> pd.DataFrame:
    orders = pd.read_csv(path, encoding="utf-16", sep="\t")
    for col in [
        "pick_lng",
        "pick_lat",
        "deliver_lng",
        "deliver_lat",
        "create_time",
        "promise_deliver_time",
    ]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    orders = orders.dropna(subset=["courier_id", "create_time", "pick_lng", "pick_lat", "deliver_lng", "deliver_lat"])
    orders["create_dt"] = pd.to_datetime(orders["create_time"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")

    # The raw order coordinates use the source dataset's obfuscated reference
    # frame. The same constant offset is used by the simulation notebook.
    offset_lon = 116.329992 - 121.453059
    offset_lat = 39.79069 - 39.009465
    orders["pick_lng_corr"] = orders["pick_lng"] + offset_lon
    orders["pick_lat_corr"] = orders["pick_lat"] + offset_lat
    orders["deliver_lng_corr"] = orders["deliver_lng"] + offset_lon
    orders["deliver_lat_corr"] = orders["deliver_lat"] + offset_lat
    return orders


def lonlat_to_xy_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat0 = np.nanmedian(lat) if len(lat) else 39.9
    earth_km = 6371.0088
    x = earth_km * np.deg2rad(lon) * math.cos(math.radians(lat0))
    y = earth_km * np.deg2rad(lat)
    return x, y


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def point_area_features(points_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for courier_id, group in points_df.groupby("courier_id"):
        x, y = lonlat_to_xy_km(group["lon"].to_numpy(), group["lat"].to_numpy())
        mask = np.isfinite(x) & np.isfinite(y)
        pts = list(zip(x[mask].round(6), y[mask].round(6)))
        hull = convex_hull(pts)
        area = polygon_area(hull)
        if len(pts) < 2:
            aspect = 1.0
        else:
            width = float(np.max(x[mask]) - np.min(x[mask]))
            height = float(np.max(y[mask]) - np.min(y[mask]))
            shorter = max(min(width, height), 1e-6)
            aspect = max(width, height) / shorter
        rows.append(
            {
                "courier_id": courier_id,
                "activity_area_km2": area,
                "area_aspect_ratio": min(aspect, 50.0),
            }
        )
    return pd.DataFrame(rows)


def active_bin_count(group: pd.DataFrame, bin_seconds: int = 600) -> int:
    active_bins: set[int] = set()
    for start, end in group[["wave_start_ts", "wave_end_ts"]].dropna().itertuples(index=False):
        start_bin = int(start // bin_seconds)
        end_bin = int(math.ceil(end / bin_seconds))
        active_bins.update(range(start_bin, max(start_bin + 1, end_bin + 1)))
    return len(active_bins)


def build_features(wave: pd.DataFrame, orders: pd.DataFrame, min_waves: int) -> pd.DataFrame:
    wave_group = wave.groupby("courier_id")
    wave_features = wave_group.agg(
        num_waves=("wave_index", "count"),
        avg_wave_distance_m=("wave_total_nav_distance_m", "mean"),
        avg_wave_duration_s=("wave_total_nav_duration_s", "mean"),
        total_wave_distance_m=("wave_total_nav_distance_m", "sum"),
        total_wave_duration_s=("wave_total_nav_duration_s", "sum"),
        total_segments=("num_original_segments", "sum"),
        stacked_order_ratio=("stacked_state_ratio", "mean"),
        rider_level=("rider_level", "mean"),
        rider_speed=("rider_speed", "mean"),
        rider_max_load=("rider_max_load", "mean"),
        first_wave_ts=("wave_start_ts", "min"),
        last_wave_ts=("wave_end_ts", "max"),
    ).reset_index()
    wave_features["avg_dist_per_action_m"] = (
        wave_features["total_wave_distance_m"] / wave_features["total_segments"].clip(lower=1)
    )
    wave_features["avg_dur_per_action_s"] = (
        wave_features["total_wave_duration_s"] / wave_features["total_segments"].clip(lower=1)
    )
    wave_features["avg_daily_work_duration_hr"] = (
        (wave_features["last_wave_ts"] - wave_features["first_wave_ts"]).clip(lower=0) / 3600.0
    )
    fragmentation = wave_group.apply(active_bin_count, include_groups=False).rename("time_fragmentation").reset_index()
    wave_features = wave_features.merge(fragmentation, on="courier_id", how="left")

    order_stats = []
    for courier_id, group in orders.groupby("courier_id"):
        hours = group["create_dt"].dt.hour + group["create_dt"].dt.minute / 60.0
        total = len(group)
        order_stats.append(
            {
                "courier_id": courier_id,
                "total_orders": total,
                "lunch_ratio": float(((hours >= 11.0) & (hours < 14.0)).sum() / total),
                "dinner_ratio": float(((hours >= 17.0) & (hours < 21.0)).sum() / total),
                "late_night_ratio": float(((hours >= 21.0) | (hours < 7.0)).sum() / total),
            }
        )
    order_features = pd.DataFrame(order_stats)

    order_points = pd.concat(
        [
            orders[["courier_id", "pick_lng_corr", "pick_lat_corr"]].rename(
                columns={"pick_lng_corr": "lon", "pick_lat_corr": "lat"}
            ),
            orders[["courier_id", "deliver_lng_corr", "deliver_lat_corr"]].rename(
                columns={"deliver_lng_corr": "lon", "deliver_lat_corr": "lat"}
            ),
        ],
        ignore_index=True,
    )
    wave_points = wave[["courier_id", "wave_start_lng", "wave_start_lat"]].rename(
        columns={"wave_start_lng": "lon", "wave_start_lat": "lat"}
    )
    area_features = point_area_features(pd.concat([order_points, wave_points], ignore_index=True))

    features = (
        wave_features.merge(order_features, on="courier_id", how="inner")
        .merge(area_features, on="courier_id", how="left")
        .fillna({"activity_area_km2": 0.0, "area_aspect_ratio": 1.0})
    )
    features = features[features["num_waves"] >= min_waves].copy()
    return features[["courier_id", "num_waves", "total_orders"] + FEATURE_COLUMNS].reset_index(drop=True)


def transform_features(features: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.Series, pd.Series]:
    transformed = features[FEATURE_COLUMNS].astype(float).copy()
    for col in SKEWED_FEATURES:
        transformed[col] = np.log1p(transformed[col].clip(lower=0))
    mean = transformed.mean(axis=0)
    std = transformed.std(axis=0, ddof=0).replace(0, 1.0)
    z = (transformed - mean) / std
    return z.to_numpy(dtype=float), z, mean, std


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)
    return np.sqrt(d2)


def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    centers = np.empty((k, x.shape[1]), dtype=float)
    first = rng.integers(n)
    centers[0] = x[first]
    closest_d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(closest_d2.sum())
        if total <= 0:
            centers[i] = x[rng.integers(n)]
        else:
            idx = rng.choice(n, p=closest_d2 / total)
            centers[i] = x[idx]
        closest_d2 = np.minimum(closest_d2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def run_kmeans(
    x: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
    tol: float = 1e-5,
) -> KMeansResult:
    best: KMeansResult | None = None
    base_rng = np.random.default_rng(seed)
    seeds = base_rng.integers(0, 2**32 - 1, size=n_init)

    for init_seed in seeds:
        rng = np.random.default_rng(int(init_seed))
        centers = kmeans_plus_plus_init(x, k, rng)
        labels = np.zeros(len(x), dtype=int)
        for iteration in range(1, max_iter + 1):
            distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = x[new_labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
                else:
                    new_centers[j] = x[rng.integers(len(x))]
            shift = float(np.sqrt(np.sum((new_centers - centers) ** 2)))
            centers = new_centers
            labels = new_labels
            if shift < tol:
                break
        inertia = float(np.sum((x - centers[labels]) ** 2))
        result = KMeansResult(labels=labels, centers=centers, inertia=inertia, iterations=iteration)
        if best is None or result.inertia < best.inertia:
            best = result

    assert best is not None
    return best


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(x):
        return float("nan")
    dist = pairwise_distances(x)
    values = []
    for i in range(len(x)):
        same = labels == labels[i]
        if same.sum() <= 1:
            a = 0.0
        else:
            a = float(dist[i, same].sum() / (same.sum() - 1))
        b = math.inf
        for cluster in unique:
            if cluster == labels[i]:
                continue
            mask = labels == cluster
            b = min(b, float(dist[i, mask].mean()))
        denom = max(a, b)
        values.append(0.0 if denom == 0 else (b - a) / denom)
    return float(np.mean(values))


def calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    n, k = len(x), len(unique)
    if k < 2 or k >= n:
        return float("nan")
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for cluster in unique:
        members = x[labels == cluster]
        center = members.mean(axis=0)
        between += len(members) * float(np.sum((center - overall) ** 2))
        within += float(np.sum((members - center) ** 2))
    return float((between / (k - 1)) / (within / (n - k))) if within > 0 else float("inf")


def davies_bouldin_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    k = len(unique)
    if k < 2:
        return float("nan")
    centers = []
    scatters = []
    for cluster in unique:
        members = x[labels == cluster]
        center = members.mean(axis=0)
        centers.append(center)
        scatters.append(float(np.sqrt(np.sum((members - center) ** 2, axis=1)).mean()))
    centers_arr = np.vstack(centers)
    center_dist = pairwise_distances(centers_arr)
    ratios = []
    for i in range(k):
        vals = []
        for j in range(k):
            if i == j:
                continue
            vals.append((scatters[i] + scatters[j]) / max(center_dist[i, j], 1e-12))
        ratios.append(max(vals))
    return float(np.mean(ratios))


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    def comb2(n: np.ndarray | float) -> np.ndarray | float:
        return n * (n - 1) / 2

    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    a_vals, a_inv = np.unique(labels_a, return_inverse=True)
    b_vals, b_inv = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((len(a_vals), len(b_vals)), dtype=float)
    for i in range(len(labels_a)):
        contingency[a_inv[i], b_inv[i]] += 1
    sum_comb = float(comb2(contingency).sum())
    row_comb = float(comb2(contingency.sum(axis=1)).sum())
    col_comb = float(comb2(contingency.sum(axis=0)).sum())
    total_comb = float(comb2(len(labels_a)))
    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return 0.0 if denom == 0 else float((sum_comb - expected) / denom)


def evaluate_k_range(x: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for k in range(args.k_min, args.k_max + 1):
        result = run_kmeans(x, k=k, seed=args.seed + k, n_init=args.n_init)
        rows.append(
            {
                "k": k,
                "inertia": result.inertia,
                "silhouette": silhouette_score(x, result.labels),
                "calinski_harabasz": calinski_harabasz_score(x, result.labels),
                "davies_bouldin": davies_bouldin_score(x, result.labels),
                "min_cluster_size": int(np.bincount(result.labels).min()),
                "max_cluster_size": int(np.bincount(result.labels).max()),
                "iterations": result.iterations,
            }
        )
    return pd.DataFrame(rows)


def stability_experiment(x: np.ndarray, k: int, seed: int, runs: int, n_init: int) -> pd.DataFrame:
    labels = []
    for i in range(runs):
        labels.append(run_kmeans(x, k=k, seed=seed + 1000 + i, n_init=max(10, n_init // 4)).labels)
    scores = []
    for i in range(runs):
        for j in range(i + 1, runs):
            scores.append(adjusted_rand_index(labels[i], labels[j]))
    return pd.DataFrame(
        {
            "metric": ["adjusted_rand_index"],
            "runs": [runs],
            "pair_count": [len(scores)],
            "mean": [float(np.mean(scores))],
            "std": [float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0],
            "min": [float(np.min(scores))],
            "median": [float(np.median(scores))],
            "max": [float(np.max(scores))],
        }
    )


def assign_persona_names(summary: pd.DataFrame) -> dict[int, str]:
    remaining = set(summary["cluster"].astype(int))
    assignments: dict[int, str] = {}

    lunch = int(summary.set_index("cluster").loc[list(remaining), "lunch_ratio"].idxmax())
    assignments[lunch] = "Lunch-Peak Specialist"
    remaining.remove(lunch)

    dinner = int(summary.set_index("cluster").loc[list(remaining), "dinner_ratio"].idxmax())
    assignments[dinner] = "Dinner-Peak Core"
    remaining.remove(dinner)

    super_scores = (
        summary.set_index("cluster")["stacked_order_ratio"].rank()
        + summary.set_index("cluster")["avg_wave_distance_m"].rank()
        + summary.set_index("cluster")["avg_wave_duration_s"].rank()
    )
    super_cluster = int(super_scores.loc[list(remaining)].idxmax())
    assignments[super_cluster] = "Super Stacker"
    remaining.remove(super_cluster)

    for cluster in remaining:
        assignments[int(cluster)] = "Full-time Workhorse"
    return assignments


def add_persona_labels(features: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    assigned = features.copy()
    assigned["cluster"] = labels.astype(int)
    raw_summary = assigned.groupby("cluster")[FEATURE_COLUMNS + ["num_waves", "total_orders"]].mean().reset_index()
    raw_summary["n_couriers"] = assigned.groupby("cluster").size().to_numpy()
    mapping = assign_persona_names(raw_summary)
    assigned["persona"] = assigned["cluster"].map(mapping)
    assigned["persona"] = pd.Categorical(assigned["persona"], categories=PERSONA_ORDER, ordered=True)

    summary = assigned.groupby("persona", observed=True).agg(
        n_couriers=("courier_id", "size"),
        num_waves=("num_waves", "mean"),
        total_orders=("total_orders", "mean"),
        **{col: (col, "mean") for col in FEATURE_COLUMNS},
    )
    summary = summary.reset_index()
    return assigned, summary


def pca_2d(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def color_for_persona(persona: str) -> str:
    return {
        "Full-time Workhorse": "#2E7D32",
        "Lunch-Peak Specialist": "#C0A000",
        "Super Stacker": "#EF7D00",
        "Dinner-Peak Core": "#1F77B4",
    }.get(persona, "#666666")


def write_svg_scatter(coords: np.ndarray, labels: pd.Series, path: Path) -> None:
    width, height = 760, 520
    margin = 70
    x = coords[:, 0]
    y = coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 1.0

    def sx(val: float) -> float:
        return margin + (val - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * (width - 2 * margin)

    def sy(val: float) -> float:
        return height - margin - (val - y_min + y_pad) / (y_max - y_min + 2 * y_pad) * (height - 2 * margin)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="16">PC1</text>',
        f'<text x="22" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 22 {height/2})">PC2</text>',
        '<text x="70" y="34" font-family="Arial" font-size="18" font-weight="bold">Persona clusters in standardized feature space</text>',
    ]
    for persona in PERSONA_ORDER:
        mask = labels.astype(str).to_numpy() == persona
        for px, py in coords[mask]:
            lines.append(
                f'<circle cx="{sx(float(px)):.2f}" cy="{sy(float(py)):.2f}" r="3.4" '
                f'fill="{color_for_persona(persona)}" fill-opacity="0.72" stroke="white" stroke-width="0.4"/>'
            )
    legend_x = width - 250
    legend_y = 55
    lines.append(f'<rect x="{legend_x-14}" y="{legend_y-24}" width="238" height="116" fill="white" stroke="#ddd"/>')
    for i, persona in enumerate(PERSONA_ORDER):
        y0 = legend_y + i * 26
        lines.append(f'<circle cx="{legend_x}" cy="{y0}" r="6" fill="{color_for_persona(persona)}"/>')
        lines.append(f'<text x="{legend_x+14}" y="{y0+5}" font-family="Arial" font-size="14">{persona}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_k_metrics(metrics: pd.DataFrame, path: Path) -> None:
    width, height = 760, 420
    margin = 64
    k_vals = metrics["k"].to_numpy()
    sil = metrics["silhouette"].to_numpy()
    db = metrics["davies_bouldin"].to_numpy()

    def scale(vals: np.ndarray, lo: float, hi: float, invert: bool = False) -> list[float]:
        denom = hi - lo if hi != lo else 1.0
        scaled = margin + (vals - lo) / denom * (height - 2 * margin)
        if invert:
            scaled = height - scaled
        return list(scaled)

    x = scale(k_vals, float(k_vals.min()), float(k_vals.max()), invert=False)
    y_sil = scale(sil, float(np.nanmin(sil)), float(np.nanmax(sil)), invert=True)
    y_db = scale(db, float(np.nanmin(db)), float(np.nanmax(db)), invert=True)

    def polyline(xs: list[float], ys: list[float]) -> str:
        return " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(xs, ys))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
        '<text x="68" y="34" font-family="Arial" font-size="18" font-weight="bold">Cluster-validity metrics across k</text>',
        f'<polyline fill="none" stroke="#2E7D32" stroke-width="2.5" points="{polyline(x, y_sil)}"/>',
        f'<polyline fill="none" stroke="#C43C35" stroke-width="2.5" points="{polyline(x, y_db)}"/>',
    ]
    for xi, yi, kval in zip(x, y_sil, k_vals):
        lines.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="5" fill="#2E7D32"/>')
        lines.append(f'<text x="{xi:.2f}" y="{height-margin+24}" text-anchor="middle" font-family="Arial" font-size="13">{int(kval)}</text>')
    for xi, yi in zip(x, y_db):
        lines.append(f'<circle cx="{xi:.2f}" cy="{yi:.2f}" r="5" fill="#C43C35"/>')
    lines.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="15">Number of clusters (k)</text>')
    lines.append(f'<rect x="{width-236}" y="48" width="185" height="58" fill="white" stroke="#ddd"/>')
    lines.append(f'<line x1="{width-218}" y1="68" x2="{width-188}" y2="68" stroke="#2E7D32" stroke-width="3"/>')
    lines.append(f'<text x="{width-178}" y="73" font-family="Arial" font-size="14">Silhouette (higher)</text>')
    lines.append(f'<line x1="{width-218}" y1="92" x2="{width-188}" y2="92" stroke="#C43C35" stroke-width="3"/>')
    lines.append(f'<text x="{width-178}" y="97" font-family="Arial" font-size="14">Davies-Bouldin (lower)</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_summary_table(summary: pd.DataFrame) -> str:
    cols = [
        ("persona", "Persona"),
        ("n_couriers", "N"),
        ("avg_daily_work_duration_hr", "Work hr"),
        ("lunch_ratio", "Lunch"),
        ("dinner_ratio", "Dinner"),
        ("late_night_ratio", "Late"),
        ("time_fragmentation", "Active bins"),
        ("avg_wave_distance_m", "Wave dist. m"),
        ("avg_wave_duration_s", "Wave dur. s"),
        ("avg_dist_per_action_m", "Dist/action m"),
        ("avg_dur_per_action_s", "Dur/action s"),
        ("stacked_order_ratio", "Stacked"),
        ("activity_area_km2", "Area km2"),
        ("area_aspect_ratio", "Aspect"),
        ("rider_level", "Level"),
        ("rider_speed", "Speed"),
        ("rider_max_load", "Max load"),
    ]
    lines = [
        "\\begin{tabular}{lrrrrrrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(header for _, header in cols) + r" \\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        values = []
        for col, _ in cols:
            val = row[col]
            if col == "persona":
                values.append(str(val))
            elif col == "n_couriers":
                values.append(f"{int(val)}")
            elif "ratio" in col or col in {"rider_speed", "area_aspect_ratio"}:
                values.append(f"{float(val):.2f}")
            else:
                values.append(f"{float(val):.1f}")
        lines.append(" & ".join(values) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def write_feature_dictionary(path: Path) -> None:
    rows = [
        ("avg_daily_work_duration_hr", "Elapsed hours between a rider's first and last routed wave event."),
        ("lunch_ratio", "Share of the rider's orders created during 11:00-14:00."),
        ("dinner_ratio", "Share of the rider's orders created during 17:00-21:00."),
        ("late_night_ratio", "Share of the rider's orders created after 21:00 or before 07:00."),
        ("time_fragmentation", "Number of 10-minute bins covered by routed wave activity."),
        ("avg_wave_distance_m", "Mean routed distance per delivery wave from Amap routing."),
        ("avg_wave_duration_s", "Mean routed duration per delivery wave from Amap routing."),
        ("avg_dist_per_action_m", "Total routed distance divided by total original wave segments."),
        ("avg_dur_per_action_s", "Total routed duration divided by total original wave segments."),
        ("stacked_order_ratio", "Mean share of pickup/drop-off states where the rider carries more than one active order."),
        ("activity_area_km2", "Convex-hull area of corrected pickup, drop-off, and wave-start points."),
        ("area_aspect_ratio", "Bounding-box elongation ratio of the rider's activity points."),
        ("rider_level", "Mean platform rider level recorded in the wave data."),
        ("rider_speed", "Mean rider speed recorded in the wave data."),
        ("rider_max_load", "Mean rider maximum load recorded in the wave data."),
    ]
    pd.DataFrame(rows, columns=["feature", "definition"]).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wave = load_wave_data(Path(args.wave_data))
    orders = load_order_data(Path(args.order_data))
    features = build_features(wave, orders, min_waves=args.min_waves)
    x, z_features, _, _ = transform_features(features)

    k_metrics = evaluate_k_range(x, args)
    final = run_kmeans(x, k=args.k, seed=args.seed, n_init=args.n_init)
    assignments, summary = add_persona_labels(features, final.labels)
    stability = stability_experiment(
        x,
        k=args.k,
        seed=args.seed,
        runs=args.stability_runs,
        n_init=args.n_init,
    )

    assignments = assignments.sort_values(["persona", "courier_id"])
    summary = summary.sort_values("persona")

    features.to_csv(output_dir / "rider_persona_features.csv", index=False)
    z_features.assign(courier_id=features["courier_id"]).to_csv(output_dir / "rider_persona_features_standardized.csv", index=False)
    assignments.to_csv(output_dir / "rider_persona_assignments.csv", index=False)
    summary.to_csv(output_dir / "persona_cluster_summary.csv", index=False)
    k_metrics.to_csv(output_dir / "k_selection_metrics.csv", index=False)
    stability.to_csv(output_dir / "cluster_stability.csv", index=False)
    write_feature_dictionary(output_dir / "feature_dictionary.csv")
    (output_dir / "persona_table.tex").write_text(latex_summary_table(summary), encoding="utf-8")

    coords = pca_2d(x)
    coord_df = pd.DataFrame(coords, columns=["pc1", "pc2"])
    coord_df["courier_id"] = features["courier_id"]
    coord_df["persona"] = assignments.sort_index()["persona"].astype(str).to_numpy()
    coord_df.to_csv(output_dir / "persona_pca_coordinates.csv", index=False)
    write_svg_scatter(coords, assignments.sort_index()["persona"].astype(str), output_dir / "persona_clusters_pca.svg")
    write_svg_k_metrics(k_metrics, output_dir / "k_selection_metrics.svg")

    report = {
        "wave_rows": int(len(wave)),
        "order_rows": int(len(orders)),
        "couriers_in_wave": int(wave["courier_id"].nunique()),
        "couriers_in_orders": int(orders["courier_id"].nunique()),
        "couriers_clustered": int(len(features)),
        "feature_count": len(FEATURE_COLUMNS),
        "k": args.k,
        "final_inertia": final.inertia,
        "final_silhouette": silhouette_score(x, final.labels),
        "final_calinski_harabasz": calinski_harabasz_score(x, final.labels),
        "final_davies_bouldin": davies_bouldin_score(x, final.labels),
        "stability_mean_ari": float(stability.loc[0, "mean"]),
        "stability_median_ari": float(stability.loc[0, "median"]),
    }
    (output_dir / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Persona experiment completed.")
    print(json.dumps(report, indent=2))
    print("\nCluster summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
