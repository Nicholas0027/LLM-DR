#!/usr/bin/env python3
"""Compare scaled simulation outputs with empirical rider/order patterns."""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_simulation_experiment as base  # noqa: E402


PERSONA_ORDER = base.PERSONA_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="0_persona/data/order_20200215.txt")
    parser.add_argument("--wave-data", default="0_persona/data/amap_routed_wave_data_20200215.csv")
    parser.add_argument("--assignments", default="outputs/persona/rider_persona_assignments.csv")
    parser.add_argument("--simulation-run", required=True, help="Scaled simulation run directory.")
    parser.add_argument("--service-date", default="20200215")
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=20)
    parser.add_argument("--grid-bins", type=int, default=12)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def literal_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def coordinate_correct_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    offset_lon = 116.329992 - 121.453059
    offset_lat = 39.79069 - 39.009465
    for col in ["pick_lng", "deliver_lng"]:
        df[f"{col}_corr"] = pd.to_numeric(df[col], errors="coerce") + offset_lon
    for col in ["pick_lat", "deliver_lat"]:
        df[f"{col}_corr"] = pd.to_numeric(df[col], errors="coerce") + offset_lat
    return df


def filter_by_time(df: pd.DataFrame, time_col: str, service_date: str, start_hour: int, end_hour: int) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    mask = (
        (out[time_col].dt.strftime("%Y%m%d") == service_date)
        & (out[time_col].dt.hour >= start_hour)
        & (out[time_col].dt.hour < end_hour)
    )
    return out[mask].copy()


def haversine_rows(df: pd.DataFrame, lon1: str, lat1: str, lon2: str, lat2: str) -> pd.Series:
    return df.apply(
        lambda row: base.haversine_km(
            base.Point(float(row[lon1]), float(row[lat1])),
            base.Point(float(row[lon2]), float(row[lat2])),
        ),
        axis=1,
    )


def project_points_km(lons: Iterable[float], lats: Iterable[float]) -> np.ndarray:
    lon_arr = np.asarray(list(lons), dtype=float)
    lat_arr = np.asarray(list(lats), dtype=float)
    if len(lon_arr) == 0:
        return np.empty((0, 2))
    lat0 = np.nanmean(lat_arr)
    lon0 = np.nanmean(lon_arr)
    x = (lon_arr - lon0) * 111.0 * math.cos(math.radians(lat0))
    y = (lat_arr - lat0) * 111.0
    return np.column_stack([x, y])


def convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points
    pts = sorted(map(tuple, points))

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def radius_gyration_km(lons: Iterable[float], lats: Iterable[float]) -> float:
    pts = project_points_km(lons, lats)
    if len(pts) == 0:
        return 0.0
    centre = pts.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pts - centre) ** 2, axis=1))))


def activity_area_km2(lons: Iterable[float], lats: Iterable[float]) -> float:
    pts = project_points_km(lons, lats)
    if len(pts) < 3:
        return 0.0
    return polygon_area(convex_hull(pts))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p + 1e-12
    q = q + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def wasserstein_1d(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    probs = np.linspace(0, 1, 101)
    return float(np.mean(np.abs(np.quantile(a, probs) - np.quantile(b, probs))))


def ks_1d(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.sort(np.asarray(list(a), dtype=float))
    b = np.sort(np.asarray(list(b), dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, values, side="right") / len(a)
    cdf_b = np.searchsorted(b, values, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def hourly_hist(times: pd.Series, start_hour: int, end_hour: int) -> np.ndarray:
    hours = pd.to_datetime(times).dt.hour
    return np.asarray([(hours == hour).sum() for hour in range(start_hour, end_hour)], dtype=float)


def grid_hist(df_a: pd.DataFrame, df_b: pd.DataFrame, lon_col: str, lat_col: str, bins: int) -> tuple[np.ndarray, np.ndarray]:
    both = pd.concat([df_a[[lon_col, lat_col]], df_b[[lon_col, lat_col]]], ignore_index=True).dropna()
    if both.empty:
        return np.zeros(bins * bins), np.zeros(bins * bins)
    lon_min, lon_max = both[lon_col].min(), both[lon_col].max()
    lat_min, lat_max = both[lat_col].min(), both[lat_col].max()
    if lon_min == lon_max:
        lon_max += 1e-6
    if lat_min == lat_max:
        lat_max += 1e-6
    hist_a, _, _ = np.histogram2d(df_a[lon_col], df_a[lat_col], bins=bins, range=[[lon_min, lon_max], [lat_min, lat_max]])
    hist_b, _, _ = np.histogram2d(df_b[lon_col], df_b[lat_col], bins=bins, range=[[lon_min, lon_max], [lat_min, lat_max]])
    return hist_a.ravel(), hist_b.ravel()


def load_empirical(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(args.assignments)[["courier_id", "persona"]]

    orders = pd.read_csv(args.orders)
    orders = coordinate_correct_orders(orders)
    orders = filter_by_time(orders, "create_time", args.service_date, args.start_hour, args.end_hour)
    orders = orders.merge(assignments, on="courier_id", how="inner")
    orders["od_distance_km"] = haversine_rows(orders, "pick_lng_corr", "pick_lat_corr", "deliver_lng_corr", "deliver_lat_corr")

    waves = pd.read_csv(args.wave_data)
    waves = waves[waves["wave_processing_status"].isin(["complete_success", "complete_partial_failure"])].copy()
    waves["tracking_ids_list"] = waves["tracking_ids_chronological"].apply(literal_list)
    waves["expect_times_list"] = waves["expect_times_chronological"].apply(literal_list)
    waves["n_orders_in_wave"] = waves["tracking_ids_list"].apply(lambda xs: len(set(map(str, xs))))
    waves["wave_start_ts"] = waves["expect_times_list"].apply(lambda xs: min(xs) if xs else np.nan)
    waves = waves.dropna(subset=["wave_start_ts"]).copy()
    waves = filter_by_time(waves, "wave_start_ts", args.service_date, args.start_hour, args.end_hour)
    waves = waves.merge(assignments, on="courier_id", how="inner")
    waves["wave_distance_km"] = pd.to_numeric(waves["wave_total_nav_distance_m"], errors="coerce") / 1000.0
    waves["wave_duration_min"] = pd.to_numeric(waves["wave_total_nav_duration_s"], errors="coerce") / 60.0
    waves["stacked_wave"] = waves["n_orders_in_wave"] > 1
    return orders, waves


def load_simulation_model(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(model_dir / "completed_orders_log.csv")
    waves = pd.read_csv(model_dir / "wave_summary.csv")
    riders = pd.read_csv(model_dir / "rider_summary.csv")
    if not orders.empty:
        orders["create_dt"] = pd.to_datetime(orders["create_time"])
        orders["od_distance_km"] = haversine_rows(orders, "pickup_lon", "pickup_lat", "delivery_lon", "delivery_lat")
    if not waves.empty:
        waves["start_dt"] = pd.to_datetime(waves["start_time"])
        waves["wave_distance_km"] = pd.to_numeric(waves["route_distance_m"], errors="coerce") / 1000.0
        waves["wave_duration_min"] = pd.to_numeric(waves["route_duration_s"], errors="coerce") / 60.0
        waves["n_orders_in_wave"] = pd.to_numeric(waves["order_count"], errors="coerce")
        waves["stacked_wave"] = waves["n_orders_in_wave"] > 1
    return orders, waves, riders


def summarise_persona_orders(
    orders: pd.DataFrame,
    waves: pd.DataFrame,
    persona: str,
    prefix: str,
    time_col: str,
    lon_col: str,
    lat_col: str,
) -> dict[str, object]:
    po = orders[orders["persona"] == persona] if "persona" in orders else orders[orders["persona_name"] == persona]
    pw = waves[waves["persona"] == persona] if "persona" in waves else waves[waves["persona_name"] == persona]
    rider_col = "courier_id" if "courier_id" in po else "assigned_rider_id"
    active_riders = int(po[rider_col].nunique()) if rider_col in po and not po.empty else 0
    return {
        "source": prefix,
        "persona": persona,
        "order_count": int(len(po)),
        "wave_count": int(len(pw)),
        "active_riders": active_riders,
        "orders_per_active_rider": round(len(po) / active_riders, 4) if active_riders else 0.0,
        "avg_bundle_size": round(float(pw["n_orders_in_wave"].mean()), 4) if not pw.empty else 0.0,
        "stacked_wave_ratio": round(float(pw["stacked_wave"].mean()), 4) if not pw.empty else 0.0,
        "mean_od_distance_km": round(float(po["od_distance_km"].mean()), 4) if not po.empty else 0.0,
        "mean_wave_distance_km": round(float(pw["wave_distance_km"].mean()), 4) if not pw.empty else 0.0,
        "mean_wave_duration_min": round(float(pw["wave_duration_min"].mean()), 4) if not pw.empty else 0.0,
        "pickup_radius_gyration_km": round(radius_gyration_km(po[lon_col], po[lat_col]), 4) if not po.empty else 0.0,
        "pickup_activity_area_km2": round(activity_area_km2(po[lon_col], po[lat_col]), 4) if not po.empty else 0.0,
    }


def compare_persona(
    emp_orders: pd.DataFrame,
    emp_waves: pd.DataFrame,
    sim_orders: pd.DataFrame,
    sim_waves: pd.DataFrame,
    persona: str,
    args: argparse.Namespace,
    model: str,
) -> dict[str, object]:
    eo = emp_orders[emp_orders["persona"] == persona]
    ew = emp_waves[emp_waves["persona"] == persona]
    so = sim_orders[sim_orders["persona_name"] == persona]
    sw = sim_waves[sim_waves["persona_name"] == persona]

    emp_hour = hourly_hist(eo["create_time"], args.start_hour, args.end_hour) if not eo.empty else np.zeros(args.end_hour - args.start_hour)
    sim_hour = hourly_hist(so["create_dt"], args.start_hour, args.end_hour) if not so.empty else np.zeros(args.end_hour - args.start_hour)
    emp_grid, sim_grid = grid_hist(
        eo.rename(columns={"pick_lng_corr": "pickup_lon", "pick_lat_corr": "pickup_lat"}),
        so,
        "pickup_lon",
        "pickup_lat",
        args.grid_bins,
    )

    emp_rg = radius_gyration_km(eo["pick_lng_corr"], eo["pick_lat_corr"]) if not eo.empty else 0.0
    sim_rg = radius_gyration_km(so["pickup_lon"], so["pickup_lat"]) if not so.empty else 0.0
    emp_area = activity_area_km2(eo["pick_lng_corr"], eo["pick_lat_corr"]) if not eo.empty else 0.0
    sim_area = activity_area_km2(so["pickup_lon"], so["pickup_lat"]) if not so.empty else 0.0
    area_ratio = (sim_area + 1e-6) / (emp_area + 1e-6)

    return {
        "model": model,
        "persona": persona,
        "emp_order_count": int(len(eo)),
        "sim_order_count": int(len(so)),
        "emp_wave_count": int(len(ew)),
        "sim_wave_count": int(len(sw)),
        "hourly_jsd": round(js_divergence(emp_hour, sim_hour), 4),
        "pickup_grid_jsd": round(js_divergence(emp_grid, sim_grid), 4),
        "od_distance_wasserstein_km": round(wasserstein_1d(eo["od_distance_km"], so["od_distance_km"]), 4),
        "od_distance_ks": round(ks_1d(eo["od_distance_km"], so["od_distance_km"]), 4),
        "wave_distance_wasserstein_km": round(wasserstein_1d(ew["wave_distance_km"], sw["wave_distance_km"]), 4),
        "wave_distance_ks": round(ks_1d(ew["wave_distance_km"], sw["wave_distance_km"]), 4),
        "bundle_size_wasserstein": round(wasserstein_1d(ew["n_orders_in_wave"], sw["n_orders_in_wave"]), 4),
        "emp_stacked_wave_ratio": round(float(ew["stacked_wave"].mean()), 4) if not ew.empty else 0.0,
        "sim_stacked_wave_ratio": round(float(sw["stacked_wave"].mean()), 4) if not sw.empty else 0.0,
        "stacked_wave_ratio_abs_diff": round(
            abs((float(ew["stacked_wave"].mean()) if not ew.empty else 0.0) - (float(sw["stacked_wave"].mean()) if not sw.empty else 0.0)),
            4,
        ),
        "pickup_radius_gyration_abs_diff_km": round(abs(emp_rg - sim_rg), 4),
        "pickup_activity_area_ratio": round(area_ratio, 4),
        "pickup_activity_area_log_ratio_abs": round(abs(math.log(area_ratio)), 4),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.simulation_run) / "empirical_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    emp_orders, emp_waves = load_empirical(args)
    empirical_summary = []
    for persona in PERSONA_ORDER:
        empirical_summary.append(
            summarise_persona_orders(
                emp_orders,
                emp_waves,
                persona,
                "empirical",
                "create_time",
                "pick_lng_corr",
                "pick_lat_corr",
            )
        )
    pd.DataFrame(empirical_summary).to_csv(output_dir / "empirical_persona_summary.csv", index=False)

    comparison_rows = []
    sim_summary_rows = []
    for model_dir in sorted(Path(args.simulation_run).iterdir()):
        if not model_dir.is_dir() or not (model_dir / "completed_orders_log.csv").exists():
            continue
        sim_orders, sim_waves, _ = load_simulation_model(model_dir)
        model = model_dir.name
        for persona in PERSONA_ORDER:
            sim_summary_rows.append(
                summarise_persona_orders(
                    sim_orders,
                    sim_waves,
                    persona,
                    model,
                    "create_dt",
                    "pickup_lon",
                    "pickup_lat",
                )
            )
            comparison_rows.append(compare_persona(emp_orders, emp_waves, sim_orders, sim_waves, persona, args, model))

    sim_summary = pd.DataFrame(sim_summary_rows)
    comparison = pd.DataFrame(comparison_rows)
    sim_summary.to_csv(output_dir / "simulation_persona_summary.csv", index=False)
    comparison.to_csv(output_dir / "persona_validation_metrics.csv", index=False)

    aggregate_cols = [
        "hourly_jsd",
        "pickup_grid_jsd",
        "od_distance_wasserstein_km",
        "od_distance_ks",
        "wave_distance_wasserstein_km",
        "wave_distance_ks",
        "bundle_size_wasserstein",
        "stacked_wave_ratio_abs_diff",
        "pickup_radius_gyration_abs_diff_km",
        "pickup_activity_area_log_ratio_abs",
    ]
    aggregate = comparison.groupby("model", observed=True)[aggregate_cols].mean().reset_index()
    aggregate.to_csv(output_dir / "model_validation_aggregate.csv", index=False)
    ranks = aggregate[["model"]].copy()
    for col in aggregate_cols:
        ranks[f"{col}_rank"] = aggregate[col].rank(method="average", ascending=True)
    rank_cols = [col for col in ranks.columns if col.endswith("_rank")]
    ranks["mean_rank"] = ranks[rank_cols].mean(axis=1)
    ranks.sort_values("mean_rank").to_csv(output_dir / "model_validation_ranks.csv", index=False)
    print(f"Wrote empirical validation outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
