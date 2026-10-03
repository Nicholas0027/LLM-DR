#!/usr/bin/env python3
"""Create publication figures for the scaled LLM-DR simulation outputs."""

from __future__ import annotations

import argparse
from io import BytesIO
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from compare_simulation_to_empirical import load_empirical, literal_list
from persona_clustering_experiment import load_profile


PERSONAS = [
    "Full-time Workhorse",
    "Lunch-Peak Specialist",
    "Super Stacker",
    "Dinner-Peak Core",
]

PERSONA_LABELS = {
    "Full-time Workhorse": "Full-time\nWorkhorse",
    "Lunch-Peak Specialist": "Lunch-Peak\nSpecialist",
    "Super Stacker": "Super\nStacker",
    "Dinner-Peak Core": "Dinner-Peak\nCore",
}

COLORS = {
    "Full-time Workhorse": "#2F7D5A",
    "Lunch-Peak Specialist": "#B8A33D",
    "Super Stacker": "#D8792C",
    "Dinner-Peak Core": "#3C6EA8",
}

ACTION_STYLES = {
    "Start": {"color": "#2F3237", "marker": "o", "label": "Start"},
    "Pickup": {"color": "#D73F3F", "marker": "^", "label": "Pickup"},
    "Drop-off": {"color": "#6A4C93", "marker": "s", "label": "Drop-off"},
}

WEB_MERCATOR_RADIUS = 6378137.0
WEB_MERCATOR_LIMIT = math.pi * WEB_MERCATOR_RADIUS
WEB_MERCATOR_MAX_LAT = 85.05112878
GCJ_A = 6378245.0
GCJ_EE = 0.006693421622965943


def lonlat_to_webmercator(lon: object, lat: object) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.clip(np.asarray(lat, dtype=float), -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT)
    x = WEB_MERCATOR_RADIUS * np.deg2rad(lon_arr)
    y = WEB_MERCATOR_RADIUS * np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat_arr) / 2.0))
    return x, y


def _gcj_transform_lat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    out += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    out += (20.0 * np.sin(y * np.pi) + 40.0 * np.sin(y / 3.0 * np.pi)) * 2.0 / 3.0
    out += (160.0 * np.sin(y / 12.0 * np.pi) + 320.0 * np.sin(y * np.pi / 30.0)) * 2.0 / 3.0
    return out


def _gcj_transform_lon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    out += (20.0 * np.sin(6.0 * x * np.pi) + 20.0 * np.sin(2.0 * x * np.pi)) * 2.0 / 3.0
    out += (20.0 * np.sin(x * np.pi) + 40.0 * np.sin(x / 3.0 * np.pi)) * 2.0 / 3.0
    out += (150.0 * np.sin(x / 12.0 * np.pi) + 300.0 * np.sin(x / 30.0 * np.pi)) * 2.0 / 3.0
    return out


def gcj02_to_wgs84(lon: object, lat: object) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_b, lat_b = np.broadcast_arrays(lon_arr, lat_arr)
    wgs_lon = lon_b.astype(float).copy()
    wgs_lat = lat_b.astype(float).copy()

    in_china = (
        (lon_b >= 72.004)
        & (lon_b <= 137.8347)
        & (lat_b >= 0.8293)
        & (lat_b <= 55.8271)
        & np.isfinite(lon_b)
        & np.isfinite(lat_b)
    )
    if not np.any(in_china):
        return wgs_lon, wgs_lat

    lon_c = lon_b[in_china]
    lat_c = lat_b[in_china]
    dlat = _gcj_transform_lat(lon_c - 105.0, lat_c - 35.0)
    dlon = _gcj_transform_lon(lon_c - 105.0, lat_c - 35.0)
    radlat = lat_c / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1.0 - GCJ_EE * magic * magic
    sqrt_magic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((GCJ_A * (1.0 - GCJ_EE)) / (magic * sqrt_magic) * np.pi)
    dlon = (dlon * 180.0) / (GCJ_A / sqrt_magic * np.cos(radlat) * np.pi)
    mg_lat = lat_c + dlat
    mg_lon = lon_c + dlon
    wgs_lon[in_china] = lon_c * 2.0 - mg_lon
    wgs_lat[in_china] = lat_c * 2.0 - mg_lat
    return wgs_lon, wgs_lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = float(np.clip(lat, -WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT))
    n = 2**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, xtile)), max(0, min(n - 1, ytile))


def tile_bounds_webmercator(xtile: int, ytile: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2**zoom
    tile_size = 2.0 * WEB_MERCATOR_LIMIT / n
    xmin = -WEB_MERCATOR_LIMIT + xtile * tile_size
    xmax = xmin + tile_size
    ymax = WEB_MERCATOR_LIMIT - ytile * tile_size
    ymin = ymax - tile_size
    return xmin, xmax, ymin, ymax


def webmercator_to_lonlat(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    lon = np.rad2deg(x_arr / WEB_MERCATOR_RADIUS)
    lat = np.rad2deg(2.0 * np.arctan(np.exp(y_arr / WEB_MERCATOR_RADIUS)) - np.pi / 2.0)
    return lon, lat


def fetch_tile(xtile: int, ytile: int, zoom: int, cache_dir: Path) -> Image.Image | None:
    cache_path = cache_dir / str(zoom) / str(xtile) / f"{ytile}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGBA")

    url = f"https://a.basemaps.cartocdn.com/light_all/{zoom}/{xtile}/{ytile}.png"
    request = Request(url, headers={"User-Agent": "LLM-DR manuscript figure generation"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError):
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return Image.open(BytesIO(payload)).convert("RGBA")


def add_light_basemap(
    ax: plt.Axes,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    cache_dir: Path,
    zoom: int = 11,
) -> None:
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    xtiles = range(min(x0, x1), max(x0, x1) + 1)
    ytiles = range(min(y0, y1), max(y0, y1) + 1)

    for xtile in xtiles:
        for ytile in ytiles:
            tile = fetch_tile(xtile, ytile, zoom, cache_dir)
            if tile is None:
                continue
            xmin, xmax, ymin, ymax = tile_bounds_webmercator(xtile, ytile, zoom)
            ax.imshow(
                np.asarray(tile),
                extent=(xmin, xmax, ymin, ymax),
                origin="upper",
                alpha=1.0,
                interpolation="bilinear",
                zorder=0,
            )


def basemap_mosaic(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    cache_dir: Path,
    zoom: int = 13,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    x0, y0 = lonlat_to_tile(lon_min, lat_max, zoom)
    x1, y1 = lonlat_to_tile(lon_max, lat_min, zoom)
    xtiles = list(range(min(x0, x1), max(x0, x1) + 1))
    ytiles = list(range(min(y0, y1), max(y0, y1) + 1))
    if not xtiles or not ytiles:
        return None

    tile_size = 256
    mosaic = Image.new("RGBA", (len(xtiles) * tile_size, len(ytiles) * tile_size), (255, 255, 255, 0))
    loaded = False
    for xi, xtile in enumerate(xtiles):
        for yi, ytile in enumerate(ytiles):
            tile = fetch_tile(xtile, ytile, zoom, cache_dir)
            if tile is None:
                continue
            loaded = True
            mosaic.paste(tile.resize((tile_size, tile_size)), (xi * tile_size, yi * tile_size))
    if not loaded:
        return None

    xmin, _, _, ymax = tile_bounds_webmercator(min(xtiles), min(ytiles), zoom)
    _, xmax, ymin, _ = tile_bounds_webmercator(max(xtiles), max(ytiles), zoom)
    return mosaic, (xmin, xmax, ymin, ymax)


def local_xy_from_lonlat(lon: object, lat: object, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_scale = 111.0 * math.cos(math.radians(lat0))
    x = (lon_arr - lon0) * lon_scale
    y = (lat_arr - lat0) * 111.0
    return x, y


def add_basemap_floor(
    ax: plt.Axes,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    lon0: float,
    lat0: float,
    z_floor: float,
    cache_dir: Path,
) -> None:
    mosaic_result = basemap_mosaic(lon_min, lat_min, lon_max, lat_max, cache_dir)
    if mosaic_result is None:
        return
    image, (x0, x1, y0, y1) = mosaic_result
    width, height = image.size
    req_x0, req_y0 = lonlat_to_webmercator(lon_min, lat_min)
    req_x1, req_y1 = lonlat_to_webmercator(lon_max, lat_max)
    req_xmin, req_xmax = sorted([float(req_x0), float(req_x1)])
    req_ymin, req_ymax = sorted([float(req_y0), float(req_y1)])
    left = int(np.clip((req_xmin - x0) / (x1 - x0) * width, 0, width - 1))
    right = int(np.clip((req_xmax - x0) / (x1 - x0) * width, left + 1, width))
    top = int(np.clip((y1 - req_ymax) / (y1 - y0) * height, 0, height - 1))
    bottom = int(np.clip((y1 - req_ymin) / (y1 - y0) * height, top + 1, height))
    image = image.crop((left, top, right, bottom)).resize((72, 72))
    rgba = np.asarray(image).astype(float) / 255.0
    rgba[..., 3] = 1.0

    west, north = webmercator_to_lonlat(req_xmin, req_ymax)
    east, south = webmercator_to_lonlat(req_xmax, req_ymin)
    xs = np.linspace(float(west), float(east), rgba.shape[1])
    ys = np.linspace(float(north), float(south), rgba.shape[0])
    xx_lon, yy_lat = np.meshgrid(xs, ys)
    xx, yy = local_xy_from_lonlat(xx_lon, yy_lat, lon0, lat0)
    zz = np.full_like(xx, z_floor)
    surface = ax.plot_surface(
        xx,
        yy,
        zz,
        rstride=1,
        cstride=1,
        facecolors=rgba,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=-100,
    )
    surface.set_zorder(-100)
    surface.set_zsort("min")
    surface.set_sort_zpos(z_floor - 50.0)
    surface.set_rasterized(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-run", required=True, help="Scaled simulation run directory.")
    parser.add_argument("--model", default="deepseek_v4", help="Model subdirectory to plot.")
    parser.add_argument("--model-label", default="LLM-DR")
    parser.add_argument("--validation-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--empirical-rider-features", default="outputs/persona/rider_persona_features.csv")
    parser.add_argument("--persona-assignments", default="outputs/persona/rider_persona_assignments.csv")
    parser.add_argument("--orders", default="0_persona/data/order_20200215.txt")
    parser.add_argument("--wave-data", default="0_persona/data/amap_routed_wave_data_20200215.csv")
    parser.add_argument("--service-date", default="20200215")
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=20)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.dpi": 160,
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def split_floats(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    out: list[float] = []
    for part in str(value).split("|"):
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def parse_route_polyline(value: object) -> tuple[list[float], list[float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return [], []
    lons: list[float] = []
    lats: list[float] = []
    text = str(value).strip()
    if not text:
        return [], []
    for part in text.replace("|", ";").split(";"):
        if "," not in part:
            continue
        lon_text, lat_text = part.split(",", 1)
        try:
            lons.append(float(lon_text))
            lats.append(float(lat_text))
        except ValueError:
            continue
    return lons, lats


def ordered_persona_frame(df: pd.DataFrame, persona_col: str = "persona") -> pd.DataFrame:
    order = {persona: idx for idx, persona in enumerate(PERSONAS)}
    out = df.copy()
    out["_persona_order"] = out[persona_col].map(order)
    return out.sort_values("_persona_order").drop(columns=["_persona_order"])


def minmax_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full_like(values, 0.5, dtype=float)
    return (values - lo) / (hi - lo)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a, dtype="float64").rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b, dtype="float64").rank(method="average").to_numpy(dtype=float)
    if np.nanstd(ra) == 0 or np.nanstd(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def project_points_km(lons: pd.Series, lats: pd.Series) -> np.ndarray:
    lon_arr = pd.to_numeric(lons, errors="coerce").to_numpy(dtype=float)
    lat_arr = pd.to_numeric(lats, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    lon_arr = lon_arr[mask]
    lat_arr = lat_arr[mask]
    if len(lon_arr) == 0:
        return np.empty((0, 2))
    lat0 = float(np.nanmean(lat_arr))
    lon0 = float(np.nanmean(lon_arr))
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


def activity_area_km2(lons: pd.Series, lats: pd.Series) -> float:
    pts = project_points_km(lons, lats)
    if len(pts) < 3:
        return 0.0
    return polygon_area(convex_hull(pts))


def source_percentile(values: pd.Series) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    valid = series.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    n = int(valid.sum())
    if n == 0:
        return out
    ranks = series[valid].rank(method="average")
    out.loc[valid] = (ranks - 0.5) / n
    return out


def load_persona_distribution_data(
    run_dir: Path,
    model: str,
    empirical_features_path: Path,
    assignments_path: Path,
    model_label: str,
    *,
    orders_path: Path = Path("0_persona/data/order_20200215.txt"),
    wave_path: Path = Path("0_persona/data/amap_routed_wave_data_20200215.csv"),
    service_date: str = "20200215",
    start_hour: int = 8,
    end_hour: int = 20,
) -> pd.DataFrame:
    # Keep the fitted persona labels fixed; recompute metrics in the evaluation window.
    cohort = pd.read_csv(empirical_features_path, usecols=["courier_id"])
    empirical_orders, empirical_waves = load_empirical(
        argparse.Namespace(
            orders=orders_path,
            wave_data=wave_path,
            assignments=assignments_path,
            service_date=service_date,
            start_hour=start_hour,
            end_hour=end_hour,
        )
    )
    empirical_orders = empirical_orders.merge(cohort, on="courier_id", validate="many_to_one")
    empirical_waves["stacked_intensity"] = empirical_waves.apply(
        lambda row: load_profile(
            literal_list(row["action_types_chronological"]), row["tracking_ids_list"]
        )[1],
        axis=1,
    )
    empirical_waves["wave_end_ts"] = empirical_waves["expect_times_list"].apply(
        lambda xs: max(xs) if xs else np.nan
    )
    empirical_wave_stats = empirical_waves.groupby("courier_id", observed=True).agg(
        avg_bundle=("n_orders_in_wave", "mean"),
        stacked_intensity=("stacked_intensity", "mean"),
        wave_distance_km=("wave_distance_km", "mean"),
        first_wave=("wave_start_ts", "min"),
        last_wave=("wave_end_ts", "max"),
    )
    empirical_wave_stats["work_span_hr"] = (
        pd.to_datetime(empirical_wave_stats["last_wave"], unit="s", utc=True)
        - empirical_wave_stats["first_wave"]
    ).dt.total_seconds().clip(lower=0) / 3600.0
    empirical = empirical_orders.groupby(["courier_id", "persona"], observed=True).agg(
        orders=("courier_id", "size")
    ).reset_index().merge(empirical_wave_stats, on="courier_id", how="left", validate="one_to_one")
    empirical_areas = [
        {"courier_id": rider_id, "activity_area_km2": activity_area_km2(group["pick_lng_corr"], group["pick_lat_corr"])}
        for rider_id, group in empirical_orders.groupby("courier_id", observed=True)
    ]
    empirical = empirical.merge(pd.DataFrame(empirical_areas), on="courier_id", validate="one_to_one")
    empirical = empirical[empirical["persona"].isin(PERSONAS)].copy()
    empirical_rows = pd.DataFrame(
        {
            "source": "Ground truth",
            "persona": empirical["persona"],
            "rider_id": empirical["courier_id"].astype(str),
            "orders": empirical["orders"],
            "avg_bundle": empirical["avg_bundle"],
            "stacked_intensity": empirical["stacked_intensity"],
            "wave_distance_km": empirical["wave_distance_km"],
            "work_span_hr": empirical["work_span_hr"],
            "activity_area_km2": empirical["activity_area_km2"],
        }
    )

    model_dir = run_dir / model
    riders = pd.read_csv(model_dir / "rider_summary.csv")
    waves = pd.read_csv(model_dir / "wave_summary.csv")
    completed = pd.read_csv(model_dir / "completed_orders_log.csv")
    waves["start_dt"] = pd.to_datetime(waves["start_time"], errors="coerce")
    waves["finish_dt"] = pd.to_datetime(waves["finish_time"], errors="coerce")
    wave_group = waves.groupby("rider_id", observed=True)
    wave_stats = wave_group.agg(
        avg_bundle=("order_count", "mean"),
        wave_distance_km=("route_distance_m", lambda s: pd.to_numeric(s, errors="coerce").mean() / 1000.0),
        first_wave=("start_dt", "min"),
        last_wave=("finish_dt", "max"),
    ).reset_index()
    stacked = (
        wave_group["order_count"]
        .apply(
            lambda s: float(
                np.nanmean(
                    np.maximum(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float) - 1.0, 0.0)
                    / np.maximum(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float), 1.0)
                )
            )
        )
        .rename("stacked_intensity")
        .reset_index()
    )
    wave_stats = wave_stats.merge(stacked, on="rider_id", how="left")
    wave_stats["work_span_hr"] = (wave_stats["last_wave"] - wave_stats["first_wave"]).dt.total_seconds().clip(lower=0) / 3600.0

    area_rows = []
    for rider_id, group in completed.groupby("assigned_rider_id", observed=True):
        area_rows.append({"rider_id": rider_id, "activity_area_km2": activity_area_km2(group["pickup_lon"], group["pickup_lat"])})
    area = pd.DataFrame(area_rows)
    sim = riders.merge(wave_stats, on="rider_id", how="left").merge(area, on="rider_id", how="left")
    for col in ["avg_bundle", "wave_distance_km", "stacked_intensity", "work_span_hr", "activity_area_km2"]:
        sim[col] = pd.to_numeric(sim[col], errors="coerce")
    sim_rows = pd.DataFrame(
        {
            "source": model_label,
            "persona": sim["persona_name"],
            "rider_id": sim["rider_id"].astype(str),
            "orders": sim["total_orders_completed"],
            "avg_bundle": sim["avg_bundle"],
            "stacked_intensity": sim["stacked_intensity"],
            "wave_distance_km": sim["wave_distance_km"],
            "work_span_hr": sim["work_span_hr"],
            "activity_area_km2": sim["activity_area_km2"],
        }
    )
    data = pd.concat([empirical_rows, sim_rows], ignore_index=True)
    data["persona"] = pd.Categorical(data["persona"], categories=PERSONAS, ordered=True)
    return data.sort_values(["source", "persona", "rider_id"]).reset_index(drop=True)


def make_persona_validation(
    run_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    model: str,
    model_label: str,
    empirical_features_path: Path,
    assignments_path: Path,
    **evaluation_settings: object,
) -> None:
    data = load_persona_distribution_data(
        run_dir, model, empirical_features_path, assignments_path, model_label, **evaluation_settings
    )
    metric_specs = [
        ("orders", "Orders completed"),
        ("avg_bundle", "Orders per wave"),
        ("stacked_intensity", "Stacked-state intensity"),
        ("wave_distance_km", "Routed wave distance"),
        ("work_span_hr", "Work span"),
        ("activity_area_km2", "Pickup activity area"),
    ]
    long_rows: list[pd.DataFrame] = []
    for metric, label in metric_specs:
        metric_df = data[["source", "persona", "rider_id", metric]].copy()
        metric_df = metric_df.rename(columns={metric: "raw_value"})
        metric_df["metric"] = metric
        metric_df["metric_label"] = label.replace("\n", " ")
        metric_df["standardized_value"] = metric_df.groupby("source", observed=True)["raw_value"].transform(source_percentile)
        long_rows.append(metric_df)
    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(output_dir / "persona_level_validation_source_data.csv", index=False)
    summary = (
        long.groupby(["metric", "metric_label", "source", "persona"], observed=True)
        .agg(
            n=("standardized_value", "count"),
            median=("standardized_value", "median"),
            q25=("standardized_value", lambda s: float(np.nanquantile(s, 0.25))),
            q75=("standardized_value", lambda s: float(np.nanquantile(s, 0.75))),
            raw_median=("raw_value", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "persona_level_validation_table.csv", index=False)

    source_colors = {"Ground truth": "#9AA1AA", model_label: "#2C3035"}
    source_offsets = {"Ground truth": -0.17, model_label: 0.17}
    x = np.arange(len(PERSONAS))
    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.45), constrained_layout=False)
    axes = axes.ravel()
    rng = np.random.default_rng(2026)
    for ax, (metric, label) in zip(axes, metric_specs):
        metric_data = long[long["metric"] == metric]
        for source in ["Ground truth", model_label]:
            arrays = [
                metric_data[(metric_data["source"] == source) & (metric_data["persona"] == persona)]["standardized_value"]
                .dropna()
                .to_numpy(dtype=float)
                for persona in PERSONAS
            ]
            positions = x + source_offsets[source]
            vp = ax.violinplot(
                arrays, positions=positions, widths=0.28, points=100, bw_method="scott",
                showmeans=False, showmedians=False, showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(source_colors[source])
                body.set_edgecolor("none")
                body.set_alpha(0.30 if source == "Ground truth" else 0.38)
            bp = ax.boxplot(
                arrays,
                positions=positions,
                widths=0.13,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                boxprops={"facecolor": "white", "edgecolor": source_colors[source], "linewidth": 0.72},
                whiskerprops={"color": source_colors[source], "linewidth": 0.62},
                capprops={"color": source_colors[source], "linewidth": 0.62},
                medianprops={"color": source_colors[source], "linewidth": 1.05},
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.92)
            medians = [float(np.nanmedian(arr)) if len(arr) else np.nan for arr in arrays]
            ax.scatter(positions, medians, s=12, color=source_colors[source], edgecolor="white", linewidth=0.35, zorder=5)
            if source == model_label:
                for pos, arr in zip(positions, arrays):
                    if len(arr) == 0:
                        continue
                    jitter = rng.normal(0.0, 0.018, size=len(arr))
                    ax.scatter(np.full(len(arr), pos) + jitter, arr, s=5, color=source_colors[source], alpha=0.28, linewidths=0, zorder=4)
        ax.set_title(label, pad=3.0, fontsize=7.0)
        ax.set_xticks(x)
        ax.set_xticklabels([PERSONA_LABELS[p] for p in PERSONAS])
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Percentile within source")
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.42)
        ax.tick_params(axis="x", labelsize=6.1, pad=1.5)
        ax.tick_params(axis="y", labelsize=6.2)
    handles = [
        Line2D([0], [0], color=source_colors["Ground truth"], lw=4.0, alpha=0.55, label="Ground truth"),
        Line2D([0], [0], color=source_colors[model_label], lw=4.0, alpha=0.65, label=model_label),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.935), w_pad=1.0, h_pad=1.0)
    save_figure(fig, output_dir / "persona_level_validation")
    plt.close(fig)


def build_wave_path(row: pd.Series) -> tuple[list[float], list[float]]:
    if "route_polyline" in row:
        lons, lats = parse_route_polyline(row.get("route_polyline", ""))
        if len(lons) >= 2:
            return lons, lats
    lons = [float(row["start_lon"])]
    lats = [float(row["start_lat"])]
    lons.extend(split_floats(row.get("pickup_lons", "")))
    lats.extend(split_floats(row.get("pickup_lats", "")))
    lons.extend(split_floats(row.get("delivery_lons", "")))
    lats.extend(split_floats(row.get("delivery_lats", "")))
    n = min(len(lons), len(lats))
    return lons[:n], lats[:n]


def make_2d_trajectory_map(model_dir: Path, output_dir: Path, model_label: str) -> None:
    waves = pd.read_csv(model_dir / "wave_summary.csv")
    if waves.empty:
        raise ValueError("wave_summary.csv is empty; cannot plot trajectories.")
    waves["start_time"] = pd.to_datetime(waves["start_time"])
    waves = waves.sort_values("start_time")

    paths: list[tuple[str, list[float], list[float]]] = []
    all_lons: list[float] = []
    all_lats: list[float] = []
    for _, row in waves.iterrows():
        lons, lats = build_wave_path(row)
        if len(lons) < 2:
            continue
        wgs_lons, wgs_lats = gcj02_to_wgs84(lons, lats)
        persona = str(row["persona_name"])
        paths.append((persona, wgs_lons.tolist(), wgs_lats.tolist()))
        all_lons.extend(wgs_lons.tolist())
        all_lats.extend(wgs_lats.tolist())

    if not all_lons or not all_lats:
        raise ValueError("No valid wave paths available for trajectory map.")

    lon_min = 116.0
    lon_max = float(np.nanmax(all_lons)) + 0.025
    lat_min = float(np.nanmin(all_lats)) - 0.025
    lat_max = 40.25
    if lat_min >= lat_max:
        lat_min = lat_max - 0.1

    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)

    fig, ax = plt.subplots(figsize=(5.9, 5.4), constrained_layout=True)
    add_light_basemap(ax, lon_min, lat_min, lon_max, lat_max, output_dir / "_tile_cache" / "carto_light")

    for persona, lons, lats in paths:
        xs, ys = lonlat_to_webmercator(lons, lats)
        if persona not in COLORS:
            continue
        ax.plot(xs, ys, color=COLORS[persona], alpha=0.52, linewidth=0.78, zorder=2)

    ax.set_title(f"{model_label} spatial action chains")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal", adjustable="box")
    lon_ticks = np.arange(math.ceil(lon_min * 10) / 10, math.floor(lon_max * 10) / 10 + 0.001, 0.1)
    lat_ticks = np.arange(math.ceil(lat_min * 10) / 10, math.floor(lat_max * 10) / 10 + 0.001, 0.1)
    if len(lon_ticks):
        tx, _ = lonlat_to_webmercator(lon_ticks, np.zeros_like(lon_ticks))
        ax.set_xticks(tx)
        ax.set_xticklabels([f"{tick:.1f}" for tick in lon_ticks])
    if len(lat_ticks):
        _, ty = lonlat_to_webmercator(np.zeros_like(lat_ticks), lat_ticks)
        ax.set_yticks(ty)
        ax.set_yticklabels([f"{tick:.1f}" for tick in lat_ticks])
    ax.grid(color="#FFFFFF", linewidth=0.45, alpha=0.75)
    ax.text(
        0.99,
        0.012,
        "Basemap: CARTO/OSM",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color="#777777",
        zorder=4,
    )
    handles = [
        Line2D([0], [0], color=COLORS[persona], lw=1.2, label=persona)
        for persona in PERSONAS
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=6)
    save_figure(fig, output_dir / "llmdr_2d_trajectory_map")
    plt.close(fig)


def representative_riders(model_dir: Path, output_dir: Path) -> pd.DataFrame:
    riders = pd.read_csv(model_dir / "rider_summary.csv")
    chosen = []
    for persona in PERSONAS:
        subset = riders[riders["persona_name"] == persona].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["total_orders_completed", "decision_count"], ascending=[False, False])
        chosen.append(subset.iloc[0])
    selected = pd.DataFrame(chosen)
    selected.to_csv(output_dir / "representative_riders.csv", index=False)
    return selected


def wave_path_3d(row: pd.Series) -> tuple[list[float], list[float], list[float]]:
    lons, lats = build_wave_path(row)
    if not lons:
        return [], [], []
    wgs_lons, wgs_lats = gcj02_to_wgs84(lons, lats)
    start = pd.to_datetime(row["start_time"])
    finish = pd.to_datetime(row["finish_time"])
    if pd.isna(finish) or finish <= start:
        finish = start + pd.Timedelta(minutes=5 * max(len(lons) - 1, 1))
    hours = np.linspace(
        start.hour + start.minute / 60.0 + start.second / 3600.0,
        finish.hour + finish.minute / 60.0 + finish.second / 3600.0,
        len(lons),
    )
    return wgs_lons.tolist(), wgs_lats.tolist(), hours.tolist()


def action_points_3d(row: pd.Series) -> list[tuple[str, float, float, float]]:
    raw_lons: list[float] = []
    raw_lats: list[float] = []
    labels: list[str] = []

    try:
        raw_lons.append(float(row["start_lon"]))
        raw_lats.append(float(row["start_lat"]))
        labels.append("Start")
    except (TypeError, ValueError):
        pass

    pickup_lons = split_floats(row.get("pickup_lons", ""))
    pickup_lats = split_floats(row.get("pickup_lats", ""))
    for lon, lat in zip(pickup_lons, pickup_lats):
        raw_lons.append(lon)
        raw_lats.append(lat)
        labels.append("Pickup")

    dropoff_lons = split_floats(row.get("delivery_lons", ""))
    dropoff_lats = split_floats(row.get("delivery_lats", ""))
    for lon, lat in zip(dropoff_lons, dropoff_lats):
        raw_lons.append(lon)
        raw_lats.append(lat)
        labels.append("Drop-off")

    if not raw_lons:
        return []

    start = pd.to_datetime(row["start_time"])
    finish = pd.to_datetime(row["finish_time"])
    if pd.isna(finish) or finish <= start:
        finish = start + pd.Timedelta(minutes=5 * max(len(raw_lons) - 1, 1))
    hours = np.linspace(
        start.hour + start.minute / 60.0 + start.second / 3600.0,
        finish.hour + finish.minute / 60.0 + finish.second / 3600.0,
        len(raw_lons),
    )
    wgs_lons, wgs_lats = gcj02_to_wgs84(raw_lons, raw_lats)
    return [
        (label, float(lon), float(lat), float(hour))
        for label, lon, lat, hour in zip(labels, wgs_lons, wgs_lats, hours)
    ]


def make_3d_representative_trajectories(model_dir: Path, output_dir: Path, model_label: str) -> None:
    waves = pd.read_csv(model_dir / "wave_summary.csv")
    if waves.empty:
        raise ValueError("wave_summary.csv is empty; cannot plot trajectories.")
    waves["start_time"] = pd.to_datetime(waves["start_time"])
    selected = representative_riders(model_dir, output_dir)

    fig = plt.figure(figsize=(7.2, 5.9))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.075, wspace=0.02, hspace=0.08)
    for idx, persona in enumerate(PERSONAS, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.computed_zorder = False
        sel = selected[selected["persona_name"] == persona]
        if sel.empty:
            ax.set_axis_off()
            continue
        rider_id = int(sel.iloc[0]["rider_id"])
        subset = waves[(waves["persona_name"] == persona) & (waves["rider_id"] == rider_id)].sort_values("start_time")
        paths = []
        for _, row in subset.iterrows():
            lons, lats, hours = wave_path_3d(row)
            if len(lons) >= 2:
                paths.append((lons, lats, hours))
        if not paths:
            ax.set_axis_off()
            continue
        action_points: list[tuple[str, float, float, float]] = []
        for _, row in subset.iterrows():
            action_points.extend(action_points_3d(row))

        all_lons = [lon for lons, _, _ in paths for lon in lons] + [point[1] for point in action_points]
        all_lats = [lat for _, lats, _ in paths for lat in lats] + [point[2] for point in action_points]
        all_hours = [hour for _, _, hours in paths for hour in hours] + [point[3] for point in action_points]
        lon0 = float(np.nanmean(all_lons))
        lat0 = float(np.nanmean(all_lats))
        lon_min, lon_max = float(np.nanmin(all_lons)), float(np.nanmax(all_lons))
        lat_min, lat_max = float(np.nanmin(all_lats)), float(np.nanmax(all_lats))
        lon_pad = max((lon_max - lon_min) * 0.10, 0.006)
        lat_pad = max((lat_max - lat_min) * 0.10, 0.006)
        lon_min -= lon_pad
        lon_max += lon_pad
        lat_min -= lat_pad
        lat_max += lat_pad

        xy_lons = np.asarray(all_lons, dtype=float)
        xy_lats = np.asarray(all_lats, dtype=float)
        all_x, all_y = local_xy_from_lonlat(xy_lons, xy_lats, lon0, lat0)
        x_pad = max((float(np.nanmax(all_x)) - float(np.nanmin(all_x))) * 0.10, 0.25)
        y_pad = max((float(np.nanmax(all_y)) - float(np.nanmin(all_y))) * 0.10, 0.25)
        z_floor = float(np.nanmin(all_hours)) - 1.25
        add_basemap_floor(
            ax,
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            lon0,
            lat0,
            z_floor,
            output_dir / "_tile_cache" / "carto_light_3d",
        )

        for lons, lats, hours in paths:
            xs, ys = local_xy_from_lonlat(lons, lats, lon0, lat0)
            ax.plot(xs, ys, hours, color=COLORS[persona], linewidth=1.15, alpha=0.86, zorder=30)
        for label, lon, lat, hour in action_points:
            style = ACTION_STYLES[label]
            x, y = local_xy_from_lonlat([lon], [lat], lon0, lat0)
            ax.scatter(
                x,
                y,
                [hour],
                color=style["color"],
                marker=style["marker"],
                s=13,
                edgecolor="white",
                linewidth=0.28,
                alpha=0.90,
                zorder=40,
            )
        ax.text2D(
            0.02,
            0.96,
            f"({chr(96 + idx)}) {persona}\nRider {rider_id}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            color="#333333",
        )
        ax.set_xlabel("East (km)", labelpad=1)
        ax.set_ylabel("North (km)", labelpad=1)
        ax.set_zlabel("Hour", labelpad=1)
        ax.set_axisbelow(False)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_zorder(80)
            axis.label.set_zorder(90)
            for tick_label in axis.get_ticklabels():
                tick_label.set_zorder(90)
        ax.set_xlim(float(np.nanmin(all_x)) - x_pad, float(np.nanmax(all_x)) + x_pad)
        ax.set_ylim(float(np.nanmin(all_y)) - y_pad, float(np.nanmax(all_y)) + y_pad)
        ax.set_zlim(z_floor, float(np.nanmax(all_hours)) + 0.25)
        ax.view_init(elev=24, azim=-58)
        ax.tick_params(axis="both", which="major", labelsize=5, pad=0)
        ax.zaxis.set_tick_params(labelsize=5, pad=0)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(True)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"]["color"] = (0.82, 0.85, 0.88, 0.30)
            axis._axinfo["grid"]["linewidth"] = 0.32
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5.2,
            label=style["label"],
        )
        for style in ACTION_STYLES.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    save_figure(fig, output_dir / "llmdr_3d_representative_trajectories")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    run_dir = Path(args.simulation_run)
    model_dir = run_dir / args.model
    validation_dir = Path(args.validation_dir) if args.validation_dir else run_dir / "empirical_validation"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    make_persona_validation(
        run_dir,
        validation_dir,
        output_dir,
        args.model,
        args.model_label,
        Path(args.empirical_rider_features),
        Path(args.persona_assignments),
        orders_path=Path(args.orders),
        wave_path=Path(args.wave_data),
        service_date=args.service_date,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
    )
    make_2d_trajectory_map(model_dir, output_dir, args.model_label)
    make_3d_representative_trajectories(model_dir, output_dir, args.model_label)
    print(f"Wrote scaled simulation figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
