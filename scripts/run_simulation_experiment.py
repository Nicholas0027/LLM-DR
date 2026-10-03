#!/usr/bin/env python3
"""Run reproducible LLM-DR simulation experiments.

This runner replaces the notebook-only workflow with a configurable experiment
entry point. It supports a deterministic heuristic backend for debugging and an
OpenAI-compatible chat-completions backend for real LLM model comparisons.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PERSONA_ORDER = [
    "Full-time Workhorse",
    "Lunch-Peak Specialist",
    "Super Stacker",
    "Dinner-Peak Core",
]

PERSONA_DESCRIPTIONS = {
    "Full-time Workhorse": (
        "You are an experienced full-time delivery rider. You value stable daily "
        "earnings, steady order completion, and broad spatial coverage across the day."
    ),
    "Lunch-Peak Specialist": (
        "You are a short-duration rider focused on the lunch rush. You prefer dense, "
        "nearby, efficient orders and avoid work that does not fit your peak window."
    ),
    "Super Stacker": (
        "You are an efficiency-oriented rider who actively looks for multi-order "
        "bundles, longer waves, and spatially coherent routes."
    ),
    "Dinner-Peak Core": (
        "You are an evening-rush rider. You prefer direct, manageable orders around "
        "the dinner period and avoid overly complex bundles."
    ),
}

START_LOCATIONS = {
    "zhongguancun": (116.31647, 39.983992),
    "xidan": (116.37420, 39.90750),
    "wangjing": (116.48210, 39.99680),
    "sanlitun": (116.455294, 39.937492),
}


@dataclass
class Point:
    lon: float
    lat: float

    @property
    def wkt(self) -> str:
        return f"POINT ({self.lon:.6f} {self.lat:.6f})"


@dataclass
class Order:
    order_id: int
    create_time: pd.Timestamp
    promise_deliver_time: pd.Timestamp
    pick_loc: Point
    deliver_loc: Point
    fee: float
    status: str = "PENDING"
    assigned_rider_id: int | None = None
    actual_duration_s: float | None = None
    actual_distance_m: float | None = None
    finish_time: pd.Timestamp | None = None


@dataclass
class PersonaProfile:
    name: str
    description: str
    stats: dict[str, float]


@dataclass
class ModelConfig:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    temperature: float = 0.2
    max_tokens: int = 1024
    response_format_json: bool = False
    thinking_type: str | None = None
    reasoning_effort: str | None = None


@dataclass
class DecisionResult:
    accepted_order_ids: list[int]
    reasoning_steps: list[str]
    raw_content: str = ""
    latency_s: float = 0.0
    ok: bool = True
    error: str = ""


@dataclass
class RouteEstimate:
    distance_m: float
    duration_s: float
    polyline: str = ""
    provider: str = "haversine"
    api_error_count: int = 0
    cache_hit_count: int = 0


@dataclass
class Rider:
    rider_id: int
    persona: PersonaProfile
    location: Point
    status: str = "IDLE"
    current_orders: list[Order] = field(default_factory=list)
    task_finish_time: pd.Timestamp | None = None
    task_distance_m: float = 0.0
    task_duration_s: float = 0.0
    task_route_provider: str = ""
    task_route_polyline: str = ""
    task_route_api_errors: int = 0
    task_route_cache_hits: int = 0
    order_history: list[Order] = field(default_factory=list)
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    total_earnings: float = 0.0
    total_mileage_km: float = 0.0

    @property
    def name(self) -> str:
        return self.persona.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="0_persona/data/order_20200215.txt")
    parser.add_argument(
        "--service-date",
        default=None,
        help="Date to keep in YYYYMMDD or YYYY-MM-DD form. Defaults to the first YYYYMMDD found in --orders.",
    )
    parser.add_argument("--persona-summary", default="outputs/persona/persona_cluster_summary.csv")
    parser.add_argument("--model-suite", default="config/model_suite.example.json")
    parser.add_argument("--models", nargs="+", default=["heuristic"])
    parser.add_argument("--locations", nargs="+", default=["zhongguancun"])
    parser.add_argument("--output-dir", default="outputs/simulation")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--start-hour", type=int, default=7)
    parser.add_argument("--end-hour", type=int, default=11)
    parser.add_argument("--time-step-minutes", type=int, default=10)
    parser.add_argument("--broadcast-radius-km", type=float, default=2.0)
    parser.add_argument("--max-offers-per-rider", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--detour-factor", type=float, default=1.35)
    parser.add_argument("--speed-mps", type=float, default=3.2)
    parser.add_argument("--service-seconds-per-order", type=float, default=120.0)
    parser.add_argument("--route-provider", choices=["haversine", "amap"], default="haversine")
    parser.add_argument("--amap-key-env", default="AMAP_API_KEY")
    parser.add_argument("--amap-keys-env", default="AMAP_API_KEYS")
    parser.add_argument("--amap-mode", choices=["bicycling", "electrobike"], default="bicycling")
    parser.add_argument("--route-cache", default="outputs/route_cache/amap_routes.jsonl")
    parser.add_argument("--route-timeout", type=int, default=20)
    parser.add_argument("--route-retries", type=int, default=1)
    parser.add_argument("--route-workers", type=int, default=4)
    parser.add_argument("--route-strict", action="store_true")
    parser.add_argument("--route-tool-in-prompt", dest="route_tool_in_prompt", action="store_true", default=True)
    parser.add_argument("--no-route-tool-in-prompt", dest="route_tool_in_prompt", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Force heuristic backend even for named LLM models.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_model_suite(path: Path) -> dict[str, ModelConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    configs: dict[str, ModelConfig] = {}
    for name, raw in data.get("models", {}).items():
        configs[name] = ModelConfig(
            name=name,
            provider=str(raw.get("provider", "")),
            base_url=str(raw["base_url"]).rstrip("/"),
            model=str(raw["model"]),
            api_key_env=str(raw["api_key_env"]),
            temperature=float(raw.get("temperature", 0.2)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            response_format_json=bool(raw.get("response_format_json", False)),
            thinking_type=str(raw["thinking_type"]) if raw.get("thinking_type") else None,
            reasoning_effort=str(raw["reasoning_effort"]) if raw.get("reasoning_effort") else None,
        )
    return configs


def load_personas(path: Path) -> list[PersonaProfile]:
    summary = pd.read_csv(path)
    summary = summary.set_index("persona")
    personas: list[PersonaProfile] = []
    stat_cols = [
        "avg_daily_work_duration_hr",
        "lunch_ratio",
        "dinner_ratio",
        "late_night_ratio",
        "avg_wave_distance_m",
        "avg_wave_duration_s",
        "stacked_order_ratio",
        "activity_area_km2",
    ]
    for persona_name in PERSONA_ORDER:
        row = summary.loc[persona_name]
        stats = {col: float(row[col]) for col in stat_cols if col in row}
        stats["n_couriers"] = float(row["n_couriers"])
        personas.append(
            PersonaProfile(
                name=persona_name,
                description=PERSONA_DESCRIPTIONS[persona_name],
                stats=stats,
            )
        )
    return personas


def haversine_km(a: Point, b: Point) -> float:
    radius_km = 6371.0088
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(x))


def estimate_route(points: list[Point], detour_factor: float, speed_mps: float) -> tuple[float, float]:
    distance_m = 0.0
    for start, end in zip(points, points[1:]):
        distance_m += haversine_km(start, end) * 1000.0 * detour_factor
    duration_s = distance_m / max(speed_mps, 0.1)
    return distance_m, duration_s


def point_polyline(points: list[Point]) -> str:
    return ";".join(f"{point.lon:.6f},{point.lat:.6f}" for point in points)


class RoutePlanner:
    """Routing tool used by rider agents and the simulation environment."""

    def __init__(
        self,
        provider: str = "haversine",
        amap_key_env: str = "AMAP_API_KEY",
        amap_keys_env: str = "AMAP_API_KEYS",
        amap_mode: str = "electrobike",
        cache_path: Path | None = None,
        timeout: int = 20,
        retries: int = 1,
        route_workers: int = 4,
        detour_factor: float = 1.35,
        speed_mps: float = 3.2,
        strict: bool = False,
    ):
        self.provider = provider
        self.amap_key_env = amap_key_env
        self.amap_keys_env = amap_keys_env
        self.amap_mode = amap_mode
        self.cache_path = cache_path
        self.timeout = timeout
        self.retries = retries
        self.route_workers = max(1, route_workers)
        self.detour_factor = detour_factor
        self.speed_mps = speed_mps
        self.strict = strict
        self.cache: dict[str, dict[str, Any]] = {}
        self.request_count = 0
        self.cache_hit_count = 0
        self.api_error_count = 0
        self._amap_conn: http.client.HTTPSConnection | None = None
        self._lock = threading.Lock()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RoutePlanner":
        return cls(
            provider=getattr(args, "route_provider", "haversine"),
            amap_key_env=getattr(args, "amap_key_env", "AMAP_API_KEY"),
            amap_keys_env=getattr(args, "amap_keys_env", "AMAP_API_KEYS"),
            amap_mode=getattr(args, "amap_mode", "bicycling"),
            cache_path=Path(getattr(args, "route_cache", "outputs/route_cache/amap_routes.jsonl")),
            timeout=getattr(args, "route_timeout", 20),
            retries=getattr(args, "route_retries", 1),
            route_workers=getattr(args, "route_workers", 4),
            detour_factor=getattr(args, "detour_factor", 1.35),
            speed_mps=getattr(args, "speed_mps", 3.2),
            strict=getattr(args, "route_strict", False),
        )

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("key", ""))
                if key:
                    self.cache[key] = row

    def _write_cache_row(self, row: dict[str, Any]) -> None:
        if not self.cache_path:
            return
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _cache_key(self, start: Point, end: Point, mode: str) -> str:
        return f"{mode}:{start.lon:.6f},{start.lat:.6f}->{end.lon:.6f},{end.lat:.6f}"

    def _fallback_segment(self, start: Point, end: Point, provider: str = "haversine") -> RouteEstimate:
        distance_m = haversine_km(start, end) * 1000.0 * self.detour_factor
        duration_s = distance_m / max(self.speed_mps, 0.1)
        return RouteEstimate(distance_m, duration_s, point_polyline([start, end]), provider)

    def _amap_keys(self) -> list[str]:
        keys: list[str] = []
        multi_value = os.getenv(self.amap_keys_env, "")
        for item in re.split(r"[,;\s]+", multi_value):
            clean = item.strip().strip('"').strip("'")
            if clean and clean not in keys:
                keys.append(clean)
        single_value = os.getenv(self.amap_key_env, "").strip().strip('"').strip("'")
        if single_value and single_value not in keys:
            keys.append(single_value)
        return keys

    def _amap_get_json(self, path: str, persistent: bool = True) -> dict[str, Any]:
        if not persistent:
            conn = http.client.HTTPSConnection("restapi.amap.com", timeout=self.timeout)
            try:
                conn.request("GET", path, headers={"Accept": "application/json", "Connection": "close"})
                response = conn.getresponse()
                body = response.read()
                if response.status >= 400:
                    raise RuntimeError(f"Amap HTTP {response.status}: {body[:300].decode('utf-8', errors='replace')}")
                return json.loads(body.decode("utf-8"))
            finally:
                conn.close()

        if self._amap_conn is None:
            self._amap_conn = http.client.HTTPSConnection("restapi.amap.com", timeout=self.timeout)
        try:
            self._amap_conn.request("GET", path, headers={"Accept": "application/json", "Connection": "keep-alive"})
            response = self._amap_conn.getresponse()
            body = response.read()
            if response.status >= 400:
                raise RuntimeError(f"Amap HTTP {response.status}: {body[:300].decode('utf-8', errors='replace')}")
            return json.loads(body.decode("utf-8"))
        except Exception:
            if self._amap_conn is not None:
                self._amap_conn.close()
            self._amap_conn = None
            raise

    def _parse_amap_response(self, data: dict[str, Any]) -> tuple[float, float, str]:
        paths: list[dict[str, Any]] = []
        if isinstance(data.get("route"), dict):
            paths = data["route"].get("paths") or []
        elif isinstance(data.get("data"), dict):
            paths = data["data"].get("paths") or []
        if not paths:
            info = data.get("info") or data.get("errmsg") or "no path returned"
            raise ValueError(f"Amap routing failed: {info}")

        path = paths[0]
        distance_m = float(path.get("distance") or 0.0)
        cost = path.get("cost") if isinstance(path.get("cost"), dict) else {}
        duration_s = float(cost.get("duration") or path.get("duration") or 0.0)
        polyline_parts: list[str] = []
        path_polyline = path.get("polyline")
        if isinstance(path_polyline, list):
            polyline_parts.extend(str(item) for item in path_polyline if item)
        elif path_polyline:
            polyline_parts.append(str(path_polyline))
        for step in path.get("steps") or []:
            polyline = step.get("polyline")
            if isinstance(polyline, list):
                polyline_parts.extend(str(item) for item in polyline if item)
            elif polyline:
                polyline_parts.append(str(polyline))
        polyline = ";".join(part.strip(";") for part in polyline_parts if part)
        if distance_m <= 0:
            raise ValueError("Amap routing returned non-positive distance.")
        if duration_s <= 0:
            duration_s = distance_m / max(self.speed_mps, 0.1)
        return distance_m, duration_s, polyline

    def _request_amap_segment(self, start: Point, end: Point, mode: str, persistent: bool = True) -> RouteEstimate:
        api_keys = self._amap_keys()
        if not api_keys:
            raise ValueError(f"Missing {self.amap_keys_env} or {self.amap_key_env}")
        with self._lock:
            api_key = api_keys[self.request_count % len(api_keys)]
            self.request_count += 1

        params_data = {
            "origin": f"{start.lon:.6f},{start.lat:.6f}",
            "destination": f"{end.lon:.6f},{end.lat:.6f}",
            "key": api_key,
            "output": "json",
        }
        if mode == "bicycling":
            path_base = "/v4/direction/bicycling"
        elif mode == "bicycling_v5":
            params_data["show_fields"] = "cost,navi"
            path_base = "/v5/direction/bicycling"
        elif mode == "walking":
            path_base = "/v3/direction/walking"
        else:
            params_data["show_fields"] = "cost,navi"
            path_base = f"/v5/direction/{mode}"
        params = urllib.parse.urlencode(params_data)
        data = self._amap_get_json(f"{path_base}?{params}", persistent=persistent)
        status = str(data.get("status", "1"))
        errcode = str(data.get("errcode", "0"))
        if status not in {"1", "None"} and errcode not in {"0", "None"}:
            info = data.get("info") or data.get("errmsg") or "unknown error"
            raise ValueError(f"Amap routing failed: {info}")
        distance_m, duration_s, polyline = self._parse_amap_response(data)
        if not polyline:
            polyline = point_polyline([start, end])
        return RouteEstimate(distance_m, duration_s, polyline, f"amap:{mode}")

    def estimate_segment(self, start: Point, end: Point, persistent: bool = True) -> RouteEstimate:
        if haversine_km(start, end) < 0.001:
            return RouteEstimate(0.0, 0.0, point_polyline([start, end]), self.provider)
        if self.provider != "amap":
            return self._fallback_segment(start, end)

        modes = [self.amap_mode]
        if self.amap_mode == "bicycling":
            modes.append("bicycling_v5")
            modes.append("walking")
        else:
            modes.append("bicycling")
        last_error = ""
        for mode in modes:
            key = self._cache_key(start, end, mode)
            with self._lock:
                row = self.cache.get(key)
                if row is not None:
                    self.cache_hit_count += 1
            if row is not None:
                return RouteEstimate(
                    float(row["distance_m"]),
                    float(row["duration_s"]),
                    str(row.get("polyline", "")),
                    str(row.get("provider", f"amap:{mode}")),
                    cache_hit_count=1,
                )

            for attempt in range(self.retries + 1):
                try:
                    estimate = self._request_amap_segment(start, end, mode, persistent=persistent)
                    row = {
                        "key": key,
                        "provider": estimate.provider,
                        "distance_m": round(estimate.distance_m, 3),
                        "duration_s": round(estimate.duration_s, 3),
                        "polyline": estimate.polyline,
                    }
                    with self._lock:
                        if key not in self.cache:
                            self.cache[key] = row
                            self._write_cache_row(row)
                    return estimate
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if attempt < self.retries:
                        time.sleep(0.5 * (attempt + 1))

        with self._lock:
            self.api_error_count += 1
        if self.strict:
            raise RuntimeError(
                f"{last_error or 'Amap routing failed.'}; "
                f"segment={start.lon:.6f},{start.lat:.6f}->{end.lon:.6f},{end.lat:.6f}; "
                f"modes={','.join(modes)}"
            )
        fallback = self._fallback_segment(start, end, "amap_fallback_haversine")
        fallback.api_error_count = 1
        return fallback

    def estimate_segments(self, pairs: list[tuple[Point, Point]]) -> list[RouteEstimate]:
        if self.route_workers <= 1 or len(pairs) <= 1:
            return [self.estimate_segment(start, end) for start, end in pairs]

        results: list[RouteEstimate | None] = [None] * len(pairs)
        with ThreadPoolExecutor(max_workers=min(self.route_workers, len(pairs))) as executor:
            futures = {
                executor.submit(self.estimate_segment, start, end, False): idx
                for idx, (start, end) in enumerate(pairs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    with self._lock:
                        if self.api_error_count > 0:
                            self.api_error_count -= 1
                    start, end = pairs[idx]
                    results[idx] = self.estimate_segment(start, end, persistent=True)
        return [result for result in results if result is not None]

    def estimate_route(self, points: list[Point]) -> RouteEstimate:
        if len(points) < 2:
            return RouteEstimate(0.0, 0.0, "", self.provider)

        total_distance_m = 0.0
        total_duration_s = 0.0
        providers: list[str] = []
        polyline_points: list[str] = []
        api_errors = 0
        cache_hits = 0
        estimates = self.estimate_segments(list(zip(points, points[1:])))
        for estimate in estimates:
            total_distance_m += estimate.distance_m
            total_duration_s += estimate.duration_s
            providers.append(estimate.provider)
            api_errors += estimate.api_error_count
            cache_hits += estimate.cache_hit_count
            segment_points = [part for part in estimate.polyline.split(";") if part]
            if polyline_points and segment_points and polyline_points[-1] == segment_points[0]:
                segment_points = segment_points[1:]
            polyline_points.extend(segment_points)

        provider = providers[0] if providers and all(item == providers[0] for item in providers) else "+".join(sorted(set(providers)))
        return RouteEstimate(
            total_distance_m,
            total_duration_s,
            ";".join(polyline_points),
            provider,
            api_error_count=api_errors,
            cache_hit_count=cache_hits,
        )


def calculate_fee(distance_km: float) -> float:
    if distance_km <= 1:
        return 3.0
    if distance_km <= 3:
        return 6.0
    if distance_km <= 5:
        return 10.0
    return 15.0


def infer_service_date(path: Path) -> str | None:
    match = re.search(r"(20\d{6})", path.name)
    return match.group(1) if match else None


def normalize_service_date(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.replace("-", "").strip()
    if not re.fullmatch(r"\d{8}", clean):
        raise ValueError("--service-date must use YYYYMMDD or YYYY-MM-DD.")
    return clean


def load_orders(path: Path, start_hour: int, end_hour: int, service_date: str | None = None) -> list[Order]:
    df = pd.read_csv(path)
    required = ["tracking_id", "pick_lng", "pick_lat", "deliver_lng", "deliver_lat", "create_time", "promise_deliver_time"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).copy()

    offset_lon = 116.329992 - 121.453059
    offset_lat = 39.79069 - 39.009465
    for col in ["pick_lng", "deliver_lng"]:
        df[col] += offset_lon
    for col in ["pick_lat", "deliver_lat"]:
        df[col] += offset_lat

    df["create_dt"] = pd.to_datetime(df["create_time"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    df["promise_dt"] = pd.to_datetime(df["promise_deliver_time"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
    if service_date:
        df = df[df["create_dt"].dt.strftime("%Y%m%d") == service_date].copy()
    df = df[(df["create_dt"].dt.hour >= start_hour) & (df["create_dt"].dt.hour < end_hour)].copy()
    df = df.sort_values("create_dt")

    orders: list[Order] = []
    for row in df.itertuples(index=False):
        pick = Point(float(row.pick_lng), float(row.pick_lat))
        deliver = Point(float(row.deliver_lng), float(row.deliver_lat))
        direct_km = haversine_km(pick, deliver)
        orders.append(
            Order(
                order_id=int(row.tracking_id),
                create_time=row.create_dt,
                promise_deliver_time=row.promise_dt,
                pick_loc=pick,
                deliver_loc=deliver,
                fee=calculate_fee(direct_km),
            )
        )
    return orders


def clean_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(content[start : end + 1])


class ChatModelClient:
    def __init__(self, config: ModelConfig, timeout: int, retries: int):
        self.config = config
        self.timeout = timeout
        self.retries = retries

    def decide(self, prompt: str) -> DecisionResult:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            return DecisionResult([], [], ok=False, error=f"Missing {self.config.api_key_env}")

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a delivery rider agent. Return concise strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if self.config.thinking_type:
            payload["thinking"] = {"type": self.config.thinking_type}
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=body,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error = ""
        started = time.time()
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                message = data["choices"][0]["message"]
                content = message.get("content") or message.get("reasoning_content") or ""
                parsed = clean_json_content(content)
                return DecisionResult(
                    accepted_order_ids=[int(x) for x in parsed.get("accepted_order_ids", [])],
                    reasoning_steps=[str(x) for x in parsed.get("reasoning_steps", [])],
                    raw_content=content,
                    latency_s=time.time() - started,
                    ok=True,
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            if attempt < self.retries:
                time.sleep(2**attempt)

        return DecisionResult([], [], latency_s=time.time() - started, ok=False, error=last_error)


class HeuristicClient:
    def __init__(self, model_name: str = "heuristic"):
        self.config = ModelConfig(model_name, "local", "", model_name, "")

    def decide_for_rider(self, rider: Rider, offers: list[Order], current_time: pd.Timestamp) -> DecisionResult:
        hour = current_time.hour + current_time.minute / 60.0
        scored: list[tuple[float, Order]] = []
        for order in offers:
            to_pick = haversine_km(rider.location, order.pick_loc)
            delivery = haversine_km(order.pick_loc, order.deliver_loc)
            score = order.fee - 1.2 * to_pick - 0.2 * delivery
            if rider.name == "Lunch-Peak Specialist":
                score += 8.0 if 11 <= hour < 14 else -5.0
                score -= 0.8 * delivery
            elif rider.name == "Dinner-Peak Core":
                score += 8.0 if 17 <= hour < 21 else -4.0
                score -= 0.5 * max(0, delivery - 4)
            elif rider.name == "Super Stacker":
                score += 0.6 * len(offers)
                score += 0.5 * order.fee
            else:
                score += 1.0
            scored.append((score, order))

        scored.sort(key=lambda x: x[0], reverse=True)
        accepted: list[int] = []
        if scored and scored[0][0] > 0:
            max_orders = 2 if rider.name in {"Super Stacker", "Full-time Workhorse"} else 1
            accepted = [order.order_id for _, order in scored[:max_orders] if _ > -2]

        return DecisionResult(
            accepted_order_ids=accepted,
            reasoning_steps=[
                f"Perception: {len(offers)} offers are available near the rider.",
                f"Recall & Evaluation: Heuristic scoring used persona={rider.name}.",
                f"Planning & Bundling: Selected up to {len(accepted)} order(s) based on fee, proximity, and persona.",
                f"Decision: Accept {accepted}." if accepted else "Decision: Reject all offers.",
            ],
            ok=True,
        )


def build_prompt(
    rider: Rider,
    offers: list[Order],
    current_time: pd.Timestamp,
    route_planner: RoutePlanner | None = None,
) -> str:
    history = "No completed orders yet."
    if rider.order_history:
        recent = rider.order_history[-5:]
        lines = [
            f"You have completed {len(rider.order_history)} orders, earned CNY {rider.total_earnings:.2f}, "
            f"and traveled {rider.total_mileage_km:.2f} km."
        ]
        for order in recent:
            on_time = order.finish_time is not None and order.finish_time <= order.promise_deliver_time
            lines.append(
                f"- Order {order.order_id}: fee {order.fee:.1f}, duration {order.actual_duration_s or 0:.0f}s, "
                f"{'on time' if on_time else 'late'}."
            )
        history = "\n".join(lines)

    route_tool_estimates: list[tuple[RouteEstimate, RouteEstimate]] = []
    if route_planner is not None and offers:
        pairs: list[tuple[Point, Point]] = []
        for order in offers:
            pairs.append((rider.location, order.pick_loc))
            pairs.append((order.pick_loc, order.deliver_loc))
        estimates = route_planner.estimate_segments(pairs)
        route_tool_estimates = [
            (estimates[i], estimates[i + 1])
            for i in range(0, len(estimates), 2)
            if i + 1 < len(estimates)
        ]

    offer_lines = []
    for idx, order in enumerate(offers, start=1):
        if route_planner is not None and idx <= len(route_tool_estimates):
            to_pick, delivery = route_tool_estimates[idx - 1]
            to_pick_km = to_pick.distance_m / 1000.0
            delivery_km = delivery.distance_m / 1000.0
            time_line = (
                f"   - Route-tool travel time: {to_pick.duration_s / 60.0:.1f} min to pickup; "
                f"{delivery.duration_s / 60.0:.1f} min pickup-to-delivery\n"
            )
            distance_label = "Route-tool distance"
        else:
            to_pick_km = haversine_km(rider.location, order.pick_loc)
            delivery_km = haversine_km(order.pick_loc, order.deliver_loc)
            time_line = ""
            distance_label = "Distance"
        offer_lines.append(
            f"{idx}. Order ID: {order.order_id}\n"
            f"   - Fee: CNY {order.fee:.1f}\n"
            f"   - {distance_label} to pickup: {to_pick_km:.2f} km\n"
            f"   - {distance_label} pickup-to-delivery: {delivery_km:.2f} km\n"
            f"{time_line}"
            f"   - Pickup: {order.pick_loc.wkt}\n"
            f"   - Delivery: {order.deliver_loc.wkt}\n"
            f"   - Promised delivery time: {order.promise_deliver_time.strftime('%H:%M:%S')}"
        )

    return f"""
# ROLE AND GOAL
You are an AI agent acting as a delivery rider in a realistic simulation of Beijing's on-demand delivery market.
Make a human-like decision according to your rider persona and current operating context.

# PERSONA PROFILE
Persona: {rider.persona.name}
Description: {rider.persona.description}
Empirical statistics: {json.dumps(rider.persona.stats, ensure_ascii=False)}

# CURRENT STATE
Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
Current location: {rider.location.wkt}
Today's earnings so far: CNY {rider.total_earnings:.2f}
Today's mileage so far: {rider.total_mileage_km:.2f} km
Routing tool: estimates road-network travel distance and time between candidate waypoints.

# MEMORY
{history}

# AVAILABLE ORDERS
{chr(10).join(offer_lines)}

# DECISION WORKFLOW
Think through four concise steps:
1. Perception: summarize the offers.
2. Recall & Evaluation: compare the offers with the persona.
3. Planning & Bundling: decide whether a single order, a bundle, or rejection is best.
4. Decision: list the accepted order IDs.

# OUTPUT FORMAT
Return only this JSON object:
{{
  "accepted_order_ids": [integer order IDs chosen from AVAILABLE ORDERS, or []],
  "reasoning_steps": [
    "Perception: ...",
    "Recall & Evaluation: ...",
    "Planning & Bundling: ...",
    "Decision: ..."
  ]
}}
""".strip()


def create_riders(personas: list[PersonaProfile], location: Point) -> list[Rider]:
    return [Rider(idx + 1, persona, Point(location.lon, location.lat)) for idx, persona in enumerate(personas)]


def validate_decision(result: DecisionResult, offers: list[Order]) -> list[int]:
    valid_ids = {order.order_id for order in offers}
    accepted: list[int] = []
    for order_id in result.accepted_order_ids:
        if order_id in valid_ids and order_id not in accepted:
            accepted.append(order_id)
    return accepted


def finish_ready_tasks(riders: list[Rider], current_time: pd.Timestamp) -> int:
    completed = 0
    for rider in riders:
        if rider.status != "DELIVERING" or rider.task_finish_time is None:
            continue
        if rider.task_finish_time > current_time:
            continue
        for order in rider.current_orders:
            order.status = "COMPLETED"
            rider.order_history.append(order)
            rider.total_earnings += order.fee
            completed += 1
        if rider.current_orders:
            rider.location = rider.current_orders[-1].deliver_loc
        rider.total_mileage_km += rider.task_distance_m / 1000.0
        rider.current_orders = []
        rider.task_finish_time = None
        rider.task_distance_m = 0.0
        rider.task_duration_s = 0.0
        rider.task_route_provider = ""
        rider.task_route_polyline = ""
        rider.task_route_api_errors = 0
        rider.task_route_cache_hits = 0
        rider.status = "IDLE"
    return completed


def assign_orders(
    rider: Rider,
    accepted_orders: list[Order],
    current_time: pd.Timestamp,
    route_planner: RoutePlanner | None,
    service_seconds_per_order: float,
) -> None:
    route = [rider.location] + [order.pick_loc for order in accepted_orders] + [order.deliver_loc for order in accepted_orders]
    if route_planner is None:
        route_planner = RoutePlanner()
    route_estimate = route_planner.estimate_route(route)
    distance_m = route_estimate.distance_m
    moving_duration_s = route_estimate.duration_s
    duration_s = moving_duration_s + service_seconds_per_order * len(accepted_orders)
    finish_time = current_time + pd.Timedelta(seconds=duration_s)
    for order in accepted_orders:
        order.status = "ASSIGNED"
        order.assigned_rider_id = rider.rider_id
        order.actual_duration_s = duration_s
        order.actual_distance_m = distance_m
        order.finish_time = finish_time
    rider.status = "DELIVERING"
    rider.current_orders = accepted_orders
    rider.task_finish_time = finish_time
    rider.task_distance_m = distance_m
    rider.task_duration_s = duration_s
    rider.task_route_provider = route_estimate.provider
    rider.task_route_polyline = route_estimate.polyline
    rider.task_route_api_errors = route_estimate.api_error_count
    rider.task_route_cache_hits = route_estimate.cache_hit_count


def run_one_simulation(
    model_name: str,
    model_config: ModelConfig | None,
    orders: list[Order],
    personas: list[PersonaProfile],
    location_name: str,
    location: Point,
    args: argparse.Namespace,
) -> tuple[list[Rider], dict[str, Any]]:
    riders = create_riders(personas, location)
    local_orders = [
        Order(
            order.order_id,
            order.create_time,
            order.promise_deliver_time,
            Point(order.pick_loc.lon, order.pick_loc.lat),
            Point(order.deliver_loc.lon, order.deliver_loc.lat),
            order.fee,
        )
        for order in orders
    ]
    if not local_orders:
        raise ValueError("No orders available for the selected time window.")

    route_planner = RoutePlanner.from_args(args)
    client: Any
    if args.dry_run or model_name == "heuristic":
        client = HeuristicClient(model_name)
        backend = "heuristic"
    else:
        if model_config is None:
            raise ValueError(f"Missing model config for {model_name}.")
        client = ChatModelClient(model_config, timeout=args.request_timeout, retries=args.retries)
        backend = "llm"

    current_time = local_orders[0].create_time.floor(f"{args.time_step_minutes}min")
    end_time = local_orders[-1].create_time.ceil(f"{args.time_step_minutes}min") + pd.Timedelta(hours=2)
    time_step = pd.Timedelta(minutes=args.time_step_minutes)
    completed_count = 0
    decision_count = 0
    api_error_count = 0
    started = time.time()

    while current_time <= end_time:
        completed_count += finish_ready_tasks(riders, current_time)
        new_orders = [
            order
            for order in local_orders
            if order.status == "PENDING" and current_time <= order.create_time < current_time + time_step
        ]
        available_pool = new_orders.copy()
        for rider in [r for r in riders if r.status == "IDLE"]:
            if args.max_decisions and decision_count >= args.max_decisions:
                break
            offers = [
                order
                for order in available_pool
                if haversine_km(rider.location, order.pick_loc) <= args.broadcast_radius_km
            ]
            offers = sorted(offers, key=lambda o: haversine_km(rider.location, o.pick_loc))[: args.max_offers_per_rider]
            if not offers:
                continue

            if backend == "heuristic":
                result = client.decide_for_rider(rider, offers, current_time)
                prompt = ""
            else:
                prompt_route_planner = route_planner if getattr(args, "route_tool_in_prompt", True) else None
                prompt = build_prompt(rider, offers, current_time, prompt_route_planner)
                result = client.decide(prompt)
            decision_count += 1
            if not result.ok:
                api_error_count += 1

            accepted_ids = validate_decision(result, offers)
            rider.decision_log.append(
                {
                    "timestamp": current_time.isoformat(),
                    "model": model_name,
                    "backend": backend,
                    "location": location_name,
                    "rider_id": rider.rider_id,
                    "persona_name": rider.name,
                    "available_orders": [order.order_id for order in offers],
                    "accepted_order_ids": accepted_ids,
                    "decision_json": {
                        "accepted_order_ids": accepted_ids,
                        "reasoning_steps": result.reasoning_steps,
                    },
                    "ok": result.ok,
                    "error": result.error,
                    "latency_s": round(result.latency_s, 3),
                    "prompt_chars": len(prompt),
                    "raw_content": result.raw_content,
                }
            )

            if accepted_ids:
                accepted_orders = [order for order in offers if order.order_id in accepted_ids]
                assign_orders(
                    rider,
                    accepted_orders,
                    current_time,
                    route_planner,
                    args.service_seconds_per_order,
                )
                accepted_set = set(accepted_ids)
                available_pool = [order for order in available_pool if order.order_id not in accepted_set]
        if args.max_decisions and decision_count >= args.max_decisions:
            break
        current_time += time_step

    # Finish tasks that are already assigned before writing outputs.
    while any(r.status == "DELIVERING" for r in riders):
        next_finish = min(r.task_finish_time for r in riders if r.task_finish_time is not None)
        completed_count += finish_ready_tasks(riders, next_finish)

    metadata = {
        "model_name": model_name,
        "model_id": model_config.model if model_config else model_name,
        "backend": backend,
        "location": location_name,
        "orders_in_window": len(local_orders),
        "completed_order_count": completed_count,
        "decision_count": decision_count,
        "api_error_count": api_error_count,
        "elapsed_s": round(time.time() - started, 3),
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "broadcast_radius_km": args.broadcast_radius_km,
        "max_decisions": args.max_decisions,
        "route_provider": args.route_provider,
        "amap_mode": args.amap_mode,
        "route_request_count": route_planner.request_count,
        "route_cache_hit_count": route_planner.cache_hit_count,
        "route_api_error_count": route_planner.api_error_count,
    }
    return riders, metadata


def write_outputs(run_dir: Path, riders: list[Rider], metadata: dict[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    rider_rows = []
    order_rows = []
    decision_rows = []
    for rider in riders:
        completed = rider.order_history
        on_time_count = sum(1 for order in completed if order.finish_time and order.finish_time <= order.promise_deliver_time)
        rider_rows.append(
            {
                "model": metadata["model_name"],
                "location": metadata["location"],
                "rider_id": rider.rider_id,
                "persona_name": rider.name,
                "total_orders_completed": len(completed),
                "total_earnings": round(rider.total_earnings, 2),
                "total_mileage_km": round(rider.total_mileage_km, 3),
                "on_time_rate": round(on_time_count / len(completed), 4) if completed else 0.0,
                "decision_count": len(rider.decision_log),
            }
        )
        for order in completed:
            order_rows.append(
                {
                    "model": metadata["model_name"],
                    "location": metadata["location"],
                    "order_id": order.order_id,
                    "assigned_rider_id": order.assigned_rider_id,
                    "persona_name": rider.name,
                    "create_time": order.create_time.isoformat(),
                    "promise_deliver_time": order.promise_deliver_time.isoformat(),
                    "finish_time": order.finish_time.isoformat() if order.finish_time else "",
                    "is_on_time": bool(order.finish_time and order.finish_time <= order.promise_deliver_time),
                    "fee": order.fee,
                    "actual_duration_s": round(order.actual_duration_s or 0.0, 2),
                    "actual_distance_m": round(order.actual_distance_m or 0.0, 2),
                    "pickup_wkt": order.pick_loc.wkt,
                    "delivery_wkt": order.deliver_loc.wkt,
                }
            )
        decision_rows.extend(rider.decision_log)

    pd.DataFrame(rider_rows).to_csv(run_dir / "rider_summary.csv", index=False)
    pd.DataFrame(order_rows).to_csv(run_dir / "completed_orders_log.csv", index=False)
    with (run_dir / "decision_log.jsonl").open("w", encoding="utf-8") as f:
        for row in decision_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    platform_on_time_rate = 0.0
    if order_rows:
        platform_on_time_rate = sum(1 for row in order_rows if row["is_on_time"]) / len(order_rows)

    totals = {
        "model": metadata["model_name"],
        "model_id": metadata["model_id"],
        "backend": metadata["backend"],
        "location": metadata["location"],
        "orders_in_window": metadata["orders_in_window"],
        "completed_order_count": sum(row["total_orders_completed"] for row in rider_rows),
        "total_earnings": round(sum(row["total_earnings"] for row in rider_rows), 2),
        "total_mileage_km": round(sum(row["total_mileage_km"] for row in rider_rows), 3),
        "mean_on_time_rate": round(sum(row["on_time_rate"] for row in rider_rows) / len(rider_rows), 4),
        "platform_on_time_rate": round(platform_on_time_rate, 4),
        "decision_count": metadata["decision_count"],
        "api_error_count": metadata["api_error_count"],
        "route_provider": metadata.get("route_provider", ""),
        "amap_mode": metadata.get("amap_mode", ""),
        "route_request_count": metadata.get("route_request_count", 0),
        "route_cache_hit_count": metadata.get("route_cache_hit_count", 0),
        "route_api_error_count": metadata.get("route_api_error_count", 0),
        "elapsed_s": metadata["elapsed_s"],
    }
    return totals


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))
    configs = load_model_suite(Path(args.model_suite))
    personas = load_personas(Path(args.persona_summary))
    orders_path = Path(args.orders)
    service_date = normalize_service_date(args.service_date) or infer_service_date(orders_path)
    orders = load_orders(orders_path, args.start_hour, args.end_hour, service_date)
    args.service_date = service_date

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(args.output_dir) / run_id
    base_output.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for model_name in args.models:
        config = configs.get(model_name)
        if model_name != "heuristic" and config is None:
            raise ValueError(f"Model '{model_name}' not found in {args.model_suite}.")
        for location_name in args.locations:
            if location_name not in START_LOCATIONS:
                raise ValueError(f"Unknown location '{location_name}'. Valid: {sorted(START_LOCATIONS)}")
            location = Point(*START_LOCATIONS[location_name])
            riders, metadata = run_one_simulation(model_name, config, orders, personas, location_name, location, args)
            run_dir = base_output / model_name / location_name
            comparison_rows.append(write_outputs(run_dir, riders, metadata))
            print(json.dumps(comparison_rows[-1], ensure_ascii=False), flush=True)

    pd.DataFrame(comparison_rows).to_csv(base_output / "model_comparison_summary.csv", index=False)
    (base_output / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Wrote simulation outputs to {base_output}", flush=True)


if __name__ == "__main__":
    main()
