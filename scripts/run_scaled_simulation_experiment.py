#!/usr/bin/env python3
"""Run scaled LLM-DR simulations with demand-density rider initialisation.

This runner extends ``run_simulation_experiment.py`` from a four-agent proof of
concept to a small fleet simulation. It keeps the same order representation,
persona profiles, LLM client, and routing tool, but adds:

* 15 riders per persona by default;
* demand-density initial locations sampled from observed pickup points;
* platform dispatch candidates and conflict resolution;
* additional rule-based baselines;
* wave-level outputs for empirical validation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_simulation_experiment as base  # noqa: E402


LOCAL_BASELINES = {
    "nearest_baseline",
    "highest_fee_baseline",
    "profit_baseline",
    "persona_heuristic",
    "heuristic",
}


@dataclass
class Bid:
    rider: base.Rider
    orders: list[base.Order]
    result: base.DecisionResult
    prompt: str
    offers: list[base.Order]

    @property
    def platform_distance_km(self) -> float:
        if not self.orders:
            return float("inf")
        return sum(base.haversine_km(self.rider.location, order.pick_loc) for order in self.orders) / len(self.orders)


class RuleClient:
    def __init__(self, strategy: str):
        self.strategy = "persona_heuristic" if strategy == "heuristic" else strategy
        self.persona_client = base.HeuristicClient(strategy)

    def decide_for_rider(self, rider: base.Rider, offers: list[base.Order], current_time: pd.Timestamp) -> base.DecisionResult:
        if self.strategy == "persona_heuristic":
            return self.persona_client.decide_for_rider(rider, offers, current_time)

        if not offers:
            return base.DecisionResult([], ["Decision: No offers available."], ok=True)

        if self.strategy == "nearest_baseline":
            chosen = min(offers, key=lambda order: base.haversine_km(rider.location, order.pick_loc))
            reason = f"Decision: Accept nearest pickup order {chosen.order_id}."
            return base.DecisionResult([chosen.order_id], [reason], ok=True)

        if self.strategy == "highest_fee_baseline":
            chosen = max(offers, key=lambda order: (order.fee, -base.haversine_km(rider.location, order.pick_loc)))
            reason = f"Decision: Accept highest-fee order {chosen.order_id}."
            return base.DecisionResult([chosen.order_id], [reason], ok=True)

        if self.strategy == "profit_baseline":
            scored: list[tuple[float, base.Order]] = []
            for order in offers:
                to_pick = base.haversine_km(rider.location, order.pick_loc)
                delivery = base.haversine_km(order.pick_loc, order.deliver_loc)
                score = order.fee / max(to_pick + delivery, 0.2)
                scored.append((score, order))
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen_orders = [order for score, order in scored[:2] if score >= 1.5]
            if not chosen_orders:
                chosen_orders = [scored[0][1]]
            ids = [order.order_id for order in chosen_orders]
            reason = f"Decision: Accept orders {ids} with the highest fee-per-distance score."
            return base.DecisionResult(ids, [reason], ok=True)

        raise ValueError(f"Unknown rule strategy: {self.strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="0_persona/data/order_20200215.txt")
    parser.add_argument("--service-date", default=None)
    parser.add_argument("--persona-summary", default="outputs/persona/persona_cluster_summary.csv")
    parser.add_argument("--model-suite", default="config/model_suite.example.json")
    parser.add_argument("--models", nargs="+", default=["persona_heuristic"])
    parser.add_argument("--output-dir", default="outputs/scaled_simulation")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--start-hour", type=int, default=8)
    parser.add_argument("--end-hour", type=int, default=20)
    parser.add_argument("--time-step-minutes", type=int, default=10)
    parser.add_argument("--riders-per-persona", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--initial-jitter-km", type=float, default=0.45)
    parser.add_argument("--broadcast-radius-km", type=float, default=2.0)
    parser.add_argument("--dispatch-candidates-per-order", type=int, default=3)
    parser.add_argument("--pending-horizon-minutes", type=int, default=30)
    parser.add_argument("--max-rider-decisions-per-step", type=int, default=12)
    parser.add_argument("--max-offers-per-rider", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=240, help="0 means unlimited.")
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
    parser.add_argument("--dry-run", action="store_true", help="Force persona heuristic backend for named LLM models.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N rider decisions. 0 disables progress logging.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Write partial checkpoints every N rider decisions. 0 disables checkpoints.")
    parser.add_argument("--checkpoint-keep-all", action="store_true", help="Also keep numbered checkpoint snapshots.")
    return parser.parse_args()


def clone_orders(orders: list[base.Order]) -> list[base.Order]:
    return [
        base.Order(
            order.order_id,
            order.create_time,
            order.promise_deliver_time,
            base.Point(order.pick_loc.lon, order.pick_loc.lat),
            base.Point(order.deliver_loc.lon, order.deliver_loc.lat),
            order.fee,
        )
        for order in orders
    ]


def jitter_point(point: base.Point, rng: random.Random, jitter_km: float) -> base.Point:
    if jitter_km <= 0:
        return base.Point(point.lon, point.lat)
    dy_km = rng.gauss(0.0, jitter_km)
    dx_km = rng.gauss(0.0, jitter_km)
    lat = point.lat + dy_km / 111.0
    lon_scale = max(111.0 * math.cos(math.radians(point.lat)), 1.0)
    lon = point.lon + dx_km / lon_scale
    return base.Point(lon, lat)


def create_density_initialised_riders(
    personas: list[base.PersonaProfile],
    orders: list[base.Order],
    riders_per_persona: int,
    seed: int,
    jitter_km: float,
) -> list[base.Rider]:
    rng = random.Random(seed)
    demand_points = [order.pick_loc for order in orders]
    riders: list[base.Rider] = []
    rider_id = 1
    for persona in personas:
        for replica_id in range(1, riders_per_persona + 1):
            sampled = rng.choice(demand_points)
            initial = jitter_point(sampled, rng, jitter_km)
            rider = base.Rider(rider_id, persona, base.Point(initial.lon, initial.lat))
            rider.initial_location = base.Point(initial.lon, initial.lat)  # type: ignore[attr-defined]
            rider.replica_id = replica_id  # type: ignore[attr-defined]
            riders.append(rider)
            rider_id += 1
    return riders


def build_dispatch_offers(
    riders: list[base.Rider],
    orders: list[base.Order],
    current_time: pd.Timestamp,
    time_step: pd.Timedelta,
    broadcast_radius_km: float,
    dispatch_candidates_per_order: int,
    pending_horizon_minutes: int,
) -> dict[int, list[base.Order]]:
    idle_riders = [rider for rider in riders if rider.status == "IDLE"]
    if not idle_riders:
        return {}

    horizon_start = current_time - pd.Timedelta(minutes=pending_horizon_minutes)
    visible_orders = [
        order
        for order in orders
        if order.status == "PENDING"
        and horizon_start <= order.create_time < current_time + time_step
        and order.promise_deliver_time > current_time
    ]

    offers_by_rider: dict[int, list[base.Order]] = defaultdict(list)
    for order in visible_orders:
        candidates = [
            (base.haversine_km(rider.location, order.pick_loc), rider)
            for rider in idle_riders
            if base.haversine_km(rider.location, order.pick_loc) <= broadcast_radius_km
        ]
        candidates.sort(key=lambda item: (item[0], item[1].rider_id))
        for _, rider in candidates[:dispatch_candidates_per_order]:
            offers_by_rider[rider.rider_id].append(order)
    return offers_by_rider


def select_decision_riders(
    riders: list[base.Rider],
    offers_by_rider: dict[int, list[base.Order]],
    max_rider_decisions_per_step: int,
) -> list[base.Rider]:
    rider_by_id = {rider.rider_id: rider for rider in riders}
    candidates = []
    for rider_id, offers in offers_by_rider.items():
        rider = rider_by_id[rider_id]
        min_distance = min(base.haversine_km(rider.location, order.pick_loc) for order in offers)
        earliest_deadline = min(order.promise_deliver_time for order in offers)
        candidates.append((earliest_deadline, min_distance, -len(offers), rider.rider_id, rider))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    selected = [item[-1] for item in candidates]
    if max_rider_decisions_per_step > 0:
        selected = selected[:max_rider_decisions_per_step]
    return selected


def sorted_offers(rider: base.Rider, offers: list[base.Order], max_offers: int) -> list[base.Order]:
    unique = {order.order_id: order for order in offers}
    ordered = sorted(
        unique.values(),
        key=lambda order: (
            order.promise_deliver_time,
            base.haversine_km(rider.location, order.pick_loc),
            -order.fee,
        ),
    )
    return ordered[:max_offers]


def client_for_model(
    model_name: str,
    config: base.ModelConfig | None,
    args: argparse.Namespace,
) -> tuple[Any, str]:
    if args.dry_run and model_name not in LOCAL_BASELINES:
        return RuleClient("persona_heuristic"), "heuristic"
    if model_name in LOCAL_BASELINES:
        return RuleClient(model_name), "heuristic"
    if config is None:
        raise ValueError(f"Model '{model_name}' not found in {args.model_suite}.")
    return base.ChatModelClient(config, timeout=args.request_timeout, retries=args.retries), "llm"


def decide_with_client(
    client: Any,
    backend: str,
    rider: base.Rider,
    offers: list[base.Order],
    current_time: pd.Timestamp,
    route_planner: base.RoutePlanner | None = None,
) -> tuple[base.DecisionResult, str]:
    if backend == "heuristic":
        return client.decide_for_rider(rider, offers, current_time), ""
    prompt = base.build_prompt(rider, offers, current_time, route_planner)
    return client.decide(prompt), prompt


def resolve_bids_and_assign(
    bids: list[Bid],
    current_time: pd.Timestamp,
    args: argparse.Namespace,
    route_planner: base.RoutePlanner,
    wave_rows: list[dict[str, Any]],
    order_wave_ids: dict[int, str],
    model_name: str,
) -> int:
    assigned_count = 0
    assigned_order_ids: set[int] = set()
    wave_index = len(wave_rows) + 1
    bids = sorted(bids, key=lambda bid: (bid.platform_distance_km, bid.rider.rider_id))
    for bid in bids:
        accepted = [order for order in bid.orders if order.status == "PENDING" and order.order_id not in assigned_order_ids]
        if not accepted:
            continue
        base.assign_orders(
            bid.rider,
            accepted,
            current_time,
            route_planner,
            args.service_seconds_per_order,
        )
        wave_id = f"{model_name}_w{wave_index:06d}"
        wave_index += 1
        for order in accepted:
            assigned_order_ids.add(order.order_id)
            order_wave_ids[order.order_id] = wave_id
        assigned_count += len(accepted)
        wave_rows.append(
            {
                "wave_id": wave_id,
                "model": model_name,
                "rider_id": bid.rider.rider_id,
                "persona_name": bid.rider.name,
                "start_time": current_time.isoformat(),
                "finish_time": bid.rider.task_finish_time.isoformat() if bid.rider.task_finish_time else "",
                "order_count": len(accepted),
                "order_ids": "|".join(str(order.order_id) for order in accepted),
                "total_fee": round(sum(order.fee for order in accepted), 2),
                "route_distance_m": round(bid.rider.task_distance_m, 2),
                "route_duration_s": round(bid.rider.task_duration_s, 2),
                "route_provider": bid.rider.task_route_provider,
                "route_polyline": bid.rider.task_route_polyline,
                "route_api_errors": bid.rider.task_route_api_errors,
                "route_cache_hits": bid.rider.task_route_cache_hits,
                "all_on_time": all(order.finish_time and order.finish_time <= order.promise_deliver_time for order in accepted),
                "any_late": any(order.finish_time and order.finish_time > order.promise_deliver_time for order in accepted),
                "start_lon": bid.rider.location.lon,
                "start_lat": bid.rider.location.lat,
                "pickup_lons": "|".join(f"{order.pick_loc.lon:.6f}" for order in accepted),
                "pickup_lats": "|".join(f"{order.pick_loc.lat:.6f}" for order in accepted),
                "delivery_lons": "|".join(f"{order.deliver_loc.lon:.6f}" for order in accepted),
                "delivery_lats": "|".join(f"{order.deliver_loc.lat:.6f}" for order in accepted),
            }
        )
    return assigned_count


def finish_remaining_tasks(riders: list[base.Rider]) -> int:
    completed = 0
    while any(rider.status == "DELIVERING" for rider in riders):
        next_finish = min(rider.task_finish_time for rider in riders if rider.task_finish_time is not None)
        completed += base.finish_ready_tasks(riders, next_finish)
    return completed


def _iso_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def checkpoint_rows(
    riders: list[base.Rider],
    orders: list[base.Order],
    wave_rows: list[dict[str, Any]],
    order_wave_ids: dict[int, str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rider_rows = []
    completed_order_rows = []
    order_state_rows = []
    decision_rows: list[dict[str, Any]] = []
    for rider in riders:
        completed = rider.order_history
        on_time_count = sum(1 for order in completed if order.finish_time and order.finish_time <= order.promise_deliver_time)
        initial = getattr(rider, "initial_location", rider.location)
        rider_rows.append(
            {
                "model": model_name,
                "rider_id": rider.rider_id,
                "replica_id": getattr(rider, "replica_id", ""),
                "persona_name": rider.name,
                "status": rider.status,
                "current_lon": rider.location.lon,
                "current_lat": rider.location.lat,
                "initial_lon": initial.lon,
                "initial_lat": initial.lat,
                "task_finish_time": _iso_time(rider.task_finish_time),
                "current_order_ids": "|".join(str(order.order_id) for order in rider.current_orders),
                "completed_order_ids": "|".join(str(order.order_id) for order in completed),
                "total_orders_completed": len(completed),
                "total_earnings": round(rider.total_earnings, 2),
                "total_mileage_km": round(rider.total_mileage_km, 3),
                "on_time_rate": round(on_time_count / len(completed), 4) if completed else 0.0,
                "decision_count": len(rider.decision_log),
            }
        )
        for order in completed:
            completed_order_rows.append(
                {
                    "model": model_name,
                    "wave_id": order_wave_ids.get(order.order_id, ""),
                    "order_id": order.order_id,
                    "assigned_rider_id": order.assigned_rider_id,
                    "persona_name": rider.name,
                    "create_time": _iso_time(order.create_time),
                    "promise_deliver_time": _iso_time(order.promise_deliver_time),
                    "finish_time": _iso_time(order.finish_time),
                    "is_on_time": bool(order.finish_time and order.finish_time <= order.promise_deliver_time),
                    "fee": order.fee,
                    "actual_duration_s": round(order.actual_duration_s or 0.0, 2),
                    "actual_distance_m": round(order.actual_distance_m or 0.0, 2),
                    "pickup_lon": order.pick_loc.lon,
                    "pickup_lat": order.pick_loc.lat,
                    "delivery_lon": order.deliver_loc.lon,
                    "delivery_lat": order.deliver_loc.lat,
                }
            )
        decision_rows.extend(rider.decision_log)

    for order in orders:
        order_state_rows.append(
            {
                "model": model_name,
                "wave_id": order_wave_ids.get(order.order_id, ""),
                "order_id": order.order_id,
                "status": order.status,
                "assigned_rider_id": order.assigned_rider_id,
                "create_time": _iso_time(order.create_time),
                "promise_deliver_time": _iso_time(order.promise_deliver_time),
                "finish_time": _iso_time(order.finish_time),
                "fee": order.fee,
                "actual_duration_s": round(order.actual_duration_s or 0.0, 2),
                "actual_distance_m": round(order.actual_distance_m or 0.0, 2),
                "pickup_lon": order.pick_loc.lon,
                "pickup_lat": order.pick_loc.lat,
                "delivery_lon": order.deliver_loc.lon,
                "delivery_lat": order.deliver_loc.lat,
            }
        )

    return (
        pd.DataFrame(rider_rows),
        pd.DataFrame(completed_order_rows),
        pd.DataFrame(order_state_rows),
        pd.DataFrame(wave_rows),
        decision_rows,
    )


def write_checkpoint(
    checkpoint_root: Path,
    model_name: str,
    riders: list[base.Rider],
    orders: list[base.Order],
    wave_rows: list[dict[str, Any]],
    order_wave_ids: dict[int, str],
    metadata: dict[str, Any],
    keep_all: bool,
) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = checkpoint_root / "latest.tmp"
    latest_dir = checkpoint_root / "latest"
    previous_dir = checkpoint_root / "previous"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    rider_rows, completed_order_rows, order_state_rows, wave_frame, decision_rows = checkpoint_rows(
        riders, orders, wave_rows, order_wave_ids, model_name
    )
    rider_rows.to_csv(tmp_dir / "rider_state.csv", index=False)
    completed_order_rows.to_csv(tmp_dir / "completed_orders_log.csv", index=False)
    order_state_rows.to_csv(tmp_dir / "order_state.csv", index=False)
    wave_frame.to_csv(tmp_dir / "wave_summary.csv", index=False)
    with (tmp_dir / "decision_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (tmp_dir / "partial_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if previous_dir.exists():
        shutil.rmtree(previous_dir)
    if latest_dir.exists():
        latest_dir.rename(previous_dir)
    tmp_dir.rename(latest_dir)

    if keep_all:
        checkpoint_name = f"checkpoint_{int(metadata.get('decision_count', 0)):06d}"
        snapshot_dir = checkpoint_root / checkpoint_name
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        shutil.copytree(latest_dir, snapshot_dir)


def run_one_scaled_simulation(
    model_name: str,
    model_config: base.ModelConfig | None,
    source_orders: list[base.Order],
    personas: list[base.PersonaProfile],
    args: argparse.Namespace,
) -> tuple[list[base.Rider], list[dict[str, Any]], dict[int, str], dict[str, Any]]:
    orders = clone_orders(source_orders)
    riders = create_density_initialised_riders(
        personas,
        orders,
        riders_per_persona=args.riders_per_persona,
        seed=args.seed,
        jitter_km=args.initial_jitter_km,
    )
    route_planner = base.RoutePlanner.from_args(args)
    client, backend = client_for_model(model_name, model_config, args)

    current_time = orders[0].create_time.floor(f"{args.time_step_minutes}min")
    end_time = orders[-1].create_time.ceil(f"{args.time_step_minutes}min")
    time_step = pd.Timedelta(minutes=args.time_step_minutes)
    decision_count = 0
    api_error_count = 0
    completed_count = 0
    wave_rows: list[dict[str, Any]] = []
    order_wave_ids: dict[int, str] = {}
    checkpoint_root_raw = getattr(args, "current_checkpoint_dir", "")
    checkpoint_root = Path(checkpoint_root_raw) if checkpoint_root_raw else None
    last_checkpoint_mark = 0
    started = time.time()

    while current_time <= end_time:
        completed_count += base.finish_ready_tasks(riders, current_time)
        offers_by_rider = build_dispatch_offers(
            riders,
            orders,
            current_time,
            time_step,
            args.broadcast_radius_km,
            args.dispatch_candidates_per_order,
            args.pending_horizon_minutes,
        )
        decision_riders = select_decision_riders(riders, offers_by_rider, args.max_rider_decisions_per_step)
        bids: list[Bid] = []

        for rider in decision_riders:
            if args.max_decisions > 0 and decision_count >= args.max_decisions:
                break
            offers = sorted_offers(rider, offers_by_rider[rider.rider_id], args.max_offers_per_rider)
            if not offers:
                continue
            prompt_route_planner = route_planner if args.route_tool_in_prompt else None
            result, prompt = decide_with_client(client, backend, rider, offers, current_time, prompt_route_planner)
            decision_count += 1
            if args.progress_every > 0 and decision_count % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "model": model_name,
                            "current_time": current_time.isoformat(),
                            "decision_count": decision_count,
                            "completed_so_far": completed_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if not result.ok:
                api_error_count += 1
            accepted_ids = base.validate_decision(result, offers)
            accepted_orders = [order for order in offers if order.order_id in accepted_ids]
            rider.decision_log.append(
                {
                    "timestamp": current_time.isoformat(),
                    "model": model_name,
                    "backend": backend,
                    "rider_id": rider.rider_id,
                    "replica_id": getattr(rider, "replica_id", ""),
                    "persona_name": rider.name,
                    "available_orders": [order.order_id for order in offers],
                    "accepted_order_ids": accepted_ids,
                    "ok": result.ok,
                    "error": result.error,
                    "latency_s": round(result.latency_s, 3),
                    "prompt_chars": len(prompt),
                    "raw_content": result.raw_content,
                    "decision_json": {
                        "accepted_order_ids": accepted_ids,
                        "reasoning_steps": result.reasoning_steps,
                    },
                }
            )
            if accepted_orders:
                bids.append(Bid(rider, accepted_orders, result, prompt, offers))

        resolve_bids_and_assign(bids, current_time, args, route_planner, wave_rows, order_wave_ids, model_name)
        if (
            checkpoint_root is not None
            and args.checkpoint_every > 0
            and decision_count > 0
            and decision_count // args.checkpoint_every > last_checkpoint_mark
        ):
            last_checkpoint_mark = decision_count // args.checkpoint_every
            checkpoint_metadata = {
                "event": "checkpoint",
                "model_name": model_name,
                "model_id": model_config.model if model_config else model_name,
                "backend": backend,
                "current_time": current_time.isoformat(),
                "end_time": end_time.isoformat(),
                "orders_in_window": len(orders),
                "rider_count": len(riders),
                "completed_order_count_so_far": completed_count,
                "assigned_wave_count_so_far": len(wave_rows),
                "decision_count": decision_count,
                "api_error_count": api_error_count,
                "elapsed_s": round(time.time() - started, 3),
                "route_provider": args.route_provider,
                "amap_mode": args.amap_mode,
                "route_request_count": route_planner.request_count,
                "route_cache_hit_count": route_planner.cache_hit_count,
                "route_api_error_count": route_planner.api_error_count,
            }
            write_checkpoint(
                checkpoint_root,
                model_name,
                riders,
                orders,
                wave_rows,
                order_wave_ids,
                checkpoint_metadata,
                keep_all=args.checkpoint_keep_all,
            )
            print(json.dumps(checkpoint_metadata, ensure_ascii=False), flush=True)
        if args.max_decisions > 0 and decision_count >= args.max_decisions:
            break
        current_time += time_step

    completed_count += finish_remaining_tasks(riders)
    unassigned_count = sum(1 for order in orders if order.status == "PENDING")

    metadata = {
        "model_name": model_name,
        "model_id": model_config.model if model_config else model_name,
        "backend": backend,
        "orders_in_window": len(orders),
        "rider_count": len(riders),
        "riders_per_persona": args.riders_per_persona,
        "completed_order_count": completed_count,
        "unassigned_order_count": unassigned_count,
        "decision_count": decision_count,
        "api_error_count": api_error_count,
        "elapsed_s": round(time.time() - started, 3),
        "start_hour": args.start_hour,
        "end_hour": args.end_hour,
        "time_step_minutes": args.time_step_minutes,
        "broadcast_radius_km": args.broadcast_radius_km,
        "dispatch_candidates_per_order": args.dispatch_candidates_per_order,
        "max_rider_decisions_per_step": args.max_rider_decisions_per_step,
        "max_offers_per_rider": args.max_offers_per_rider,
        "max_decisions": args.max_decisions,
        "seed": args.seed,
        "route_provider": args.route_provider,
        "amap_mode": args.amap_mode,
        "route_request_count": route_planner.request_count,
        "route_cache_hit_count": route_planner.cache_hit_count,
        "route_api_error_count": route_planner.api_error_count,
    }
    if checkpoint_root is not None and args.checkpoint_every > 0:
        final_checkpoint_metadata = {
            "event": "checkpoint_final",
            **metadata,
        }
        write_checkpoint(
            checkpoint_root,
            model_name,
            riders,
            orders,
            wave_rows,
            order_wave_ids,
            final_checkpoint_metadata,
            keep_all=args.checkpoint_keep_all,
        )
    return riders, wave_rows, order_wave_ids, metadata


def write_scaled_outputs(
    run_dir: Path,
    riders: list[base.Rider],
    wave_rows: list[dict[str, Any]],
    order_wave_ids: dict[int, str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rider_rows = []
    order_rows = []
    decision_rows = []
    for rider in riders:
        completed = rider.order_history
        on_time_count = sum(1 for order in completed if order.finish_time and order.finish_time <= order.promise_deliver_time)
        initial = getattr(rider, "initial_location", rider.location)
        rider_rows.append(
            {
                "model": metadata["model_name"],
                "rider_id": rider.rider_id,
                "replica_id": getattr(rider, "replica_id", ""),
                "persona_name": rider.name,
                "initial_lon": initial.lon,
                "initial_lat": initial.lat,
                "final_lon": rider.location.lon,
                "final_lat": rider.location.lat,
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
                    "wave_id": order_wave_ids.get(order.order_id, ""),
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
                    "pickup_lon": order.pick_loc.lon,
                    "pickup_lat": order.pick_loc.lat,
                    "delivery_lon": order.deliver_loc.lon,
                    "delivery_lat": order.deliver_loc.lat,
                }
            )
        decision_rows.extend(rider.decision_log)

    pd.DataFrame(rider_rows).to_csv(run_dir / "rider_summary.csv", index=False)
    pd.DataFrame(order_rows).to_csv(run_dir / "completed_orders_log.csv", index=False)
    pd.DataFrame(wave_rows).to_csv(run_dir / "wave_summary.csv", index=False)
    with (run_dir / "decision_log.jsonl").open("w", encoding="utf-8") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    platform_on_time_rate = 0.0
    if order_rows:
        platform_on_time_rate = sum(1 for row in order_rows if row["is_on_time"]) / len(order_rows)
    totals = {
        "model": metadata["model_name"],
        "model_id": metadata["model_id"],
        "backend": metadata["backend"],
        "orders_in_window": metadata["orders_in_window"],
        "rider_count": metadata["rider_count"],
        "completed_order_count": sum(row["total_orders_completed"] for row in rider_rows),
        "unassigned_order_count": metadata["unassigned_order_count"],
        "total_earnings": round(sum(row["total_earnings"] for row in rider_rows), 2),
        "total_mileage_km": round(sum(row["total_mileage_km"] for row in rider_rows), 3),
        "mean_on_time_rate": round(sum(row["on_time_rate"] for row in rider_rows) / len(rider_rows), 4),
        "platform_on_time_rate": round(platform_on_time_rate, 4),
        "decision_count": metadata["decision_count"],
        "api_error_count": metadata["api_error_count"],
        "api_error_rate": round(metadata["api_error_count"] / metadata["decision_count"], 4)
        if metadata["decision_count"]
        else 0.0,
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
    base.load_env_file(Path(args.env_file))
    configs = base.load_model_suite(Path(args.model_suite))
    personas = base.load_personas(Path(args.persona_summary))
    orders_path = Path(args.orders)
    service_date = base.normalize_service_date(args.service_date) or base.infer_service_date(orders_path)
    orders = base.load_orders(orders_path, args.start_hour, args.end_hour, service_date)
    if not orders:
        raise ValueError("No orders available for the selected date and hour window.")
    args.service_date = service_date

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id
    base_output = Path(args.output_dir) / run_id
    base_output.mkdir(parents=True, exist_ok=True)
    (base_output / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    comparison_rows = []
    for model_name in args.models:
        config = configs.get(model_name)
        if model_name not in LOCAL_BASELINES and not args.dry_run and config is None:
            raise ValueError(f"Model '{model_name}' not found in {args.model_suite}.")
        args.current_checkpoint_dir = str(base_output / model_name / "checkpoints")
        riders, wave_rows, order_wave_ids, metadata = run_one_scaled_simulation(model_name, config, orders, personas, args)
        run_dir = base_output / model_name
        comparison_rows.append(write_scaled_outputs(run_dir, riders, wave_rows, order_wave_ids, metadata))
        print(json.dumps(comparison_rows[-1], ensure_ascii=False), flush=True)

    pd.DataFrame(comparison_rows).to_csv(base_output / "model_comparison_summary.csv", index=False)
    (base_output / "experiment_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"Wrote scaled simulation outputs to {base_output}", flush=True)


if __name__ == "__main__":
    main()
