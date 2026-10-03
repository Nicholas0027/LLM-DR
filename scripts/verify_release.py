#!/usr/bin/env python3
"""Offline integrity and record checks for the frozen release."""

import ast
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

import persona_clustering_experiment as persona
import run_simulation_experiment as simulation


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "deepseek_v4": 1428, "nearest_baseline": 1461,
    "highest_fee_baseline": 1221, "profit_baseline": 1508, "persona_heuristic": 1047,
}


def main():
    manifest = json.loads((ROOT / "manifest.sha256.json").read_text())
    for name, digest in manifest.items():
        path = ROOT / name
        assert path.is_file(), f"Missing release file: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Changed release file: {name}"
        assert path.stat().st_size < 100_000_000, f"File exceeds normal GitHub upload limit: {name}"
        if path.suffix in (".csv", ".json", ".jsonl", ".tex", ".md", ".py", ".cff"):
            text = path.read_text()
            assert not re.search(r"sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}", text), f"Possible credential: {name}"
            assert ("/" + "Users/") not in text, f"Local host path: {name}"
    env = (ROOT / ".env.example").read_text().splitlines()
    assert all(not line.partition("=")[2].strip() for line in env if line and not line.startswith("#"))

    orders = pd.read_csv(ROOT / "data/empirical/orders.csv")
    filtered = pd.read_csv(ROOT / "data/empirical/orders_filtered.tsv", sep="\t", encoding="utf-16")
    waves = pd.read_csv(ROOT / "data/empirical/waves.csv")
    assignments = pd.read_csv(ROOT / "data/persona/rider_persona_assignments.csv")
    assert (len(orders), len(filtered), len(waves), len(assignments)) == (9199, 9025, 2508, 710)
    assert orders.tracking_id.is_unique and assignments.courier_id.is_unique
    assert filtered.tracking_id.is_unique
    assert set(filtered.tracking_id).issubset(set(orders.tracking_id))
    assert set(filtered.courier_id) == set(assignments.courier_id)
    common = orders.set_index("tracking_id").loc[filtered.tracking_id].reset_index()[filtered.columns]
    pd.testing.assert_frame_equal(filtered.reset_index(drop=True), common, check_dtype=False)
    assert orders.tracking_id.between(6_000_000_000_000_000_001, 6_999_999_999_999_999_999).all()
    selected = simulation.load_orders(ROOT / "data/empirical/orders.csv", 8, 20, "20200215")
    selected_ids = {order.order_id for order in selected}
    assert len(selected) == len(selected_ids) == 8656
    order_ids = set(orders.tracking_id) | set(filtered.tracking_id)
    for row in waves.itertuples():
        ids = [int(value) for value in ast.literal_eval(row.tracking_ids_chronological)]
        actions = ast.literal_eval(row.action_types_chronological)
        times = ast.literal_eval(row.expect_times_chronological)
        assert actions[0] == "ASSIGN"
        assert len(actions) == len(times) == len(ids) + 1
        assert set(actions[1:]).issubset({"PICKUP", "DELIVERY"})
        assert set(ids).issubset(order_ids)
        assert len(set(ids)) >= 2

    calculated = persona.build_features(
        persona.load_wave_data(ROOT / "data/empirical/waves.csv"),
        persona.load_order_data(ROOT / "data/empirical/orders_filtered.tsv"), min_waves=1,
    )
    frozen = pd.read_csv(ROOT / "data/persona/rider_persona_features.csv")
    pd.testing.assert_frame_equal(calculated, frozen, check_dtype=False, rtol=1e-7, atol=1e-7)

    for model, count in MODELS.items():
        folder = ROOT / "reference/simulation" / model
        metadata = json.loads((folder / "run_metadata.json").read_text())
        riders = pd.read_csv(folder / "rider_summary.csv")
        completed = pd.read_csv(folder / "completed_orders_log.csv")
        simulated_waves = pd.read_csv(folder / "wave_summary.csv")
        decisions = [json.loads(line) for line in (folder / "decision_log.jsonl").read_text().splitlines()]
        assert len(riders) == metadata["rider_count"] == 60
        assert (riders.groupby("persona_name").size() == 15).all()
        assert len(completed) == riders.total_orders_completed.sum() == metadata["completed_order_count"] == count
        assert completed.order_id.is_unique and set(completed.order_id).issubset(selected_ids)
        assert len(decisions) == metadata["decision_count"] == riders.decision_count.sum()
        assert simulated_waves.order_count.sum() == count
        assert set(completed.wave_id) == set(simulated_waves.wave_id)
        assert metadata["orders_in_window"] == count + metadata["unassigned_order_count"] == 8656
        assert metadata["max_decisions"] == metadata["max_rider_decisions_per_step"] == 0
        assert metadata["route_provider"] == "amap" and metadata["amap_mode"] == "bicycling"
        for decision in decisions:
            assert set(decision["available_orders"]).issubset(selected_ids)
            assert set(decision["accepted_order_ids"]).issubset(decision["available_orders"])

    config = json.loads((ROOT / "config/main_experiment.json").read_text())
    assert config["max_decisions"] == config["max_rider_decisions_per_step"] == 0
    assert config["route_strict"] and config["route_tool_in_prompt"]
    assert config["riders_per_persona"] == 15 and config["route_provider"] == "amap"
    print(f"Verified {len(manifest)} release files, empirical features, and five frozen 60-rider runs.")


if __name__ == "__main__":
    main()
