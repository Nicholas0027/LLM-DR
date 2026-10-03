#!/usr/bin/env python3
"""Probe OpenAI-compatible model endpoints without running the simulation."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-suite", default="config/model_suite.example.json")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--chat", action="store_true", help="Also send a minimal chat request.")
    parser.add_argument(
        "--response-format",
        action="store_true",
        help="Request OpenAI JSON mode during the chat probe. Some proxy models reject this.",
    )
    parser.add_argument("--timeout", type=int, default=45)
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


def request_json(url: str, api_key: str, timeout: int, payload: dict | None = None) -> tuple[bool, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if payload is None else "POST"
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        return False, {"status": exc.code, "error": detail}
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}


def main() -> None:
    args = parse_args()
    load_env_file(Path(args.env_file))
    suite = json.loads(Path(args.model_suite).read_text(encoding="utf-8"))["models"]

    rows = []
    for name in args.models:
        cfg = suite[name]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            rows.append({"model_name": name, "ok": False, "stage": "env", "error": f"Missing {cfg['api_key_env']}"})
            continue

        base_url = cfg["base_url"].rstrip("/")
        ok_models, models_payload = request_json(f"{base_url}/models", api_key, args.timeout)
        row = {
            "model_name": name,
            "provider": cfg.get("provider", ""),
            "model_id": cfg["model"],
            "base_url": base_url,
            "models_endpoint_ok": ok_models,
        }
        if ok_models and isinstance(models_payload, dict):
            ids = []
            for item in models_payload.get("data", []):
                if isinstance(item, dict) and "id" in item:
                    ids.append(item["id"])
            row["model_id_listed"] = cfg["model"] in ids if ids else None
            row["available_model_count"] = len(ids)
            row["available_model_sample"] = ids[:12]
        else:
            row["models_error"] = models_payload

        if args.chat:
            payload = {
                "model": cfg["model"],
                "messages": [
                    {"role": "system", "content": "Return strict JSON."},
                    {"role": "user", "content": "Return {\"ok\": true, \"label\": \"probe\"} as JSON."},
                ],
                "temperature": 0,
                "max_tokens": 80,
            }
            if args.response_format:
                payload["response_format"] = {"type": "json_object"}
            ok_chat, chat_payload = request_json(f"{base_url}/chat/completions", api_key, args.timeout, payload)
            row["chat_endpoint_ok"] = ok_chat
            if ok_chat and isinstance(chat_payload, dict):
                content = chat_payload.get("choices", [{}])[0].get("message", {}).get("content", "")
                row["chat_content_sample"] = content[:200]
            else:
                row["chat_error"] = chat_payload
        rows.append(row)

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
