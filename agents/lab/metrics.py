# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Shared JSON state for local lab dashboard and experiment runner."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path.home() / ".taos" / "lab"
_LOCK = threading.Lock()


def lab_dir() -> Path:
    d = Path(os.environ.get("TAOS_LAB_DIR", _DEFAULT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path() -> Path:
    return lab_dir() / "state.json"


def _default_state() -> dict[str, Any]:
    return {
        "updated_at": 0.0,
        "phase": "idle",
        "profile": "lite",
        "host": {},
        "current_trial": None,
        "live": {
            "sim_timestamp": 0,
            "proxy_step": 0,
            "respond_ms": 0.0,
            "orders_last_tick": 0,
            "cancels_last_tick": 0,
            "books_active": 0,
            "score_median": 0.0,
            "round_trips": 0,
            "per_book": {},
        },
        "trials": [],
        "best_trial": None,
        "log_tail": [],
    }


def load_state() -> dict[str, Any]:
    path = state_path()
    if not path.exists():
        return _default_state()
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return _default_state()


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    path = state_path()
    tmp = path.with_suffix(".tmp")
    with _LOCK:
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(path)


def patch_state(**sections: Any) -> dict[str, Any]:
    state = load_state()
    for key, val in sections.items():
        if isinstance(val, dict) and isinstance(state.get(key), dict):
            state[key].update(val)
        else:
            state[key] = val
    save_state(state)
    return state


def append_log(line: str, max_lines: int = 80) -> None:
    state = load_state()
    tail = state.setdefault("log_tail", [])
    tail.append(f"{time.strftime('%H:%M:%S')} {line}")
    state["log_tail"] = tail[-max_lines:]
    save_state(state)


def publish_proxy_tick(
    *,
    step: int,
    sim_timestamp: int,
    respond_ms: float,
    orders: int,
    cancels: int,
) -> None:
    live = load_state().get("live", {})
    live.update(
        {
            "proxy_step": step,
            "sim_timestamp": sim_timestamp,
            "respond_ms": respond_ms,
            "orders_last_tick": orders,
            "cancels_last_tick": cancels,
        }
    )
    patch_state(live=live, phase="running")


def publish_agent_metrics(
    agent: Any,
    ctx: Any,
    *,
    orders: int,
    cancels: int,
) -> None:
    per_book: dict[str, dict] = {}
    round_trips = 0
    validator = ctx.validator
    for book_id in ctx.plans.keys():
        fp = agent._fingerprint(validator, book_id)
        tier = ctx.tiers.get(book_id)
        tier_s = tier.name if hasattr(tier, "name") else str(tier)
        per_book[str(book_id)] = {
            "tier": tier_s,
            "score_proxy": round(fp.score_proxy, 4),
            "position": round(fp.signed_position, 4),
            "round_trips": len(fp.round_trip_pnls),
        }
        round_trips += len(fp.round_trip_pnls)

    fps = agent.fingerprints.get(validator, {})
    proxies = [fp.score_proxy for fp in fps.values()]
    score_median = float(sum(proxies) / len(proxies)) if proxies else 0.0
    warm = agent._warm_snapshots.get(validator, {})
    if "score_median" in warm:
        score_median = float(warm["score_median"])

    live = {
        "sim_timestamp": ctx.timestamp,
        "books_active": len(ctx.plans),
        "score_median": round(score_median, 4),
        "round_trips": round_trips,
        "orders_last_tick": orders,
        "cancels_last_tick": cancels,
        "per_book": per_book,
    }
    patch_state(live=live)


def record_trial_result(trial: dict[str, Any]) -> None:
    state = load_state()
    trials = state.setdefault("trials", [])
    trials.append(trial)
    state["trials"] = trials[-50:]
    best = state.get("best_trial")
    if best is None or trial.get("score", 0) > best.get("score", 0):
        state["best_trial"] = trial
    save_state(state)
