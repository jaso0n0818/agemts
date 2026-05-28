#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Offline + smoke tests for MedianAlignedTierAgent (no C++ simulator required)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from types import SimpleNamespace

from pathlib import Path

SN79_ROOT = str(Path(__file__).resolve().parents[1])

sys.path.insert(0, f"{SN79_ROOT}/agents")

from MedianAlignedTierAgent import (  # noqa: E402
    BookFingerprint,
    BookTier,
    MedianAlignedTierAgent,
)


def make_agent() -> MedianAlignedTierAgent:
    import os

    log_dir = "/tmp/median_tier_experiment"
    os.makedirs(log_dir, exist_ok=True)
    config = SimpleNamespace(
        quantity=0.25,
        max_quantity=0.75,
        expiry_period=30_000_000_000,
        alpha_fraction=0.2,
        defensive_fraction=0.2,
        lazy_load=False,
        data_dir="/tmp/median_tier_experiment/data",
    )
    agent = MedianAlignedTierAgent(0, config, log_dir=log_dir)
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        priceDecimals=2,
        volumeDecimals=4,
        publish_interval=1_000_000_000,
    )
    return agent


def test_tier_assignment(agent: MedianAlignedTierAgent) -> dict:
    validator = "test_validator"
    book_ids = list(range(16))

    for i, book_id in enumerate(book_ids):
        fp = agent._fingerprint(validator, book_id)
        if i < 3:
            fp.round_trip_pnls.extend([50.0, 40.0, 45.0])
            fp.score_proxy = 0.85
        elif i < 13:
            fp.round_trip_pnls.extend([1.0, -0.5, 0.2])
            fp.score_proxy = 0.52
        else:
            fp.round_trip_pnls.extend([-8.0, -5.0, -3.0])
            fp.score_proxy = 0.28

    state = SimpleNamespace(
        accounts={
            0: {
                i: SimpleNamespace(traded_volume=100.0, own_base=0.0, own_quote=30_000.0)
                for i in book_ids
            }
        },
        books={
            i: SimpleNamespace(
                bids=[SimpleNamespace(price=300.0)],
                asks=[SimpleNamespace(price=301.0)],
            )
            for i in book_ids
        },
    )
    agent.accounts = state.accounts[0]
    agent.uid = 0
    net = agent._build_network_context(state, validator, book_ids)
    tiers = agent._assign_tiers(validator, book_ids, net, state)
    counts = {t.value: 0 for t in BookTier}
    for t in tiers.values():
        counts[t.value] += 1

    alpha_ids = [b for b, t in tiers.items() if t == BookTier.ALPHA]
    defensive_ids = [b for b, t in tiers.items() if t == BookTier.DEFENSIVE]
    return {
        "book_count": len(book_ids),
        "tier_counts": counts,
        "alpha_books": alpha_ids,
        "defensive_books": defensive_ids,
        "max_alpha_cap": max(1, int(len(book_ids) * agent.alpha_fraction)),
        "max_defensive_cap": max(1, int(len(book_ids) * agent.defensive_fraction)),
    }


def test_score_proxy_update(agent: MedianAlignedTierAgent) -> dict:
    fp = BookFingerprint()
    fp.round_trip_pnls.extend([100.0, 50.0])
    agent._update_score_proxy(fp)
    high = fp.score_proxy

    fp2 = BookFingerprint()
    fp2.round_trip_pnls.extend([-50.0, -30.0])
    agent._update_score_proxy(fp2)
    low = fp2.score_proxy

    return {"high_pnl_proxy": round(high, 4), "low_pnl_proxy": round(low, 4), "ordered": high > low}


def test_agent_http(port: int = 18888, wait_s: float = 6.0) -> dict:
    python = f"{SN79_ROOT}/.venv/bin/python"
    agent_script = f"{SN79_ROOT}/agents/MedianAlignedTierAgent.py"
    proc = subprocess.Popen(
        [
            python,
            agent_script,
            "--port",
            str(port),
            "--agent_id",
            "0",
            "--params",
            "quantity=0.25",
            "max_quantity=0.75",
            "expiry_period=30000000000",
            f"data_dir=/tmp/median_tier_agent_{port}/data",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=f"{SN79_ROOT}/agents",
    )
    result = {"port": port, "started": False, "openapi_ok": False, "exit_code": None}
    try:
        time.sleep(wait_s)
        if proc.poll() is None:
            result["started"] = True
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=3) as resp:
                    data = json.loads(resp.read().decode())
                    result["openapi_ok"] = "/handle" in str(data.get("paths", {}))
            except (urllib.error.URLError, TimeoutError) as exc:
                result["http_error"] = str(exc)
        else:
            out, _ = proc.communicate(timeout=2)
            result["early_exit_output"] = (out or "")[-2000:]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        result["exit_code"] = proc.returncode
    return result


def check_simulator() -> dict:
    candidates = [
        f"{SN79_ROOT}/simulate/trading/run/build/src/cpp/taosim",
        f"{SN79_ROOT}/simulate/trading/build/src/cpp/taosim",
    ]
    found = [p for p in candidates if __import__("os").path.isfile(p)]
    return {
        "binary_found": found,
        "e2e_ready": bool(found),
        "note": "Full proxy+sim experiment needs taosim binary (install_validator.sh / build).",
    }


def main() -> int:
    print("=" * 60)
    print("MedianAlignedTierAgent experiment (venv)")
    print("=" * 60)

    results: dict = {"python": sys.executable, "sn79_root": SN79_ROOT}

    try:
        agent = make_agent()
        results["import"] = "ok"
    except Exception as exc:
        results["import"] = f"FAIL: {exc}"
        print(json.dumps(results, indent=2))
        return 1

    results["tier_assignment"] = test_tier_assignment(agent)
    results["score_proxy"] = test_score_proxy_update(agent)
    results["simulator"] = check_simulator()
    results["agent_http"] = test_agent_http()

    print(json.dumps(results, indent=2))

    ok = (
        results["score_proxy"].get("ordered")
        and results["agent_http"].get("started")
        and results["agent_http"].get("openapi_ok")
    )
    print("=" * 60)
    print("OVERALL:", "PASS (partial — no simulator E2E)" if ok else "PARTIAL/FAIL — see details")
    print("=" * 60)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
