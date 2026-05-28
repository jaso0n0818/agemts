# SPDX-FileCopyrightText: 2025 Rayleigh Research
# SPDX-License-Identifier: MIT
"""Grid-search MedianAlignedTierAgent params on local proxy + simulator."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from itertools import product
from pathlib import Path

SN79_ROOT = Path(__file__).resolve().parents[2]
PROXY_DIR = SN79_ROOT / "agents/proxy"
SIM_DIR = SN79_ROOT / "simulate/trading/run"
PY = SN79_ROOT / ".venv/bin/python"
CONFIG_PATH = PROXY_DIR / "config_median_lab.json"

# Small grid — expand via TAOS_LAB_GRID=full
GRID_LITE = {
    "quantity": [0.2, 0.25, 0.35],
    "max_quantity": [0.5, 0.75],
    "alpha_fraction": [0.15, 0.2, 0.25],
    "defensive_fraction": [0.15, 0.2],
}

GRID_FULL = {
    **GRID_LITE,
    "inventory_limit": [0.25, 0.30, 0.35],
    "flow_threshold": [0.30, 0.35, 0.40],
}


def param_combos(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))]


def score_trial(live: dict, respond_ms: float) -> float:
    """Higher is better — aligned with validator weights (kappa-heavy)."""
    sm = float(live.get("score_median", 0))
    books = float(live.get("books_active", 0))
    rts = float(live.get("round_trips", 0))
    latency_pen = max(0.0, (respond_ms - 800.0) / 2000.0)
    churn_pen = max(0.0, float(live.get("orders_last_tick", 0)) - 8.0) * 0.02
    kappa_proxy = sm * 0.79
    activity_proxy = min(books / 16.0, 1.0) * 0.10
    return kappa_proxy + activity_proxy + min(rts, 20) * 0.05 - latency_pen - churn_pen


def write_proxy_config(base: dict, params: dict) -> None:
    cfg = deepcopy(base)
    agent_list = cfg["agents"]["MedianAlignedTierAgent"]
    agent_list[0]["params"].update(params)
    agent_list[0]["params"]["lazy_load"] = 1
    agent_list[0]["params"]["round_trip_history"] = 64
    sim_name = os.environ.get("TAOS_LAB_PROFILE", "lite")
    cfg["proxy"]["simulation_xml"] = (
        f"../../simulate/trading/run/config/simulation_lab_{sim_name}.xml"
    )
    CONFIG_PATH.write_text(json.dumps(cfg, indent=4))


def run_trial(params: dict, trial_idx: int, wall_seconds: int) -> dict:
    from lab.metrics import append_log, load_state, patch_state, record_trial_result

    base = json.loads(CONFIG_PATH.read_text())
    write_proxy_config(base, params)

    trial_id = f"t{trial_idx:03d}"
    patch_state(
        phase="trial",
        current_trial={"id": trial_id, "params": params, "started_at": time.time()},
    )
    append_log(f"Trial {trial_id} start {params}")

    env = os.environ.copy()
    env["TAOS_LAB_METRICS"] = "1"
    env["TAOS_PROXY_SKIP_SEED"] = "1"
    env["PYTHONPATH"] = str(SN79_ROOT / "agents")
    env["PATH"] = f"{PY.parent}:{env.get('PATH', '')}"

    launcher = subprocess.Popen(
        [str(PY), "launcher.py", "--config", "config_median_lab.json"],
        cwd=str(PROXY_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    sim_xml = f"config/simulation_lab_{env.get('TAOS_LAB_PROFILE', 'lite')}.xml"
    sim = subprocess.Popen(
        [str(SIM_DIR.parent / "build/src/cpp/taosim"), "-f", sim_xml],
        cwd=str(SIM_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + wall_seconds
    try:
        while time.time() < deadline:
            time.sleep(2)
            if sim.poll() is not None:
                break
    finally:
        for proc in (sim, launcher):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        time.sleep(1)
        for proc in (sim, launcher):
            if proc.poll() is None:
                proc.kill()
        for proc in (sim, launcher):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    live = load_state().get("live", {})
    respond_ms = float(live.get("respond_ms", 0))
    sc = score_trial(live, respond_ms)
    result = {
        "id": trial_id,
        "params": params,
        "score": round(sc, 4),
        "score_median": live.get("score_median"),
        "books_active": live.get("books_active"),
        "round_trips": live.get("round_trips"),
        "respond_ms": respond_ms,
        "finished_at": time.time(),
    }
    record_trial_result(result)
    append_log(f"Trial {trial_id} score={sc:.4f}")
    patch_state(current_trial=None, phase="between_trials")
    return result


def run_experiment(max_trials: int | None = None) -> dict:
    from lab.cpu_profile import choose_profile, detect_host
    from lab.metrics import patch_state

    host = detect_host()
    profile = choose_profile(host)
    os.environ["TAOS_LAB_PROFILE"] = profile.name

    sys.path.insert(0, str(SN79_ROOT / "agents"))
    gen = subprocess.run(
        [str(PY), str(SN79_ROOT / "agents/lab/gen_sim_config.py"), "--profile", profile.name],
        check=True,
        capture_output=True,
        text=True,
    )
    sim_path = gen.stdout.strip()
    patch_state(
        host={"cpu_count": host.cpu_count, "ram_gb": host.ram_gb, "model": host.model},
        profile=profile.name,
        phase="experiment",
    )

    grid = GRID_FULL if os.environ.get("TAOS_LAB_GRID") == "full" else GRID_LITE
    combos = param_combos(grid)
    if max_trials:
        combos = combos[:max_trials]

    results = []
    for i, params in enumerate(combos):
        results.append(run_trial(params, i, profile.trial_seconds))

    best = max(results, key=lambda r: r["score"]) if results else None
    patch_state(phase="done", best_trial=best)
    out_path = Path(os.environ.get("TAOS_LAB_DIR", Path.home() / ".taos/lab")) / "best_params.json"
    if best:
        out_path.write_text(json.dumps(best, indent=2))
    return {"trials": len(results), "best": best}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Local param search for MedianAlignedTierAgent")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()
    summary = run_experiment(max_trials=args.max_trials)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
