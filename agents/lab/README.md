# Local Agent Lab

CPU-aware local simulation, parameter search, and real-time dashboard for `MedianAlignedTierAgent`.

## Quick start

```bash
cd /home/administrator/workspace/sn-79
./scripts/run_local_lab.sh
```

Open **http://127.0.0.1:8765**

## Modes

| Command | Description |
|---------|-------------|
| `./scripts/run_local_lab.sh` | Dashboard + ~120s live run (current params) |
| `./scripts/run_local_lab.sh --experiment` | Grid search; writes `~/.taos/lab/best_params.json` |
| `./scripts/run_local_lab.sh --experiment --max-trials 4` | Limit trials |
| `./scripts/run_local_lab.sh --dashboard-only` | Dashboard only |

## CPU profiles

Auto-selected from cores/RAM (`agents/lab/cpu_profile.py`):

| Profile | Books | Typical host |
|---------|-------|----------------|
| `lite` | 4 | ≤6 cores, ≤16GB |
| `medium` | 8 | 6+ cores, 12GB+ |
| `full` | 16 | 12+ cores, 24GB+ |

Regenerate sim XML:

```bash
PYTHONPATH=agents .venv/bin/python agents/lab/gen_sim_config.py --profile medium
```

## State files

- `~/.taos/lab/state.json` — live metrics + trial leaderboard (dashboard SSE)
- `~/.taos/lab/best_params.json` — best experiment result

## Environment

- `TAOS_LAB_METRICS=1` — agent/proxy publish metrics (set by script)
- `TAOS_LAB_DIR` — state directory (default `~/.taos/lab`)
- `TAOS_LAB_GRID=full` — larger parameter grid
