"""
Repeat local lab runs with fixed simulator seed and compute delta metrics per run.

Why deltas:
The simulator/proxy may append to the same CSV directory across runs. We therefore
measure per-run deltas by comparing row counts before/after each run.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeltaStats:
    run_idx: int
    started_at: float
    ended_at: float
    orders: int
    cancels: int
    cancel_fail: int
    trades: int

    @property
    def cancel_to_fill(self) -> float:
        return self.cancels / self.trades if self.trades else float("inf")

    @property
    def failed_cancel_rate(self) -> float:
        return self.cancel_fail / self.cancels if self.cancels else 0.0

    @property
    def trades_per_order(self) -> float:
        return self.trades / self.orders if self.orders else 0.0


def _line_count(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def _count_failed_cancels_from_row(path: Path, start_row_idx: int) -> int:
    """
    Count failures in cancellations.csv from data row index start_row_idx (0-based, excluding header).
    """
    fail = 0
    with path.open() as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i < start_row_idx:
                continue
            ok = row.get("success", "").strip().lower() in ("true", "1", "yes")
            if not ok:
                fail += 1
    return fail


def _latest_csv_dir(root: Path) -> Path:
    # Use cancellations.csv mtime; same as analyze_cancel_to_fill.py but required.
    candidates = sorted(
        root.rglob("cancellations.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        base = c.parent
        if (base / "orders.csv").exists() and (base / "trades.csv").exists():
            return base
    raise FileNotFoundError(f"No run found under {root}")


def run_once(repo_root: Path) -> None:
    subprocess.run([str(repo_root / "scripts/run_local_lab.sh")], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="sn-79 repo root",
    )
    ap.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="agents/data root",
    )
    args = ap.parse_args()

    repo = Path(args.repo_root)
    data_root = Path(args.data_root)

    base = _latest_csv_dir(data_root)
    orders_p = base / "orders.csv"
    cancels_p = base / "cancellations.csv"
    trades_p = base / "trades.csv"

    print(f"csv_dir: {base}")
    results: list[DeltaStats] = []

    # Track pre-run data-row counts (exclude header).
    pre_orders = max(_line_count(orders_p) - 1, 0)
    pre_cancels = max(_line_count(cancels_p) - 1, 0)
    pre_trades = max(_line_count(trades_p) - 1, 0)

    for i in range(args.runs):
        started = time.time()
        run_once(repo)
        ended = time.time()

        post_orders = max(_line_count(orders_p) - 1, 0)
        post_cancels = max(_line_count(cancels_p) - 1, 0)
        post_trades = max(_line_count(trades_p) - 1, 0)

        d_orders = max(post_orders - pre_orders, 0)
        d_cancels = max(post_cancels - pre_cancels, 0)
        d_trades = max(post_trades - pre_trades, 0)
        d_fail = _count_failed_cancels_from_row(cancels_p, pre_cancels)

        st = DeltaStats(
            run_idx=i + 1,
            started_at=started,
            ended_at=ended,
            orders=d_orders,
            cancels=d_cancels,
            cancel_fail=d_fail,
            trades=d_trades,
        )
        results.append(st)

        print(
            f"run {st.run_idx}/{args.runs}: "
            f"orders={st.orders} cancels={st.cancels} trades={st.trades} "
            f"cancel_to_fill={st.cancel_to_fill:.3f} "
            f"failed_cancel_rate={st.failed_cancel_rate:.3%} "
            f"trades_per_order={st.trades_per_order:.3%}"
        )

        pre_orders, pre_cancels, pre_trades = post_orders, post_cancels, post_trades

    # Summary (mean over finite cancel_to_fill only)
    finite_ctf = [r.cancel_to_fill for r in results if r.cancel_to_fill != float("inf")]
    if finite_ctf:
        mean_ctf = sum(finite_ctf) / len(finite_ctf)
        print(f"mean_cancel_to_fill: {mean_ctf:.3f}")
    mean_fcr = sum(r.failed_cancel_rate for r in results) / len(results)
    mean_tpo = sum(r.trades_per_order for r in results) / len(results)
    print(f"mean_failed_cancel_rate: {mean_fcr:.3%}")
    print(f"mean_trades_per_order: {mean_tpo:.3%}")


if __name__ == "__main__":
    main()

