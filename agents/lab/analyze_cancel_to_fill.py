"""
Local lab run analyzer.

Computes cancel-to-fill and failed-cancel ratios from the latest simulator CSV outputs.
Designed for C-phase: determine whether FAILED TO CANCEL is benign "warning residue"
or inefficient churn.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunStats:
    base_dir: Path
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


def _count_rows(path: Path) -> int:
    with path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def _count_failed_cancels(path: Path) -> int:
    fail = 0
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            ok = row.get("success", "").strip().lower() in ("true", "1", "yes")
            if not ok:
                fail += 1
    return fail


def find_latest_run(root: Path) -> Path | None:
    # Look for cancellations.csv because it has failure info.
    candidates = sorted(
        root.rglob("cancellations.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        base = c.parent
        if (base / "orders.csv").exists() and (base / "trades.csv").exists():
            return base
    return None


def compute_stats(base: Path) -> RunStats:
    orders = _count_rows(base / "orders.csv")
    cancels = _count_rows(base / "cancellations.csv")
    cancel_fail = _count_failed_cancels(base / "cancellations.csv")
    trades = _count_rows(base / "trades.csv")
    return RunStats(
        base_dir=base,
        orders=orders,
        cancels=cancels,
        cancel_fail=cancel_fail,
        trades=trades,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="agents/data root (default: agents/data)",
    )
    ap.add_argument("--base", default=None, help="Explicit run dir containing CSVs")
    args = ap.parse_args()

    if args.base:
        base = Path(args.base)
    else:
        base = find_latest_run(Path(args.root))
        if base is None:
            raise SystemExit(f"No run found under {args.root}")

    st = compute_stats(base)
    print(f"run_dir: {st.base_dir}")
    print(f"orders: {st.orders}")
    print(f"cancels: {st.cancels}")
    print(f"cancel_fail: {st.cancel_fail}")
    print(f"trades: {st.trades}")
    print(f"cancel_to_fill: {st.cancel_to_fill:.3f}")
    print(f"failed_cancel_rate: {st.failed_cancel_rate:.3%}")
    print(f"trades_per_order: {st.trades_per_order:.3%}")


if __name__ == "__main__":
    main()

