# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Trading agent aligned with validator scoring in reward.py:
  - Per-book Kappa from realized round-trip PnL
  - Median aggregation across books (not mean of stars)
  - Left-tail outlier penalty on weak books
  - Inactive-book tolerance (~37.5%) but all books should stay active

Architecture (organic ensemble):
  - PRIMARY: tiered maker (_process_book) — sole source of orders
  - ADVISORY (no independent orders): trend guard, flow/imbalance, momentum hint
    → only veto or scale the primary path
  - EMERGENCY (rare): market unwind when toxic trend + trapped inventory (not for scoring)
  - PREPARE: one pass per tick → BookActionPlan per book; EXECUTE is append-only
  - WARM: onTrade refreshes score snapshot between validator requests
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import os

import bittensor as bt
import numpy as np

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.instructions import STP, TimeInForce
from taos.im.protocol.models import Book, OrderDirection


class BookTier(Enum):
    ALPHA = "alpha"
    NEUTRAL = "neutral"
    DEFENSIVE = "defensive"


@dataclass
class TierParams:
    quantity_mult: float
    max_volume_ratio: float
    inventory_limit: float
    force_activity: bool


def _deque(maxlen: int) -> deque:
    return deque(maxlen=maxlen)


@dataclass
class BookFingerprint:
    midquotes: deque = field(default_factory=lambda: _deque(120))
    returns: deque = field(default_factory=lambda: _deque(120))
    reactions: deque = field(default_factory=lambda: _deque(120))
    round_trip_pnls: deque = field(default_factory=lambda: _deque(64))
    previous_mid: float | None = None
    previous_flow: float = 0.0
    last_roundtrip: int = 0
    signed_position: float = 0.0
    last_position_open_ts: int = 0
    entry_price: float | None = None
    score_proxy: float = 0.5
    score_proxy_dirty: bool = False
    last_bid_price: float | None = None
    last_ask_price: float | None = None
    last_plan_ts: int = 0
    flow_signed_volumes: deque = field(default_factory=lambda: _deque(40))
    last_emergency_ts: int = 0
    # Last observed fill timestamp (maker or taker). Used to detect "dead" books.
    last_fill_ts: int = 0
    # Maker fills since last respond — skip cancel on IDs already gone (saves instruction budget).
    recent_maker_fill_ids: set[int] = field(default_factory=set)


@dataclass
class NetworkContext:
    """
    Cross-book portfolio + peer-activity context (updated each respond).

    Note: book_score_median is the median of *my* books' score_proxy — not subnet Kappa.
    Use network_median_kappa_hint for optional external calibration.
    peer_activity_ratio compares aggregate peer vs own traded volume across all books.
    coverage_pressure rises when too few books have kappa-ready round-trip history
    (mainnet: stay below validator max_inactive_books_ratio).
    """

    book_score_median: float = 0.5
    book_score_p25: float = 0.5
    book_score_p75: float = 0.5
    portfolio_wealth_ratio: float = 1.0
    peer_activity_ratio: float = 0.0
    alpha_unlocked: bool = False
    alpha_absolute_min: float = 0.58
    defensive_absolute_max: float = 0.42
    stress_multiplier: float = 1.0
    max_alpha_slots: int = 0
    kappa_ready_ratio: float = 0.0
    coverage_pressure: float = 0.0


@dataclass(frozen=True)
class BookSignal:
    mid: float
    spread: float
    depth_imbalance: float
    flow: float
    reaction: float
    volatility: float
    toxic: bool
    regime: str
    vpin: float = 0.0


@dataclass
class AdvisoryGuidance:
    """Information-only overlay: widens quotes / scales size; does not place orders."""

    bid_ticks_back: int = 0
    ask_ticks_back: int = 0
    bid_size_mult: float = 1.0
    ask_size_mult: float = 1.0
    veto_bid: bool = False
    veto_ask: bool = False
    reasons: list[str] = field(default_factory=list)


@dataclass
class TickBookPrep:
    """Per-book snapshot computed once per validator tick."""

    signal: BookSignal | None
    trend_short: float = 0.0
    trend_long: float = 0.0
    momentum_align: float = 0.0
    coverage_push: bool = False


@dataclass
class BookActionPlan:
    """Pre-compiled instructions for one book (prepare phase → fast execute)."""

    book_id: int
    cancel_ids: list[int] = field(default_factory=list)
    place_bid: bool = False
    place_ask: bool = False
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    emergency_dir: OrderDirection | None = None
    emergency_qty: float = 0.0
    emergency_cancel_all: bool = False


@dataclass
class TickContext:
    validator: str
    timestamp: int
    net: NetworkContext
    tiers: dict[int, BookTier]
    books: dict[int, TickBookPrep]
    peer_ratio_by_book: dict[int, float] = field(default_factory=dict)
    fleet_inventory_skew: float = 0.0
    plans: dict[int, BookActionPlan] = field(default_factory=dict)
    focus_books: set[int] = field(default_factory=set)


class MedianAlignedTierAgent(FinanceSimulationAgent):
    """
    Validator-aligned tiered maker agent.

    Mirrors calculate_kappa_score goals:
      - Keep weak books above the left-tail outlier band (defensive tier)
      - Push strong books without starving the median (alpha tier capped)
      - Maintain round-trip activity on every book (neutral baseline)
    """

    def initialize(self):
        self.mainnet_mode = bool(int(getattr(self.config, "mainnet_mode", 0)))
        self.quantity = float(getattr(self.config, "quantity", 0.25))
        self.max_quantity = float(getattr(self.config, "max_quantity", self.quantity * 3))
        self.expiry_period = int(getattr(self.config, "expiry_period", 30_000_000_000))
        self.depth = int(getattr(self.config, "depth", 5))
        self.min_edge_bps = float(getattr(self.config, "min_edge_bps", 0.5)) / 10_000
        self.max_maker_fee = float(getattr(self.config, "max_maker_fee", 0.003))
        self.volume_cap_turnover = float(getattr(self.config, "volume_cap_turnover", 10.0))
        self.base_inventory_limit = float(getattr(self.config, "inventory_limit", 0.30))
        self.flow_threshold = float(getattr(self.config, "flow_threshold", 0.35))
        self.reaction_threshold = float(getattr(self.config, "reaction_threshold", 0.00001))
        self.toxic_volatility_ratio = float(getattr(self.config, "toxic_volatility_ratio", 1.25))
        _default_activity = 600_000_000_000 if self.mainnet_mode else 900_000_000_000
        self.activity_period = int(
            getattr(self.config, "activity_period", _default_activity)
        )
        self.alpha_fraction = float(getattr(self.config, "alpha_fraction", 0.20))
        self.defensive_fraction = float(getattr(self.config, "defensive_fraction", 0.20))
        self.kappa_norm_min = float(getattr(self.config, "kappa_norm_min", -2.5))
        self.kappa_norm_max = float(getattr(self.config, "kappa_norm_max", 2.5))
        # Align score_proxy with validator's kappa_3 (tau is typically 0).
        self.kappa_tau = float(getattr(self.config, "kappa_tau", 0.0))
        self.min_realized_observations = int(
            getattr(self.config, "min_realized_observations", 3)
        )
        self.pnl_reference_fraction = float(getattr(self.config, "pnl_reference_fraction", 0.002))
        self.signal_window = int(getattr(self.config, "signal_window", 120))
        self.defensive_activity_margin = float(
            getattr(self.config, "defensive_activity_margin", 0.08)
        )
        self.alpha_absolute_min = float(
            getattr(
                self.config,
                "alpha_absolute_min",
                0.62 if self.mainnet_mode else 0.58,
            )
        )
        self.fleet_alpha_unlock_median = float(
            getattr(
                self.config,
                "fleet_alpha_unlock_median",
                0.52 if self.mainnet_mode else 0.48,
            )
        )
        self.fleet_defensive_median = float(
            getattr(
                self.config,
                "fleet_defensive_median",
                0.42 if self.mainnet_mode else 0.40,
            )
        )
        self.peer_volume_defensive_ratio = float(
            getattr(self.config, "peer_volume_defensive_ratio", 2.5)
        )
        self.min_my_volume_for_peer_ratio = float(
            getattr(self.config, "min_my_volume_for_peer_ratio", 50.0)
        )
        self.peer_activity_stress_ratio = float(
            getattr(self.config, "peer_activity_stress_ratio", 1.8)
        )
        self.network_median_kappa_hint = float(
            getattr(
                self.config,
                "network_median_kappa_hint",
                0.10 if self.mainnet_mode else -1.0,
            )
        )
        # Validator allows ~37.5% books without kappa; target higher to avoid outlier penalty.
        self.max_inactive_books_ratio = float(
            getattr(self.config, "max_inactive_books_ratio", 0.375)
        )
        self.coverage_safety_margin = float(
            getattr(self.config, "coverage_safety_margin", 0.05)
        )
        self.kappa_bootstrap_ticks_back = int(
            getattr(
                self.config,
                "kappa_bootstrap_ticks_back",
                2 if self.mainnet_mode else 1,
            )
        )
        self.weak_book_proxy_floor = float(
            getattr(self.config, "weak_book_proxy_floor", 0.42)
        )
        self.fingerprints = defaultdict(lambda: defaultdict(BookFingerprint))
        self._tier_cache: dict[str, dict[int, BookTier]] = {}
        self._network_ctx: dict[str, NetworkContext] = {}
        self._book_count_hint: int = 16
        self.min_quote_quantity = float(getattr(self.config, "min_quote_quantity", 0.0))
        if self.min_quote_quantity <= 0:
            self.min_quote_quantity = max(self.quantity * 0.4, 0.1)
        self.trend_lookback_short = int(getattr(self.config, "trend_lookback_short", 8))
        self.trend_lookback_long = int(getattr(self.config, "trend_lookback_long", 32))
        self.trend_block_bps = float(getattr(self.config, "trend_block_bps", 3.0)) / 10_000
        self.momentum_block_bps = float(getattr(self.config, "momentum_block_bps", 2.0)) / 10_000
        self.imbalance_veto_threshold = float(
            getattr(self.config, "imbalance_veto_threshold", 0.18)
        )
        self.advisory_log = bool(int(getattr(self.config, "advisory_log", 0)))
        self.round_trip_history = int(getattr(self.config, "round_trip_history", 64))
        self.tier_hysteresis = float(getattr(self.config, "tier_hysteresis", 0.04))
        self.max_advisory_ticks_back = int(getattr(self.config, "max_advisory_ticks_back", 6))
        # Round-trip completion: when holding inventory, pull the exit-side quote closer
        # (inside the spread when possible) to realize PnL sooner and unlock per-book Kappa.
        # Keep modest to avoid turning into a taker; postOnly constraints still apply.
        self.roundtrip_complete_ticks = int(
            getattr(self.config, "roundtrip_complete_ticks", 1 if self.mainnet_mode else 0)
        )
        self.roundtrip_complete_only_when_cold = bool(
            int(getattr(self.config, "roundtrip_complete_only_when_cold", 1))
        )
        # Focus books (worst 1–N): complete round-trips faster to raise min_roundtrip_volume.
        _default_focus_add = 2 if self.mainnet_mode else 0
        self.focus_roundtrip_complete_ticks_add = int(
            getattr(self.config, "focus_roundtrip_complete_ticks_add", _default_focus_add)
        )
        # Focus books: if a book stays totally flat and cold/under-active, maker quotes can
        # fail to get filled for long periods on low-liquidity books. Optionally force a
        # tiny taker trade (market) to "seed" activity; subsequent logic will try to
        # complete the round-trip and/or force-close after timeout.
        _default_focus_force_trade = 0  # disabled unless explicitly enabled
        self.focus_force_trade_after = int(
            getattr(self.config, "focus_force_trade_after", _default_focus_force_trade)
        )
        # Fast bootstrap: if a book holds inventory too long without completing a round-trip,
        # force a small flatten to generate realized PnL observations (Kappa unlock) and help
        # keep some books recently active for validator activity sampling.
        _default_force_close = 60_000_000_000 if self.mainnet_mode else 0
        self.force_roundtrip_close_after = int(
            getattr(self.config, "force_roundtrip_close_after", _default_force_close)
        )
        self.force_roundtrip_close_only_when_cold = bool(
            int(getattr(self.config, "force_roundtrip_close_only_when_cold", 1))
        )
        self.force_roundtrip_close_cancel_all = bool(
            int(getattr(self.config, "force_roundtrip_close_cancel_all", 1))
        )
        # Reduce churn: on mainnet, only force-close on focus books by default.
        _default_force_focus_only = 1 if self.mainnet_mode else 0
        self.force_roundtrip_close_focus_only = bool(
            int(getattr(self.config, "force_roundtrip_close_focus_only", _default_force_focus_only))
        )
        # Min sim time between full quote refresh (reduces cancel storms).
        self.min_refresh_interval = int(
            getattr(self.config, "min_refresh_interval", 10_000_000_000)
        )
        self.max_instructions_per_book = int(
            getattr(self.config, "max_instructions_per_book", 4)
        )
        self.volume_soft_cap = float(getattr(self.config, "volume_soft_cap", 0.80))
        # Exchange/simulator constraint: orders with volume below this floor are rejected
        # with MINIMUM_ORDER_SIZE_VIOLATION. Defaulting to `quantity` keeps the common
        # case (quantity==minOrderSize) safe while still allowing experiments to override.
        _mos = float(getattr(self.config, "min_order_size", self.quantity))
        if _mos <= 0:
            _mos = self.quantity
        self.min_order_size = _mos
        # Optional: flip "survival widening" into directional skewing in trend/toxic regimes.
        # Defaults off to preserve prior behaviour and validator-aligned risk controls.
        self.aggressive_trend_mode = bool(
            int(getattr(self.config, "aggressive_trend_mode", 0))
        )
        self.aggressive_toxic_mode = bool(
            int(getattr(self.config, "aggressive_toxic_mode", 0))
        )
        if self.mainnet_mode:
            self.aggressive_trend_mode = False
            self.aggressive_toxic_mode = False
        # Forced bootstrap / activity only when defending subnet scoring (not pure alpha).
        self.scoring_defense = bool(
            int(getattr(self.config, "scoring_defense", int(self.mainnet_mode)))
        )
        self.emergency_unwind = bool(int(getattr(self.config, "emergency_unwind", 1)))
        self.emergency_inventory_ratio = float(
            getattr(self.config, "emergency_inventory_ratio", 0.40)
        )
        self.emergency_loss_bps = float(getattr(self.config, "emergency_loss_bps", 12.0)) / 10_000
        self.emergency_cooldown = int(
            getattr(self.config, "emergency_cooldown", 30_000_000_000)
        )
        self.vpin_window = int(getattr(self.config, "vpin_window", 24))
        self.vpin_threshold = float(getattr(self.config, "vpin_threshold", 0.62))
        self.toxic_refresh_factor = float(
            getattr(self.config, "toxic_refresh_factor", 0.45)
        )
        self.toxic_max_cancels = int(getattr(self.config, "toxic_max_cancels", 2))
        self.volume_scan_interval = int(
            getattr(self.config, "volume_scan_interval", 15)
        )
        # Low-volume scoring mode: only trade on a subset of books (others left inactive).
        # Validator tolerates up to ~37.5% inactive books; targeting ~80/128 keeps us above
        # the threshold while concentrating volume to unlock per-book Kappa faster.
        self.active_books_target = int(getattr(self.config, "active_books_target", 0))
        self._active_books_by_validator: dict[str, set[int]] = defaultdict(set)
        # Ensure the worst 1–N books (by "kappa readiness") stay inside the active set so
        # validator `min_roundtrip_volume = min(book_roundtrip_volume)` can actually rise.
        _default_focus = 3 if self.mainnet_mode else 0
        self.focus_books_target = int(getattr(self.config, "focus_books_target", _default_focus))
        self._last_focus_log_ts: dict[str, int] = {}
        # Tight-spread / competitive-book protection (postOnly + STP.CANCEL_BOTH checks).
        self.postonly_buffer_ticks = int(
            getattr(
                self.config,
                "postonly_buffer_ticks",
                1 if self.mainnet_mode else 0,
            )
        )
        # Ignore sub-tick repricing noise — keeps queue priority vs faster competitors.
        self.requote_price_tolerance_ticks = int(
            getattr(
                self.config,
                "requote_price_tolerance_ticks",
                1 if self.mainnet_mode else 0,
            )
        )
        # Protocol only allows CO/CN/CB/DC (not NO_STP). Mainnet default CO vs CB reduces
        # CONTRACT_VIOLATION on competitive sells (see OrderPlacementValidator checkPostOnly).
        _default_maker_stp = STP.CANCEL_OLDEST if self.mainnet_mode else STP.CANCEL_BOTH
        self.maker_limit_stp = STP(
            int(getattr(self.config, "maker_limit_stp", int(_default_maker_stp)))
        )
        self._pnl_baseline_wealth: float | None = None
        self._tier_prev: dict[str, dict[int, BookTier]] = defaultdict(dict)
        self._warm_snapshots: dict[str, dict] = defaultdict(dict)
        self._volume_scan_cache: dict[str, dict] = {}
        self._respond_seq: dict[str, int] = defaultdict(int)

    def _sync_min_order_size_from_sim(self) -> None:
        """Ensure we never submit sizes below simulator minOrderSize (mainnet is strict)."""
        mos = getattr(self.simulation_config, "minOrderSize", None)
        if mos is None:
            return
        try:
            mos_f = float(mos)
        except Exception:
            return
        if mos_f > 0 and self.min_order_size < mos_f:
            self.min_order_size = mos_f

    def _fingerprint(self, validator: str, book_id: int) -> BookFingerprint:
        return self.fingerprints[validator][book_id]

    def _tier_params(self, tier: BookTier, net: NetworkContext | None = None) -> TierParams:
        stress = net.stress_multiplier if net else 1.0
        if tier == BookTier.ALPHA:
            return TierParams(
                quantity_mult=1.75 * stress,
                max_volume_ratio=0.72 * stress,
                inventory_limit=self.base_inventory_limit * 1.15,
                force_activity=False,
            )
        if tier == BookTier.DEFENSIVE:
            return TierParams(
                quantity_mult=0.72,
                max_volume_ratio=0.48,
                inventory_limit=self.base_inventory_limit * 0.65,
                force_activity=True,
            )
        return TierParams(
            quantity_mult=1.0,
            max_volume_ratio=0.62,
            inventory_limit=self.base_inventory_limit,
            force_activity=False,
        )

    def _book_count(self) -> int:
        if hasattr(self.simulation_config, "book_count") and self.simulation_config.book_count:
            return max(int(self.simulation_config.book_count), 1)
        return max(self._book_count_hint, 1)

    def _kappa_ready_book_count(self, validator: str, book_ids: list[int]) -> int:
        return sum(
            1
            for book_id in book_ids
            if len(self._fingerprint(validator, book_id).round_trip_pnls)
            >= self.min_realized_observations
        )

    def _coverage_target_ratio(self) -> float:
        return max(
            0.5,
            1.0 - self.max_inactive_books_ratio - self.coverage_safety_margin,
        )

    def _compute_coverage_pressure(
        self, validator: str, book_ids: list[int]
    ) -> tuple[float, float]:
        n = len(book_ids)
        if n == 0:
            return 0.0, 0.0
        ready = self._kappa_ready_book_count(validator, book_ids)
        ready_ratio = ready / n
        target = self._coverage_target_ratio()
        if ready_ratio >= target:
            return ready_ratio, 0.0
        gap = target - ready_ratio
        pressure = min(1.0, gap / max(target * 0.15, 0.04))
        return ready_ratio, pressure

    def _effective_activity_period(self, coverage_pressure: float) -> int:
        if coverage_pressure <= 0:
            return self.activity_period
        shrink = 0.5 * min(coverage_pressure, 1.0)
        return max(int(self.activity_period * (1.0 - shrink)), 60_000_000_000)

    def _book_needs_coverage_push(
        self,
        validator: str,
        book_id: int,
        timestamp: int,
        net: NetworkContext,
    ) -> bool:
        if net.coverage_pressure <= 0:
            return False
        fp = self._fingerprint(validator, book_id)
        period = self._effective_activity_period(net.coverage_pressure)
        under_activity = (
            fp.last_roundtrip == 0 or timestamp - fp.last_roundtrip > period
        )
        kappa_cold = len(fp.round_trip_pnls) < self.min_realized_observations
        weak_tail = (
            len(fp.round_trip_pnls) >= 2
            and fp.score_proxy < min(net.book_score_p25, self.weak_book_proxy_floor)
        )
        return kappa_cold or under_activity or weak_tail

    def _ensure_pnl_baseline(self) -> float:
        """Fixed wealth anchor for score_proxy — not tied to live miner_wealth (avoids loss spiral)."""
        if self._pnl_baseline_wealth is None:
            wealth = float(getattr(self.simulation_config, "miner_wealth", 0.0) or 0.0)
            if wealth <= 0:
                wealth = float(getattr(self.config, "pnl_baseline_wealth", 50_000.0))
            self._pnl_baseline_wealth = max(wealth, 1e-9)
        return self._pnl_baseline_wealth

    def _book_pnl_reference(self, fingerprint: BookFingerprint) -> float:
        """
        Per-book PnL scale for tier proxy (agent-side only; validator uses MAD per book).
        Baseline capital is fixed at session start; only local return vol scales the ref.
        """
        capital_per_book = self._ensure_pnl_baseline() / self._book_count()
        base_ref = capital_per_book * self.pnl_reference_fraction
        if fingerprint.returns:
            vol = self._deque_tail_median(fingerprint.returns, 24)
            # Wider window when local moves are large (high-vol / illiquid books).
            vol_scale = max(0.35, min(2.5, vol / 0.00005))
        else:
            vol_scale = 1.0
        return max(base_ref * vol_scale, 1e-9)

    def _update_score_proxy(self, fingerprint: BookFingerprint) -> None:
        """
        Per-book proxy aligned with validator kappa_3() formula.

        The validator computes Kappa-3 on realized PnL time-series normalized by MAD,
        then normalizes raw kappa to [0,1] via (kappa - norm_min)/(norm_max - norm_min).
        Here we approximate the same calculation using per-round-trip realized PnLs.
        """
        pnls = list(fingerprint.round_trip_pnls)
        if len(pnls) < self.min_realized_observations:
            fingerprint.score_proxy = 0.5
            return

        x = np.asarray(pnls, dtype=np.float64)
        tau = float(self.kappa_tau)

        # Normalize by MAD (scale-invariant) as in validator.
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        mad = max(mad, 1e-6)
        r = x / mad

        mu = float(r.mean())
        downside = np.maximum(tau - r, 0.0)
        lpm3 = float(np.power(downside, 3).mean())
        upside = np.maximum(r - tau, 0.0)
        upm3 = float(np.power(upside, 3).mean())

        typical_scale = abs(mu) + float(np.std(r))
        regularization = float(np.power(typical_scale * 0.1, 3))
        epsilon = 1e-2 if mu > tau else 1e-6

        if lpm3 > epsilon:
            raw_kappa = (mu - tau) / float(np.cbrt(lpm3 + regularization))
        elif mu > tau:
            raw_kappa = (mu - tau) / float(np.cbrt(upm3 + regularization))
        else:
            raw_kappa = 0.0

        span = self.kappa_norm_max - self.kappa_norm_min
        if span <= 0:
            fingerprint.score_proxy = 0.5
            return
        fingerprint.score_proxy = max(
            0.0, min(1.0, (raw_kappa - self.kappa_norm_min) / span)
        )

    def _flush_score_proxies(
        self,
        validator: str,
        book_ids: list[int] | None = None,
    ) -> None:
        """Recompute kappa proxy on respond — not on every onTrade (hot path)."""
        fps = self.fingerprints.get(validator)
        if not fps:
            return
        targets = book_ids if book_ids is not None else list(fps.keys())
        updated = False
        for book_id in targets:
            fp = fps.get(book_id)
            if fp is None or not fp.score_proxy_dirty:
                continue
            self._update_score_proxy(fp)
            fp.score_proxy_dirty = False
            updated = True
        if updated:
            self._refresh_warm_scores(validator)

    def _scoring_defense_active(self, ctx: TickContext) -> bool:
        if not self.scoring_defense:
            return False
        return ctx.net.coverage_pressure > 0.01

    def _deque_tail_median(self, values: deque, tail: int) -> float:
        n = len(values)
        if n == 0:
            return 0.0
        start = max(0, n - tail)
        if start >= n:
            return 0.0
        return float(np.median([values[i] for i in range(start, n)]))

    def _recent_vol_bps(
        self, fingerprint: BookFingerprint, signal: BookSignal | None
    ) -> float:
        if fingerprint.returns:
            return self._deque_tail_median(fingerprint.returns, 12) * 10_000
        if signal is not None and signal.mid > 0:
            return (signal.spread / signal.mid) * 10_000
        return self.min_edge_bps * 10_000

    def _vol_scaled_ticks(
        self,
        fingerprint: BookFingerprint,
        signal: BookSignal,
        base_ticks: int,
    ) -> int:
        vol_bps = self._recent_vol_bps(fingerprint, signal)
        spread_bps = (signal.spread / signal.mid) * 10_000 if signal.mid > 0 else 1.0
        ref = max(spread_bps, self.min_edge_bps * 10_000, 0.5)
        scale = max(0.55, min(2.25, vol_bps / ref))
        return min(
            self.max_advisory_ticks_back,
            max(0, int(round(base_ticks * scale))),
        )

    def _vol_scaled_size_cap(
        self,
        fingerprint: BookFingerprint,
        signal: BookSignal,
        base_cap: float,
    ) -> float:
        vol_bps = self._recent_vol_bps(fingerprint, signal)
        spread_bps = (signal.spread / signal.mid) * 10_000 if signal.mid > 0 else 1.0
        ref = max(spread_bps, 0.5)
        stress = max(0.0, min(1.0, (vol_bps / ref - 0.8) * 0.35))
        return max(0.35, base_cap * (1.0 - stress))

    def _account_wealth(self, account, mid: float) -> float:
        own_base_value = max(account.own_base, 0.0) * mid
        own_quote = max(account.own_quote, 0.0)
        return own_base_value + own_quote

    def _my_volumes(self, book_ids: list[int]) -> tuple[dict[int, float], float]:
        """O(books) — uses agent accounts, not full network scan."""
        my_by_book: dict[int, float] = {}
        my_total = 0.0
        for book_id in book_ids:
            account = self.accounts.get(book_id)
            vol = float(account.traded_volume or 0.0) if account else 0.0
            my_by_book[book_id] = vol
            my_total += vol
        return my_by_book, my_total

    def _scan_peer_volumes(
        self,
        state: MarketSimulationStateUpdate,
        book_ids: list[int],
    ) -> tuple[dict[int, float], float]:
        """O(users×books) peer pass — cached between ticks."""
        book_set = set(book_ids)
        peer_by_book = dict.fromkeys(book_ids, 0.0)
        peer_total = 0.0
        if not state.accounts:
            return peer_by_book, 0.0
        for uid, uid_accounts in state.accounts.items():
            agent_id = int(uid)
            if agent_id == self.uid or agent_id < 0:
                continue
            for book_id, account in uid_accounts.items():
                if book_id not in book_set:
                    continue
                vol = account.traded_volume or 0.0
                peer_by_book[book_id] = peer_by_book.get(book_id, 0.0) + vol
                peer_total += vol
        return peer_by_book, peer_total

    def _peer_ratios(
        self,
        book_ids: list[int],
        my_by_book: dict[int, float],
        peer_by_book: dict[int, float],
    ) -> dict[int, float]:
        ratios: dict[int, float] = {}
        for book_id in book_ids:
            my_vol = my_by_book.get(book_id, 0.0)
            if my_vol < self.min_my_volume_for_peer_ratio:
                ratios[book_id] = 0.0
            else:
                ratios[book_id] = min(peer_by_book.get(book_id, 0.0) / my_vol, 10.0)
        return ratios

    def _scan_account_volumes(
        self,
        state: MarketSimulationStateUpdate,
        book_ids: list[int],
    ) -> tuple[dict[int, float], float, float]:
        """My volumes every tick; full peer scan cached (volume_scan_interval)."""
        my_by_book, my_total = self._my_volumes(book_ids)
        peer_by_book, peer_total = self._scan_peer_volumes(state, book_ids)
        ratios = self._peer_ratios(book_ids, my_by_book, peer_by_book)
        return ratios, my_total, peer_total

    def _cached_account_volumes(
        self,
        state: MarketSimulationStateUpdate,
        validator: str,
        book_ids: list[int],
    ) -> tuple[dict[int, float], float, float]:
        seq = self._respond_seq[validator]
        self._respond_seq[validator] = seq + 1
        my_by_book, my_total = self._my_volumes(book_ids)

        cache = self._volume_scan_cache.get(validator)
        book_key = tuple(book_ids)
        peer_fresh = (
            cache is None
            or cache.get("book_key") != book_key
            or seq - cache.get("seq", -1) >= self.volume_scan_interval
        )
        if peer_fresh:
            peer_by_book, peer_total = self._scan_peer_volumes(state, book_ids)
            self._volume_scan_cache[validator] = {
                "book_key": book_key,
                "seq": seq,
                "peer_by_book": peer_by_book,
                "peer_total": peer_total,
            }
        else:
            peer_by_book = cache["peer_by_book"]
            peer_total = cache["peer_total"]

        ratios = self._peer_ratios(book_ids, my_by_book, peer_by_book)
        return ratios, my_total, peer_total

    def _build_network_context(
        self,
        state: MarketSimulationStateUpdate,
        validator: str,
        book_ids: list[int],
        volume_totals: tuple[float, float] | None = None,
    ) -> NetworkContext:
        proxies = []
        for book_id in book_ids:
            fp = self._fingerprint(validator, book_id)
            if len(fp.round_trip_pnls) >= 2:
                proxies.append(fp.score_proxy)

        if proxies:
            book_median = float(np.median(proxies))
            book_p25 = float(np.percentile(proxies, 25))
            book_p75 = float(np.percentile(proxies, 75))
        else:
            book_median = book_p25 = book_p75 = 0.5

        if self.network_median_kappa_hint >= 0.0:
            book_median = 0.5 * book_median + 0.5 * self.network_median_kappa_hint
            book_p75 = max(book_p75, self.network_median_kappa_hint)

        if volume_totals is not None:
            my_total_vol, peer_total_vol = volume_totals
        else:
            _, my_total_vol, peer_total_vol = self._scan_account_volumes(state, book_ids)
        peer_activity_ratio = (
            peer_total_vol / my_total_vol
            if my_total_vol >= self.min_my_volume_for_peer_ratio
            else 0.0
        )

        total_wealth = 0.0
        baseline = self._ensure_pnl_baseline()
        capital_per_book = baseline / max(len(book_ids), 1)
        for book_id in book_ids:
            account = self.accounts.get(book_id)
            book = state.books.get(book_id)
            if account is None or book is None or not book.bids or not book.asks:
                continue
            mid = (book.bids[0].price + book.asks[0].price) / 2
            total_wealth += self._account_wealth(account, mid)
        portfolio_ratio = total_wealth / max(
            capital_per_book * len(book_ids), 1e-9
        )

        alpha_unlocked = book_median >= self.fleet_alpha_unlock_median
        if portfolio_ratio < 0.92:
            alpha_unlocked = False
        if peer_activity_ratio >= self.peer_activity_stress_ratio:
            alpha_unlocked = False

        kappa_ready_ratio, coverage_pressure = self._compute_coverage_pressure(
            validator, book_ids
        )
        if coverage_pressure > 0.5:
            alpha_unlocked = False

        stress = 1.0
        if book_median < self.fleet_defensive_median:
            stress = 0.65
        elif book_median < self.fleet_alpha_unlock_median:
            stress = 0.85
        if portfolio_ratio < 0.95:
            stress = min(stress, 0.75)
        if peer_activity_ratio >= self.peer_activity_stress_ratio:
            stress = min(stress, 0.7)

        n = len(book_ids)
        max_alpha = 0 if not alpha_unlocked else max(1, int(n * self.alpha_fraction))
        alpha_floor = max(self.alpha_absolute_min, book_p75)
        defensive_ceiling = min(0.45, book_p25 + 0.05)

        return NetworkContext(
            book_score_median=book_median,
            book_score_p25=book_p25,
            book_score_p75=book_p75,
            portfolio_wealth_ratio=portfolio_ratio,
            peer_activity_ratio=peer_activity_ratio,
            alpha_unlocked=alpha_unlocked,
            alpha_absolute_min=alpha_floor,
            defensive_absolute_max=defensive_ceiling,
            stress_multiplier=stress,
            max_alpha_slots=max_alpha,
            kappa_ready_ratio=kappa_ready_ratio,
            coverage_pressure=coverage_pressure,
        )

    def _assign_tiers(
        self,
        validator: str,
        book_ids: list[int],
        net: NetworkContext,
        peer_ratio_by_book: dict[int, float],
    ) -> dict[int, BookTier]:
        rows: list[tuple[int, float, int]] = []
        for book_id in book_ids:
            fp = self._fingerprint(validator, book_id)
            obs = len(fp.round_trip_pnls)
            rows.append((book_id, fp.score_proxy, obs))

        if len(rows) < 3:
            return {book_id: BookTier.NEUTRAL for book_id, _, _ in rows}

        proxies = np.array([r[1] for r in rows], dtype=np.float64)
        q25, q75 = np.percentile(proxies, [25, 75])
        alpha_cut = max(q75, net.alpha_absolute_min)
        defensive_cut = min(q25, net.defensive_absolute_max)

        tier_by_book: dict[int, BookTier] = {}
        prev_tiers = self._tier_prev.get(validator, {})
        h = self.tier_hysteresis
        for book_id, proxy, obs in rows:
            peer_ratio = peer_ratio_by_book.get(book_id, 0.0)
            prev = prev_tiers.get(book_id)
            if obs < self.min_realized_observations:
                tier_by_book[book_id] = (
                    BookTier.DEFENSIVE
                    if proxy <= net.book_score_p25
                    else BookTier.NEUTRAL
                )
                continue
            if obs < 2:
                tier_by_book[book_id] = BookTier.NEUTRAL
            elif (
                net.alpha_unlocked
                and proxy >= alpha_cut
                and peer_ratio < self.peer_volume_defensive_ratio
            ):
                tier_by_book[book_id] = BookTier.ALPHA
            elif (
                proxy <= defensive_cut
                or proxy < self.weak_book_proxy_floor
                or peer_ratio >= self.peer_volume_defensive_ratio
                or (
                    not net.alpha_unlocked
                    and proxy < net.book_score_median
                )
            ):
                tier_by_book[book_id] = BookTier.DEFENSIVE
            else:
                tier_by_book[book_id] = BookTier.NEUTRAL
            if prev == BookTier.ALPHA and tier_by_book[book_id] != BookTier.ALPHA:
                if proxy >= alpha_cut - h:
                    tier_by_book[book_id] = BookTier.ALPHA
            elif prev == BookTier.DEFENSIVE and tier_by_book[book_id] != BookTier.DEFENSIVE:
                if proxy <= defensive_cut + h:
                    tier_by_book[book_id] = BookTier.DEFENSIVE

        alpha_ids = sorted(
            [bid for bid, t in tier_by_book.items() if t == BookTier.ALPHA],
            key=lambda bid: self._fingerprint(validator, bid).score_proxy,
            reverse=True,
        )
        if net.max_alpha_slots <= 0:
            for bid in alpha_ids:
                tier_by_book[bid] = BookTier.NEUTRAL
        elif len(alpha_ids) > net.max_alpha_slots:
            for bid in alpha_ids[net.max_alpha_slots :]:
                tier_by_book[bid] = BookTier.NEUTRAL

        max_defensive = max(1, int(len(rows) * self.defensive_fraction))
        defensive_ids = sorted(
            [bid for bid, t in tier_by_book.items() if t == BookTier.DEFENSIVE],
            key=lambda bid: self._fingerprint(validator, bid).score_proxy,
        )
        if len(defensive_ids) > max_defensive:
            for bid in defensive_ids[max_defensive:]:
                tier_by_book[bid] = BookTier.NEUTRAL

        self._tier_prev[validator] = dict(tier_by_book)
        return tier_by_book

    def _trade_flow(self, book: Book) -> float:
        trades = [event for event in (book.events or []) if event.type == "t"]
        total = sum(trade.quantity for trade in trades)
        if total <= 0:
            return 0.0
        signed = sum(
            trade.quantity if trade.side == OrderDirection.BUY else -trade.quantity
            for trade in trades
        )
        return max(-1.0, min(1.0, signed / total))

    def _record_tick_flow(self, fingerprint: BookFingerprint, book: Book) -> None:
        for event in book.events or []:
            if event.type != "t":
                continue
            signed = (
                event.quantity
                if event.side == OrderDirection.BUY
                else -event.quantity
            )
            fingerprint.flow_signed_volumes.append(signed)

    def _vpin(self, fingerprint: BookFingerprint) -> float:
        d = fingerprint.flow_signed_volumes
        n = len(d)
        if n == 0:
            return 0.0
        start = max(0, n - self.vpin_window)
        buy = 0.0
        sell = 0.0
        for i in range(start, n):
            v = d[i]
            if v > 0:
                buy += v
            else:
                sell -= v
        total = buy + sell
        if total <= 0:
            return 0.0
        return abs(buy - sell) / total

    def _unrealized_return(self, fingerprint: BookFingerprint, mid: float) -> float:
        if fingerprint.entry_price is None or fingerprint.entry_price <= 0 or mid <= 0:
            return 0.0
        pos = fingerprint.signed_position
        if pos > 1e-8:
            return (mid - fingerprint.entry_price) / fingerprint.entry_price
        if pos < -1e-8:
            return (fingerprint.entry_price - mid) / fingerprint.entry_price
        return 0.0

    def _position_notional(self, fingerprint: BookFingerprint, mid: float) -> float:
        return abs(fingerprint.signed_position) * max(mid, 0.0)

    def _adverse_to_position(
        self,
        fingerprint: BookFingerprint,
        prep: TickBookPrep,
        signal: BookSignal,
    ) -> bool:
        pos = fingerprint.signed_position
        if abs(pos) < 1e-8:
            return False
        if not signal.toxic and signal.vpin < self.vpin_threshold:
            return False
        if pos > 0:
            return (
                prep.trend_short <= -self.trend_block_bps
                or signal.flow <= -self.flow_threshold
            )
        return (
            prep.trend_short >= self.trend_block_bps
            or signal.flow >= self.flow_threshold
        )

    def _effective_min_refresh_interval(
        self,
        prep: TickBookPrep | None,
        fingerprint: BookFingerprint,
    ) -> int:
        base = self.min_refresh_interval
        signal = prep.signal if prep else None
        if signal is None:
            return base
        if signal.toxic or signal.vpin >= self.vpin_threshold:
            return max(int(base * self.toxic_refresh_factor), 2_000_000_000)
        if len(fingerprint.returns) >= 4:
            vol = self._deque_tail_median(fingerprint.returns, 8)
            if vol > self.momentum_block_bps * 2:
                return max(int(base * 0.65), 3_000_000_000)
        return base

    def _max_cancel_batch(self, prep: TickBookPrep | None) -> int:
        signal = prep.signal if prep else None
        if signal and (signal.toxic or signal.vpin >= self.vpin_threshold):
            return self.toxic_max_cancels
        return 1

    def _needs_emergency_unwind(
        self,
        ctx: TickContext,
        book_id: int,
        book: Book,
        prep: TickBookPrep,
    ) -> tuple[OrderDirection, float] | None:
        if not self.emergency_unwind or prep.signal is None:
            return None
        signal = prep.signal
        account = self.accounts.get(book_id)
        if account is None:
            return None
        fingerprint = self._fingerprint(ctx.validator, book_id)
        if ctx.timestamp - fingerprint.last_emergency_ts < self.emergency_cooldown:
            return None
        pos = fingerprint.signed_position
        if abs(pos) < 1e-8:
            return None
        mid = signal.mid
        min_notional = self.quantity * mid * 1.25
        if self._position_notional(fingerprint, mid) < min_notional:
            return None
        inv_ratio = self._inventory_ratio(account, mid)
        ur = self._unrealized_return(fingerprint, mid)
        adverse = self._adverse_to_position(fingerprint, prep, signal)
        loss_hit = ur <= -self.emergency_loss_bps
        inv_hit = abs(inv_ratio) >= self.emergency_inventory_ratio
        fleet_stress = ctx.net.portfolio_wealth_ratio < 0.92
        if not adverse and not (loss_hit and inv_hit):
            return None
        if loss_hit and not adverse and not fleet_stress:
            return None
        if prep.coverage_push and len(fingerprint.round_trip_pnls) < self.min_realized_observations:
            if not loss_hit or ur > -self.emergency_loss_bps * 2:
                return None
        direction = (
            OrderDirection.SELL if pos > 0 else OrderDirection.BUY
        )
        vol_dec = self.simulation_config.volumeDecimals
        qty = round(min(abs(pos), self.max_quantity), vol_dec)
        if qty < self.min_order_size:
            return None
        if direction == OrderDirection.SELL:
            qty = round(min(qty, max(account.base_balance.free, 0.0)), vol_dec)
        else:
            max_buy = account.quote_balance.free / max(best_ask := book.asks[0].price, 1e-9)
            qty = round(min(qty, max_buy), vol_dec)
        if qty < self.min_order_size:
            return None
        return direction, qty

    def _depth_imbalance(self, book: Book) -> float:
        bid_depth = sum(level.quantity for level in book.bids[: self.depth])
        ask_depth = sum(level.quantity for level in book.asks[: self.depth])
        total_depth = bid_depth + ask_depth
        if total_depth <= 0:
            return 0.0
        return (bid_depth - ask_depth) / total_depth

    def _mean(self, values: deque) -> float:
        return sum(values) / len(values) if values else 0.0

    def _signal(self, validator: str, book_id: int, book: Book) -> BookSignal | None:
        if not book.bids or not book.asks:
            return None

        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        spread = best_ask - best_bid
        if best_bid <= 0 or spread <= 0:
            return None

        mid = (best_bid + best_ask) / 2
        flow = self._trade_flow(book)
        fingerprint = self._fingerprint(validator, book_id)
        self._record_tick_flow(fingerprint, book)
        vpin = self._vpin(fingerprint)

        if fingerprint.previous_mid:
            realized_return = (mid - fingerprint.previous_mid) / fingerprint.previous_mid
            fingerprint.returns.append(abs(realized_return))
            if fingerprint.previous_flow:
                fingerprint.reactions.append(fingerprint.previous_flow * realized_return)

        fingerprint.midquotes.append(mid)
        fingerprint.previous_mid = mid
        fingerprint.previous_flow = flow

        reaction = self._mean(fingerprint.reactions)
        volatility = self._mean(fingerprint.returns)
        depth_imbalance = self._depth_imbalance(book)
        relative_spread = spread / mid
        directional_flow = abs(flow) >= self.flow_threshold
        continuing_flow = reaction > self.reaction_threshold
        volatile_for_spread = volatility > relative_spread * self.toxic_volatility_ratio
        toxic = (
            volatile_for_spread
            or (directional_flow and continuing_flow)
            or vpin >= self.vpin_threshold
        )

        if reaction > self.reaction_threshold:
            regime = "trend"
        elif reaction < -self.reaction_threshold:
            regime = "reversion"
        else:
            regime = "spread"

        return BookSignal(
            mid=mid,
            spread=spread,
            depth_imbalance=depth_imbalance,
            flow=flow,
            reaction=reaction,
            volatility=volatility,
            toxic=toxic,
            regime=regime,
            vpin=vpin,
        )

    def _volume_ratio(self, account) -> float:
        traded_volume = account.traded_volume
        if traded_volume is None or self.simulation_config.miner_wealth <= 0:
            return 0.0
        volume_cap = self.volume_cap_turnover * self.simulation_config.miner_wealth
        return traded_volume / volume_cap if volume_cap > 0 else 0.0

    def _inventory_ratio(self, account, mid: float) -> float:
        own_base_value = max(account.own_base, 0.0) * mid
        own_quote = max(account.own_quote, 0.0)
        wealth = own_base_value + own_quote
        if wealth <= 0:
            return 0.0
        return (own_base_value / wealth) - 0.5

    def _prices(
        self,
        book: Book,
        guidance: AdvisoryGuidance | None = None,
        *,
        bootstrap: bool = False,
    ) -> tuple[float, float]:
        """
        Default: join BBO (passive). Advisory adds ticks_back to widen away from adverse flow.
        bootstrap=True widens both sides when building kappa observations on cold books.
        """
        tick = 10 ** (-self.simulation_config.priceDecimals)
        dec = self.simulation_config.priceDecimals
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        bid_back = guidance.bid_ticks_back if guidance else 0
        ask_back = guidance.ask_ticks_back if guidance else 0
        if bootstrap:
            bid_back = max(bid_back, self.kappa_bootstrap_ticks_back)
            ask_back = max(ask_back, self.kappa_bootstrap_ticks_back)
        bid_back = min(max(bid_back, 0), self.max_advisory_ticks_back)
        ask_back = min(max(ask_back, 0), self.max_advisory_ticks_back)
        bid = best_bid - bid_back * tick
        ask = best_ask + ask_back * tick
        if bid >= ask:
            mid = (best_bid + best_ask) / 2
            bid = round(mid - tick, dec)
            ask = round(mid + tick, dec)
        bid, ask = self._apply_postonly_buffer(book, bid, ask)
        return round(bid, dec), round(ask, dec)

    def _apply_roundtrip_completion_prices(
        self,
        book: Book,
        fingerprint: BookFingerprint,
        bid: float,
        ask: float,
        *,
        allow_bid: bool,
        allow_ask: bool,
        under_activity: bool,
        focus: bool = False,
    ) -> tuple[float, float]:
        """
        If we hold inventory, bias quotes to complete the round-trip faster.
        This targets validator Kappa unlocking: book needs >= min_realized_observations realized round-trips.
        """
        ticks = self.roundtrip_complete_ticks + (
            self.focus_roundtrip_complete_ticks_add if focus else 0
        )
        if ticks <= 0 or not book.bids or not book.asks:
            return bid, ask
        if fingerprint.signed_position == 0:
            return bid, ask
        if fingerprint.signed_position > 0 and not allow_ask:
            return bid, ask
        if fingerprint.signed_position < 0 and not allow_bid:
            return bid, ask
        if (not focus) and self.roundtrip_complete_only_when_cold and (
            len(fingerprint.round_trip_pnls) >= self.min_realized_observations and not under_activity
        ):
            return bid, ask

        tick = 10 ** (-self.simulation_config.priceDecimals)
        dec = self.simulation_config.priceDecimals
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        spread = best_ask - best_bid
        if spread < 2 * tick:
            # Too tight to improve without risking postOnly/contract violations.
            return bid, ask

        ticks = max(0, min(int(ticks), 6))
        improve = ticks * tick
        if fingerprint.signed_position > 0:
            # Long inventory: improve ask (sell) towards the touch to realize.
            target_ask = max(best_bid + tick, min(best_ask - improve, ask))
            ask = round(target_ask, dec)
        else:
            # Short inventory: improve bid (buy) towards the touch to cover.
            target_bid = min(best_ask - tick, max(best_bid + improve, bid))
            bid = round(target_bid, dec)

        # Re-apply postOnly buffer safety in case spread tightens further.
        bid, ask = self._apply_postonly_buffer(book, bid, ask)
        return bid, ask

    def _apply_postonly_buffer(
        self, book: Book, bid: float, ask: float
    ) -> tuple[float, float]:
        """Pull quotes off the touch when spread is tight — avoids CONTRACT_VIOLATION on place."""
        if self.postonly_buffer_ticks <= 0 or not book.bids or not book.asks:
            return bid, ask
        tick = 10 ** (-self.simulation_config.priceDecimals)
        dec = self.simulation_config.priceDecimals
        buf = self.postonly_buffer_ticks * tick
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        spread = best_ask - best_bid
        if spread > (1 + self.postonly_buffer_ticks) * tick:
            return bid, ask
        safe_bid = round(best_ask - tick - buf, dec)
        safe_ask = round(best_bid + tick + buf, dec)
        bid = min(bid, safe_bid)
        ask = max(ask, safe_ask)
        if bid >= ask:
            mid = (best_bid + best_ask) / 2
            bid = round(mid - tick - buf, dec)
            ask = round(mid + tick + buf, dec)
        return bid, ask

    def _postonly_place_allowed(
        self, book: Book, direction: OrderDirection, price: float
    ) -> bool:
        """Preflight at plan time — skip doomed placements and spend budget on live quotes."""
        if not book.bids or not book.asks:
            return False
        if direction == OrderDirection.BUY:
            return price < book.asks[0].price
        return price > book.bids[0].price

    def _prices_changed_materially(
        self,
        fingerprint: BookFingerprint,
        bid_price: float,
        ask_price: float,
    ) -> bool:
        tick = 10 ** (-self.simulation_config.priceDecimals)
        tol = self.requote_price_tolerance_ticks * tick
        if fingerprint.last_bid_price is None or fingerprint.last_ask_price is None:
            return True
        if abs(bid_price - fingerprint.last_bid_price) > tol:
            return True
        if abs(ask_price - fingerprint.last_ask_price) > tol:
            return True
        return False

    def _quote_size(
        self,
        fingerprint: BookFingerprint,
        signal: BookSignal,
        account,
        tier: TierParams,
        under_activity: bool,
    ) -> float:
        size = self.quantity * tier.quantity_mult
        if signal.regime == "spread" and not signal.toxic:
            size *= 1.35
        if under_activity:
            size = max(size, self.quantity * 0.85)
        if fingerprint.last_roundtrip == 0:
            size = max(size, self.quantity * 0.9)
        vol_ratio = self._volume_ratio(account)
        if vol_ratio > tier.max_volume_ratio * 0.55:
            size *= 0.55
        elif vol_ratio > tier.max_volume_ratio * 0.35:
            size *= 0.8
        cap = self.max_quantity * tier.quantity_mult
        return round(
            max(self.min_order_size, self.quantity * 0.5, min(size, cap)),
            self.simulation_config.volumeDecimals,
        )

    def _has_live_quote(self, account, side: OrderDirection, price: float) -> bool:
        return any(order.side == side and order.price == price for order in account.orders)

    def _open_orders(self, account) -> dict[int, object]:
        return {order.id: order for order in (account.orders or [])}

    def _book_instruction_count(
        self, response: FinanceAgentResponse, book_id: int
    ) -> int:
        return sum(1 for i in response.instructions if i.bookId == book_id)

    def _missing_target_quotes(
        self,
        account,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
    ) -> bool:
        if allow_bid and not self._has_live_quote(
            account, OrderDirection.BUY, bid_price
        ):
            return True
        if allow_ask and not self._has_live_quote(
            account, OrderDirection.SELL, ask_price
        ):
            return True
        return False

    def _has_stale_quotes(
        self,
        account,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
        fingerprint: BookFingerprint,
    ) -> bool:
        live = self._open_orders(account)
        pos = fingerprint.signed_position
        for order in live.values():
            if pos > 1e-8 and order.side == OrderDirection.SELL:
                continue
            if pos < -1e-8 and order.side == OrderDirection.BUY:
                continue
            # More conservative stale detection:
            # - If a side is currently allowed, we don't force requote just because
            #   we have extra quotes at other prices. This reduces unnecessary
            #   cancel attempts and associated "Order IDs do not exist" churn.
            if order.side == OrderDirection.BUY:
                if not allow_bid:
                    return True
                continue
            if not allow_ask:
                return True
            continue
        return False

    def _should_requote(
        self,
        fingerprint: BookFingerprint,
        timestamp: int,
        under_activity: bool,
        account,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
        prices_changed: bool,
        min_refresh: int | None = None,
    ) -> bool:
        if self._missing_target_quotes(account, bid_price, ask_price, allow_bid, allow_ask):
            return True
        if self._has_stale_quotes(
            account, bid_price, ask_price, allow_bid, allow_ask, fingerprint
        ):
            return True
        elapsed = timestamp - fingerprint.last_plan_ts
        refresh_cap = min_refresh if min_refresh is not None else self.min_refresh_interval
        if fingerprint.last_plan_ts and elapsed < refresh_cap:
            return False
        if prices_changed:
            return True
        return False

    def _apply_inventory_skew(
        self,
        fingerprint: BookFingerprint,
        allow_bid: bool,
        allow_ask: bool,
    ) -> tuple[bool, bool]:
        """After a one-sided fill, prioritize completing the round-trip."""
        pos = fingerprint.signed_position
        if pos > 1e-8:
            return False, allow_ask or True
        if pos < -1e-8:
            return allow_bid or True, False
        return allow_bid, allow_ask

    def _collect_cancel_ids(
        self,
        account,
        book_id: int,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
        fingerprint: BookFingerprint,
        prices_changed: bool,
        max_cancels: int = 1,
    ) -> list[int]:
        live = self._open_orders(account)
        pos = fingerprint.signed_position
        cancel_ids: list[int] = []
        for order_id, order in live.items():
            if len(cancel_ids) >= max_cancels:
                break
            if pos > 1e-8 and order.side == OrderDirection.SELL:
                continue
            if pos < -1e-8 and order.side == OrderDirection.BUY:
                continue
            if order.side == OrderDirection.BUY:
                keep = allow_bid and order.price == bid_price
            else:
                keep = allow_ask and order.price == ask_price
            if not prices_changed and keep:
                continue
            if not keep:
                cancel_ids.append(order_id)
        return cancel_ids

    def _cancel_vulnerable_quotes(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        account,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
        fingerprint: BookFingerprint,
        prices_changed: bool,
    ) -> None:
        for order_id in self._collect_cancel_ids(
            account,
            book_id,
            bid_price,
            ask_price,
            allow_bid,
            allow_ask,
            fingerprint,
            prices_changed,
        ):
            if self._book_instruction_count(response, book_id) >= self.max_instructions_per_book:
                break
            response.cancel_order(book_id, order_id)

    def _maker_edge_is_positive(self, account, signal: BookSignal) -> bool:
        maker_fee = account.fees.maker_fee_rate if account.fees else 0.0
        if maker_fee > self.max_maker_fee:
            return False
        required_edge = max(self.min_edge_bps, (2 * maker_fee) + self.min_edge_bps)
        return signal.spread > signal.mid * required_edge

    def _return_over_lookback_deque(self, mids: deque, lookback: int) -> float:
        n = len(mids)
        if n <= lookback:
            return 0.0
        start = mids[-lookback - 1]
        if start <= 0:
            return 0.0
        end = mids[-1]
        return (end - start) / start

    def _trend_from_fingerprint(self, fingerprint: BookFingerprint) -> tuple[float, float, float]:
        mids = fingerprint.midquotes
        n = len(mids)
        if n < self.trend_lookback_short + 2:
            return 0.0, 0.0, 0.0
        trend_short = self._return_over_lookback_deque(mids, self.trend_lookback_short)
        trend_long = self._return_over_lookback_deque(
            mids, min(self.trend_lookback_long, n - 2)
        )
        momentum_align = 0.0
        if trend_short * trend_long > 0 and abs(trend_short) >= self.momentum_block_bps:
            momentum_align = 1.0 if trend_short > 0 else -1.0
        return trend_short, trend_long, momentum_align

    def _compute_fleet_inventory_skew(
        self,
        state: MarketSimulationStateUpdate,
        book_ids: list[int],
    ) -> float:
        """Mean inventory ratio across books — cross-book exposure hint."""
        ratios: list[float] = []
        for book_id in book_ids:
            book = state.books.get(book_id)
            account = self.accounts.get(book_id)
            if book is None or account is None or not book.bids or not book.asks:
                continue
            mid = (book.bids[0].price + book.asks[0].price) / 2
            ratios.append(self._inventory_ratio(account, mid))
        return float(np.mean(ratios)) if ratios else 0.0

    def _build_advisory(
        self,
        prep: TickBookPrep,
        tier: BookTier,
        net: NetworkContext,
        fingerprint: BookFingerprint,
        fleet_inventory_skew: float = 0.0,
    ) -> AdvisoryGuidance:
        """Widen spread / shrink size — keep at least one side quotable for activity."""
        guidance = AdvisoryGuidance()
        signal = prep.signal
        if signal is None:
            return guidance

        ts, tl, mom = prep.trend_short, prep.trend_long, prep.momentum_align

        if mom < 0 and ts <= -self.trend_block_bps and tl <= -self.momentum_block_bps:
            tb = self._vol_scaled_ticks(fingerprint, signal, 3)
            guidance.bid_ticks_back += tb
            guidance.bid_size_mult = min(
                guidance.bid_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.55),
            )
            guidance.reasons.append("trend:passive_bid")
        elif mom > 0 and ts >= self.trend_block_bps and tl >= self.momentum_block_bps:
            tb = self._vol_scaled_ticks(fingerprint, signal, 3)
            guidance.ask_ticks_back += tb
            guidance.ask_size_mult = min(
                guidance.ask_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.55),
            )
            guidance.reasons.append("trend:passive_ask")

        if signal.vpin >= self.vpin_threshold:
            tb = self._vol_scaled_ticks(fingerprint, signal, 1)
            guidance.bid_ticks_back += tb
            guidance.ask_ticks_back += tb
            cap = self._vol_scaled_size_cap(fingerprint, signal, 0.7)
            guidance.bid_size_mult = min(guidance.bid_size_mult, cap)
            guidance.ask_size_mult = min(guidance.ask_size_mult, cap)
            guidance.reasons.append(f"vpin:{signal.vpin:.2f}")

        if signal.toxic:
            if signal.flow >= self.flow_threshold:
                tb = self._vol_scaled_ticks(fingerprint, signal, 4)
                guidance.ask_ticks_back += tb
                guidance.ask_size_mult = min(
                    guidance.ask_size_mult,
                    self._vol_scaled_size_cap(fingerprint, signal, 0.45),
                )
                guidance.reasons.append("flow:toxic_ask_wide")
            elif signal.flow <= -self.flow_threshold:
                tb = self._vol_scaled_ticks(fingerprint, signal, 4)
                guidance.bid_ticks_back += tb
                guidance.bid_size_mult = min(
                    guidance.bid_size_mult,
                    self._vol_scaled_size_cap(fingerprint, signal, 0.45),
                )
                guidance.reasons.append("flow:toxic_bid_wide")
            else:
                tb = self._vol_scaled_ticks(fingerprint, signal, 2)
                guidance.bid_ticks_back += tb
                guidance.ask_ticks_back += tb
                cap = self._vol_scaled_size_cap(fingerprint, signal, 0.6)
                guidance.bid_size_mult = min(guidance.bid_size_mult, cap)
                guidance.ask_size_mult = min(guidance.ask_size_mult, cap)
                guidance.reasons.append("flow:toxic_both_wide")

        # Optional directional skewing (maker-only): keep with-flow side near BBO and
        # widen the opposing side. This cannot place inside the spread; it only biases
        # which side remains competitive under adverse regimes.
        if tier != BookTier.DEFENSIVE:
            if self.aggressive_toxic_mode and signal.toxic and abs(signal.flow) >= self.flow_threshold:
                if signal.flow > 0:
                    guidance.bid_ticks_back = min(guidance.bid_ticks_back, 0)
                    guidance.bid_size_mult = max(guidance.bid_size_mult, 1.30)
                    guidance.ask_ticks_back = max(guidance.ask_ticks_back, min(15, self.max_advisory_ticks_back))
                    guidance.ask_size_mult = min(guidance.ask_size_mult, 0.35)
                    guidance.reasons.append("toxic:skew_long")
                else:
                    guidance.ask_ticks_back = min(guidance.ask_ticks_back, 0)
                    guidance.ask_size_mult = max(guidance.ask_size_mult, 1.30)
                    guidance.bid_ticks_back = max(guidance.bid_ticks_back, min(15, self.max_advisory_ticks_back))
                    guidance.bid_size_mult = min(guidance.bid_size_mult, 0.35)
                    guidance.reasons.append("toxic:skew_short")
            elif self.aggressive_trend_mode and signal.regime == "trend" and abs(mom) > 0:
                if mom > 0:
                    guidance.bid_ticks_back = min(guidance.bid_ticks_back, 0)
                    guidance.bid_size_mult = max(guidance.bid_size_mult, 1.15)
                    guidance.ask_ticks_back = max(guidance.ask_ticks_back, min(10, self.max_advisory_ticks_back))
                    guidance.ask_size_mult = min(guidance.ask_size_mult, 0.35)
                    guidance.reasons.append("trend:skew_long")
                else:
                    guidance.ask_ticks_back = min(guidance.ask_ticks_back, 0)
                    guidance.ask_size_mult = max(guidance.ask_size_mult, 1.15)
                    guidance.bid_ticks_back = max(guidance.bid_ticks_back, min(10, self.max_advisory_ticks_back))
                    guidance.bid_size_mult = min(guidance.bid_size_mult, 0.35)
                    guidance.reasons.append("trend:skew_short")

        imb = signal.depth_imbalance
        if imb >= self.imbalance_veto_threshold and ts > 0:
            tb = self._vol_scaled_ticks(fingerprint, signal, 2)
            guidance.ask_ticks_back += tb
            guidance.ask_size_mult = min(
                guidance.ask_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.65),
            )
            guidance.reasons.append("imbalance:passive_ask")
        elif imb <= -self.imbalance_veto_threshold and ts < 0:
            tb = self._vol_scaled_ticks(fingerprint, signal, 2)
            guidance.bid_ticks_back += tb
            guidance.bid_size_mult = min(
                guidance.bid_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.65),
            )
            guidance.reasons.append("imbalance:passive_bid")

        if tier == BookTier.DEFENSIVE and net.peer_activity_ratio >= self.peer_activity_stress_ratio:
            tb = self._vol_scaled_ticks(fingerprint, signal, 1)
            guidance.bid_ticks_back += tb
            guidance.ask_ticks_back += tb
            guidance.bid_size_mult *= 0.85
            guidance.ask_size_mult *= 0.85
            guidance.reasons.append("peer:defensive_throttle")

        if fingerprint.signed_position > 1e-8 and ts < -self.momentum_block_bps:
            tb = self._vol_scaled_ticks(fingerprint, signal, 2)
            guidance.bid_ticks_back += tb
            guidance.bid_size_mult = min(
                guidance.bid_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.4),
            )
            guidance.reasons.append("inventory:long_passive_bid")
        elif fingerprint.signed_position < -1e-8 and ts > self.momentum_block_bps:
            tb = self._vol_scaled_ticks(fingerprint, signal, 2)
            guidance.ask_ticks_back += tb
            guidance.ask_size_mult = min(
                guidance.ask_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.4),
            )
            guidance.reasons.append("inventory:short_passive_ask")

        if fleet_inventory_skew > 0.18:
            guidance.bid_ticks_back += self._vol_scaled_ticks(fingerprint, signal, 1)
            guidance.bid_size_mult = min(
                guidance.bid_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.75),
            )
            guidance.reasons.append("fleet:net_long")
        elif fleet_inventory_skew < -0.18:
            guidance.ask_ticks_back += self._vol_scaled_ticks(fingerprint, signal, 1)
            guidance.ask_size_mult = min(
                guidance.ask_size_mult,
                self._vol_scaled_size_cap(fingerprint, signal, 0.75),
            )
            guidance.reasons.append("fleet:net_short")

        guidance.bid_ticks_back = min(guidance.bid_ticks_back, self.max_advisory_ticks_back)
        guidance.ask_ticks_back = min(guidance.ask_ticks_back, self.max_advisory_ticks_back)
        guidance.bid_size_mult = max(guidance.bid_size_mult, 0.35)
        guidance.ask_size_mult = max(guidance.ask_size_mult, 0.35)
        return guidance

    def _apply_advisory_sides(
        self,
        allow_bid: bool,
        allow_ask: bool,
        guidance: AdvisoryGuidance,
    ) -> tuple[bool, bool]:
        if guidance.veto_bid:
            allow_bid = False
        if guidance.veto_ask:
            allow_ask = False
        return allow_bid, allow_ask

    def _apply_advisory_size(
        self,
        quantity: float,
        side: OrderDirection,
        guidance: AdvisoryGuidance,
    ) -> float:
        mult = guidance.bid_size_mult if side == OrderDirection.BUY else guidance.ask_size_mult
        mult = max(0.35, min(mult, 1.5))
        sim_floor = float(getattr(self.simulation_config, "minOrderSize", 0.0) or 0.0)
        floor = max(self.min_order_size, sim_floor, self.quantity)
        # If the exchange floor is binding, shrinking size is not an option — prefer widening price only.
        if quantity < floor:
            quantity = floor
        if self.mainnet_mode or quantity <= floor * 1.001:
            mult = max(mult, 1.0)
        qty = round(
            quantity * mult,
            self.simulation_config.volumeDecimals,
        )
        if qty < floor:
            return 0.0
        return qty

    def _prepare_tick(self, state: MarketSimulationStateUpdate) -> TickContext:
        validator = state.dendrite.hotkey
        book_ids = sorted(state.books.keys())
        if book_ids:
            self._book_count_hint = len(book_ids)
        self._flush_score_proxies(validator, book_ids)
        peer_ratio_by_book, my_total, peer_total = self._cached_account_volumes(
            state, validator, book_ids
        )
        net = self._build_network_context(
            state, validator, book_ids, volume_totals=(my_total, peer_total)
        )
        self._network_ctx[validator] = net
        tiers = self._assign_tiers(validator, book_ids, net, peer_ratio_by_book)
        self._tier_cache[validator] = tiers

        books: dict[int, TickBookPrep] = {}
        for book_id, book in state.books.items():
            signal = self._signal(validator, book_id, book)
            fp = self._fingerprint(validator, book_id)
            ts, tl, mom = self._trend_from_fingerprint(fp)
            books[book_id] = TickBookPrep(
                signal=signal,
                trend_short=ts,
                trend_long=tl,
                momentum_align=mom,
                coverage_push=self._book_needs_coverage_push(
                    validator, book_id, state.timestamp, net
                ),
            )
        fleet_skew = self._compute_fleet_inventory_skew(state, book_ids)
        ctx = TickContext(
            validator=validator,
            timestamp=state.timestamp,
            net=net,
            tiers=tiers,
            books=books,
            peer_ratio_by_book=peer_ratio_by_book,
            fleet_inventory_skew=fleet_skew,
        )
        ctx.plans = self._compile_tick_plans(ctx, state)
        return ctx

    def _compile_tick_plans(
        self,
        ctx: TickContext,
        state: MarketSimulationStateUpdate,
    ) -> dict[int, BookActionPlan]:
        plans: dict[int, BookActionPlan] = {}
        # If configured, concentrate on a subset of books to reduce volume while still
        # meeting validator max_inactive_books constraints.
        allowed: set[int] | None = None
        focus_set: set[int] = set()
        if self.active_books_target and self.active_books_target > 0:
            target = min(int(self.active_books_target), len(state.books))
            prev = self._active_books_by_validator.get(ctx.validator, set())
            # Always include books needing coverage push (kappa-cold / under-activity / weak tail).
            needs_base = [bid for bid, prep in ctx.books.items() if prep.coverage_push]

            # Additionally pin the worst 1–N books to the active set so the *minimum*
            # per-book roundtrip metrics can improve (min-roundtrip is dominated by one laggard).
            focus: list[int] = []
            if self.focus_books_target and self.focus_books_target > 0:
                candidates: list[tuple[int, int, int, int]] = []
                for bid in state.books.keys():
                    fp = self._fingerprint(ctx.validator, bid)
                    # Smaller: fewer realized round-trips; older last activity is worse.
                    rt_count = len(fp.round_trip_pnls)
                    last = fp.last_roundtrip if fp.last_roundtrip > 0 else fp.last_position_open_ts
                    last = last if last > 0 else 0
                    candidates.append((rt_count, last, bid, bid))
                # Prefer books with fewer samples; then those not recently round-tripped.
                candidates.sort(key=lambda x: (x[0], x[1], x[2]))
                focus = [bid for *_ignore, bid in candidates[: self.focus_books_target]]
                focus_set = set(focus)

                # Light, throttled logging so we can verify focus selection from logs.
                last_log = self._last_focus_log_ts.get(ctx.validator, 0)
                if last_log == 0 or state.timestamp - last_log >= 300_000_000_000:  # first tick or ~5 min sim
                    bt.logging.info(
                        f"FOCUS books (target={self.focus_books_target}): {focus} "
                        f"rt_counts={[len(self._fingerprint(ctx.validator, b).round_trip_pnls) for b in focus]}"
                    )
                    self._last_focus_log_ts[ctx.validator] = state.timestamp

            # Order needs with focus first to guarantee inclusion.
            needs = list(dict.fromkeys(focus + needs_base))
            allowed = set(needs[:target])
            if len(allowed) < target:
                # Keep prior active set for stability, then fill with lowest-id remainder.
                for bid in sorted(prev):
                    if len(allowed) >= target:
                        break
                    allowed.add(bid)
            if len(allowed) < target:
                for bid in sorted(state.books.keys()):
                    if len(allowed) >= target:
                        break
                    allowed.add(bid)
            self._active_books_by_validator[ctx.validator] = allowed
        # Persist focus books for execution-phase logic.
        ctx.focus_books = focus_set

        for book_id, book in state.books.items():
            if allowed is not None and book_id not in allowed:
                continue
            prep = ctx.books[book_id]
            tier = ctx.tiers.get(book_id, BookTier.NEUTRAL)
            emergency = self._needs_emergency_unwind(ctx, book_id, book, prep)
            if emergency is not None:
                direction, qty = emergency
                plans[book_id] = BookActionPlan(
                    book_id=book_id,
                    emergency_dir=direction,
                    emergency_qty=qty,
                    emergency_cancel_all=True,
                )
                continue
            if prep.signal is None:
                plan = self._plan_minimum_book(
                    ctx, state, book_id, book, prep, tier
                )
            else:
                plan = self._plan_full_book(ctx, state, book_id, book)
            if plan is not None:
                plans[book_id] = plan
        self._apply_fleet_plan_adjustments(plans, ctx)
        return plans

    def _apply_fleet_plan_adjustments(
        self,
        plans: dict[int, BookActionPlan],
        ctx: TickContext,
    ) -> None:
        """Cross-book: fleet wealth stress and net inventory skew scale all plans."""
        vol_dec = self.simulation_config.volumeDecimals
        sim_floor = float(getattr(self.simulation_config, "minOrderSize", 0.0) or 0.0)
        floor = max(self.min_order_size, sim_floor, self.quantity)
        stress = ctx.net.stress_multiplier
        if ctx.net.portfolio_wealth_ratio < 0.93:
            stress = min(stress, 0.85)
        skew = ctx.fleet_inventory_skew
        for plan in plans.values():
            mult = stress
            if skew > 0.2:
                if plan.place_bid:
                    mult *= 0.88
            elif skew < -0.2:
                if plan.place_ask:
                    mult *= 0.88
            if mult < 1.0:
                if plan.place_bid:
                    plan.bid_qty = round(plan.bid_qty * mult, vol_dec)
                if plan.place_ask:
                    plan.ask_qty = round(plan.ask_qty * mult, vol_dec)
            # Never let cross-book scaling push size below exchange floor.
            if plan.place_bid and plan.bid_qty < floor:
                plan.bid_qty = round(floor, vol_dec)
            if plan.place_ask and plan.ask_qty < floor:
                plan.ask_qty = round(floor, vol_dec)

    def _plan_minimum_book(
        self,
        ctx: TickContext,
        state: MarketSimulationStateUpdate,
        book_id: int,
        book: Book,
        prep: TickBookPrep,
        tier: BookTier,
    ) -> BookActionPlan | None:
        if not book.bids or not book.asks:
            return None
        account = self.accounts[book_id]
        fingerprint = self._fingerprint(ctx.validator, book_id)
        if self._volume_ratio(account) >= self.volume_soft_cap:
            return None
        quantity = round(
            max(self.min_order_size, self.min_quote_quantity),
            self.simulation_config.volumeDecimals,
        )
        allow_bid, allow_ask = self._apply_inventory_skew(fingerprint, True, True)
        guidance = self._build_advisory(
            prep, tier, ctx.net, fingerprint, ctx.fleet_inventory_skew
        )
        bootstrap = prep.coverage_push
        bid_price, ask_price = self._prices(
            book, guidance, bootstrap=bootstrap
        )
        allow_bid, allow_ask = self._apply_advisory_sides(allow_bid, allow_ask, guidance)
        prices_changed = self._prices_changed_materially(
            fingerprint, bid_price, ask_price
        )
        period = self._effective_activity_period(ctx.net.coverage_pressure)
        under_activity = (
            fingerprint.last_roundtrip == 0
            or state.timestamp - fingerprint.last_roundtrip > period
        )
        bid_price, ask_price = self._apply_roundtrip_completion_prices(
            book,
            fingerprint,
            bid_price,
            ask_price,
            allow_bid=allow_bid,
            allow_ask=allow_ask,
            under_activity=under_activity,
            focus=book_id in ctx.focus_books,
        )
        min_refresh = self._effective_min_refresh_interval(prep, fingerprint)
        if not self._should_requote(
            fingerprint,
            state.timestamp,
            under_activity,
            account,
            bid_price,
            ask_price,
            allow_bid,
            allow_ask,
            prices_changed,
            min_refresh=min_refresh,
        ):
            return BookActionPlan(book_id=book_id)
        cancel_ids = self._collect_cancel_ids(
            account,
            book_id,
            bid_price,
            ask_price,
            allow_bid,
            allow_ask,
            fingerprint,
            prices_changed,
            max_cancels=self._max_cancel_batch(prep),
        )
        bid_qty = self._apply_advisory_size(quantity, OrderDirection.BUY, guidance)
        ask_qty = self._apply_advisory_size(quantity, OrderDirection.SELL, guidance)
        place_bid = (
            allow_bid
            and bid_qty > 0
            and not self._has_live_quote(account, OrderDirection.BUY, bid_price)
            and self._postonly_place_allowed(book, OrderDirection.BUY, bid_price)
        )
        place_ask = (
            allow_ask
            and ask_qty > 0
            and not self._has_live_quote(account, OrderDirection.SELL, ask_price)
            and self._postonly_place_allowed(book, OrderDirection.SELL, ask_price)
        )
        return BookActionPlan(
            book_id=book_id,
            cancel_ids=cancel_ids,
            place_bid=place_bid,
            place_ask=place_ask,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
        )

    def _plan_full_book(
        self,
        ctx: TickContext,
        state: MarketSimulationStateUpdate,
        book_id: int,
        book: Book,
    ) -> BookActionPlan | None:
        prep = ctx.books.get(book_id)
        if prep is None or prep.signal is None:
            return None
        signal = prep.signal
        account = self.accounts[book_id]
        tier = ctx.tiers.get(book_id, BookTier.NEUTRAL)
        tier_params = self._tier_params(tier, ctx.net)
        fingerprint = self._fingerprint(ctx.validator, book_id)
        guidance = self._build_advisory(
            prep, tier, ctx.net, fingerprint, ctx.fleet_inventory_skew
        )
        bid_price, ask_price = self._prices(book, guidance)
        prices_changed = self._prices_changed_materially(
            fingerprint, bid_price, ask_price
        )
        inventory_ratio = self._inventory_ratio(account, signal.mid)
        period = self._effective_activity_period(ctx.net.coverage_pressure)
        under_activity = (
            fingerprint.last_roundtrip == 0
            or state.timestamp - fingerprint.last_roundtrip > period
        )
        bid_price, ask_price = self._apply_roundtrip_completion_prices(
            book,
            fingerprint,
            bid_price,
            ask_price,
            allow_bid=True,  # allow flags refined below; completion uses inventory direction.
            allow_ask=True,
            under_activity=under_activity,
            focus=book_id in ctx.focus_books,
        )
        prices_changed = self._prices_changed_materially(
            fingerprint, bid_price, ask_price
        )
        vol_ratio = self._volume_ratio(account)
        defense = self._scoring_defense_active(ctx)
        kappa_cold = len(fingerprint.round_trip_pnls) < self.min_realized_observations
        bootstrap = (
            defense
            and prep.coverage_push
            and kappa_cold
        )
        if bootstrap:
            bid_price, ask_price = self._prices(book, guidance, bootstrap=True)
            prices_changed = self._prices_changed_materially(
                fingerprint, bid_price, ask_price
            )
        can_trade = (
            vol_ratio < tier_params.max_volume_ratio
            and vol_ratio < self.volume_soft_cap
            and self._maker_edge_is_positive(account, signal)
        )
        if bootstrap and vol_ratio < self.volume_soft_cap * 0.95:
            can_trade = True
        allow_bid = can_trade and inventory_ratio < tier_params.inventory_limit
        allow_ask = can_trade and inventory_ratio > -tier_params.inventory_limit
        if signal.toxic:
            if signal.flow >= self.flow_threshold:
                allow_ask = allow_ask and tier_params.force_activity
            elif signal.flow <= -self.flow_threshold:
                allow_bid = allow_bid and tier_params.force_activity
        elif (
            defense
            and tier_params.force_activity
            and under_activity
            and can_trade
        ):
            margin = self.defensive_activity_margin
            if fingerprint.score_proxy < 0.4:
                margin += 0.04
            allow_bid = inventory_ratio < tier_params.inventory_limit + margin
            allow_ask = inventory_ratio > -(tier_params.inventory_limit + margin)
        elif signal.regime == "trend" and tier != BookTier.DEFENSIVE:
            if signal.flow >= self.flow_threshold and signal.depth_imbalance >= 0:
                allow_ask = False
            elif signal.flow <= -self.flow_threshold and signal.depth_imbalance <= 0:
                allow_bid = False
        elif signal.regime == "trend" and tier == BookTier.DEFENSIVE:
            allow_bid = allow_bid and signal.flow <= self.flow_threshold * 0.5
            allow_ask = allow_ask and signal.flow >= -self.flow_threshold * 0.5
        if (
            tier == BookTier.ALPHA
            and ctx.net.alpha_unlocked
            and signal.regime == "spread"
            and not signal.toxic
        ):
            if signal.depth_imbalance > 0.12:
                allow_ask = allow_ask and inventory_ratio < tier_params.inventory_limit * 0.85
            elif signal.depth_imbalance < -0.12:
                allow_bid = allow_bid and inventory_ratio > -tier_params.inventory_limit * 0.85
        allow_bid, allow_ask = self._apply_inventory_skew(fingerprint, allow_bid, allow_ask)
        if (
            defense
            and under_activity
            and can_trade
            and not allow_bid
            and not allow_ask
            and not signal.toxic
        ):
            allow_bid = True
            allow_ask = True
        allow_bid, allow_ask = self._apply_advisory_sides(allow_bid, allow_ask, guidance)
        min_refresh = self._effective_min_refresh_interval(prep, fingerprint)
        if not self._should_requote(
            fingerprint,
            state.timestamp,
            under_activity,
            account,
            bid_price,
            ask_price,
            allow_bid,
            allow_ask,
            prices_changed,
            min_refresh=min_refresh,
        ):
            return BookActionPlan(book_id=book_id)
        cancel_ids = self._collect_cancel_ids(
            account,
            book_id,
            bid_price,
            ask_price,
            allow_bid,
            allow_ask,
            fingerprint,
            prices_changed,
            max_cancels=self._max_cancel_batch(prep),
        )
        quantity = self._quote_size(
            fingerprint, signal, account, tier_params, under_activity
        )
        if bootstrap:
            quantity = max(
                self.min_order_size,
                round(quantity * 0.75, self.simulation_config.volumeDecimals),
            )
        bid_qty = self._apply_advisory_size(quantity, OrderDirection.BUY, guidance)
        ask_qty = self._apply_advisory_size(quantity, OrderDirection.SELL, guidance)
        place_bid = (
            allow_bid
            and bid_qty > 0
            and not self._has_live_quote(account, OrderDirection.BUY, bid_price)
            and self._postonly_place_allowed(book, OrderDirection.BUY, bid_price)
        )
        place_ask = (
            allow_ask
            and ask_qty > 0
            and not self._has_live_quote(account, OrderDirection.SELL, ask_price)
            and self._postonly_place_allowed(book, OrderDirection.SELL, ask_price)
        )
        if self.advisory_log and guidance.reasons:
            bt.logging.debug(
                f"BOOK {book_id} plan tier={tier.value} "
                f"trend={prep.trend_short:.6f}/{prep.trend_long:.6f} "
                f"bid={place_bid} ask={place_ask} reasons={','.join(guidance.reasons)}"
            )
        return BookActionPlan(
            book_id=book_id,
            cancel_ids=cancel_ids,
            place_bid=place_bid,
            place_ask=place_ask,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
        )

    def _execute_plan(
        self,
        response: FinanceAgentResponse,
        state: MarketSimulationStateUpdate,
        ctx: TickContext,
        plan: BookActionPlan,
    ) -> None:
        """Apply pre-built plan — minimal work on the hot path."""
        book_id = plan.book_id
        account = self.accounts[book_id]
        fingerprint = self._fingerprint(ctx.validator, book_id)
        live_ids = set(self._open_orders(account))
        acted = False
        cap = self.max_instructions_per_book
        sim_floor = float(getattr(self.simulation_config, "minOrderSize", 0.0) or 0.0)
        floor = max(self.min_order_size, sim_floor, self.quantity)

        # Force-close stuck inventory to accelerate realized observations (Kappa unlock) and
        # keep books recently active (validator activity sampling).
        if (
            self.force_roundtrip_close_after > 0
            and (not self.force_roundtrip_close_focus_only or book_id in ctx.focus_books)
            and abs(fingerprint.signed_position) >= floor
            and self._book_instruction_count(response, book_id) < cap
        ):
            period = self._effective_activity_period(ctx.net.coverage_pressure)
            under_activity = (
                fingerprint.last_roundtrip == 0
                or state.timestamp - fingerprint.last_roundtrip > period
            )
            kappa_cold = len(fingerprint.round_trip_pnls) < self.min_realized_observations
            should_force = True
            if self.force_roundtrip_close_only_when_cold:
                should_force = kappa_cold or under_activity
            # last_roundtrip updates only on completed round-trips; for a never-closed position,
            # fall back to when the position was opened.
            anchor_ts = (
                fingerprint.last_roundtrip
                if fingerprint.last_roundtrip > 0
                else fingerprint.last_position_open_ts
            )
            if (
                should_force
                and anchor_ts > 0
                and state.timestamp - anchor_ts >= self.force_roundtrip_close_after
            ):
                cancel_batch = list(live_ids) if self.force_roundtrip_close_cancel_all else []
                for order_id in cancel_batch:
                    if self._book_instruction_count(response, book_id) >= cap - 1:
                        break
                    response.cancel_order(book_id, order_id)

                qty = round(
                    min(abs(fingerprint.signed_position), self.max_quantity),
                    self.simulation_config.volumeDecimals,
                )
                if qty >= floor and self._book_instruction_count(response, book_id) < cap:
                    direction = (
                        OrderDirection.SELL
                        if fingerprint.signed_position > 0
                        else OrderDirection.BUY
                    )
                    response.market_order(
                        book_id=book_id,
                        direction=direction,
                        quantity=qty,
                        stp=STP.CANCEL_BOTH,
                    )
                    bt.logging.info(
                        f"BOOK {book_id} : FORCED ROUNDTRIP CLOSE dir={direction.name} qty={qty} "
                        f"pos={fingerprint.signed_position:.6f} kappa_cold={kappa_cold} under_activity={under_activity}"
                    )
                    fingerprint.last_plan_ts = state.timestamp
                    fingerprint.last_bid_price = None
                    fingerprint.last_ask_price = None
                    return

        # Focus-only "seed trade" for low-liquidity books: if we're flat and the book is
        # cold/under-active for long enough, do a minimal market order to guarantee at
        # least one fill on that book. This addresses the common failure mode where the
        # weakest books never trade → min_roundtrip_volume stagnates.
        if (
            self.focus_force_trade_after > 0
            and book_id in ctx.focus_books
            and fingerprint.signed_position == 0
            and self._book_instruction_count(response, book_id) < cap
        ):
            period = self._effective_activity_period(ctx.net.coverage_pressure)
            under_activity = (
                fingerprint.last_roundtrip == 0
                or state.timestamp - fingerprint.last_roundtrip > period
            )
            kappa_cold = len(fingerprint.round_trip_pnls) < self.min_realized_observations
            anchor_ts = fingerprint.last_fill_ts if fingerprint.last_fill_ts > 0 else fingerprint.last_roundtrip
            if (kappa_cold or under_activity) and (
                anchor_ts == 0 or state.timestamp - anchor_ts >= self.focus_force_trade_after
            ):
                # Alternate direction deterministically to avoid building one-sided fleet skew.
                direction = OrderDirection.BUY if (book_id % 2 == 0) else OrderDirection.SELL
                qty = round(min(floor, self.max_quantity), self.simulation_config.volumeDecimals)
                if qty >= floor:
                    response.market_order(
                        book_id=book_id,
                        direction=direction,
                        quantity=qty,
                        stp=STP.CANCEL_BOTH,
                    )
                    bt.logging.info(
                        f"BOOK {book_id} : FOCUS SEED TRADE dir={direction.name} qty={qty} "
                        f"kappa_cold={kappa_cold} under_activity={under_activity}"
                    )
                    fingerprint.last_plan_ts = state.timestamp
                    fingerprint.last_bid_price = None
                    fingerprint.last_ask_price = None
                    return

        if plan.emergency_dir is not None and plan.emergency_qty > 0:
            cancel_batch = list(live_ids) if plan.emergency_cancel_all else []
            for order_id in cancel_batch:
                if self._book_instruction_count(response, book_id) >= cap:
                    break
                response.cancel_order(book_id, order_id)
            if self._book_instruction_count(response, book_id) < cap:
                response.market_order(
                    book_id=book_id,
                    direction=plan.emergency_dir,
                    quantity=plan.emergency_qty,
                    stp=STP.CANCEL_BOTH,
                )
                fingerprint.last_emergency_ts = state.timestamp
                fingerprint.last_plan_ts = state.timestamp
                fingerprint.last_bid_price = None
                fingerprint.last_ask_price = None
            return

        for order_id in plan.cancel_ids:
            if order_id not in live_ids:
                continue
            if order_id in fingerprint.recent_maker_fill_ids:
                continue
            if self._book_instruction_count(response, book_id) >= cap:
                break
            response.cancel_order(book_id, order_id)
            acted = True

        # Execution-time postOnly safety: BBO can move between plan and execute, turning a
        # previously safe postOnly price into a crossing order (CONTRACT_VIOLATION). Re-check
        # against latest book snapshot and buffer away if needed.
        book = state.books.get(book_id)
        bid_price = plan.bid_price
        ask_price = plan.ask_price
        if book is not None and book.bids and book.asks and (plan.place_bid or plan.place_ask):
            bid_price, ask_price = self._apply_postonly_buffer(book, bid_price, ask_price)
            if plan.place_bid and not self._postonly_place_allowed(
                book, OrderDirection.BUY, bid_price
            ):
                plan.place_bid = False
            if plan.place_ask and not self._postonly_place_allowed(
                book, OrderDirection.SELL, ask_price
            ):
                plan.place_ask = False
        vol_dec = self.simulation_config.volumeDecimals
        bid_qty = round(max(plan.bid_qty, floor), vol_dec) if plan.place_bid else 0.0
        if plan.place_bid and bid_qty < floor:
            bt.logging.info(f"BOOK {book_id} : SKIP BID qty<{floor} (bid_qty={bid_qty})")
            bid_qty = 0.0
        if plan.place_bid and account.quote_balance.free >= bid_qty * bid_price:
            if self._book_instruction_count(response, book_id) >= cap:
                if acted:
                    fingerprint.last_plan_ts = state.timestamp
                return
            response.limit_order(
                book_id=book_id,
                direction=OrderDirection.BUY,
                quantity=bid_qty,
                price=bid_price,
                postOnly=True,
                stp=self.maker_limit_stp,
                timeInForce=TimeInForce.GTT,
                expiryPeriod=self.expiry_period,
            )
            fingerprint.last_bid_price = bid_price
            acted = True
        ask_qty = round(max(plan.ask_qty, floor), vol_dec) if plan.place_ask else 0.0
        if plan.place_ask and ask_qty < floor:
            bt.logging.info(f"BOOK {book_id} : SKIP ASK qty<{floor} (ask_qty={ask_qty})")
            ask_qty = 0.0
        if plan.place_ask and account.base_balance.free >= ask_qty:
            if self._book_instruction_count(response, book_id) >= cap:
                if acted:
                    fingerprint.last_plan_ts = state.timestamp
                return
            response.limit_order(
                book_id=book_id,
                direction=OrderDirection.SELL,
                quantity=ask_qty,
                price=ask_price,
                postOnly=True,
                stp=self.maker_limit_stp,
                timeInForce=TimeInForce.GTT,
                expiryPeriod=self.expiry_period,
            )
            fingerprint.last_ask_price = ask_price
            acted = True
        if acted:
            fingerprint.last_plan_ts = state.timestamp

    def _save_warm_snapshot(self, ctx: TickContext) -> None:
        """After respond: snapshot for incremental refresh before next validator tick."""
        proxies = [
            self._fingerprint(ctx.validator, bid).score_proxy
            for bid in ctx.plans.keys()
        ]
        self._warm_snapshots[ctx.validator] = {
            "timestamp": ctx.timestamp,
            "score_median": float(np.median(proxies)) if proxies else ctx.net.book_score_median,
            "fleet_skew": ctx.fleet_inventory_skew,
            "portfolio_ratio": ctx.net.portfolio_wealth_ratio,
            "tier_count": {t.value: sum(1 for x in ctx.tiers.values() if x == t) for t in BookTier},
        }

    def _refresh_warm_scores(self, validator: str) -> None:
        """Between validator ticks: refresh fleet score from trade history."""
        fps = self.fingerprints.get(validator)
        if not fps:
            return
        proxies = [fp.score_proxy for fp in fps.values() if len(fp.round_trip_pnls) >= 2]
        if not proxies:
            return
        warm = self._warm_snapshots[validator]
        warm["score_median"] = float(np.median(proxies))
        warm["last_trade_ts"] = warm.get("last_trade_ts", 0)

    def _execution_order(self, ctx: TickContext) -> list[int]:
        """Cold / weak / defensive books first so instruction budget helps median kappa."""

        def sort_key(book_id: int) -> tuple:
            prep = ctx.books[book_id]
            fp = self._fingerprint(ctx.validator, book_id)
            tier = ctx.tiers.get(book_id, BookTier.NEUTRAL)
            obs = len(fp.round_trip_pnls)
            tier_rank = {
                BookTier.DEFENSIVE: 0,
                BookTier.NEUTRAL: 1,
                BookTier.ALPHA: 2,
            }[tier]
            return (
                0 if prep.coverage_push else 1,
                tier_rank,
                0 if obs < self.min_realized_observations else 1,
                fp.score_proxy,
                book_id,
            )

        return sorted(ctx.plans.keys(), key=sort_key)

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        response = FinanceAgentResponse(agent_id=self.uid)
        self._sync_min_order_size_from_sim()
        ctx = self._prepare_tick(state)
        for book_id in self._execution_order(ctx):
            self._execute_plan(response, state, ctx, ctx.plans[book_id])
        self._save_warm_snapshot(ctx)
        validator = ctx.validator
        if validator:
            for fp in self.fingerprints.get(validator, {}).values():
                fp.recent_maker_fill_ids.clear()
        if os.environ.get("TAOS_LAB_METRICS"):
            try:
                from lab.metrics import publish_agent_metrics

                plans = ctx.plans.values()
                orders = sum(
                    (1 if p.place_bid else 0) + (1 if p.place_ask else 0) for p in plans
                )
                cancels = sum(len(p.cancel_ids) for p in plans)
                publish_agent_metrics(self, ctx, orders=orders, cancels=cancels)
            except Exception:
                pass
        return response

    def _update_entry_on_add(
        self,
        fingerprint: BookFingerprint,
        previous_position: float,
        signed_quantity: float,
        next_position: float,
        trade_price: float,
    ) -> None:
        if trade_price <= 0:
            return
        if abs(previous_position) < 1e-12 and abs(next_position) > 1e-12:
            fingerprint.entry_price = trade_price
            return
        if previous_position > 1e-12 and signed_quantity > 0:
            fingerprint.entry_price = (
                fingerprint.entry_price * previous_position + trade_price * signed_quantity
            ) / next_position
        elif previous_position < -1e-12 and signed_quantity < 0:
            prev_abs = abs(previous_position)
            add_abs = abs(signed_quantity)
            fingerprint.entry_price = (
                fingerprint.entry_price * prev_abs + trade_price * add_abs
            ) / abs(next_position)

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if validator is None or event.bookId is None:
            return

        fingerprint = self._fingerprint(validator, event.bookId)
        is_maker = event.makerAgentId == self.uid
        is_taker = event.takerAgentId == self.uid
        is_buy = (is_taker and event.side == OrderDirection.BUY) or (
            is_maker and event.side == OrderDirection.SELL
        )
        signed_quantity = event.quantity if is_buy else -event.quantity

        previous_position = fingerprint.signed_position
        next_position = previous_position + signed_quantity
        closes_long = previous_position > 1e-12 and signed_quantity < 0
        closes_short = previous_position < -1e-12 and signed_quantity > 0

        self._update_entry_on_add(
            fingerprint, previous_position, signed_quantity, next_position, event.price
        )
        if abs(previous_position) < 1e-12 and abs(next_position) > 1e-12:
            fingerprint.last_position_open_ts = event.timestamp

        # Any fill counts as "activity" for this book.
        if event.timestamp and event.timestamp > 0:
            fingerprint.last_fill_ts = event.timestamp

        if closes_long or closes_short:
            if fingerprint.entry_price is not None and event.price > 0:
                closed_qty = min(abs(signed_quantity), abs(previous_position))
                if previous_position > 0:
                    pnl = (event.price - fingerprint.entry_price) * closed_qty
                else:
                    pnl = (fingerprint.entry_price - event.price) * closed_qty
                fingerprint.round_trip_pnls.append(pnl)
                fingerprint.score_proxy_dirty = True
                fingerprint.last_roundtrip = event.timestamp
                if validator:
                    warm = self._warm_snapshots[validator]
                    warm["last_trade_ts"] = event.timestamp
            if abs(next_position) < 1e-12:
                fingerprint.entry_price = None
                fingerprint.last_position_open_ts = 0
            elif closes_long and next_position < -1e-12:
                fingerprint.entry_price = event.price
            elif closes_short and next_position > 1e-12:
                fingerprint.entry_price = event.price
            # Partial close: entry_price (average cost) retained for remaining size.

        fingerprint.signed_position = next_position
        if is_maker:
            maker_id = getattr(event, "makerOrderId", None)
            if maker_id is not None:
                fingerprint.recent_maker_fill_ids.add(int(maker_id))


if __name__ == "__main__":
    """
    Local:
      python MedianAlignedTierAgent.py --port 8888 --agent_id 0 \\
        --params quantity=0.25 expiry_period=30000000000

    Miner (mainnet-aligned scoring):
      --agent.name MedianAlignedTierAgent \\
      --agent.params mainnet_mode=1 quantity=0.25 max_quantity=0.75 \\
        expiry_period=30000000000 lazy_load=1 max_instructions_per_book=4 \\
        min_refresh_interval=10000000000 volume_soft_cap=0.80
    """
    launch(MedianAlignedTierAgent)
