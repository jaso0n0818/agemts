# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Bootstrap scoring variant for a newly registered SN79 miner.

This file intentionally does not modify MedianAlignedTierAgent. It subclasses the
current production agent and only supplies more activity-oriented defaults when
the same key is not already provided in --agent.params.
"""

from MedianAlignedTierAgent import MedianAlignedTierAgent


class MedianBootstrapScoreAgent(MedianAlignedTierAgent):
    """
    Fast score-unlock profile.

    Goal:
      - Build >=3 realized round-trip observations on enough books quickly.
      - Keep PnL damage controlled by requiring small non-negative force closes.
      - Reduce over-sensitive defensive tiering during the first scoring window.
    """

    BOOTSTRAP_DEFAULTS = {
        "mainnet_mode": 1,
        "scoring_defense": 1,
        "fast_scoring_mode": 1,
        "peer_scan_enabled": 0,
        "portfolio_scan_enabled": 0,
        "alpha_fraction": 0.0,
        # More activity than the safe baseline, but not a full 2x size jump.
        "quantity": 0.30,
        "min_order_size": 0.25,
        "min_quote_quantity": 0.30,
        "max_quantity": 1.00,
        "volume_soft_cap": 0.98,
        # New miners need a wider active set than the bare 62.5% threshold.
        "active_books_target": 96,
        "active_books_min_ratio": 0.625,
        "active_books_target_margin": 0,
        "max_inactive_books_ratio": 0.25,
        "min_realized_observations": 3,
        # Faster completion and gentler tiny-profit thresholds for kappa readiness.
        "roundtrip_complete_ticks": 5,
        "roundtrip_complete_only_when_cold": 0,
        "roundtrip_min_profit_bps": 0.05,
        "force_roundtrip_close_after": 25_000_000_000,
        "force_roundtrip_close_only_when_cold": 1,
        "force_roundtrip_close_cancel_all": 0,
        "force_roundtrip_close_min_return_bps": 0.00,
        "market_close_reentry_cooldown": 25_000_000_000,
        # Keep most slots focused on weak/small books until kappa_ready improves.
        "small_book_reserve_fraction": 0.82,
        "profit_book_reserve_fraction": 0.12,
        "small_book_volume_percentile": 45.0,
        "small_book_activity_period": 60_000_000_000,
        "small_book_requote_interval": 4_000_000_000,
        "small_book_roundtrip_ticks_add": 3,
        "small_book_min_profit_bps": 0.00,
        "small_book_force_close_after": 20_000_000_000,
        "small_book_force_close_min_return_bps": 0.00,
        "small_book_max_cancels": 1,
        # Defensive exists, but should not shrink activity too early.
        "defensive_entry_buffer": 0.04,
        "defensive_quantity_mult": 1.00,
        "defensive_max_volume_ratio": 0.68,
        "defensive_inventory_mult": 0.90,
        "defensive_fraction": 0.12,
        "weak_book_proxy_floor": 0.38,
        # Local proxy should not scare the miner away from slightly noisy books.
        "kappa_proxy_mu_boost": 1.15,
        "kappa_proxy_downside_weight": 0.65,
        "kappa_proxy_regularization_scale": 0.07,
        # Preserve contract safety while increasing touch coverage.
        "postonly_buffer_ticks": 3,
        "requote_price_tolerance_ticks": 1,
        "maker_limit_stp": 1,
        "max_instructions_per_book": 4,
        "lazy_load": 1,
        "expiry_period": 30_000_000_000,
        "event_log_interval": 200,
        "response_timing_interval": 20,
        "compact_report_interval": 50,
        "slow_response_warn_s": 1.0,
        "forward_timing_interval": 20,
        "forward_slow_warn_s": 1.0,
    }

    def initialize(self):
        for key, value in self.BOOTSTRAP_DEFAULTS.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
        super().initialize()


if __name__ == "__main__":
    from taos.common.agents import launch

    launch(MedianBootstrapScoreAgent)
