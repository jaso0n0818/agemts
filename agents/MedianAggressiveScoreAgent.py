# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""
Aggressive scoring variant for an already scoring SN79 miner.

This variant keeps the production MedianAlignedTierAgent logic intact and only
changes defaults toward higher volume on good books and less defensive throttling.
Use after the miner already has stable activity/kappa coverage.
"""

from MedianAlignedTierAgent import MedianAlignedTierAgent


class MedianAggressiveScoreAgent(MedianAlignedTierAgent):
    """
    Higher-upside profile.

    Goal:
      - Keep enough active books to avoid inactivity pressure.
      - Push larger maker size on profitable/high-proxy books.
      - Maintain a real postOnly buffer because the profile is more aggressive.
    """

    AGGRESSIVE_DEFAULTS = {
        "mainnet_mode": 1,
        "scoring_defense": 1,
        "fast_scoring_mode": 1,
        "peer_scan_enabled": 0,
        "portfolio_scan_enabled": 0,
        # Allow a small alpha lane once the fleet median is healthy.
        "alpha_fraction": 0.08,
        "fleet_alpha_unlock_median": 0.53,
        "quantity": 0.45,
        "min_order_size": 0.25,
        "min_quote_quantity": 0.35,
        "max_quantity": 1.75,
        "volume_soft_cap": 1.00,
        "active_books_target": 88,
        "active_books_min_ratio": 0.625,
        "active_books_target_margin": 0,
        "max_inactive_books_ratio": 0.30,
        "min_realized_observations": 3,
        # Do not force loss-making round trips too quickly in aggressive mode.
        "roundtrip_complete_ticks": 4,
        "roundtrip_complete_only_when_cold": 0,
        "roundtrip_min_profit_bps": 0.20,
        "force_roundtrip_close_after": 45_000_000_000,
        "force_roundtrip_close_only_when_cold": 1,
        "force_roundtrip_close_cancel_all": 0,
        "force_roundtrip_close_min_return_bps": 0.15,
        "market_close_reentry_cooldown": 30_000_000_000,
        # More capacity for good books while preserving a small-book lane.
        "small_book_reserve_fraction": 0.58,
        "profit_book_reserve_fraction": 0.36,
        "small_book_volume_percentile": 35.0,
        "small_book_activity_period": 90_000_000_000,
        "small_book_requote_interval": 5_000_000_000,
        "small_book_roundtrip_ticks_add": 2,
        "small_book_min_profit_bps": 0.05,
        "small_book_force_close_after": 35_000_000_000,
        "small_book_force_close_min_return_bps": 0.05,
        "small_book_max_cancels": 1,
        "profit_book_min_proxy": 0.58,
        "profit_book_size_mult": 1.75,
        "profit_book_min_profit_bps": 0.40,
        "profit_book_force_close_after": 110_000_000_000,
        "profit_book_force_close_min_return_bps": 0.40,
        # Defensive should be harder to enter and less size-destructive.
        "defensive_entry_buffer": 0.035,
        "defensive_quantity_mult": 1.05,
        "defensive_max_volume_ratio": 0.70,
        "defensive_inventory_mult": 0.92,
        "defensive_fraction": 0.10,
        "weak_book_proxy_floor": 0.37,
        "kappa_proxy_mu_boost": 1.18,
        "kappa_proxy_downside_weight": 0.60,
        "kappa_proxy_regularization_scale": 0.07,
        # More aggressive size needs a little more postOnly protection.
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
        for key, value in self.AGGRESSIVE_DEFAULTS.items():
            if not hasattr(self.config, key):
                setattr(self.config, key, value)
        super().initialize()


if __name__ == "__main__":
    from taos.common.agents import launch

    launch(MedianAggressiveScoreAgent)
