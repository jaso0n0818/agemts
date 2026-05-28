# Tier gating: no ALPHA when fleet median is weak (macro lock).
from types import SimpleNamespace

import numpy as np

from MedianAlignedTierAgent import BookTier, MedianAlignedTierAgent


def make_agent():
    config = SimpleNamespace(
        lazy_load=False,
        data_dir="/tmp/median_network_test/data",
        alpha_absolute_min=0.58,
        fleet_alpha_unlock_median=0.48,
    )
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_network_test")
    agent.simulation_config = SimpleNamespace(miner_wealth=50_000.0, book_count=8)
    return agent


def test_fleet_lock_blocks_alpha():
    agent = make_agent()
    validator = "v"
    book_ids = list(range(8))
    for i in book_ids:
        fp = agent._fingerprint(validator, i)
        # Losses large enough that score_proxy stays below fleet_alpha_unlock_median (0.48).
        fp.round_trip_pnls.extend([-8.0, -5.0, -3.0])
        agent._update_score_proxy(fp)

    state = SimpleNamespace(
        accounts={0: {i: SimpleNamespace(traded_volume=10.0, own_base=0, own_quote=30000) for i in book_ids}},
        books={i: SimpleNamespace(bids=[SimpleNamespace(price=300)], asks=[SimpleNamespace(price=301)]) for i in book_ids},
    )
    agent.accounts = state.accounts[0]
    net = agent._build_network_context(state, validator, book_ids)
    assert not net.alpha_unlocked
    assert net.max_alpha_slots == 0
    ratios, _, _ = agent._scan_account_volumes(state, book_ids)
    tiers = agent._assign_tiers(validator, book_ids, net, ratios)
    assert BookTier.ALPHA not in tiers.values()


def test_absolute_alpha_requires_high_proxy():
    agent = make_agent()
    validator = "v"
    book_ids = list(range(4))
    for i, pnl in enumerate([100.0, 80.0, -5.0, -8.0]):
        fp = agent._fingerprint(validator, i)
        fp.round_trip_pnls.extend([pnl, pnl * 0.9, pnl * 0.8])
        agent._update_score_proxy(fp)

    state = SimpleNamespace(
        accounts={0: {i: SimpleNamespace(traded_volume=100.0, own_base=0, own_quote=30000) for i in book_ids}},
        books={i: SimpleNamespace(bids=[SimpleNamespace(price=300)], asks=[SimpleNamespace(price=301)]) for i in book_ids},
    )
    agent.accounts = state.accounts[0]
    net = agent._build_network_context(state, validator, book_ids)
    ratios, _, _ = agent._scan_account_volumes(state, book_ids)
    tiers = agent._assign_tiers(validator, book_ids, net, ratios)
    alpha_books = [b for b, t in tiers.items() if t == BookTier.ALPHA]
    for bid in alpha_books:
        assert agent._fingerprint(validator, bid).score_proxy >= agent.alpha_absolute_min - 0.01


def test_cold_start_peer_ratio_not_defensive():
    agent = make_agent()
    state = SimpleNamespace(
        accounts={
            0: {0: SimpleNamespace(traded_volume=0.0, own_base=0, own_quote=30000)},
            1: {0: SimpleNamespace(traded_volume=5000.0, own_base=0, own_quote=30000)},
        },
        books={0: SimpleNamespace(bids=[SimpleNamespace(price=300)], asks=[SimpleNamespace(price=301)])},
    )
    ratios, _, _ = agent._scan_account_volumes(state, [0])
    assert ratios[0] == 0.0


if __name__ == "__main__":
    test_fleet_lock_blocks_alpha()
    test_absolute_alpha_requires_high_proxy()
    test_cold_start_peer_ratio_not_defensive()
    print("PASS: network tier gates")
