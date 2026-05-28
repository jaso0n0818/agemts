# Mainnet scoring: coverage pressure, tier gates, execution order.
from types import SimpleNamespace

from MedianAlignedTierAgent import BookTier, MedianAlignedTierAgent, TickBookPrep, TickContext


def make_agent(**overrides):
    params = {
        "lazy_load": False,
        "mainnet_mode": 1,
        "min_realized_observations": 3,
        "max_inactive_books_ratio": 0.375,
        "coverage_safety_margin": 0.05,
    }
    params.update(overrides)
    config = SimpleNamespace(**params)
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_mainnet_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=128,
        priceDecimals=2,
        volumeDecimals=4,
    )
    return agent


def test_coverage_pressure_when_fleet_cold():
    agent = make_agent()
    validator = "v"
    book_ids = list(range(16))
    for i in book_ids:
        fp = agent._fingerprint(validator, i)
        if i < 2:
            fp.round_trip_pnls.extend([1.0, 0.5, -0.2])
        else:
            fp.round_trip_pnls.append(0.1)

    state = SimpleNamespace(
        accounts={
            0: {
                i: SimpleNamespace(
                    traded_volume=10.0, own_base=0, own_quote=30_000, orders=[]
                )
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
    net = agent._build_network_context(state, validator, book_ids)
    assert net.kappa_ready_ratio == 0.125
    assert net.coverage_pressure > 0.5
    assert not net.alpha_unlocked


def test_kappa_cold_books_never_alpha():
    agent = make_agent()
    validator = "v"
    book_ids = [0, 1, 2, 3]
    for i, pnls in enumerate([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0], [1.0], [0.5]]):
        fp = agent._fingerprint(validator, i)
        fp.round_trip_pnls.extend(pnls)
        agent._update_score_proxy(fp)

    state = SimpleNamespace(
        accounts={
            0: {
                i: SimpleNamespace(
                    traded_volume=200.0, own_base=0, own_quote=30_000, orders=[]
                )
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
    net = agent._build_network_context(state, validator, book_ids)
    net.alpha_unlocked = True
    net.max_alpha_slots = 2
    ratios, _, _ = agent._scan_account_volumes(state, book_ids)
    tiers = agent._assign_tiers(validator, book_ids, net, ratios)
    assert tiers[2] != BookTier.ALPHA
    assert tiers[3] != BookTier.ALPHA


def test_execution_order_prioritizes_coverage_push():
    agent = make_agent()
    validator = "v"
    ctx = TickContext(
        validator=validator,
        timestamp=1_000_000_000_000,
        net=SimpleNamespace(coverage_pressure=0.8),
        tiers={0: BookTier.ALPHA, 1: BookTier.DEFENSIVE, 2: BookTier.NEUTRAL},
        books={
            0: TickBookPrep(signal=None, coverage_push=False),
            1: TickBookPrep(signal=None, coverage_push=True),
            2: TickBookPrep(signal=None, coverage_push=False),
        },
        plans={0: None, 1: None, 2: None},
    )
    order = agent._execution_order(ctx)
    assert order[0] == 1


if __name__ == "__main__":
    test_coverage_pressure_when_fleet_cold()
    test_kappa_cold_books_never_alpha()
    test_execution_order_prioritizes_coverage_push()
    print("PASS: mainnet scoring")
