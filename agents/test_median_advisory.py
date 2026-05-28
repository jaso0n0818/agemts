# Advisory layer: widen quotes; primary maker remains sole order source.
from types import SimpleNamespace

from MedianAlignedTierAgent import (
    AdvisoryGuidance,
    BookSignal,
    BookTier,
    MedianAlignedTierAgent,
    NetworkContext,
    TickBookPrep,
)


def make_agent():
    config = SimpleNamespace(
        lazy_load=False,
        data_dir="/tmp/median_advisory_test/data",
        trend_block_bps=3.0,
        momentum_block_bps=2.0,
    )
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_advisory_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=8,
        priceDecimals=2,
        volumeDecimals=2,
    )
    return agent


def test_trend_guard_widens_bid_not_full_veto():
    agent = make_agent()
    validator = "v"
    fp = agent._fingerprint(validator, 0)
    for p in [100.0, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0, 96.5, 96.0, 95.5]:
        fp.midquotes.append(p)
    ts, tl, mom = agent._trend_from_fingerprint(fp)
    assert mom < 0
    prep = TickBookPrep(
        signal=BookSignal(
            mid=95.5,
            spread=0.1,
            depth_imbalance=0.0,
            flow=0.0,
            reaction=0.0,
            volatility=0.0,
            toxic=False,
            regime="trend",
        ),
        trend_short=ts,
        trend_long=tl,
        momentum_align=mom,
    )
    g = agent._build_advisory(prep, BookTier.NEUTRAL, NetworkContext(), fp)
    assert g.bid_ticks_back >= 1
    assert not g.veto_bid
    bid, ask = agent._apply_advisory_sides(True, True, g)
    assert bid and ask


def test_pnl_reference_ignores_wealth_drop():
    agent = make_agent()
    agent._pnl_baseline_wealth = 50_000.0
    fp = agent._fingerprint("v", 0)
    agent.simulation_config.miner_wealth = 10_000.0
    ref_low = agent._book_pnl_reference(fp)
    agent.simulation_config.miner_wealth = 50_000.0
    ref_high = agent._book_pnl_reference(fp)
    assert ref_low == ref_high


def test_prices_join_bbo_by_default():
    agent = make_agent()
    book = SimpleNamespace(
        bids=[SimpleNamespace(price=100.0)],
        asks=[SimpleNamespace(price=100.2)],
    )
    bid, ask = agent._prices(book)
    assert bid == 100.0
    assert ask == 100.2


def test_advisory_never_enables_without_primary():
    agent = make_agent()
    g = AdvisoryGuidance()
    bid, ask = agent._apply_advisory_sides(False, False, g)
    assert not bid and not ask


if __name__ == "__main__":
    test_trend_guard_widens_bid_not_full_veto()
    test_pnl_reference_ignores_wealth_drop()
    test_prices_join_bbo_by_default()
    test_advisory_never_enables_without_primary()
    print("PASS: median advisory")
