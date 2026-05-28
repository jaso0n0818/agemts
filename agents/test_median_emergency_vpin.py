# Emergency unwind + VPIN signal tests.
from types import SimpleNamespace

from taos.im.protocol.models import OrderDirection

from MedianAlignedTierAgent import BookSignal, MedianAlignedTierAgent, TickBookPrep


def make_agent():
    config = SimpleNamespace(
        lazy_load=False,
        emergency_unwind=1,
        emergency_inventory_ratio=0.35,
        emergency_loss_bps=10.0,
        emergency_cooldown=0,
        vpin_threshold=0.6,
        quantity=0.25,
        min_order_size=0.25,
    )
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_emergency_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=8,
        priceDecimals=2,
        volumeDecimals=4,
    )
    agent.accounts = {
        0: SimpleNamespace(
            own_base=1.0,
            own_quote=30_000.0,
            orders=[],
            traded_volume=0.0,
            base_balance=SimpleNamespace(free=1.0),
            quote_balance=SimpleNamespace(free=30_000.0),
            fees=SimpleNamespace(maker_fee_rate=0.0),
        )
    }
    return agent


def test_vpin_detects_one_sided_flow():
    agent = make_agent()
    fp = agent._fingerprint("v", 0)
    for _ in range(20):
        fp.flow_signed_volumes.append(1.0)
    assert agent._vpin(fp) >= 0.99


def test_emergency_unwind_long_in_toxic_downtrend():
    agent = make_agent()
    validator = "v"
    fp = agent._fingerprint(validator, 0)
    fp.signed_position = 1.0
    fp.entry_price = 310.0
    fp.round_trip_pnls.extend([1.0, 0.5, -0.2])
    for _ in range(24):
        fp.flow_signed_volumes.append(-1.0)

    book = SimpleNamespace(
        bids=[SimpleNamespace(price=298.0)],
        asks=[SimpleNamespace(price=299.0)],
        events=[],
    )
    signal = BookSignal(
        mid=298.5,
        spread=1.0,
        depth_imbalance=-0.3,
        flow=-0.8,
        reaction=0.0,
        volatility=0.01,
        toxic=True,
        regime="trend",
        vpin=0.95,
    )
    prep = TickBookPrep(
        signal=signal,
        trend_short=-0.01,
        trend_long=-0.008,
        momentum_align=-1.0,
    )
    ctx = SimpleNamespace(
        validator=validator,
        timestamp=100_000_000_000,
        net=SimpleNamespace(portfolio_wealth_ratio=0.95, coverage_pressure=0.0),
        tiers={0: agent._tier_params.__class__},
    )
    from MedianAlignedTierAgent import BookTier

    ctx.tiers = {0: BookTier.DEFENSIVE}
    result = agent._needs_emergency_unwind(ctx, 0, book, prep)
    assert result is not None
    direction, qty = result
    assert direction == OrderDirection.SELL
    assert qty >= agent.min_order_size


if __name__ == "__main__":
    test_vpin_detects_one_sided_flow()
    test_emergency_unwind_long_in_toxic_downtrend()
    print("PASS: emergency + vpin")
