# Execution hygiene: postOnly buffer, material requote, maker STP defaults.
from types import SimpleNamespace

from MedianAlignedTierAgent import BookFingerprint, MedianAlignedTierAgent
from taos.im.protocol.instructions import STP
from taos.im.protocol.models import OrderDirection


def make_agent(mainnet: bool = True):
    config = SimpleNamespace(
        lazy_load=False,
        mainnet_mode=int(mainnet),
        postonly_buffer_ticks=1,
        requote_price_tolerance_ticks=1,
    )
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_exec_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=16,
        priceDecimals=2,
        volumeDecimals=4,
    )
    return agent


def tight_book():
    bid = SimpleNamespace(price=100.0, quantity=1.0)
    ask = SimpleNamespace(price=100.01, quantity=1.0)
    return SimpleNamespace(bids=[bid], asks=[ask])


def test_mainnet_defaults_cancel_oldest():
    agent = make_agent(mainnet=True)
    assert agent.maker_limit_stp == STP.CANCEL_OLDEST
    assert agent.postonly_buffer_ticks == 1


def test_postonly_buffer_widens_tight_spread():
    agent = make_agent()
    book = tight_book()
    bid, ask = agent._apply_postonly_buffer(book, 100.0, 100.01)
    assert bid < 100.0
    assert ask > 100.01
    assert bid < ask


def test_postonly_place_allowed_rejects_crossing_sell():
    agent = make_agent()
    book = tight_book()
    assert not agent._postonly_place_allowed(book, OrderDirection.SELL, 100.0)
    assert agent._postonly_place_allowed(book, OrderDirection.SELL, 100.02)


def test_prices_changed_materially_ignores_noise():
    agent = make_agent()
    fp = BookFingerprint()
    fp.last_bid_price = 100.0
    fp.last_ask_price = 101.0
    assert not agent._prices_changed_materially(fp, 100.0, 101.0)
    assert agent._prices_changed_materially(fp, 100.02, 101.0)


def test_skip_cancel_for_recent_maker_fill():
    agent = make_agent()
    fp = agent._fingerprint("v", 0)
    fp.recent_maker_fill_ids.add(42)
    order = SimpleNamespace(id=42, side=OrderDirection.BUY, price=99.0)
    account = SimpleNamespace(orders=[order])
    ids = agent._collect_cancel_ids(
        account,
        0,
        98.0,
        102.0,
        True,
        True,
        fp,
        True,
        max_cancels=2,
    )
    assert 42 in ids
    plan = SimpleNamespace(
        book_id=0,
        cancel_ids=ids,
        place_bid=False,
        place_ask=False,
        bid_price=98.0,
        ask_price=102.0,
        bid_qty=0.0,
        ask_qty=0.0,
        emergency_dir=None,
        emergency_qty=0.0,
        emergency_cancel_all=False,
    )
    from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate

    state = SimpleNamespace(timestamp=1)
    ctx = SimpleNamespace(validator="v", plans={0: plan})
    response = FinanceAgentResponse(agent_id=0)
    agent.accounts = {0: account}
    agent._execute_plan(response, state, ctx, plan)
    assert not any(
        getattr(i, "orderId", None) == 42 or getattr(i, "o", None) == 42
        for i in response.instructions
    )


if __name__ == "__main__":
    test_mainnet_defaults_cancel_oldest()
    test_postonly_buffer_widens_tight_spread()
    test_postonly_place_allowed_rejects_crossing_sell()
    test_prices_changed_materially_ignores_noise()
    test_skip_cancel_for_recent_maker_fill()
    print("PASS: median execution")
