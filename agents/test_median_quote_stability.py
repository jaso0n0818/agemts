# Quote stability / requote gating (no simulator).
from types import SimpleNamespace

from MedianAlignedTierAgent import MedianAlignedTierAgent, BookFingerprint
from taos.im.protocol.models import OrderDirection


def make_agent():
    config = SimpleNamespace(lazy_load=False, min_refresh_interval=10_000_000_000)
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_quote_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=16,
        priceDecimals=2,
        volumeDecimals=4,
    )
    return agent


def test_should_requote_skips_when_stable():
    agent = make_agent()
    fp = BookFingerprint()
    fp.last_bid_price = 100.0
    fp.last_ask_price = 101.0
    fp.last_plan_ts = 1_000_000_000

    order = SimpleNamespace(
        id=1, side=OrderDirection.BUY, price=100.0
    )
    order2 = SimpleNamespace(id=2, side=OrderDirection.SELL, price=101.0)
    account = SimpleNamespace(orders=[order, order2])

    assert not agent._should_requote(
        fp,
        5_000_000_000,
        False,
        account,
        100.0,
        101.0,
        True,
        True,
        False,
    )


def test_should_requote_when_missing_quote():
    agent = make_agent()
    fp = BookFingerprint()
    account = SimpleNamespace(orders=[])
    assert agent._should_requote(
        fp,
        0,
        False,
        account,
        100.0,
        101.0,
        True,
        True,
        False,
    )


if __name__ == "__main__":
    test_should_requote_skips_when_stable()
    test_should_requote_when_missing_quote()
    print("PASS: quote stability")
