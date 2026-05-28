# Quick unit test for partial-fill entry_price handling (no simulator).
from types import SimpleNamespace

from MedianAlignedTierAgent import MedianAlignedTierAgent, BookFingerprint
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.models import OrderDirection


def make_agent():
    config = SimpleNamespace(lazy_load=False, data_dir="/tmp/median_partial_test/data")
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_partial_test")
    agent.simulation_config = SimpleNamespace(miner_wealth=50_000.0, book_count=16)
    return agent


def trade(agent, book_id, qty, side, price, ts):
    event = SimpleNamespace(
        bookId=book_id,
        makerAgentId=agent.uid,
        takerAgentId=999,
        side=side,
        quantity=qty,
        price=price,
        timestamp=ts,
    )
    agent.onTrade(event, validator="v")


def test_partial_long_close_keeps_entry():
    agent = make_agent()
    fp = agent._fingerprint("v", 0)
    # Maker buy (bid filled): taker sells -> event.side SELL
    trade(agent, 0, 10.0, OrderDirection.SELL, 100.0, 1)
    assert abs(fp.signed_position - 10.0) < 1e-9
    assert fp.entry_price == 100.0

    # Maker sell (reduce long): taker buys -> event.side BUY
    trade(agent, 0, 3.0, OrderDirection.BUY, 102.0, 2)
    assert abs(fp.signed_position - 7.0) < 1e-9
    assert fp.entry_price == 100.0, "partial close must not wipe average entry"
    assert len(fp.round_trip_pnls) == 1
    assert abs(fp.round_trip_pnls[-1] - 6.0) < 1e-9

    trade(agent, 0, 7.0, OrderDirection.BUY, 101.0, 3)
    assert abs(fp.signed_position) < 1e-9
    assert fp.entry_price is None
    assert len(fp.round_trip_pnls) == 2


if __name__ == "__main__":
    test_partial_long_close_keeps_entry()
    print("PASS: partial fill entry_price")
