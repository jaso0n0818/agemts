# Hot path: onTrade defers heavy kappa proxy to respond().
from types import SimpleNamespace

from MedianAlignedTierAgent import MedianAlignedTierAgent
from taos.im.protocol.models import OrderDirection


def make_agent():
    config = SimpleNamespace(lazy_load=False, scoring_defense=0, mainnet_mode=0)
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_hotpath_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=4,
        priceDecimals=2,
        volumeDecimals=4,
    )
    return agent


def test_on_trade_marks_dirty_not_eager_proxy():
    agent = make_agent()
    fp = agent._fingerprint("v", 0)
    fp.signed_position = 1.0
    fp.entry_price = 300.0
    fp.score_proxy = 0.9

    # Buy-initiated trade: taker hits our resting sell → closes long.
    event = SimpleNamespace(
        bookId=0,
        makerAgentId=0,
        takerAgentId=1,
        side=OrderDirection.BUY,
        quantity=1.0,
        price=299.0,
        timestamp=1_000,
    )
    agent.onTrade(event, validator="v")
    assert fp.score_proxy_dirty
    assert fp.score_proxy == 0.9

    agent._flush_score_proxies("v", [0])
    assert not fp.score_proxy_dirty
    assert fp.score_proxy != 0.9 or len(fp.round_trip_pnls) < 3


def test_scoring_defense_off_skips_bootstrap_override():
    agent = make_agent()
    ctx = SimpleNamespace(
        net=SimpleNamespace(coverage_pressure=0.8),
    )
    assert not agent._scoring_defense_active(ctx)


def test_volume_scan_cache_reduces_peer_pass():
    agent = make_agent()
    agent.volume_scan_interval = 100
    validator = "v"
    book_ids = [0, 1]
    state = SimpleNamespace(
        accounts={
            0: {0: SimpleNamespace(traded_volume=10.0), 1: SimpleNamespace(traded_volume=5.0)},
            1: {0: SimpleNamespace(traded_volume=50.0), 1: SimpleNamespace(traded_volume=20.0)},
        }
    )
    agent.accounts = {
        0: SimpleNamespace(traded_volume=10.0),
        1: SimpleNamespace(traded_volume=5.0),
    }
    agent.uid = 0

    calls = 0
    orig = agent._scan_peer_volumes

    def counted(state, book_ids):
        nonlocal calls
        calls += 1
        return orig(state, book_ids)

    agent._scan_peer_volumes = counted
    agent._cached_account_volumes(state, validator, book_ids)
    agent._cached_account_volumes(state, validator, book_ids)
    assert calls == 1


if __name__ == "__main__":
    test_on_trade_marks_dirty_not_eager_proxy()
    test_scoring_defense_off_skips_bootstrap_override()
    test_volume_scan_cache_reduces_peer_pass()
    print("PASS: hotpath")
