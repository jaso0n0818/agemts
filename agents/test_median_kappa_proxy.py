from types import SimpleNamespace

from MedianAlignedTierAgent import MedianAlignedTierAgent, BookFingerprint


def make_agent():
    config = SimpleNamespace(
        lazy_load=False,
        kappa_tau=0.0,
        kappa_norm_min=-2.5,
        kappa_norm_max=2.5,
        min_realized_observations=3,
    )
    agent = MedianAlignedTierAgent(0, config, log_dir="/tmp/median_kappa_test")
    agent.simulation_config = SimpleNamespace(
        miner_wealth=50_000.0,
        book_count=16,
        priceDecimals=2,
        volumeDecimals=4,
    )
    return agent


def test_kappa_proxy_is_bounded():
    agent = make_agent()
    fp = BookFingerprint()
    fp.round_trip_pnls.extend([1.0, -0.5, 0.2, -0.1, 0.3])
    agent._update_score_proxy(fp)
    assert 0.0 <= fp.score_proxy <= 1.0


def test_kappa_proxy_defaults_when_insufficient_obs():
    agent = make_agent()
    fp = BookFingerprint()
    fp.round_trip_pnls.extend([1.0, -1.0])  # < min_realized_observations
    agent._update_score_proxy(fp)
    assert fp.score_proxy == 0.5


if __name__ == "__main__":
    test_kappa_proxy_is_bounded()
    test_kappa_proxy_defaults_when_insufficient_obs()
    print("PASS: kappa proxy")

