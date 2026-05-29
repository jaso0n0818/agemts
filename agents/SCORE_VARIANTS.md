# Median score variants

These files are additive profiles. They do not modify `MedianAlignedTierAgent.py`.

## MedianBootstrapScoreAgent

Use this when a miner has just registered, has score near zero, or needs to
build kappa-ready observations quickly.

Main behavior:

- Targets 96 active books.
- Uses moderate size: `quantity=0.30`, `max_quantity=1.00`.
- Completes round trips faster with near-zero positive/flat force-close
  thresholds.
- Keeps most capacity in small/weak books until enough books have at least
  three realized PnL observations.
- Makes DEFENSIVE harder to enter and less size-destructive.

Run:

```bash
cd /home/administrator/hyj/agemts
./run_sn79_8081_bootstrap.sh jason apex_2 8081
```

## MedianAggressiveScoreAgent

Use this after activity and kappa readiness are already established, and the
goal is to push higher trading/PnL score.

Main behavior:

- Targets 88 active books.
- Uses larger size: `quantity=0.45`, `max_quantity=1.75`.
- Opens a small alpha lane when the fleet score proxy is healthy.
- Allocates more books to the profit lane and increases profit-book size.
- Keeps `postonly_buffer_ticks=3` to offset the larger quote size.

Run:

```bash
cd /home/administrator/hyj/agemts
./run_sn79_8081_aggressive.sh jason apex_2 8081
```

## Switching back

To return to the current production profile:

```bash
cd /home/administrator/hyj/agemts
./run_sn79_8081.sh jason apex_2 8081
```

## Independent multi-hotkey runners

Use these when three different hotkeys should run at the same time. Each script
uses a unique PM2 process name derived from the profile, hotkey name, and port.
They do not stop or delete the other profiles.

Example layout:

```bash
cd /home/administrator/hyj/agemts

./run_sn79_stable_independent.sh jason apex_2 8081
./run_sn79_bootstrap_independent.sh jason first 8099
./run_sn79_aggressive_independent.sh jason second 8101
```

Resulting PM2 process names:

```text
sn79_stable_apex_2_8081
sn79_bootstrap_first_8099
sn79_aggressive_second_8101
```

Check logs separately:

```bash
pm2 logs sn79_stable_apex_2_8081 --lines 100
pm2 logs sn79_bootstrap_first_8099 --lines 100
pm2 logs sn79_aggressive_second_8101 --lines 100
```
