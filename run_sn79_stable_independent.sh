#!/bin/sh
set -e

# Independent stable runner.
# Usage: ./run_sn79_stable_independent.sh <COLDKEY_NAME> <HOTKEY_NAME> <AXON_PORT>
#
# PM2 process name is unique per hotkey/port:
#   sn79_stable_<HOTKEY_NAME>_<AXON_PORT>

export BT_NO_PARSE_CLI_ARGS=0

ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
NETUID="79"
WALLET_PATH="$HOME/.bittensor/wallets"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv_sn79_8081/bin/python"
VENV_PIP="$SCRIPT_DIR/.venv_sn79_8081/bin/pip"

AGENT_PATH="$SCRIPT_DIR/agents"
AGENT_NAME="MedianAlignedTierAgent"
AGENT_PARAMS="mainnet_mode=1 scoring_defense=1 fast_scoring_mode=1 peer_scan_enabled=0 portfolio_scan_enabled=0 alpha_fraction=0 quantity=0.35 min_order_size=0.25 min_quote_quantity=0.30 max_quantity=1.25 min_realized_observations=3 activity_period=180000000000 max_inactive_books_ratio=0.30 volume_soft_cap=0.95 min_refresh_interval=10000000000 max_instructions_per_book=4 postonly_buffer_ticks=2 requote_price_tolerance_ticks=1 maker_limit_stp=1 emergency_unwind=1 vpin_threshold=0.62 toxic_refresh_factor=0.45 toxic_max_cancels=1 volume_scan_interval=60 lazy_load=1 expiry_period=30000000000 roundtrip_complete_ticks=4 roundtrip_complete_only_when_cold=0 roundtrip_min_profit_bps=0.2 active_books_target=88 active_books_min_ratio=0.625 active_books_target_margin=0 defensive_entry_buffer=0.03 defensive_quantity_mult=0.95 defensive_max_volume_ratio=0.62 defensive_inventory_mult=0.85 defensive_fraction=0.16 weak_book_proxy_floor=0.39 kappa_proxy_mu_boost=1.12 kappa_proxy_downside_weight=0.70 kappa_proxy_regularization_scale=0.08 small_book_reserve_fraction=0.70 profit_book_reserve_fraction=0.25 small_book_volume_percentile=40 small_book_activity_period=90000000000 small_book_requote_interval=5000000000 small_book_roundtrip_ticks_add=2 small_book_min_profit_bps=0.05 small_book_force_close_after=30000000000 small_book_force_close_min_return_bps=0.05 small_book_max_cancels=1 profit_book_min_proxy=0.62 profit_book_size_mult=1.45 profit_book_min_profit_bps=0.35 profit_book_force_close_after=90000000000 profit_book_force_close_min_return_bps=0.35 force_roundtrip_close_after=45000000000 force_roundtrip_close_only_when_cold=1 force_roundtrip_close_cancel_all=0 force_roundtrip_close_min_return_bps=0.2 market_close_reentry_cooldown=30000000000 event_log_interval=200 response_timing_interval=20 compact_report_interval=50 slow_response_warn_s=1.0 forward_timing_interval=20 forward_slow_warn_s=1.0"
LOG_LEVEL="info"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: venv not found at $VENV_PYTHON"
  exit 1
fi

if [ $# -lt 3 ]; then
  echo "Usage: $0 <COLDKEY_NAME> <HOTKEY_NAME> <AXON_PORT>"
  exit 2
fi

WALLET_NAME="$1"
HOTKEY_NAME="$2"
AXON_PORT="$3"
SAFE_HOTKEY="$(printf '%s' "$HOTKEY_NAME" | tr -c 'A-Za-z0-9_' '_')"
PROCESS_NAME="sn79_stable_${SAFE_HOTKEY}_${AXON_PORT}"

"$VENV_PIP" install -e .

MINER_DIR="$SCRIPT_DIR/taos/im/neurons"
pm2 stop "$PROCESS_NAME" >/dev/null 2>&1 || true
pm2 delete "$PROCESS_NAME" >/dev/null 2>&1 || true

pm2 start "$MINER_DIR/miner.py" \
  --name "$PROCESS_NAME" \
  --interpreter "$VENV_PYTHON" \
  --cwd "$MINER_DIR" \
  -- \
  --netuid "$NETUID" \
  --subtensor.chain_endpoint "$ENDPOINT" \
  --wallet.path "$WALLET_PATH" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$HOTKEY_NAME" \
  --axon.port "$AXON_PORT" \
  --agent.path "$AGENT_PATH" \
  --agent.name "$AGENT_NAME" \
  --agent.params $AGENT_PARAMS \
  --logging."$LOG_LEVEL"

pm2 save
echo "Started $PROCESS_NAME"
echo "Logs: pm2 logs $PROCESS_NAME --lines 100"
