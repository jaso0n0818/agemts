#!/bin/sh
set -e

# SN79 miner runner (isolated venv + isolated agent path + 8081-series ports).
# Usage:
#   ./run_sn79_8081.sh <COLDKEY_NAME> <HOTKEY_NAME> [AXON_PORT]
#
# Notes:
# - Does NOT touch any existing PM2 processes by default (process name: sn79_8081).
# - Uses a dedicated venv under this repo: .venv_sn79_8081 (Python 3.10.9).

export BT_NO_PARSE_CLI_ARGS=0

ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
NETUID="79"
WALLET_PATH="$HOME/.bittensor/wallets"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv_sn79_8081/bin/python"
VENV_PIP="$SCRIPT_DIR/.venv_sn79_8081/bin/pip"

AGENT_PATH="$HOME/.taos/agents_sn79_8081"
AGENT_NAME="MedianAlignedTierAgent"
AGENT_PARAMS="mainnet_mode=1 scoring_defense=1 fast_scoring_mode=1 peer_scan_enabled=0 portfolio_scan_enabled=0 alpha_fraction=0 quantity=0.25 min_order_size=0.25 min_quote_quantity=0.25 max_quantity=0.75 min_realized_observations=3 activity_period=180000000000 volume_soft_cap=0.90 min_refresh_interval=10000000000 max_instructions_per_book=4 postonly_buffer_ticks=2 requote_price_tolerance_ticks=1 maker_limit_stp=1 emergency_unwind=1 vpin_threshold=0.62 toxic_refresh_factor=0.45 toxic_max_cancels=1 volume_scan_interval=60 lazy_load=1 expiry_period=30000000000 roundtrip_complete_ticks=4 roundtrip_complete_only_when_cold=0 roundtrip_min_profit_bps=0.2 active_books_target=80 active_books_min_ratio=0.625 active_books_target_margin=0 small_book_reserve_fraction=0.70 profit_book_reserve_fraction=0.25 small_book_volume_percentile=40 small_book_activity_period=90000000000 small_book_requote_interval=5000000000 small_book_roundtrip_ticks_add=2 small_book_min_profit_bps=0.05 small_book_force_close_after=30000000000 small_book_force_close_min_return_bps=0.05 small_book_max_cancels=1 profit_book_min_proxy=0.62 profit_book_size_mult=1.25 profit_book_min_profit_bps=0.35 profit_book_force_close_after=90000000000 profit_book_force_close_min_return_bps=0.35 force_roundtrip_close_after=45000000000 force_roundtrip_close_only_when_cold=1 force_roundtrip_close_cancel_all=0 force_roundtrip_close_min_return_bps=0.2 market_close_reentry_cooldown=30000000000 event_log_interval=200 response_timing_interval=20 compact_report_interval=50 slow_response_warn_s=1.0 forward_timing_interval=20 forward_slow_warn_s=1.0"
LOG_LEVEL="info"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: venv not found at $VENV_PYTHON"
  echo "Run the environment setup first (creates .venv_sn79_8081)."
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <COLDKEY_NAME> <HOTKEY_NAME> [AXON_PORT]"
  exit 2
fi

WALLET_NAME="$1"
HOTKEY_NAME="$2"
AXON_PORT="${3:-8081}"

# Prevent concurrent runs.
LOCK_DIR="$SCRIPT_DIR/.run_sn79_8081.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_DIR"
  if ! flock -n 9; then
    echo "Another run_sn79_8081.sh is already running (lock held at $LOCK_DIR)."
    echo "Status: pm2 ls"
    echo "Logs:   pm2 logs sn79_8081 --lines 100"
    exit 0
  fi
fi

# Ensure agent file exists in the isolated agent directory.
if [ ! -f "$AGENT_PATH/$AGENT_NAME.py" ]; then
  echo "ERROR: agent file missing: $AGENT_PATH/$AGENT_NAME.py"
  echo "Expected setup to copy it from repo agents/."
  exit 1
fi

echo "ENDPOINT: $ENDPOINT"
echo "NETUID: $NETUID"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "AXON_PORT: $AXON_PORT"
echo "AGENT_PATH: $AGENT_PATH"
echo "AGENT_NAME: $AGENT_NAME"
echo "AGENT_PARAMS: $AGENT_PARAMS"
echo "VENV_PYTHON: $VENV_PYTHON"

# Start (or restart) dedicated PM2 process name to avoid collisions.
MINER_DIR="$SCRIPT_DIR/taos/im/neurons"
if pm2 describe sn79_8081 >/dev/null 2>&1; then
  pm2 stop sn79_8081 >/dev/null 2>&1 || true
  pm2 delete sn79_8081 >/dev/null 2>&1 || true
fi

pm2 start "$MINER_DIR/miner.py" \
  --name sn79_8081 \
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
echo ""
echo "Status: pm2 ls"
echo "Logs:   pm2 logs sn79_8081 --lines 100"
