#!/bin/sh
set -e

# Independent bootstrap runner.
# Usage: ./run_sn79_bootstrap_independent.sh <COLDKEY_NAME> <HOTKEY_NAME> <AXON_PORT>
#
# PM2 process name is unique per hotkey/port:
#   sn79_bootstrap_<HOTKEY_NAME>_<AXON_PORT>

export BT_NO_PARSE_CLI_ARGS=0

ENDPOINT="wss://entrypoint-finney.opentensor.ai:443"
NETUID="79"
WALLET_PATH="$HOME/.bittensor/wallets"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv_sn79_8081/bin/python"
VENV_PIP="$SCRIPT_DIR/.venv_sn79_8081/bin/pip"

AGENT_PATH="$SCRIPT_DIR/agents"
AGENT_NAME="MedianBootstrapScoreAgent"
AGENT_PARAMS="mainnet_mode=1"
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
PROCESS_NAME="sn79_bootstrap_${SAFE_HOTKEY}_${AXON_PORT}"

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
