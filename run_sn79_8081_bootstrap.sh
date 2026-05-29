#!/bin/sh
set -e

# Bootstrap profile: use when a UID has just joined or has no score yet.
# It replaces the normal sn79_8081 PM2 process on the same port.

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

if [ $# -lt 2 ]; then
  echo "Usage: $0 <COLDKEY_NAME> <HOTKEY_NAME> [AXON_PORT]"
  exit 2
fi

WALLET_NAME="$1"
HOTKEY_NAME="$2"
AXON_PORT="${3:-8081}"

"$VENV_PIP" install -e .

MINER_DIR="$SCRIPT_DIR/taos/im/neurons"
pm2 stop sn79_8081 >/dev/null 2>&1 || true
pm2 delete sn79_8081 >/dev/null 2>&1 || true

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
echo "Bootstrap miner started. Logs: pm2 logs sn79_8081 --lines 100"
