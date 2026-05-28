#!/bin/sh
# bittensor 10 defaults to BT_NO_PARSE_CLI_ARGS=true; must enable CLI parsing for pm2 args.
export BT_NO_PARSE_CLI_ARGS=0

ENDPOINT=wss://test.finney.opentensor.ai:443
WALLET_PATH=~/.bittensor/wallets/
WALLET_NAME=vanta_test
HOTKEY_NAME=miner1
NETUID=366
AXON_PORT=8099
AGENT_PATH=~/.taos/agents
AGENT_NAME=MedianAlignedTierAgent
AGENT_PARAMS="quantity=0.25 max_quantity=0.75 expiry_period=30000000000 lazy_load=1 active_books_target=80"
LOG_LEVEL=info
FORCE_RESTART=0
while getopts e:p:w:h:u:a:g:n:m:l:f flag
do
    case "${flag}" in
        e) ENDPOINT=${OPTARG};;
        p) WALLET_PATH=${OPTARG};;
        w) WALLET_NAME=${OPTARG};;
        h) HOTKEY_NAME=${OPTARG};;
        u) NETUID=${OPTARG};;
        a) AXON_PORT=${OPTARG};;
        g) AGENT_PATH=${OPTARG};;
        n) AGENT_NAME=${OPTARG};;
        m) AGENT_PARAMS=${OPTARG};;
        l) LOG_LEVEL=${OPTARG};;
        f) FORCE_RESTART=1;;
    esac
done
echo "ENDPOINT: $ENDPOINT"
echo "WALLET_PATH: $WALLET_PATH"
echo "WALLET_NAME: $WALLET_NAME"
echo "HOTKEY_NAME: $HOTKEY_NAME"
echo "NETUID: $NETUID"
echo "AXON_PORT: $AXON_PORT"
echo "AGENT_PATH: $AGENT_PATH"
echo "AGENT_NAME: $AGENT_NAME"
echo "AGENT_PARAMS: $AGENT_PARAMS"
echo "FORCE_RESTART: $FORCE_RESTART"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"
VENV_PIP="$SCRIPT_DIR/.venv/bin/pip"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Virtualenv not found. Create it with:"
    echo "  cd $SCRIPT_DIR && python3.10 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# Prevent concurrent runs (e.g., two shells invoking this script).
# Use a repo-local lock so it works consistently across environments.
LOCK_DIR="$SCRIPT_DIR/.run_miner.lock"
if command -v flock >/dev/null 2>&1; then
    LOCK_FD=9
    # shellcheck disable=SC2094
    exec 9>"$LOCK_DIR"
    if ! flock -n 9; then
        echo "Another run_miner.sh is already running (lock held at $LOCK_DIR)."
        echo "If you're trying to inspect status/logs, use: pm2 ls ; pm2 logs miner --lines 100"
        exit 0
    fi
else
    echo "Warning: flock not found; cannot guarantee single-instance execution."
fi

if pm2 describe miner >/dev/null 2>&1; then
    if pm2 jlist | grep -q '"name":"miner".*"status":"online"' && [ "$FORCE_RESTART" -ne 1 ]; then
        echo "pm2 process 'miner' is already online. Not restarting (use -f to force)."
        echo "Status: pm2 ls"
        echo "Logs:   pm2 logs miner --lines 100"
        exit 0
    fi
fi

git pull
"$VENV_PIP" install -e .

MINER_DIR="$SCRIPT_DIR/taos/im/neurons"
if [ "$FORCE_RESTART" -eq 1 ]; then
    pm2 delete miner 2>/dev/null || true
fi
pm2 start "$MINER_DIR/miner.py" \
  --name miner \
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
echo "Verify venv interpreter:"
pm2 describe miner | grep -E 'script path|interpreter|exec cwd'
echo "Logs: pm2 logs miner --lines 100"