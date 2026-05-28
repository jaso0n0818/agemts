# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import asyncio
import json
import argparse
import os
import sys
from datetime import datetime
from functools import wraps
from loguru import logger

#--------------------------------------------------------------------------

def coro_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

async def watch(name, stream, stderr=False):
    async for line in stream:
        text = line.decode().strip()
        parsed_log = '|'.join(text.split('|')[2:]) if datetime.now().strftime('%Y-%m-%d') in text and len(text.split('|')) >=3 else text
        if not stderr:
            if not '| SUCCESS  |' in text:
                logger.info(f"{name} | {parsed_log}")
            else:
                logger.success(f"{name} | {parsed_log}")
        else:
            logger.error(f"{name} | {parsed_log}")

#--------------------------------------------------------------------------

async def launch_proxy(config_file):
    try:
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "proxy.py", "--config", config_file,
            limit = 1024 * 512,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # This is required for SIGINT to also kill the subprocess.
        await asyncio.gather(watch("PROXY", proc.stdout), watch("PROXY", proc.stderr, True))
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise

#--------------------------------------------------------------------------

async def launch_agent(name, args):
    try:
        logger.info(f"Launching agent {name}")
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, *args,
            limit = 1024 * 512,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # This is required for SIGINT to also kill the subprocess.
        await asyncio.gather(watch(name, proc.stdout), watch(name, proc.stderr, True))
    except asyncio.CancelledError:
        proc.terminate()
        await proc.wait()
        raise
    except Exception as ex:
        logger.error(f"Failed to launch agent {name}  : {ex}")

#--------------------------------------------------------------------------

@coro_wrapper
async def main(config_file):
    config = json.load(open(config_file))
    port = config['agents']['start_port']
    agents_dir = config['agents']['path']
    agent_id = 0
    argss = []
    for agent, agent_configs in config['agents'].items():
        if agent in ['start_port', 'path']: continue
        for agent_config in agent_configs:
            base_agent_name = f"{agent}_{'_'.join(list([str(a) for a in agent_config['params'].values()]))}"
            for i in range(agent_config['count']):
                agent_name = f"{base_agent_name}_{i}"
                argss.append([
                    agent_name,
                    [f"{agents_dir}/{agent}.py", "--port", str(port),
                    "--agent_id", str(agent_id),
                    "--params"] + [f"{param}={val}" for param, val in agent_config['params'].items()]
                ])
                port += 1
                agent_id += 1
    proxy_port = config['proxy']['port']
    logger.info(f"Running proxy at port {proxy_port} and agents on ports {config['agents']['start_port']}-{port}...")
    try:
        await asyncio.gather(*([launch_proxy(config_file)] + [launch_agent(name, args) for name, args in argss]))
    except asyncio.CancelledError:
        logger.success("\nServers successfully shut down.")

#--------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)

#--------------------------------------------------------------------------
