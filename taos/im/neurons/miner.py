# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2025 Rayleigh Research

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if __name__ != "__mp_main__":
    import time
    import typing
    import traceback
    import bittensor as bt

    from taos.common.neurons.miner import BaseMinerNeuron
    from taos.im.protocol import MarketSimulationStateUpdate
    from taos.im.protocol.gentrx import GenTRXAssignment

    class Miner(BaseMinerNeuron):
        """
        Miner class implementation for intelligent market simulations.

        Overrides state processing methods to provide the correct signature when attaching axon handlers.
        """
        def __init__(self):
            super().__init__()
            self.agent.subtensor = self.subtensor
            self.agent.metagraph = self.metagraph
            self.agent.config.netuid = self.config.netuid
            _gtx = getattr(self.agent, "_gtx", None)
            if _gtx is not None and getattr(_gtx, "model", None) is None:
                self.agent._ensure_model_version()
            self.axon.attach(
                forward_fn=self.forward_gentrx_assignment,
                blacklist_fn=self.blacklist_gentrx_assignment,
                priority_fn=self.priority_gentrx_assignment,
            )
            self._commit_gentrx_bucket()

        def _commit_gentrx_bucket(self) -> None:
            """Commit GenTRX S3 bucket credentials when GenTRX is installed/configured."""
            try:
                from GenTRX.src.chain import BucketInfo, GenTRXChain
            except ImportError:
                bt.logging.debug("GenTRX not installed - skipping bucket commitment")
                return

            bucket_info = BucketInfo.from_env()
            if bucket_info is None:
                bt.logging.info(
                    "GenTRX env vars not set - skipping bucket commitment "
                    "(miner not participating in GenTRX)"
                )
                return

            try:
                chain = GenTRXChain(self.subtensor, self.config.netuid, self.metagraph)
                chain.commit_bucket(self.wallet, bucket_info)
                bt.logging.info(
                    f"GenTRX bucket committed on-chain: account={bucket_info.account_id}"
                )
            except Exception as exc:
                bt.logging.warning(
                    f"GenTRX bucket commitment failed (will retry on next start): {exc}"
                )

        async def forward_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> GenTRXAssignment:
            """Accept latest GenTRX assignment synapses; non-GenTRX agents ignore them."""
            try:
                bt.logging.info(
                    f"[GTX] assignment received: round={synapse.round} "
                    f"model_version={synapse.model_version} books={synapse.books} "
                    f"files={len(synapse.data)} validator_uid={synapse.validator_uid}"
                )
                gtx = getattr(self.agent, "_gtx", None)
                if gtx is None:
                    bt.logging.debug("[GTX] agent has no _gtx; assignment ignored")
                    return synapse
                gtx.pending_assignments.append(
                    {
                        "round": synapse.round,
                        "model_version": synapse.model_version,
                        "books": synapse.books,
                        "ts_start": synapse.ts_start,
                        "ts_end": synapse.ts_end,
                        "data": synapse.data,
                        "data_source": synapse.data_source,
                        "data_endpoint": synapse.data_endpoint,
                        "data_bucket": synapse.data_bucket,
                        "data_access_key": synapse.data_access_key,
                        "data_secret_key": synapse.data_secret_key,
                        "validator_uid": synapse.validator_uid,
                    }
                )
            except Exception:
                bt.logging.error(
                    f"[GTX] forward_gentrx_assignment failed:\n{traceback.format_exc()}"
                )
                raise
            return synapse

        def blacklist_gentrx_assignment(
            self, synapse: GenTRXAssignment
        ) -> typing.Tuple[bool, str]:
            return self.blacklist(synapse)

        def priority_gentrx_assignment(self, synapse: GenTRXAssignment) -> float:
            return self.priority(synapse)

        async def forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> MarketSimulationStateUpdate:
            """
            Processes incoming market simulation state synapse by forwarding to the associated agent class for handling.

            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                taos.im.protocol.MarketSimulationStateUpdate: The synapse object with the 'response' field updated with any instructions generated by the agent.
            """
            total_start = time.time()
            start = total_start
            synapse.decompress(lazy=self.config.agent.params.lazy_load)
            decompress_s = time.time() - start
            start = time.time()
            synapse.response = self.agent.handle(synapse)
            handle_s = time.time() - start
            instruction_count = (
                len(synapse.response.instructions)
                if synapse.response is not None and synapse.response.instructions is not None
                else 0
            )
            start = time.time()
            compressed = synapse.clear_inputs().compress()
            compress_s = time.time() - start
            total_s = time.time() - total_start
            seq = getattr(self, "_forward_timing_seq", 0)
            self._forward_timing_seq = seq + 1
            params = self.config.agent.params
            interval = int(float(getattr(params, "forward_timing_interval", 20)))
            slow_s = float(getattr(params, "forward_slow_warn_s", 1.0))
            if total_s >= slow_s or (interval > 0 and seq % interval == 0):
                bt.logging.info(
                    "FORWARD_TIMING "
                    f"total={total_s:.4f}s decompress={decompress_s:.4f}s "
                    f"handle={handle_s:.4f}s compress={compress_s:.4f}s "
                    f"instructions={instruction_count}"
                )
            return compressed
        
        def blacklist_forward(
            self, synapse: MarketSimulationStateUpdate
        ) -> typing.Tuple[bool, str]:
            """
            Apply default blacklisting to all received market simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                (bool, str): Tuple containing [1] boolean indicating if the request was blacklisted [2] string containing the message indicating reason for blacklisting.
            """
            return self.blacklist(synapse)
        
        def priority_forward(self, synapse: MarketSimulationStateUpdate) -> float:
            """
            Apply default prioritization to all received simulation state synapses.
            
            Args:
                synapse (taos.im.protocol.MarketSimulationStateUpdate): The synapse object containing the latest simulation state update.

            Returns:
                float: A priority score calculated using the standard priority function.
            """
            return self.priority(synapse)

# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            time.sleep(5)
