# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT

from collections import defaultdict, deque
from dataclasses import dataclass, field

from taos.common.agents import launch
from taos.im.agents import FinanceSimulationAgent
from taos.im.protocol import FinanceAgentResponse, MarketSimulationStateUpdate
from taos.im.protocol.events import TradeEvent
from taos.im.protocol.instructions import STP, TimeInForce
from taos.im.protocol.models import Book, OrderDirection


@dataclass
class BookFingerprint:
    midquotes: deque = field(default_factory=lambda: deque(maxlen=24))
    returns: deque = field(default_factory=lambda: deque(maxlen=24))
    reactions: deque = field(default_factory=lambda: deque(maxlen=24))
    previous_mid: float | None = None
    previous_flow: float = 0.0
    last_roundtrip: int = 0
    signed_position: float = 0.0


@dataclass(frozen=True)
class BookSignal:
    mid: float
    spread: float
    depth_imbalance: float
    flow: float
    reaction: float
    volatility: float
    toxic: bool
    regime: str


class ScoreAwareFlowMakerAgent(FinanceSimulationAgent):
    """
    Maker-first agent built around the score surface visible in the validator:
    realized round-trips, per-book consistency, and fee-adjusted execution.
    """

    def initialize(self):
        self.quantity = float(getattr(self.config, "quantity", 0.25))
        self.max_quantity = float(getattr(self.config, "max_quantity", self.quantity * 3))
        self.expiry_period = int(getattr(self.config, "expiry_period", 30_000_000_000))
        self.depth = int(getattr(self.config, "depth", 5))
        self.min_edge_bps = float(getattr(self.config, "min_edge_bps", 0.5)) / 10_000
        self.max_maker_fee = float(getattr(self.config, "max_maker_fee", 0.003))
        self.max_volume_ratio = float(getattr(self.config, "max_volume_ratio", 0.85))
        self.volume_cap_turnover = float(getattr(self.config, "volume_cap_turnover", 10.0))
        self.inventory_limit = float(getattr(self.config, "inventory_limit", 0.30))
        self.flow_threshold = float(getattr(self.config, "flow_threshold", 0.35))
        self.reaction_threshold = float(getattr(self.config, "reaction_threshold", 0.00001))
        self.toxic_volatility_ratio = float(getattr(self.config, "toxic_volatility_ratio", 1.25))
        self.activity_period = int(getattr(self.config, "activity_period", 900_000_000_000))
        self.fingerprints = defaultdict(lambda: defaultdict(BookFingerprint))

    def _fingerprint(self, validator: str, book_id: int) -> BookFingerprint:
        return self.fingerprints[validator][book_id]

    def _trade_flow(self, book: Book) -> float:
        trades = [event for event in (book.events or []) if event.type == "t"]
        total = sum(trade.quantity for trade in trades)
        if total <= 0:
            return 0.0
        signed = sum(
            trade.quantity if trade.side == OrderDirection.BUY else -trade.quantity
            for trade in trades
        )
        return max(-1.0, min(1.0, signed / total))

    def _depth_imbalance(self, book: Book) -> float:
        bid_depth = sum(level.quantity for level in book.bids[: self.depth])
        ask_depth = sum(level.quantity for level in book.asks[: self.depth])
        total_depth = bid_depth + ask_depth
        if total_depth <= 0:
            return 0.0
        return (bid_depth - ask_depth) / total_depth

    def _mean(self, values: deque) -> float:
        return sum(values) / len(values) if values else 0.0

    def _signal(self, validator: str, book_id: int, book: Book) -> BookSignal | None:
        if not book.bids or not book.asks:
            return None

        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        spread = best_ask - best_bid
        if best_bid <= 0 or spread <= 0:
            return None

        mid = (best_bid + best_ask) / 2
        flow = self._trade_flow(book)
        fingerprint = self._fingerprint(validator, book_id)

        if fingerprint.previous_mid:
            realized_return = (mid - fingerprint.previous_mid) / fingerprint.previous_mid
            fingerprint.returns.append(abs(realized_return))
            if fingerprint.previous_flow:
                fingerprint.reactions.append(fingerprint.previous_flow * realized_return)

        fingerprint.midquotes.append(mid)
        fingerprint.previous_mid = mid
        fingerprint.previous_flow = flow

        reaction = self._mean(fingerprint.reactions)
        volatility = self._mean(fingerprint.returns)
        depth_imbalance = self._depth_imbalance(book)
        relative_spread = spread / mid
        directional_flow = abs(flow) >= self.flow_threshold
        continuing_flow = reaction > self.reaction_threshold
        volatile_for_spread = volatility > relative_spread * self.toxic_volatility_ratio
        toxic = volatile_for_spread or (directional_flow and continuing_flow)

        if reaction > self.reaction_threshold:
            regime = "trend"
        elif reaction < -self.reaction_threshold:
            regime = "reversion"
        else:
            regime = "spread"

        return BookSignal(
            mid=mid,
            spread=spread,
            depth_imbalance=depth_imbalance,
            flow=flow,
            reaction=reaction,
            volatility=volatility,
            toxic=toxic,
            regime=regime,
        )

    def _volume_ratio(self, account) -> float:
        traded_volume = account.traded_volume
        if traded_volume is None or self.simulation_config.miner_wealth <= 0:
            return 0.0
        volume_cap = self.volume_cap_turnover * self.simulation_config.miner_wealth
        return traded_volume / volume_cap if volume_cap > 0 else 0.0

    def _inventory_ratio(self, account, mid: float) -> float:
        own_base_value = max(account.own_base, 0.0) * mid
        own_quote = max(account.own_quote, 0.0)
        wealth = own_base_value + own_quote
        if wealth <= 0:
            return 0.0
        return (own_base_value / wealth) - 0.5

    def _prices(self, book: Book) -> tuple[float, float]:
        tick = 10 ** (-self.simulation_config.priceDecimals)
        best_bid = book.bids[0].price
        best_ask = book.asks[0].price
        bid = best_bid + tick if best_bid + tick < best_ask else best_bid
        ask = best_ask - tick if best_ask - tick > best_bid else best_ask
        return (
            round(bid, self.simulation_config.priceDecimals),
            round(ask, self.simulation_config.priceDecimals),
        )

    def _quote_size(self, fingerprint: BookFingerprint, signal: BookSignal, account) -> float:
        size = self.quantity
        if signal.regime == "spread" and not signal.toxic:
            size *= 1.5
        if fingerprint.last_roundtrip == 0:
            size = min(size, self.quantity)
        if self._volume_ratio(account) > self.max_volume_ratio * 0.65:
            size *= 0.5
        return round(
            max(self.quantity, min(size, self.max_quantity)),
            self.simulation_config.volumeDecimals,
        )

    def _has_live_quote(self, account, side: OrderDirection, price: float) -> bool:
        return any(order.side == side and order.price == price for order in account.orders)

    def _cancel_vulnerable_quotes(
        self,
        response: FinanceAgentResponse,
        book_id: int,
        account,
        bid_price: float,
        ask_price: float,
        allow_bid: bool,
        allow_ask: bool,
    ) -> None:
        # Two slots are reserved for the replacement bid and ask in this response.
        for order in account.orders:
            if len([i for i in response.instructions if i.bookId == book_id]) >= 3:
                break
            if order.side == OrderDirection.BUY:
                keep = allow_bid and order.price == bid_price
            else:
                keep = allow_ask and order.price == ask_price
            if not keep:
                response.cancel_order(book_id, order.id)

    def _maker_edge_is_positive(self, account, signal: BookSignal) -> bool:
        maker_fee = account.fees.maker_fee_rate if account.fees else 0.0
        if maker_fee > self.max_maker_fee:
            return False
        required_edge = max(self.min_edge_bps, (2 * maker_fee) + self.min_edge_bps)
        return signal.spread > signal.mid * required_edge

    def respond(self, state: MarketSimulationStateUpdate) -> FinanceAgentResponse:
        validator = state.dendrite.hotkey
        response = FinanceAgentResponse(agent_id=self.uid)

        for book_id, book in state.books.items():
            account = self.accounts[book_id]
            signal = self._signal(validator, book_id, book)
            if signal is None:
                continue

            bid_price, ask_price = self._prices(book)
            fingerprint = self._fingerprint(validator, book_id)
            inventory_ratio = self._inventory_ratio(account, signal.mid)
            under_activity_target = (
                fingerprint.last_roundtrip == 0
                or state.timestamp - fingerprint.last_roundtrip > self.activity_period
            )
            can_trade = (
                self._volume_ratio(account) < self.max_volume_ratio
                and self._maker_edge_is_positive(account, signal)
            )

            allow_bid = can_trade and inventory_ratio < self.inventory_limit
            allow_ask = can_trade and inventory_ratio > -self.inventory_limit

            if signal.toxic and not under_activity_target:
                if signal.flow >= self.flow_threshold:
                    allow_ask = False
                elif signal.flow <= -self.flow_threshold:
                    allow_bid = False
                else:
                    allow_bid = False
                    allow_ask = False
            elif signal.regime == "trend":
                if signal.flow >= self.flow_threshold and signal.depth_imbalance >= 0:
                    allow_ask = False
                elif signal.flow <= -self.flow_threshold and signal.depth_imbalance <= 0:
                    allow_bid = False

            self._cancel_vulnerable_quotes(
                response,
                book_id,
                account,
                bid_price,
                ask_price,
                allow_bid,
                allow_ask,
            )

            quantity = self._quote_size(fingerprint, signal, account)
            if allow_bid and not self._has_live_quote(account, OrderDirection.BUY, bid_price):
                if account.quote_balance.free >= quantity * bid_price:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.BUY,
                        quantity=quantity,
                        price=bid_price,
                        postOnly=True,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period,
                    )
            if allow_ask and not self._has_live_quote(account, OrderDirection.SELL, ask_price):
                if account.base_balance.free >= quantity:
                    response.limit_order(
                        book_id=book_id,
                        direction=OrderDirection.SELL,
                        quantity=quantity,
                        price=ask_price,
                        postOnly=True,
                        stp=STP.CANCEL_BOTH,
                        timeInForce=TimeInForce.GTT,
                        expiryPeriod=self.expiry_period,
                    )

        return response

    def onTrade(self, event: TradeEvent, validator: str = None) -> None:
        if validator is None or event.bookId is None:
            return

        fingerprint = self._fingerprint(validator, event.bookId)
        is_maker = event.makerAgentId == self.uid
        is_taker = event.takerAgentId == self.uid
        is_buy = (is_taker and event.side == OrderDirection.BUY) or (
            is_maker and event.side == OrderDirection.SELL
        )
        signed_quantity = event.quantity if is_buy else -event.quantity

        previous_position = fingerprint.signed_position
        next_position = previous_position + signed_quantity
        closes_long = previous_position > 0 and signed_quantity < 0
        closes_short = previous_position < 0 and signed_quantity > 0
        if closes_long or closes_short:
            fingerprint.last_roundtrip = event.timestamp
        fingerprint.signed_position = next_position


if __name__ == "__main__":
    """
    Example local run:
    python ScoreAwareFlowMakerAgent.py --port 8888 --agent_id 0 --params quantity=0.25 expiry_period=30000000000
    """
    launch(ScoreAwareFlowMakerAgent)
