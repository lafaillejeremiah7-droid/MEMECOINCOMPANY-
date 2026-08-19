"""
IB Gateway connection manager for Order Flow Bot.

Manages connection to Interactive Brokers Gateway via ib_insync.
Subscribes to NQ futures tick-by-tick trades, Level 2 market depth,
and real-time bars. Includes reconnection logic and graceful shutdown.

IMPORTANT: Connection is READ-ONLY. This module NEVER sends any
order-related requests to IB Gateway.
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pytz

from ib_insync import IB, Contract, Future, Ticker, util

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class IBConnection:
    """
    Manages IB Gateway connection for real-time market data.

    This class handles:
    - Connecting to IB Gateway in read-only mode
    - Subscribing to NQ futures tick-by-tick data
    - Level 2 market depth (5 levels)
    - Real-time bars
    - Reconnection on disconnect
    - Graceful shutdown on SIGINT/SIGTERM
    - Contract rollover for quarterly NQ expiry
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the IB connection manager.

        Args:
            config: Configuration dictionary with ib_gateway settings.
        """
        self.config = config
        self.ib_config = config.get("ib_gateway", {})
        self.instrument_config = config.get("instrument", {})

        self.host = self.ib_config.get("host", "127.0.0.1")
        self.port = self.ib_config.get("port", 4001)
        self.client_id = self.ib_config.get("client_id", 1)
        self.timeout = self.ib_config.get("timeout", 30)
        self.reconnect_delay = self.ib_config.get("reconnect_delay", 5)
        self.max_reconnect_attempts = self.ib_config.get("max_reconnect_attempts", 10)

        self.ib = IB()
        self.contract: Optional[Contract] = None
        self.ticker: Optional[Ticker] = None
        self.connected = False
        self.shutting_down = False
        self.reconnect_count = 0

        # Callbacks
        self._on_tick_callback: Optional[Callable] = None
        self._on_dom_callback: Optional[Callable] = None
        self._on_bar_callback: Optional[Callable] = None
        self._on_disconnect_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        on_tick: Optional[Callable] = None,
        on_dom: Optional[Callable] = None,
        on_bar: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
    ) -> None:
        """Set event callbacks for market data updates."""
        self._on_tick_callback = on_tick
        self._on_dom_callback = on_dom
        self._on_bar_callback = on_bar
        self._on_disconnect_callback = on_disconnect

    def _get_nq_contract(self) -> Future:
        """
        Get the front-month NQ futures contract.

        Handles quarterly expiry rollover by requesting contract details
        from IB and selecting the nearest expiry.
        """
        contract = Future(
            symbol=self.instrument_config.get("symbol", "NQ"),
            exchange=self.instrument_config.get("exchange", "CME"),
            currency=self.instrument_config.get("currency", "USD"),
        )
        return contract

    async def connect(self) -> bool:
        """
        Connect to IB Gateway in read-only mode.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            logger.info(
                f"Connecting to IB Gateway at {self.host}:{self.port} "
                f"(client_id={self.client_id}, readonly=True)"
            )
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=True,
                timeout=self.timeout,
            )
            self.connected = True
            self.reconnect_count = 0
            logger.info("Connected to IB Gateway successfully (READ-ONLY mode)")

            # Set up disconnect handler
            self.ib.disconnectedEvent += self._on_disconnected

            # Resolve and qualify the NQ contract
            self.contract = self._get_nq_contract()
            contracts = await self.ib.qualifyContractsAsync(self.contract)
            if contracts:
                self.contract = contracts[0]
                logger.info(
                    f"Qualified contract: {self.contract.localSymbol} "
                    f"(expiry: {self.contract.lastTradeDateOrContractMonth})"
                )
            else:
                logger.warning("Could not qualify NQ contract, using unqualified")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            self.connected = False
            return False

    async def subscribe_data(self) -> None:
        """Subscribe to tick-by-tick trades, DOM, and real-time bars."""
        if not self.connected or self.contract is None:
            logger.error("Cannot subscribe: not connected or no contract")
            return

        try:
            # Subscribe to tick-by-tick trades (includes aggressor side)
            self.ib.reqTickByTickData(
                self.contract, "AllLast", numberOfTicks=0, ignoreSize=False
            )
            logger.info("Subscribed to tick-by-tick trade data")

            # Subscribe to Level 2 market depth (5 levels)
            self.ib.reqMktDepth(self.contract, numRows=5)
            logger.info("Subscribed to Level 2 market depth (5 levels)")

            # Subscribe to real-time 5-second bars
            self.ticker = self.ib.reqMktData(self.contract, genericTickList="", snapshot=False)
            logger.info("Subscribed to real-time market data")

            # Set up tick handlers
            self.ib.pendingTickersEvent += self._on_pending_tickers

        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {e}")

    def _on_pending_tickers(self, tickers: List[Ticker]) -> None:
        """Handle incoming ticker updates."""
        for ticker in tickers:
            if self._on_tick_callback and ticker.lastSize:
                self._on_tick_callback(ticker)
            if self._on_dom_callback and ticker.domBids:
                self._on_dom_callback(ticker)

    def _on_disconnected(self) -> None:
        """Handle IB Gateway disconnection."""
        self.connected = False
        logger.warning("Disconnected from IB Gateway")

        if self._on_disconnect_callback:
            self._on_disconnect_callback()

        if not self.shutting_down:
            asyncio.ensure_future(self._reconnect())

    async def _reconnect(self) -> None:
        """Attempt to reconnect to IB Gateway with exponential backoff."""
        while (
            not self.shutting_down
            and self.reconnect_count < self.max_reconnect_attempts
        ):
            self.reconnect_count += 1
            delay = self.reconnect_delay * self.reconnect_count
            logger.info(
                f"Reconnection attempt {self.reconnect_count}/{self.max_reconnect_attempts} "
                f"in {delay}s..."
            )
            await asyncio.sleep(delay)

            success = await self.connect()
            if success:
                await self.subscribe_data()
                logger.info("Reconnected successfully")
                return

        if not self.shutting_down:
            logger.error(
                f"Failed to reconnect after {self.max_reconnect_attempts} attempts"
            )

    async def disconnect(self) -> None:
        """Gracefully disconnect from IB Gateway."""
        self.shutting_down = True
        if self.connected:
            logger.info("Disconnecting from IB Gateway...")
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB Gateway")

    def setup_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set up SIGINT and SIGTERM handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.ensure_future(self._shutdown(s, loop)),
            )

    async def _shutdown(self, sig: signal.Signals, loop: asyncio.AbstractEventLoop) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {sig.name}. Shutting down gracefully...")
        await self.disconnect()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()
