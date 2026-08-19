"""
Market data fetching for NAS100 Signal Bot.

Uses yfinance to fetch NAS100 data including:
- Daily bars (for multi-day setups like RSI, consecutive red days, rolling decline)
- Hourly bars (for intraday setups like first candle, PDH/PDL sweeps)
- Computes PDH (Previous Day High), PDL (Previous Day Low)
- ATR (Average True Range) for stop loss calculations
- RSI (Relative Strength Index) via ta library
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class MarketData:
    """Fetches and caches NAS100 market data."""

    def __init__(self, config: Dict):
        """
        Initialize market data fetcher.

        Args:
            config: Configuration dictionary with market settings.
        """
        self.ticker = config.get("market", {}).get("ticker", "^IXIC")
        self.daily_lookback = config.get("market", {}).get("daily_lookback_days", 30)
        self.hourly_lookback = config.get("market", {}).get("hourly_lookback_days", 5)
        self.cache_minutes = config.get("market", {}).get("data_cache_minutes", 5)

        self._daily_cache: Optional[pd.DataFrame] = None
        self._hourly_cache: Optional[pd.DataFrame] = None
        self._cache_time: Optional[datetime] = None

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache_time is None:
            return False
        elapsed = (datetime.now() - self._cache_time).total_seconds() / 60
        return elapsed < self.cache_minutes

    def fetch_daily(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch daily OHLCV data.

        Args:
            force: Force re-fetch even if cache is valid.

        Returns:
            DataFrame with daily OHLCV data.
        """
        if not force and self._is_cache_valid() and self._daily_cache is not None:
            return self._daily_cache

        logger.info(f"Fetching daily data for {self.ticker} ({self.daily_lookback} days)")
        try:
            ticker = yf.Ticker(self.ticker)
            df = ticker.history(period=f"{self.daily_lookback}d", interval="1d")

            if df.empty:
                logger.warning(f"No daily data returned for {self.ticker}")
                return pd.DataFrame()

            self._daily_cache = df
            self._cache_time = datetime.now()
            logger.info(f"Fetched {len(df)} daily bars")
            return df

        except Exception as e:
            logger.error(f"Error fetching daily data: {e}")
            if self._daily_cache is not None:
                logger.info("Returning cached daily data")
                return self._daily_cache
            return pd.DataFrame()

    def fetch_hourly(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch hourly OHLCV data.

        Args:
            force: Force re-fetch even if cache is valid.

        Returns:
            DataFrame with hourly OHLCV data.
        """
        if not force and self._is_cache_valid() and self._hourly_cache is not None:
            return self._hourly_cache

        logger.info(f"Fetching hourly data for {self.ticker} ({self.hourly_lookback} days)")
        try:
            ticker = yf.Ticker(self.ticker)
            df = ticker.history(period=f"{self.hourly_lookback}d", interval="1h")

            if df.empty:
                logger.warning(f"No hourly data returned for {self.ticker}")
                return pd.DataFrame()

            self._hourly_cache = df
            self._cache_time = datetime.now()
            logger.info(f"Fetched {len(df)} hourly bars")
            return df

        except Exception as e:
            logger.error(f"Error fetching hourly data: {e}")
            if self._hourly_cache is not None:
                logger.info("Returning cached hourly data")
                return self._hourly_cache
            return pd.DataFrame()

    def get_pdh_pdl(self, daily_df: Optional[pd.DataFrame] = None) -> Tuple[float, float]:
        """
        Get Previous Day High (PDH) and Previous Day Low (PDL).

        These are key levels: price reaches PDH or PDL 90% of days.

        Returns:
            Tuple of (PDH, PDL). Returns (0.0, 0.0) if insufficient data.
        """
        if daily_df is None:
            daily_df = self.fetch_daily()

        if daily_df.empty or len(daily_df) < 2:
            logger.warning("Insufficient data for PDH/PDL")
            return (0.0, 0.0)

        # Previous day is the second-to-last bar
        prev_day = daily_df.iloc[-2]
        pdh = float(prev_day["High"])
        pdl = float(prev_day["Low"])

        logger.debug(f"PDH: {pdh:.2f}, PDL: {pdl:.2f}")
        return (pdh, pdl)

    def get_atr(self, daily_df: Optional[pd.DataFrame] = None, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).

        Used for stop loss placement and sweep threshold calculations.

        Args:
            daily_df: Daily DataFrame. Fetched if None.
            period: ATR period (default 14).

        Returns:
            Current ATR value. Returns 0.0 if insufficient data.
        """
        if daily_df is None:
            daily_df = self.fetch_daily()

        if daily_df.empty or len(daily_df) < period + 1:
            logger.warning(f"Insufficient data for ATR({period})")
            return 0.0

        atr_indicator = AverageTrueRange(
            high=daily_df["High"],
            low=daily_df["Low"],
            close=daily_df["Close"],
            window=period,
        )
        atr_values = atr_indicator.average_true_range()

        if atr_values.empty or pd.isna(atr_values.iloc[-1]):
            return 0.0

        return float(atr_values.iloc[-1])

    def get_rsi(self, daily_df: Optional[pd.DataFrame] = None, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index).

        RSI < 30 triggers the oversold edge (70.1% win rate on 5-day hold).

        Args:
            daily_df: Daily DataFrame. Fetched if None.
            period: RSI period (default 14).

        Returns:
            Current RSI value. Returns 50.0 (neutral) if insufficient data.
        """
        if daily_df is None:
            daily_df = self.fetch_daily()

        if daily_df.empty or len(daily_df) < period + 1:
            logger.warning(f"Insufficient data for RSI({period})")
            return 50.0

        rsi_indicator = RSIIndicator(close=daily_df["Close"], window=period)
        rsi_values = rsi_indicator.rsi()

        if rsi_values.empty or pd.isna(rsi_values.iloc[-1]):
            return 50.0

        return float(rsi_values.iloc[-1])

    def get_current_price(self, daily_df: Optional[pd.DataFrame] = None) -> float:
        """Get the most recent closing price."""
        if daily_df is None:
            daily_df = self.fetch_daily()

        if daily_df.empty:
            return 0.0

        return float(daily_df["Close"].iloc[-1])

    def get_first_hourly_candle(
        self, hourly_df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Get the first 1-hour candle of the current/most recent trading day.

        Returns:
            Dict with open, high, low, close, change_pct or None.
        """
        if hourly_df is None:
            hourly_df = self.fetch_hourly()

        if hourly_df.empty:
            return None

        # Get today's date in ET
        now_et = datetime.now(ET)
        today = now_et.date()

        # Filter for today's bars
        if hourly_df.index.tzinfo is None:
            hourly_df.index = hourly_df.index.tz_localize("UTC")

        hourly_et = hourly_df.copy()
        hourly_et.index = hourly_et.index.tz_convert(ET)

        today_bars = hourly_et[hourly_et.index.date == today]

        # If no bars today, use the most recent day
        if today_bars.empty:
            if len(hourly_et) > 0:
                last_date = hourly_et.index[-1].date()
                today_bars = hourly_et[hourly_et.index.date == last_date]

        if today_bars.empty:
            return None

        first_bar = today_bars.iloc[0]
        open_price = float(first_bar["Open"])
        close_price = float(first_bar["Close"])

        if open_price == 0:
            return None

        change_pct = (close_price - open_price) / open_price

        return {
            "open": open_price,
            "high": float(first_bar["High"]),
            "low": float(first_bar["Low"]),
            "close": close_price,
            "change_pct": change_pct,
        }

    def get_daily_changes(self, daily_df: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Get daily percentage changes.

        Returns:
            Series of daily percentage changes.
        """
        if daily_df is None:
            daily_df = self.fetch_daily()

        if daily_df.empty or len(daily_df) < 2:
            return pd.Series(dtype=float)

        return daily_df["Close"].pct_change().dropna()
