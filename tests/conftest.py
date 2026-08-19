"""
Shared pytest fixtures for NAS100 Signal Bot tests.

Provides mock market data, sample configurations, and test helpers.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import pytest

ET = pytz.timezone("US/Eastern")


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        "telegram": {
            "bot_token": "TEST_TOKEN",
            "chat_id": "TEST_CHAT_ID",
        },
        "account": {
            "balance": 10000.0,
            "default_risk_percent": 1.0,
            "max_kelly_fraction": 0.5,
            "currency": "USD",
        },
        "market": {
            "ticker": "^IXIC",
            "daily_lookback_days": 30,
            "hourly_lookback_days": 5,
            "data_cache_minutes": 5,
        },
        "schedule": {
            "timezone": "US/Eastern",
            "check_times": ["09:30", "10:30", "11:00", "14:00", "15:00", "15:45"],
        },
        "thresholds": {
            "min_confluence": 1,
            "first_candle_threshold": 0.003,
            "pdl_sweep_threshold": 0.3,
            "pdh_sweep_threshold": 0.2,
            "rsi_oversold": 30,
            "consecutive_red_days": 5,
            "rolling_decline_pct": 5.0,
            "large_drop_pct": 4.0,
            "large_rally_pct": 3.0,
        },
        "logging": {
            "level": "DEBUG",
            "file": "test_nas100bot.log",
            "max_bytes": 10485760,
            "backup_count": 5,
        },
    }


@pytest.fixture
def sample_daily_df():
    """Sample daily OHLCV DataFrame with 20 trading days."""
    dates = pd.date_range(end="2024-01-20", periods=20, freq="B")
    np.random.seed(42)

    # Generate realistic price data starting at 15000
    prices = [15000.0]
    for i in range(19):
        change = np.random.normal(0.001, 0.01)  # Slight upward bias
        prices.append(prices[-1] * (1 + change))

    data = {
        "Open": [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        "High": [p * (1 + abs(np.random.normal(0.005, 0.003))) for p in prices],
        "Low": [p * (1 - abs(np.random.normal(0.005, 0.003))) for p in prices],
        "Close": prices,
        "Volume": [np.random.randint(1000000, 5000000) for _ in range(20)],
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def bearish_daily_df():
    """Daily DataFrame with 5 consecutive red days (for bounce edge)."""
    dates = pd.date_range(end="2024-01-20", periods=10, freq="B")

    # First 5 days neutral, last 5 days red
    prices = [15000, 15050, 15020, 15080, 15060,  # Neutral
              15000, 14900, 14800, 14700, 14600]  # 5 red days

    data = {
        "Open": [p + 50 for p in prices],  # Open higher than close for red days
        "High": [p + 100 for p in prices],
        "Low": [p - 50 for p in prices],
        "Close": prices,
        "Volume": [3000000] * 10,
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def bullish_first_candle():
    """First hourly candle that's bullish (>+0.3%)."""
    return {
        "open": 15000.0,
        "high": 15080.0,
        "low": 14980.0,
        "close": 15060.0,  # +0.4% from open
        "change_pct": 0.004,  # 0.4%
    }


@pytest.fixture
def bearish_first_candle():
    """First hourly candle that's bearish (<-0.3%)."""
    return {
        "open": 15000.0,
        "high": 15020.0,
        "low": 14920.0,
        "close": 14940.0,  # -0.4% from open
        "change_pct": -0.004,  # -0.4%
    }


@pytest.fixture
def neutral_first_candle():
    """First hourly candle that's neutral (within +/-0.3%)."""
    return {
        "open": 15000.0,
        "high": 15030.0,
        "low": 14980.0,
        "close": 15010.0,  # +0.07% from open
        "change_pct": 0.0007,
    }


@pytest.fixture
def kill_zone_time():
    """DateTime during NY Kill Zone (10:00 AM ET on a Wednesday)."""
    return ET.localize(datetime(2024, 1, 17, 10, 0, 0))  # Wednesday 10:00 AM ET


@pytest.fixture
def dead_zone_time():
    """DateTime during Dead Zone (1:00 PM ET)."""
    return ET.localize(datetime(2024, 1, 17, 13, 0, 0))  # Wednesday 1:00 PM ET


@pytest.fixture
def weak_period_time():
    """DateTime during weak period (Thursday 3:00 PM ET)."""
    return ET.localize(datetime(2024, 1, 18, 15, 0, 0))  # Thursday 3:00 PM ET


@pytest.fixture
def daily_changes_red_streak():
    """Daily percentage changes with 5 consecutive red days."""
    # First 5 neutral, then 5 red
    return pd.Series([
        0.003, -0.001, 0.002, 0.004, -0.002,
        -0.008, -0.012, -0.009, -0.015, -0.011,
    ])


@pytest.fixture
def daily_changes_large_drop():
    """Daily percentage changes with a >4% drop on the last day."""
    return pd.Series([
        0.003, -0.001, 0.002, 0.004, -0.002,
        0.001, -0.003, 0.002, -0.001, -0.045,  # Last day: -4.5%
    ])


@pytest.fixture
def daily_changes_large_rally():
    """Daily percentage changes with a >3% rally on the last day."""
    return pd.Series([
        0.003, -0.001, 0.002, 0.004, -0.002,
        0.001, -0.003, 0.002, -0.001, 0.035,  # Last day: +3.5%
    ])
