"""
NAS100 Order Flow Bot - Real-time order flow analysis and signal generation.

Connects to Interactive Brokers Gateway via ib_insync to analyze NQ futures
order flow in real-time. Generates trading signals based on delta divergence,
absorption, DOM imbalance, large prints, and volume profile analysis.

SIGNAL-ONLY: This bot NEVER places orders or executes trades.
It observes market microstructure and alerts the user via Telegram.
The user decides all risk, sizing, and execution manually.
"""

__version__ = "1.0.0"
__description__ = "NAS100 Order Flow Signal Bot (Read-Only)"
