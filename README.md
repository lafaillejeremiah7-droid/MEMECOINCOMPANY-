# NAS100 Signal Bot

A quantitative, signal-only trading assistant for NAS100 (Nasdaq) based on statistically-proven edges from research data (2014-2026, 3175 daily bars + 2 years of hourly data).

---

## IMPORTANT DISCLAIMER

**This bot is SIGNAL-ONLY. It will NEVER auto-execute trades.**

- All position sizing suggestions are **advisory only**
- **YOU** decide final risk, lot size, and trade execution
- The bot observes, classifies, generates signals, computes quant stats, and suggests sizes
- The user then decides whether to trade and how much to risk
- Never risk more than you can afford to lose

---

## Features

- 10 statistically-proven edges (6 long + 4 short) with exact win rates from research
- Kelly criterion optimal position sizing suggestions
- Confluence scoring (multiple aligned edges = stronger signal)
- Full quant stats on every signal: win rate, expected value, Kelly %, R:R ratio
- Time-of-day awareness (kill zone, dead zone, weak periods)
- Telegram alerts with rich formatting
- Scheduled checks at key market times
- Dry-run mode for testing without Telegram
- Modular, clean Python codebase

---

## Architecture

```
User -> [Config] -> Bot Scheduler
                        |
                    [Market Data (yfinance)]
                        |
                    [Edge Detection (10 edges)]
                        |
                    [Signal Generation + Confluence Scoring]
                        |
                    [Kelly Criterion Math]
                        |
                    [Telegram Alert / Console Log]
                        |
                    User reviews signal -> User decides to trade (or not)
```

The bot's role: observe market -> classify -> generate signal -> compute quant stats -> suggest size -> alert user.

---

## Statistically-Proven Edges

### Long Setups (6)

| Edge | Condition | Win Rate | Avg Win | Samples |
|------|-----------|----------|---------|---------|
| First 1H Candle Bullish | First hourly candle closes >+0.3% | 87.6% | +0.85% | 201 |
| PDL Sweep Reclaim | PDL swept >0.3R, price reclaims above | 76.4% | +0.92% | 55 |
| RSI Oversold | RSI(14) daily < 30 | 70.1% | +1.36% | 144 |
| 5 Red Days Bounce | 5 consecutive red days | 63.4% | +0.50% | 41 |
| Rolling Decline | 5-day decline >5% | 67.3% | +1.44% | 107 |
| Large Drop Bounce | Single day drop >4% | 66.7% | +0.59% | 27 |

### Short Setups (4)

| Edge | Condition | Win Rate | Avg Win | Samples |
|------|-----------|----------|---------|---------|
| First 1H Candle Bearish | First hourly candle closes <-0.3% | 84.8% | +0.78% | 178 |
| PDH Sweep Rejection | PDH swept >0.2R, price rejects below | 83.1% | +0.72% | 83 |
| Large Rally Fade | Single-day rally >3% | 58.5% | +0.73% | 53 |
| Weak Period Short | Thursday/Friday 3PM ET | 53.5% | +0.45% | ~200 |

---

## Timing Rules

- **NY Kill Zone (9:30-11:00 AM ET):** Highest volatility, 35.4 bps avg move/hour. Best scalping window.
- **Dead Zone (12:00-2:00 PM ET):** Volatility drops 37%. Avoid scalping.
- **Weak Period (Thu/Fri 3PM ET):** Only 46-47% green rate. Favors shorts.
- **High of Day:** Forms 9:30-10:30 ET 43.6% of the time.
- **Low of Day:** Forms 9:30-10:30 ET 53.4% of the time.
- **Price reaches PDH or PDL:** 90% of days.
- **Best days for longs:** Monday, Wednesday.
- **Worst periods:** Thursday/Friday afternoon.

---

## Kelly Criterion

The bot uses the Kelly criterion to suggest optimal position sizing:

```
kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
```

By default, the bot suggests **half-Kelly** (more conservative). The `max_kelly_fraction` config option controls this cap.

**Remember:** Kelly suggestions are advisory. YOU decide your actual position size.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your:
- Telegram bot token (from @BotFather)
- Telegram chat ID (from @userinfobot)
- Account balance (for Kelly suggestions)
- Risk preferences

### 3. Run

```bash
# Full mode with Telegram alerts
python -m nas100bot.main

# Dry-run mode (no Telegram, log to console)
python -m nas100bot.main --dry-run

# Single check and exit
python -m nas100bot.main --once

# Custom config path
python -m nas100bot.main --config /path/to/config.yaml
```

---

## Configuration

See `config.example.yaml` for all options with documentation. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `telegram.bot_token` | Telegram bot token | Required |
| `telegram.chat_id` | Your Telegram chat ID | Required |
| `account.balance` | Account balance for Kelly suggestions | 10000 |
| `account.max_kelly_fraction` | Cap Kelly at this fraction (0.5 = half-Kelly) | 0.5 |
| `market.ticker` | Market ticker symbol | ^IXIC |
| `thresholds.min_confluence` | Minimum edges for signal | 1 |
| `schedule.check_times` | Times to check (ET) | 09:30, 10:30, etc. |

---

## Signal Output Example

When a signal is generated, you receive a message like:

```
LONG SIGNAL - NAS100

Ticker: ^IXIC
Price: $15,060.00
Time: 10:00 ET (Wednesday)

Confluence Score: 3/6

Active Edges:
  1. First 1H candle closed >+0.3% -> 87.6% chance day ends green (201 samples)
  2. RSI(14) = 25.3 < 30 -> 70.1% win rate on 5-day hold (144 samples)
  3. Single day drop >4% -> 66.7% next day green (27 samples)

Quant Statistics:
  - Win Rate: 82.3% (weighted)
  - Expected Value: +0.7124% per trade
  - Kelly Fraction: 68.42%
  - Suggested Risk: 34.21% of account
  - Suggested Amount: $3,421.00

Price Levels:
  - Stop Loss: $14,850.00
  - Target: $15,200.00
  - Risk:Reward: 1.52R

Hold Period: Intraday (until close)

Time Context:
  IN KILL ZONE (highest volatility)
  Best day for longs (Mon/Wed)

---
SIGNAL ONLY - NOT FINANCIAL ADVICE
You decide: entry, risk, lot size, execution.
Bot suggests, YOU decide.
```

---

## Project Structure

```
nas100bot/
  __init__.py       # Package metadata
  main.py           # Entry point (--dry-run flag)
  config.py         # YAML config loader
  data.py           # yfinance data fetching (PDH/PDL, RSI, ATR)
  edges.py          # All edge detection functions (6 long + 4 short)
  signals.py        # Signal generation, confluence scoring
  kelly.py          # Kelly criterion math
  telegram_bot.py   # Telegram alert formatting and sending
  timing.py         # Time-of-day utilities (kill zone, dead zone)
  scheduler.py      # Schedule-based runner
tests/
  conftest.py       # Shared fixtures
  test_edges.py     # Edge detection tests
  test_kelly.py     # Kelly criterion tests
  test_signals.py   # Signal generation tests
  test_timing.py    # Timing utility tests
config.example.yaml # Example configuration
requirements.txt    # Dependencies
README.md           # This file
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## License

For personal use only. Not financial advice. Trade at your own risk.
