# Trader Development Journal

A measurement and development tool for traders implementing the **observe -> decompose -> test -> iterate** learning cycle.

> **IMPORTANT: This system is JOURNAL-ONLY. It NEVER auto-executes trades.**
> It observes, measures, and alerts. YOU decide all actions, risk, and sizing.

## Features

- **Trade Logging**: Full trade lifecycle with entry/exit, P&L, R-multiples, emotional state, execution quality
- **Setup Management**: Define named setups with expected performance, rules, and checklists
- **Rolling Statistics**: Per-setup win rate, avg R, expectancy, streak tracking (last 20 trades)
- **Account Statistics**: Total P&L, profit factor, max drawdown, Sharpe-like metric, trades/week
- **Setup Decay Detection**: Alerts when a setup's live win rate drops >15pp below expected
- **Development Loop**: Structured learning cycle (observe/decompose/test/iterate)
- **Hypothesis Testing**: Create and evaluate testable hypotheses about your setups
- **Telegram Integration**: Daily summaries, weekly reports, decay alerts, drawdown alerts

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp journal_config.example.yaml journal_config.yaml
# Edit journal_config.yaml with your Telegram token and preferences
```

### 3. Initialize and Use

```bash
# Add a setup
python -m journal setup add "RSI<30 swing" --win-rate 70 --avg-r 1.5 --hold-period "3-5 days"

# Log a trade entry
python -m journal trade entry long 17500 --stop-loss 17450 --take-profit 17600 \
  --setup "RSI<30 swing" --emotion 3 --thesis "RSI below 30 with strong support"

# Close the trade
python -m journal trade exit 1 17580 --review "Held through pullback, good patience" \
  --execution 4 --emotion 3

# View stats
python -m journal stats account
python -m journal stats setup "RSI<30 swing"
```

## CLI Reference

### Trade Commands

```bash
# Log entry
python -m journal trade entry <long|short> <price> [options]
  --stop-loss, -sl       Stop loss price
  --take-profit, -tp     Take profit price
  --setup, -s            Setup name (must exist)
  --instrument, -i       Instrument (default: NAS100)
  --entry-time, -t       Entry time ISO format (default: now)
  --confluence           Confluence notes
  --screenshot           Screenshot file path
  --thesis               Pre-trade thesis
  --emotion, -e          Emotional state 1-5
  --hypothesis           Hypothesis ID to tag trade

# Close trade
python -m journal trade exit <trade_id> <exit_price> [options]
  --exit-time, -t        Exit time ISO format (default: now)
  --pnl                  P&L in dollars (overrides auto-calc)
  --r-multiple, -r       R-multiple (overrides auto-calc)
  --review               Post-trade review notes
  --execution            Execution quality 1-5
  --emotion, -e          Emotional state 1-5
  --quantity, -q         Position quantity for P&L calculation

# List/view trades
python -m journal trade list [--limit N]
python -m journal trade open
```

### Setup Commands

```bash
python -m journal setup add <name> --win-rate <pct> --avg-r <float> [options]
  --min-confluence       Minimum confluence count
  --description, -d      Setup description
  --hold-period          Expected hold period
  --rules                Rules/checklist text

python -m journal setup edit <name> [--win-rate] [--avg-r] [--description] ...
python -m journal setup delete <name>
python -m journal setup list [--all]
```

### Statistics Commands

```bash
python -m journal stats account       # Overall account stats
python -m journal stats setup <name>  # Per-setup rolling stats (window=20)
  --window, -w           Rolling window size (default: 20)
```

### Development Loop Commands

```bash
python -m journal observe                    # OBSERVE: Review all setups
python -m journal decompose <setup_name>     # DECOMPOSE: Analyze losses

python -m journal hypothesis create <setup> <description> [--target N]
python -m journal hypothesis evaluate <id>
python -m journal hypothesis list [--all]
```

### Telegram Commands

```bash
python -m journal telegram daily [--dry-run]    # Send daily P&L summary
python -m journal telegram weekly [--dry-run]   # Send weekly performance report
```

## The Development Loop

This journal implements a structured learning cycle:

### 1. OBSERVE (every 20 trades)
The system prompts you to review setup performance. It shows which setups are working (meeting expected win rate) and which are underperforming (drifting below expected).

### 2. DECOMPOSE
For underperforming setups, the system shows your losing trades and analyzes patterns (emotional state, execution quality, time of day, confluence). The key question: "What variable was different?"

### 3. TEST
Create a testable hypothesis (e.g., "This setup only works when VIX is above 20") and tag future trades that meet the hypothesis condition.

### 4. ITERATE
After collecting enough tagged trades (default: 20), evaluate the hypothesis. The system compares hypothesis-tagged trade performance against overall setup performance and recommends: KEEP, DISCARD, or MODIFY.

## Trade Data Logged

Each trade captures:
- Entry/exit time and prices
- Direction (long/short) and instrument
- Stop loss and take profit levels
- Setup tag (which pattern triggered the trade)
- Confluence notes (what confirmed the setup)
- Screenshot path (optional chart capture)
- Pre-trade thesis (why you took the trade)
- Post-trade review (what happened, what you learned)
- Emotional state (1-5 scale)
- Execution quality (1-5 scale)
- P&L in dollars and R-multiple

## Statistics Computed

### Per-Setup (Rolling 20 Trades)
- Live win rate vs expected win rate
- Live avg R vs expected avg R
- Expectancy: WR * avgWin - (1-WR) * avgLoss
- Win rate drift detection (alerts at 15pp below expected)
- Current streak and max win/loss streaks

### Account-Wide
- Total P&L
- Overall win rate
- Best and worst trade
- Average hold time
- Profit factor (gross wins / gross losses)
- Max drawdown
- Sharpe-like metric (avg daily P&L / std of daily P&L)
- Trades per week

## Configuration

See `journal_config.example.yaml` for all options. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `alerts.wr_decay_threshold_pp` | 15 | Alert when WR drops this many pp below expected |
| `alerts.max_drawdown_threshold` | 5000 | Alert when max DD exceeds this (in $) |
| `alerts.review_interval_trades` | 20 | Prompt review after this many trades |
| `database.path` | journal.db | SQLite database file location |

## Architecture

```
journal/
  __init__.py           - Package metadata
  __main__.py           - Entry point (python -m journal)
  cli.py                - CLI argument parsing and command routing
  config.py             - YAML config loading and validation
  database.py           - SQLite database layer (trades, setups, hypotheses)
  stats.py              - Statistics engine (setup stats, account stats, alerts)
  development_loop.py   - Observe/decompose/test/iterate cycle
  telegram.py           - Telegram message formatting and sending
```

## License

Part of the NAS100 trading toolkit. For personal use.
