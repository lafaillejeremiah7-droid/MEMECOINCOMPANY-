# NAS100 Order Flow Bot

Real-time order flow analysis and signal generation for NQ futures (Nasdaq 100 E-mini) via Interactive Brokers Gateway.

**SIGNAL-ONLY**: This bot NEVER places orders or executes trades. It observes market microstructure and alerts you via Telegram. You decide all risk, sizing, and execution manually.

## Architecture

```
orderflow/
├── __init__.py          # Package initialization
├── __main__.py          # python -m orderflow entry point
├── config.py            # YAML configuration loader
├── connection.py        # IB Gateway connection manager
├── delta.py             # Cumulative delta & divergence detection
├── volume_profile.py    # Session volume profile, POC, HVN/LVN
├── dom.py               # DOM (Depth of Market) analysis
├── large_prints.py      # Large print & cluster detection
├── absorption.py        # Absorption pattern detection
├── signals.py           # Unified signal engine with cooldowns
├── database.py          # SQLite persistence (aiosqlite)
├── adaptation.py        # Self-adaptation engine
├── telegram.py          # Telegram alert formatting & sending
├── main.py              # Main orchestrator and event loop
└── README.md            # This file
```

## Signal Types

| Signal | Trigger | Direction |
|--------|---------|-----------|
| **Delta Divergence** | Price new high + delta declining | SHORT |
| **Delta Divergence** | Price new low + delta rising | LONG |
| **Absorption** | Heavy selling absorbed at level, price holds | LONG |
| **Absorption** | Heavy buying absorbed at level, price holds | SHORT |
| **Large Print Cluster** | Multiple large prints in same direction within N seconds | Follow flow |
| **DOM Imbalance Flip** | Bid/ask ratio flips from extreme one side to other | Reversal |
| **POC Reclaim** | Price sweeps below POC then reclaims above | LONG |

## Self-Adaptation

The bot tracks the forward result of every signal at 5min, 15min, 30min, and 1h intervals:

- **Rolling Win Rate**: Computed per signal type over the last 30 signals
- **Auto-Disable**: If a signal type's rolling WR drops below 50% over 30 instances, it is automatically disabled and you are alerted
- **Auto-Re-Enable**: If a disabled signal's theoretical WR recovers above 55% over the next 20 instances, it is re-enabled and you are alerted
- **Weekly Report**: Every Friday, a self-report shows active/disabled signals and performance

## Prerequisites

- **Python 3.9+**
- **Interactive Brokers account** (live or paper trading)
- **IB Gateway** installed and running
- **Telegram bot** for receiving alerts

## Linux Setup Instructions

### 1. Install IB Gateway

Download the stable release from IBKR:

```bash
# Download IB Gateway installer
wget https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh

# Make executable
chmod +x ibgateway-stable-standalone-linux-x64.sh

# Install (follow prompts)
./ibgateway-stable-standalone-linux-x64.sh
```

### 2. Configure IB Gateway API

1. Launch IB Gateway and log in
2. Navigate to **Configure > Settings > API > Settings**
3. **Enable "Enable ActiveX and Socket Clients"**
4. **Set Socket port to 4001** (live) or 4002 (paper)
5. **CHECK "Read-Only API"** (critical for safety)
6. Uncheck "Create API message log file" (optional)
7. Set "Master API client ID" if needed
8. Click OK and restart IB Gateway

### 3. Install Python Dependencies

```bash
cd /path/to/hii
pip install -r requirements.txt
```

### 4. Configure the Bot

```bash
# Copy example config
cp orderflow_config.example.yaml orderflow_config.yaml

# Edit with your settings
nano orderflow_config.yaml
```

Required settings to change:
- `telegram.bot_token`: Your Telegram bot token from @BotFather
- `telegram.chat_id`: Your Telegram chat ID from @userinfobot

### 5. Run the Bot

```bash
# Run with Telegram alerts (production)
python -m orderflow

# Run in dry-run mode (no Telegram, no IB connection)
python -m orderflow --dry-run

# Run with custom config path
python -m orderflow --config /path/to/orderflow_config.yaml
```

### 6. Run as a Service (systemd)

Create `/etc/systemd/system/orderflow-bot.service`:

```ini
[Unit]
Description=NAS100 Order Flow Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/hii
ExecStart=/usr/bin/python3 -m orderflow --config /path/to/orderflow_config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable orderflow-bot
sudo systemctl start orderflow-bot
sudo systemctl status orderflow-bot
```

## Important Notes

- **SIGNAL-ONLY**: The bot never places orders. It uses read-only API mode.
- **Contract Rollovers**: NQ futures expire quarterly (March, June, September, December). The bot handles contract qualification automatically via IB.
- **All times are in US/Eastern** timezone.
- **Reconnection**: If IB Gateway disconnects, the bot will attempt to reconnect with exponential backoff (up to max_reconnect_attempts).
- **Graceful Shutdown**: Send SIGINT (Ctrl+C) or SIGTERM to stop the bot cleanly.

## Data Flow

```
IB Gateway (localhost:4001)
    |
    ├── Tick-by-tick trades (with aggressor side)
    ├── Level 2 Market Depth (5 levels)
    └── Real-time market data
         |
         v
    ┌─────────────────────┐
    │   Order Flow Bot     │
    ├─────────────────────┤
    │ - Cumulative Delta   │
    │ - Volume Profile     │
    │ - DOM Analysis       │
    │ - Large Prints       │
    │ - Absorption         │
    ├─────────────────────┤
    │   Signal Engine      │
    │ (cooldowns, scores)  │
    ├─────────────────────┤
    │  Adaptation Engine   │
    │ (disable/re-enable)  │
    ├─────────────────────┤
    │  SQLite Database     │
    │ (signal log, results)│
    └─────────────────────┘
         |
         v
    Telegram Alerts
    (signals, summaries,
     daily & weekly reports)
```
