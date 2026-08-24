# Memescanner - Solana Memecoin Scanner Bot

A signal-only memecoin scanner that monitors Pump.fun and DEXScreener for newly launched Solana tokens, scores them using research-backed metrics, and sends Telegram alerts with probability estimates.

> **IMPORTANT: This is a SIGNAL-ONLY system. It NEVER auto-executes trades. It scans, scores, calculates probability, and alerts. You decide whether to trade.**

## Features

- **Real-time Pump.fun scanning** - Monitors new token launches and graduations every 10 seconds
- **DEXScreener integration** - Fetches comprehensive trading data (price, volume, liquidity, buy/sell counts)
- **Research-backed scoring** - 6-factor scoring engine with weights derived from actual data on winning vs losing tokens
- **Probability estimates** - Calculates probability of reaching target market caps (100k, 300k, 1M, 5M)
- **Expected Value (EV)** - Shows whether a token has positive or negative EV per dollar risked
- **Hard filters** - Eliminates obvious rugs and dumps before scoring (liquidity, buy/sell ratio, dev holdings)
- **Narrative matching** - Identifies trending themes (AI, political, celebrity, meme) with temperature ratings
- **Self-adaptation** - Tracks outcomes and adjusts scoring weights based on what actually predicts pumps
- **Telegram alerts** - Formatted alerts with all metrics, probabilities, and risk assessment
- **SQLite persistence** - Tracks all scanned tokens, narratives, and weight history

## Architecture

```
memescanner/
  __init__.py          - Package metadata
  config.py            - YAML config loading with typed access
  database.py          - Async SQLite via aiosqlite
  pump_fun.py          - Pump.fun API client
  dexscreener.py       - DEXScreener API client
  scoring.py           - 6-factor scoring engine
  probability.py       - Probability and EV calculator
  filters.py           - Hard rejection filters
  narrative.py         - Narrative matching and temperature tracking
  telegram_bot.py      - Telegram alert formatting and sending
  adaptation.py        - Self-adapting weight optimization
  main.py              - Async main loop orchestrator
```

## Scoring Methodology

Weights derived from actual research data comparing winning and losing tokens:

| Factor | Weight | Multiplier | Signal |
|--------|--------|-----------|--------|
| Buy/Sell Ratio | 25% | 4.9x | Buyers dominating = bullish |
| Liquidity | 25% | 11.9x | Strongest signal of all |
| Volume Turnover | 20% | 4.1x | High volume vs MC = interest |
| Engagement Velocity | 15% | 4.8x | Replies per hour on Pump.fun |
| Narrative Match | 10% | +3-11pp | Hot narratives outperform |
| Momentum | 5% | - | Near ATH = still pumping |

## Self-Adaptation

The bot tracks every alerted token and checks its price at 1h, 6h, and 24h after alert. After 50 tracked tokens:

1. Computes which score factors correlated with actual pumps
2. Adjusts weights accordingly (increase what predicted, decrease what did not)
3. Updates narrative temperatures based on which themes produced winners
4. Sends a weekly performance report to Telegram

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your settings:
- `telegram.bot_token` - Get from @BotFather on Telegram
- `telegram.chat_id` - Your chat or group ID

### 3. Run

```bash
python -m memescanner.main
```

Or:

```bash
python -c "import asyncio; from memescanner.main import main; asyncio.run(main())"
```

## Configuration

See `config.example.yaml` for all available settings:

- **scanner** - Check interval, minimum score threshold, max token age
- **filters** - Hard filter thresholds (liquidity, buy/sell ratio, dev holdings)
- **scoring.weights** - Factor weights (adjust or let the bot self-adapt)
- **adaptation** - Outcome tracking intervals, reweight schedule
- **database** - SQLite database file path
- **logging** - Log level and file path

## Alert Format

```
HIGH CONVICTION (Score: 82/100)

$CATALORIAN | Elon's Space Cat
CA: 4cvZwC17oM...
Age: 16 minutes

Metrics:
MC: $142,000 | Liq: $28,000
Buy/Sell: 3.2 (buyers dominating)
Volume: $95,000 (turnover: 0.67x)
Engagement: 34 replies/hr
Narrative: "cat + elon" (HOT)
Momentum: 91% of ATH

Probability Estimates:
-> 100k MC (from here): 18.2%
-> 300k MC: 6.4%
-> 1M MC: 1.2%
-> EV per $100 risked: +$47.20

Risk: MEDIUM
- LP not burned
- Top holder: 8.2%

https://dexscreener.com/solana/4cvZwC17oM...
```

## Disclaimer

This bot provides SIGNALS ONLY. It does NOT execute trades. All trading decisions are yours. Cryptocurrency trading, especially memecoins, carries extreme risk. Most memecoins go to zero. Never risk more than you can afford to lose.

## License

MIT
