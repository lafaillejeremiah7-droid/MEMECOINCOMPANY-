# Memescanner — Telegram Signal Company

Memescanner researches Solana memecoins and sends Telegram alerts. You decide whether
to trade manually. Production never starts a paper trader, loads wallet keys, signs
transactions, or places trades—even if an old paper-trading setting is enabled.

## What the eight employees do

| Employee | Job |
|---|---|
| Scout | Find active Solana tokens in the $100k–$200k market-cap window |
| Investigator | Require available contract and social evidence |
| Risk Defender | Veto dangerous authorities, ownership, taxes, or suspicious wallets |
| Market Analyst | Recheck liquidity, trading flow, momentum and missed entries |
| Trade Strategist | Calculate a small position, stop, target, holding time and modeled net payoff |
| Referee | Independently recompute net economics and honor worker vetoes |
| Alert Delivery | Send the ticket; persist accepted, rejected or uncertain delivery without blind retries |
| Operations Boss | Expire old evidence and stale tickets; expose startup/delivery failures |

These are bounded software roles sharing collectors, not eight LLMs or independent
data feeds. Scores indicate checks, not odds of profit. Production tickets replace
the old multi-x/runner targets with 8–15% gross targets and 5–8% planned stops.

## Signal rules and limits

The reference budget is $11, with $1 default/$2 maximum sizing, a $5 reserve,
one position, no DCA/leverage, and a stop after $1 daily loss or three losses.
**Signals cannot enforce your real balance, positions or losses. You must check those
manual limits yourself.** This service has no wallet access.

Net reward/risk must be at least 1.31:1 before rounding. Modeled costs must consume
no more than 25% of projected gross profit. Costs are screening assumptions, not
executable swap quotes: actual network fees, rent, transfer taxes, slippage and
price impact can exceed them. Payoffs are conditional on hitting the target,
not guaranteed returns or calibrated expectancy. Stops do not guarantee a fill.

Every alert includes the contract, entry amount/price, stop, target, 15-minute
maximum planned hold, cost estimate, gross/net payoff, liquidity, critical risks,
employee reports and BUY/WATCH/REJECT decision. Quotes are public market snapshots,
not transaction offers. Evidence expires after 120 seconds; market data is fetched
again before review, must pass the five-second handoff check, and the ticket expires
30 seconds after receipt. Recheck the market and never chase a missed entry.

**Current limitation: the on-chain collector does not verify pool locks. It now
records unknown, not false evidence of removable liquidity. Unknown LP safety or
holder-history evidence can only produce WATCH (do not buy yet), never BUY.**
Explicit hazards are rejected and kept in the audit history rather than pushed.
Until a verified liquidity-safety provider is implemented, production will not
issue BUY signals. No safety gate is bypassed to manufacture recommendations.

SQLite preserves observations, company tickets and delivery claims. One alert per
mint includes WATCH; this version does not resend a WATCH as BUY later. An uncertain
Telegram delivery retains its claim for operator review, preventing blind duplicates.
Historical paper simulation remains in the codebase but is not started by production.
At least 100 completed forward paper signals and human approval remain prerequisites
for any future real execution; no automatic live switch exists.

See the [research/company blueprint](docs/early_discovery_company.md) for background;
this signal-only scope supersedes its execution roadmap.

## Default architecture

The default runtime uses one normalized pipeline:

1. Discover recent Solana candidates independently from DEXScreener token profiles, DEXScreener latest paid boosts, CoinGecko/GeckoTerminal Solana new pools, and Pump.fun.
2. Normalize and merge duplicates by `(chain_id, mint)`, unioning source membership and preserving social, creator, creation-time, and paid-boost metadata.
3. Select only a Solana DEX pair—there is no cross-chain fallback.
4. Apply the same age, X-presence, liquidity, trading-flow, on-chain, holder, rug/scam-evidence, and evidence-availability checks to every source.
5. Persist every observation and decision (qualified, rejected, or deferred) in SQLite, then emit at most one deduplicated alert per mint.

“All-platform” means launchpad-neutral discovery across supported Solana DEX data sources. It is not a guarantee that every token will be discovered. A public source can be unavailable or rate-limited; failures are isolated and other adapters continue. DEX market enrichment is capped per cycle (40 by default); over-budget candidates are persisted as deferred and rotate into later cycles rather than being discarded.

## Evidence semantics

- "New" requires a known token/pair timestamp and an age of 0-120 minutes by default. Unknown age is rejected as `AGE_UNKNOWN_NOT_NEW`. Existing custom configs may retain an older minimum age.
- An X link is required. A completed public search with no result is recorded as `X_DATA_NOT_FOUND_OR_NOT_INDEXED`; that is partial OSINT, not proof that a token has no attention. A missing credential or search outage is separately marked unavailable and defers the candidate.
- A minimum of 5 X search results (mapping to roughly 10-20 real tweets) is required. This gate is bypassed when a big/celebrity account posted about the token or when evidence content indicates viral reach (high view/impression counts).
- Market cap must be at least $50,000. This filters dead and micro tokens (research shows Solana high-return tokens have a median market cap around $214K).
- 24-hour volume must be at least $25,000. This filters most wash-traded tokens (median fake volume on Solana is approximately $10K).
- Top-10 holder concentration must be 20% or lower (stricter than the 30% industry standard).
- Paid DEXScreener boosts are retained as metadata and never counted as organic popularity or predictive evidence.
- Celebrity names and generic buzz are neutral. `VERIFIED` requires an exact canonical X handle whose evidence contains the exact mint address. Fan/copycat handles, Unicode confusables, unrelated results, and keywords do not verify a link. Scam evidence prevents verification and positive classification.
- Missing critical mint, extension, supply, or holder evidence is `UNVERIFIED`; it earns no safety bonus and cannot alert or open a virtual paper position. Some launchpad-neutral feeds do not expose a creator wallet. In that case creator holdings remain explicitly unknown and neutral—never converted to 0%—while verified authority, extension, supply, concentration, and coordination checks can still qualify the candidate. If a source does provide a creator but that wallet cannot be resolved, evaluation is deferred.
- Token-2022 checks reject active mint/freeze control, default-frozen state, permanent delegates, non-transferable tokens, and transfer hooks unless the program is explicitly allowlisted **and** its mutation authority is explicitly revoked (the default allowlist is empty). Transfer fees default to a maximum of 100 basis points (1%); mutable, excessive, or unknown fee configurations are rejected.

## Configuration

Install dependencies and copy the example:

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
python -m memescanner
```

`MEMESCANNER_CONFIG` selects another YAML file. Secrets should normally be supplied as environment variables:

- `MEMESCANNER_TELEGRAM_BOT_TOKEN`
- `MEMESCANNER_TELEGRAM_CHAT_ID`
- `MEMESCANNER_TAVILY_API_KEY` (accepts both Tavily `tvly-` keys and X.ai `xai-` keys; when an X.ai key is detected, the scanner uses the X.ai Responses API with the `x_search` tool instead of Tavily)
- `MEMESCANNER_HELIUS_RPC_URL` (preferred complete RPC endpoint)
- `MEMESCANNER_HELIUS_API_KEY` (used only when no complete RPC URL is set)
- `MEMESCANNER_TRANSFER_HOOK_ALLOWLIST` (comma-separated exact Token-2022 hook program IDs; empty rejects hooks)
- `MEMESCANNER_ENABLE_PAPER_TRADING` is a legacy simulation setting; production ignores it and always sends signals only.
- `MEMESCANNER_COLLECT_OUTCOMES=false` disables prospective public-market capture; collection is enabled by default and never changes signals or position sizing

Missing Telegram credentials disable delivery clearly. A completed Tavily search with no indexed result remains partial OSINT, while a missing Tavily credential or outage defers candidates. Missing Helius/Solana RPC evidence also defers candidates and prevents alerts rather than silently treating them as safe. RPC errors never log credential-bearing request URLs.

Before committing, scan the exact staged Git index without printing any matched value:

```bash
python3 scripts/check_secrets.py
```

A repository-local pre-commit configuration and the `Secret pattern guard` GitHub workflow run the same staged/current-tree detector. Install the optional local hook with `pre-commit install`. `.env*`, private-key/certificate files, alternate secret YAML files, and common credential JSON files are ignored. These repository controls prevent new literals; they cannot revoke credentials that were previously exposed. Provider-side Telegram, Tavily, and Helius rotation remains mandatory, followed by updating the `MEMESCANNER_*` deployment secrets.

## Deployment via GitHub Actions

You can run the scanner continuously using the included GitHub Actions workflow. The workflow triggers manually and runs for up to 6 hours (the GitHub Actions per-job maximum).

### 1. Add repository secrets

Go to your repository **Settings > Secrets and variables > Actions** and add the following secrets:

| Secret name | Description |
|---|---|
| `MEMESCANNER_TAVILY_API_KEY` | Tavily API key (`tvly-...`) or X.ai key (`xai-...`) |
| `MEMESCANNER_HELIUS_RPC_URL` | Helius Solana RPC endpoint URL |
| `MEMESCANNER_TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather |
| `MEMESCANNER_TELEGRAM_CHAT_ID` | Telegram chat ID where alerts are delivered |

The workflow forces signal-only mode. Missing required configuration fails before scanning; secrets are never displayed. Telegram delivery is tested at startup.

### 2. Start the workflow

1. Go to the **Actions** tab in your repository.
2. Select **"Run Scanner"** from the workflow list on the left.
3. Click **"Run workflow"** and choose **main**. Code pushes to main also start a session.

Each deployment runs a **30-minute session**, not a 24/7 service. GitHub Actions may queue or fail; check the Run Scanner job and the startup Telegram message. Continuous hosting is not configured.

### 3. Restarting

Once the 30-minute session ends, manually start another session from Actions.
There is no automatic restart or scheduled 24/7 coverage. The workflow backs up
SQLite (including its WAL) to the `signal-state` artifact, then restores that
history for the next session. Artifacts expire after 90 days. Missing checkpoints
from an earlier started session halt restart rather than silently forgetting
delivery claims; recover the history before retrying. Use one process per database.

## Persistence and prospective calibration

SQLite stores two separate layers:

- `candidate_observations` and `discovery_cycles` preserve longitudinal decisions, evidence, source health, and alert state.
- `cohort_candidates` enrolls each normalized `(chain_id, mint)` exactly once, immediately after discovery and before filtering. `outcome_jobs`, `market_observations`, and `candidate_outcomes` then capture real USD-price observations at baseline, 1h, 6h, and 24h. Missing pairs, transient failures, and missed windows stay explicit missing data; they are never converted into zero returns.

Prospective outcome collection and calibration run on independent background tasks and a separate SQLite connection. Provider latency or retries cannot delay screening, alert delivery, or paper monitoring. Baseline jobs retain queue priority, while bounded concurrent requests reduce avoidable capture-window misses; provider outages still remain explicit missing data.

The runtime generates version-isolated chronological holdout reports with a 24-hour purge gap. Reports remain `INSUFFICIENT_DATA_FOR_CALIBRATION` until conservative sample, class-balance, capture-coverage, feature-coverage, and score-band gates all pass. The reporter is read-only: it cannot update scanner weights, alerts, paper positions, real position sizes, or execution behavior.

Operational one-shot commands are available for recovery and inspection:

```bash
python -m memescanner.outcomes
python -m memescanner.calibration --horizon 24h
```

## Calibration status

The repository starts without a complete prospective cohort, so it cannot immediately make honest win-rate, expected-return, or “high chance of skyrocketing” claims. It now collects the evidence required to remove that limitation over time and automatically suppresses predictive output until the predeclared gates pass. Screening scores remain ranking heuristics—not probabilities—until a versioned holdout report explicitly reaches `EMPIRICAL_HOLDOUT_CALIBRATION_READY`. Narrative, deployer, celebrity, and paid-boost signals remain neutral. See [`docs/evidence-and-calibration.md`](docs/evidence-and-calibration.md).

## Tests

```bash
python3 -m pytest tests/test_unified_scanner.py tests/test_onchain.py tests/test_x_search.py tests/test_celebrity_scanner.py -q
python3 -m pytest -q
```

## Disclaimer

Signals are informational only. Memecoins are extremely risky and many become worthless. The scanner does not provide financial advice or execute trades; users make all trading and risk decisions.
