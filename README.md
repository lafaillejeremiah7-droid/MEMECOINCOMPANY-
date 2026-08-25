# Memescanner — Solana Signal Scanner

Memescanner is a **signal-only** Solana token discovery and safety-screening service. The default `python -m memescanner` runtime never loads wallet keys, signs transactions, submits transactions, or executes live trades. Optional PaperTrader behavior is virtual accounting only, is disabled by default, and is capped at three open positions.

## Default architecture

The default runtime uses one normalized pipeline:

1. Discover recent Solana candidates independently from DEXScreener token profiles, DEXScreener latest paid boosts, CoinGecko/GeckoTerminal Solana new pools, and Pump.fun.
2. Normalize and merge duplicates by `(chain_id, mint)`, unioning source membership and preserving social, creator, creation-time, and paid-boost metadata.
3. Select only a Solana DEX pair—there is no cross-chain fallback.
4. Apply the same age, X-presence, liquidity, trading-flow, on-chain, holder, rug/scam-evidence, and evidence-availability checks to every source.
5. Persist every observation and decision (qualified, rejected, or deferred) in SQLite, then emit at most one deduplicated alert per mint.

“All-platform” means launchpad-neutral discovery across supported Solana DEX data sources. It is not a guarantee that every token will be discovered. A public source can be unavailable or rate-limited; failures are isolated and other adapters continue. DEX market enrichment is capped per cycle (40 by default); over-budget candidates are persisted as deferred and rotate into later cycles rather than being discarded.

## Evidence semantics

- “New” requires a known token/pair timestamp and an age of 10–60 minutes. Unknown age is rejected as `AGE_UNKNOWN_NOT_NEW`.
- An X link is required. A completed public search with no result is recorded as `X_DATA_NOT_FOUND_OR_NOT_INDEXED`; that is partial OSINT, not proof that a token has no attention. A missing credential or search outage is separately marked unavailable and defers the candidate.
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
- `MEMESCANNER_TAVILY_API_KEY`
- `MEMESCANNER_HELIUS_RPC_URL` (preferred complete RPC endpoint)
- `MEMESCANNER_HELIUS_API_KEY` (used only when no complete RPC URL is set)
- `MEMESCANNER_TRANSFER_HOOK_ALLOWLIST` (comma-separated exact Token-2022 hook program IDs; empty rejects hooks)
- `MEMESCANNER_ENABLE_PAPER_TRADING=true` enables virtual PaperTrader accounting, five-minute position checks, hourly portfolio updates, and daily P&L summaries

Missing Telegram credentials disable delivery clearly. A completed Tavily search with no indexed result remains partial OSINT, while a missing Tavily credential or outage defers candidates. Missing Helius/Solana RPC evidence also defers candidates and prevents alerts rather than silently treating them as safe.

## Persistence

SQLite’s `candidate_observations` table stores observation time, outcome-ready `(chain_id, mint)` identity, source memberships, boost metadata, age provenance, decision-time market data and screening rank, evidence availability, filter decision/reasons, and alert state. `discovery_cycles` separately stores source health even when every adapter is unavailable and no candidate exists. The tables and indexes are created with backward-compatible additive migration behavior alongside existing tables.

## Research and calibration limitation

Historical predictive calibration is unavailable for this repository. Screening scores are ranking heuristics, **not probabilities**. Narrative, deployer, celebrity, and paid-boost signals are not calibrated probability multipliers and must not be presented as an edge or expected return. See [`docs/evidence-and-calibration.md`](docs/evidence-and-calibration.md).

## Tests

```bash
python3 -m pytest tests/test_unified_scanner.py tests/test_onchain.py tests/test_x_search.py tests/test_celebrity_scanner.py -q
python3 -m pytest -q
```

## Disclaimer

Signals are informational only. Memecoins are extremely risky and many become worthless. The scanner does not provide financial advice or execute trades; users make all trading and risk decisions.
