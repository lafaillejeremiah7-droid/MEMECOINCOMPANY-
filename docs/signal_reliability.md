# Signal reliability and operating limits

Production sends research signals to Telegram; it has no wallet, signing, withdrawal
or swap-execution authority. No AI model receives wallet keys. BUY means the software
checks passed, not a promise of profit or a command to skip the user's manual limits.

## What changed

- DEXScreener enrichment now passes five-minute momentum and pool/mint identifiers
  through to the analyst and liquidity checker. Missing momentum cannot approve BUY.
- Social HTTP 401/402/403 errors are marked unavailable and disable that backend for
  the process lifetime. There is no blind retry or permission workaround. Temporary
  failures have a cooldown. RPC transient errors have bounded retries; long
  Retry-After delays defer work. Error logging omits request URLs and secret values.
- A Telegram operations update summarizes candidate counts, common rejection reasons,
  API errors and forward-validation progress every 15 minutes, with earlier notice
  for a newly observed access denial. Telegram acceptance does not prove a person read it.
- Durable, atomic claims allow one WATCH followed by one BUY per mint. Fresh checks
  are mandatory. Definite delivery failures release only their own claim; uncertain
  deliveries block retries pending operator review. Historical unknown delivery is
  treated conservatively. The observation-index migration keeps old rows intact.
- A local process lock rejects a second scanner using the same database. Keep one
  deployment and one persistent database; separate hosts/databases cannot deduplicate
  each other's alerts. Do not run Actions and Docker against independent histories.

## Exact-pool liquidity evidence

The verifier uses confirmed read-only Solana RPC data. It verifies the pool's owner,
binary layout, trading status where defined, token mint, quote mint and LP mint.
It then fetches the pool and LP mint together, checks the response slot, and compares
the pool's LP reserve/reference supply with the current LP mint supply using integer
arithmetic. At least 99% must be burned. A refreshed DEX snapshot must still use that
same pool. The check never signs transactions.

Supported layouts:

- [Raydium AMM v4 layout](https://github.com/raydium-io/raydium-sdk-V2/blob/master/src/raydium/liquidity/layout.ts):
  program `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8`, 752-byte account,
  trading status 1 or 6; base/quote/LP mints at offsets 400/432/464, LP reserve at 720.
- [PumpSwap IDL](https://github.com/pump-fun/pump-public-docs/blob/main/idl/pump_amm.json):
  program `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA`, Pool discriminator,
  base/quote/LP mints at offsets 43/75/107, LP reference supply at 203.
  Mayhem mode is not supported.

Unsupported layouts, pool changes, missing RPC evidence, invalid supply relationships
and unverified time locks stay UNKNOWN. Not finding burned LP is not proof that
liquidity is removable. LP burns do not eliminate program upgrades, exploits, token
risks, insider selling or changing execution liquidity. Other vetoes remain active.
Fixtures test the parser; they do not replace live protocol/account verification.

The [X.ai X Search documentation](https://docs.x.ai/developers/tools/x-search) specifies
the flat `{"type":"x_search"}` tool declaration used by the client. An accepted
request still depends on a valid account, allowed model/tool and available credits.

## Forward validation is not fabricated fills

Only delivered, fresh BUY setups can enter the reference ledger. It permits one
position, applies $11 starting capital, the $2 maximum, $5 reserve, $1 daily-loss
halt and three-consecutive-loss halt. WATCHs, duplicates and stale entries do not count.
The observer checks public market snapshots approximately every five seconds.
Observed stop/target, lost momentum or maximum holding time closes a sample using
the observed price, less the plan's estimated round-trip costs exactly once.

A gap longer than 15 seconds, missing/invalid price or changed pool makes the path
INCOMPLETE. It contributes no invented P&L and blocks additional reference entries
until an operator reviews the missing path. Restarts never clear this state or the
loss halts. Preserve the database for review; do not delete rows to obtain a pass.
The production signal stream is advisory and does not read your actual holdings.

Inspect the reference report without starting the scanner:

```bash
python -m memescanner.validation --report memescanner.db
```

The 100-sample counter is version-isolated. Even after 100 completed samples, status
is only NEEDS_HUMAN_REVIEW; `live_execution_allowed` and `executable_fills_verified`
remain false. These are sampled, modeled outcomes, not executable quotes or a
validated strategy win rate. HTTP latency, API price caching, liquidity, rent,
slippage and actual exit costs can make realizable results worse. No live switch exists.

## Continuous-host handoff

Actions remains a finite 30-minute runner. No paid host, schedule or new account is
created by this change. On a host explicitly approved and funded by the operator:

1. Stop any other scanner deployment and recover the latest `signal-state` database.
2. Configure the required `MEMESCANNER_*` variables in an untracked `.env` file.
   Never add wallet keys. Use the existing Telegram destination.
3. Restore the database into the Compose `signal-data` volume, owned by UID 10001,
   before starting. Do not silently replace an existing database with a fresh one.
4. Run `docker compose up -d --build` and check `docker compose logs --tail=100`.
5. Confirm the Telegram startup notice and operations update. Correct provider
   permissions/billing in the provider account if they report access denial.
6. Back up SQLite with its backup API/checkpoint, protect the volume and review
   incomplete samples and uncertain delivery claims. Never rely on container logs
   alone as a validation record.

Compose uses a persistent volume, a non-root process, dropped capabilities, a
single-process lock, and restart-unless-stopped. It is deployment scaffolding, not
evidence that an always-on host is running. Startup checks presence of required
configuration and Telegram acceptance; this does not certify all providers healthy.
