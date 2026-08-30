# Early Discovery Company

Research and implementation specification — 2026-08-30.

Revision: the user subsequently accepted 1.31:1 minimum net reward/risk.
The prototype now uses that floor, independent two-second exit supervision,
and a zero-minute default discovery-age floor. Six independent workers remain
planned work; the current six scores are deterministic evidence summaries.

Status: proposed six-worker architecture, not a deployed trading company. This
document does not enable execution, subscribe to paid feeds, or certify the
existing prototype as production-safe. Exactly $11 is the total budget; no
additional infrastructure budget is assumed.

## Thesis

Identify a fresh, verifiable story that is beginning to attract independent
buyers, while executable liquidity still supports a small trade. Market cap
alone is not evidence of value, actual capital inflow, or exit liquidity.

Discover below $100,000 when possible; evaluate the $100,000–$200,000 band. If
verification finishes after the permitted entry window, record a missed entry.
Never raise the entry ceiling merely because the price ran away.

This is a hypothesis to test, not a demonstrated predictor of breakouts. Many
failed tokens share the same superficial characteristics as winners.

## Case research: evidence versus interpretation

Names are provisional matches: Official TRUMP, Pistacio, apeonfone (FONE), and
The Black Bull (ANSEM). Tickers are not identities. Bind each future case study
to a verified chain and mint before doing transaction-level replay.

| Case | Evidence found | Interpretation to test | Limitation |
| --- | --- | --- | --- |
| Official TRUMP | Official branding and a published allocation assigning 80% to two affiliated entities. | An existing mass audience can create immediate attention and demand. | This concentration conflicts with the conservative strategy. No public executable $100k–$200k entry was established. |
| Pistacio | Marcello publicly acknowledged a token based on his drawing and receiving fees; the community site distinguishes itself from the artist. | Recognizable original art plus acknowledgment can attract an existing community and encourage new content. | Acknowledgment came after token creation. It cannot be credited to an earlier signal before it was published and received. |
| apeonfone | The community account explicitly frames the meme around an ape trading from a phone. | A simple image and recognizable trader identity may encourage independent remixes and sharing. | Promotional positioning does not prove what caused purchases or establish an early profitable entry. |
| ANSEM | Ansem publicly described token-holder airdrop infrastructure and a creator-oriented ecosystem. | Influencer association and participation incentives may help sustain attention. | Later ecosystem announcements cannot explain an earlier launch without a timestamped reconstruction. Concentration and incentives need separate scrutiny. |

Primary evidence:

- [TRUMP official allocation](https://gettrumpmemes.com/).
- [Marcello acknowledgment](https://x.com/Marcello_695/status/2092347600510861350).
- [Pistacio community website and disclaimer](https://pistacio.world/).
- [FONE community narrative](https://x.com/apeonfone/status/2092764499669316013).
- [Ansem ecosystem statement](https://x.com/blknoiz06/status/2089381520821674132).

The X posts were available through search-indexed excerpts; direct page access
was limited. They are leads for archival verification, not a complete event
dataset. A [community Pistacio chronology](https://memecoin.wiki/wiki/pistacio)
links original posts and reports the sequence from art to launch to artist
acknowledgment. Its price and timing claims require independent reconstruction
before use in a backtest. No complete early-window replay was performed here.

## The six employees

| Worker | Owned task | Required deliverable | Authority |
| --- | --- | --- | --- |
| Scout | Ingest new mint/pool events; maintain low-cap watchlist; detect new narrative references. | Candidate identity, discovery time, supply basis, cap, liquidity, event references. | Nominate only. |
| Investigator | Bind the narrative to the exact mint; inspect deployer history and original social evidence. | Verified facts, unresolved claims, independent source count, provenance. | Block unverifiable identity. |
| Risk Defender | Inspect sell restrictions, token authorities, liquidity withdrawal controls, ownership clusters and manipulation. | PASS/FAIL/UNKNOWN per check, raw evidence, hard-veto reasons. | Safety veto; another worker cannot overrule it. |
| Market Analyst | Measure independent buyers, net quote-asset inflow, persistence, volatility and executable liquidity. | Timestamped feature windows, uncertainty, entry validity and invalidation conditions. | Block stale, exhausted or unsupported setups. |
| Trade Strategist | Price both legs; calculate target payoff, stop loss, costs, minimum net profit and holding limit. | Complete trade ticket with independently reproducible arithmetic. | Propose only. |
| Referee | Recompute hard gates, check conflicting evidence and freshness, then reserve treasury atomically. | BUY, WATCH or REJECT with evidence IDs and reasons. | Sole paper authorization; no signing keys. |

Workers may run concurrently after discovery, but dependent decisions wait for
their inputs. They are separate bounded services with structured outputs, not
six copies of one chat prompt. A shared immutable ledger and event queue are
supporting infrastructure, not a seventh employee.

Every worker report contains worker/version, chain/mint, evidence IDs,
source_event_time, received_at, evaluated_at, expires_at, score components,
confidence, missing fields, and vetoes. Duplicate reports from the same source
do not become independent corroboration. Scores are auditable rankings, not
probabilities of profit. UNKNOWN must never be converted to PASS.

Use deterministic code for ingestion, arithmetic, risk limits and position
management. Use a language model only where useful for interpreting narrative
evidence. Treat social text as untrusted data, never as instructions to the
system. No model can change treasury policy or access signing credentials.

## Early detection experiment

The following starting settings are provisional and must be frozen before
forward testing, then changed only in a new strategy version:

1. Discover immediately from supported launch/pool events. Do not enforce a
   blanket ten-minute wait before recording a candidate.
2. Track price and supply separately. Display circulating market cap and FDV;
   do not silently substitute one for the other. Unknown supply basis blocks
   the claimed $100k–$200k classification.
3. Record social changes in rolling 30-second, 2-minute and 5-minute windows:
   distinct original authors, growth relative to the author's normal activity,
   new independent communities, and original remixes versus duplicate posts.
   Missing social access means unknown, not zero activity.
4. Record corresponding on-chain windows: independently funded buyer clusters,
   quote-asset net inflow, repeat buying, successful sells, top-holder changes,
   and execution depth. Shared funding is a warning, not conclusive identity:
   common exchange withdrawals can create false links.
5. Require persistence across at least two completed windows and evidence of
   both attention and buying. Do not invent validated numeric thresholds from
   four winning examples; derive and freeze them using winners and failures.
6. Mark paid promotion separately. DEX Screener explicitly sells boosts; a
   trending placement is not proof of organic interest. See its
   [boosting documentation](https://docs.dexscreener.com/boosting).
7. Request fresh entry and exit quotes. Check sellability and token/program
   restrictions independently; a quote alone cannot guarantee execution.
8. Expire the entry authorization after a short, explicitly tested interval
   (initial experiment: five seconds). Requote at authorization. An expired
   report or missed entry produces WATCH, not an automatic chase.

For liquidity, inspect the actual venue, active depth, executable sale size
and withdrawal powers. A generic LP-locked boolean is insufficient across
bonding curves, concentrated-liquidity pools and migrations. Unsupported pool
semantics block trading. No universal market-cap/liquidity ratio proves safety.

## Data plumbing and latency

- [Birdeye new-listing subscriptions](https://docs.birdeye.so/reference/new-token-listing)
  expose listing events, including supported launch-platform sources. Coverage
  is provider-specific, not a guarantee of every mint at creation.
- [Solana log subscriptions](https://solana.com/docs/rpc/websocket/logssubscribe)
  can feed program-specific collectors; transactions still need decoding,
  deduplication, commitment handling and gap recovery.
- [X filtered stream](https://docs.x.com/x-api/posts/filtered-stream/introduction)
  supplies near-real-time posts matching rules. Authorized access, pricing and
  delivery latency constrain feasibility; do not assume free access.

Use independent asynchronous loops for discovery and position supervision.
Measure source-to-receipt, receipt-to-decision and decision-to-fill latency.
Slow narrative analysis must not block exits. Queue overflow, stale streams or
RPC failure suspend new entries and trigger an operator alert. Paper tests
must model missing quotes, delayed fills and unfillable exits explicitly.

Do not buy subscriptions or operate six paid model calls on every token with
an $11 treasury. Prefer available authorized feeds, deterministic filtering,
caching and event-driven model use. Record all compute/API costs. If required
infrastructure cannot fit the budget, remain in offline research/shadow mode.

## Treasury and payoff gates

Preserve the user's rules: $11 total; $1 default and $2 maximum position; one
open position; at least $5 uncommitted after entry and reserved execution costs;
no leverage, borrowing, DCA or averaging down. Halt new entries at $1 daily
loss or three consecutive losses. Include costs in P&L and account for open
exposure when evaluating remaining loss capacity. Exits remain allowed during
a halt. Define the accounting day as UTC and persist halt state across restarts.

Let P be principal, g the target return, s the planned stop return, Cwin the
round-trip cost at the target, and Closs the cost under the stop scenario:

- Target gross payoff = P * g.
- Target net payoff = P * g - Cwin.
- Planned net loss = P * s + Closs.
- Net reward/risk = (P * g - Cwin) / (P * s + Closs).
- Require Cwin <= 25% of target gross payoff and net reward/risk >= 1.31.

Keep the proposed gross target within 8–15% and planned stop within 5–8%.
Do not select a tighter stop merely to make the arithmetic pass if observed
volatility makes it unsuitable. Actual loss can exceed the stop, potentially
the full position plus costs, if liquidity vanishes or execution fails.

Suggested additional paper-only experiment: require at least $0.08 target net
payoff per completed trade. This is a proposed absolute floor, not a user-
approved production setting or an empirically established optimum. Reject a
marginal trade rather than increasing frequency to manufacture activity.

Example: $2 principal, 10% target, 5% stop and $0.03 round-trip costs gives
$0.20 gross, $0.17 net target and $0.13 planned loss. Costs consume only 15%
of gross profit, and net reward/risk is approximately 1.30769:1. It still falls
just below a strict 1.31 minimum; do not round upward to authorize the trade.

Include network and priority fees, DEX/platform fees, transfer taxes, expected
slippage, price impact, required account creation/capital lockup and failed
transaction costs without double-counting charges already included in quotes.
Book unrecoverable setup costs as costs; reserve refundable deposits rather
than calling them profit or immediately available cash. Report applicable
personal-tax assumptions separately; do not label trading P&L after-income-tax
profit without the user's tax context.

Target payoff is not statistical expected profit. Estimate probability-weighted
expectancy only from suitable out-of-sample outcomes, including failed exits
and tail losses. Do not invent win probabilities or map employee scores
directly to expected returns.

Use a predeclared time stop (initial paper experiment: no more than 15 minutes)
and earlier exit on thesis invalidation or loss of momentum. Do not keep a
position open hoping for recovery. Reserve exit costs before entering.

## Required ticket and final decisions

Every authorized paper entry must first persist and output:

```text
Token:
Contract:
Entry amount:
Entry price:
Stop:
Profit target:
Maximum holding time:
Estimated round-trip costs:
Expected gross profit:
Expected net profit:
Liquidity and price impact:
Critical risks:
Employee scores:
Final decision:
```

Add mode, quote timestamp, expiry, source links and rejection reasons. Explain
that profit fields are conditional target payoffs unless probabilities have
actually been calibrated. BUY authorizes a paper entry only; WATCH means
incomplete/stale evidence or an unready/missed entry; REJECT means a failed
safety or economic gate. Never send a misleading optimistic headline alert
that contradicts the Referee's ticket.

## Validation and release gates

1. Reconstruct each verified example from creation through the target cap band.
   Freeze features at the time they were actually received. If historical
   receipt times are unavailable, model realistic delays and label the replay
   approximate. Never insert later endorsements into an earlier signal.
2. Include all discovered launches in the selected time interval, including
   dead tokens, scams, flat outcomes and missing-data cases. Record WATCH and
   REJECT decisions as well as entries; report missed winners without removing
   safety rules to fit them retrospectively.
3. Split development and evaluation chronologically. Measure net expectancy,
   drawdown, cost/profit ratio, failure rate, latency, missed-entry rate and
   P&L after overhead. A strategy that depends on one huge winner fails the
   stated small-repeatable-profit objective.
4. Complete at least 100 distinct forward paper trade lifecycles before any
   live-capital review. Duplicate alerts and partial exits do not count as new
   completed signals. Retain at least 100 resolved candidate decisions too;
   sample size is a minimum operational gate, not proof of profitability.
5. Test missing/NaN/infinite prices, missing taxes, stale evidence, spoofed
   contracts, clustered wallets, unsupported pools, failed sells, feed outages,
   duplicate events, simultaneous authorizations and process restarts.
6. Human approval is mandatory and never automatic at signal 100. A future
   deterministic wallet service must enforce a hard-coded $2 transaction
   limit, aggregate one-position/reserve/daily-loss gates, allowlisted swap
   programs, bounded fees, and no arbitrary transfer/withdrawal authority.
   No private keys, seed phrases or signing secrets reach an AI model.

## Repository implementation gaps observed

The local prototype has a single planning function generating six scores, not
six independently evidenced workers. The previous ten-minute discovery floor
and five-minute main-loop position checks have been replaced. Streaming launch
coverage, executable quotes and forward validation still require implementation
before claiming reliable early discovery or live micro-trade protection.

Implementation order: evidence ledger and replay harness; streaming collectors;
six bounded workers and independent Referee; atomic treasury and independent
exit watchdog; forward shadow validation. Existing unfinished trading changes
must receive a separate safety review. This specification does not certify or
deploy them. The accompanying paper-safety revision changes runtime defaults
as described above without adding real execution or streaming collectors.
