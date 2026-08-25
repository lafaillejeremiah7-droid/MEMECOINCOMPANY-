# Evidence and calibration protocol

The repository begins without a sufficiently documented point-in-time historical cohort. It therefore does not claim a historical win rate, expected return, calibrated probability, or proven trading edge. The prospective workflow below is designed to collect the missing evidence without allowing research results to alter signals or risk automatically.

## Cohort enrollment

Every normalized Solana `(chain_id, mint)` is enrolled in `cohort_candidates` immediately after source merging and before age, market, X, on-chain, holder, scam, alert, or paper filters run. First discovery time is insert-only. Repeated source appearances and alert-state observations never become additional samples.

Enrollment also creates idempotent outcome jobs for:

- baseline: target at first discovery, two-minute capture window;
- 1 hour: five-minute capture window;
- 6 hours: fifteen-minute capture window;
- 24 hours: sixty-minute capture window.

The outcome worker uses the same strict DEXScreener rule as screening: Solana only, and the candidate mint must be the base token because the endpoint’s price and transaction fields are base-oriented. It records the provider, pair, requested horizon, actual capture time, lag, USD price, market cap, liquidity, and status.

Missing evidence is never assigned a synthetic return. No pair, invalid price, transient errors, and expired windows remain explicit missingness in the denominator and coverage report. A return is computed only when both a positive baseline USD price and a positive horizon USD price were actually captured.

## Predeclared outcome

`price-return-2x-v1` defines the research event as a USD-price return of at least +100% from the prospective baseline. Continuous price return is retained alongside that binary event. This definition does not include transaction costs, slippage, market impact, or guaranteed executability, so even a statistically valid report would not itself establish tradable expected value.

## Leakage controls

Candidate features are frozen from the first discovery cycle, including explicit deferred/missing states. Later rediscovery cannot backfill a predictor. A row is calibration-eligible only when its frozen feature timestamp is no later than that horizon’s outcome target.

Reports isolate one policy version, feature-schema version, outcome-definition version, and horizon at a time. Candidate order is determined only by first discovery time. The chronological boundary is established from the full due cohort before missing rows are removed. Development data precedes holdout data, with a 24-hour purge gap between them, and outcome/feature coverage must pass separately in both partitions. The holdout is never used to update scanner weights.

The default report gate requires all of the following:

- at least 90% due-outcome capture coverage;
- at least 90% first-evaluation feature coverage;
- at least 500 development candidates after the purge;
- at least 500 holdout candidates;
- at least 50 positive and 50 negative holdout outcomes;
- at least 100 development and 100 holdout candidates in each displayed score band;
- at least two reportable score bands.

Before every gate passes, the status is `INSUFFICIENT_DATA_FOR_CALIBRATION`, score-band rates are suppressed, and no probability or predictive-edge claim is permitted. Once gates pass, `EMPIRICAL_HOLDOUT_CALIBRATION_READY` exposes versioned train-band rates, holdout event rates, Wilson 95% intervals, and holdout Brier scores using frozen development rates. These remain empirical research estimates—not promises of future returns.

## Strict separation from trading behavior

Prospective collection and calibration are observation/read-only paths. They cannot:

- change hard safety gates or scanner weights;
- raise celebrity, narrative, deployer, or paid-boost importance;
- open or alter a paper position;
- choose a real position size or risk amount;
- load a wallet, sign, submit, or execute a transaction.

Any future scoring change requires a new policy/feature version and a new prospective holdout. The user remains solely responsible for whether to trade and how much to risk.

## Operational commands

The default `python -m memescanner` runtime enrolls candidates immediately, while independent background workers prioritize baseline capture, service due outcome jobs with bounded concurrency, and write daily calibration-gate reports without blocking signal evaluation. One-shot recovery/report commands are also available:

```bash
python -m memescanner.outcomes
python -m memescanner.calibration --horizon 1h
python -m memescanner.calibration --horizon 6h
python -m memescanner.calibration --horizon 24h
```

All artifacts are stored in SQLite. Legacy alert-only outcome fields are excluded from this prospective cohort because they are selected after filtering and cannot provide an unbiased denominator.
