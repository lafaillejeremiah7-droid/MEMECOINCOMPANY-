# Recorded provider fixtures

Real HTTP responses from DexScreener, GeckoTerminal, Pump.fun, Helius RPC and
X.ai, captured from live providers and replayed offline by
`tests/test_provider_contracts.py` and `tests/test_pipeline_offline.py`.

## Why these exist

Every hand-written mock in this repository encoded an assumption about a provider
rather than the provider's real behaviour, and one of those assumptions hid a
production bug.

`tests/test_x_search.py` asserted that X.ai returns citations shaped like
`https://x.com/trader1/status/12345`. It does not. It returns
`https://x.com/i/status/<id>` — `i` is a placeholder and **no account handle
appears anywhere in the URL**. Because the mock supplied handles that real
responses never contain, the suite stayed green while the live mention counter
returned `1` for every token in existence, including BONK and dogwifhat.

A mock can only confirm what its author already believed. These fixtures can only
be wrong if reality changed — and `--check` detects that.

## Re-recording

```bash
PYTHONPATH=. python scripts/record_fixtures.py
```

Requires `MEMESCANNER_TAVILY_API_KEY` (or `MEMESCANNER_XAI_API_KEY`) and
`MEMESCANNER_HELIUS_RPC_URL`. The recorder runs the real pipeline and captures
whatever it actually requests — which is how `getSignaturesForAddress` and
`getTransaction` ended up covered, having been missed when the endpoint list was
enumerated by reading call sites.

## Drift detection

```bash
PYTHONPATH=. python scripts/record_fixtures.py --check
```

Re-fetches each recorded GET and compares response **shape** — nested key names
and value types — not values. A changed price or timestamp is normal and does not
fail. A renamed field, a removed field, or a changed type is drift, and is
reported.

This matters because recorded fixtures introduce their own failure mode: a stale
fixture keeps the suite green while production misparses live data. `--check` is
the guard against trading one blind spot for another.

## Layout

| Path | Contents |
| --- | --- |
| `http/index.json` | request signature → fixture filename |
| `http/_meta.json` | `recorded_at` epoch, used to freeze the clock on replay |
| `http/<slug>_<hash>.json` | one recorded response: status, content-type, body |

`recorded_at` is load-bearing. Candidate ages derive from `pair_created_at`, so a
replay that does not freeze the clock sees every recorded token drift past
`max_candidate_age_minutes`, and the age gates stop being exercised at all.

## Credentials

Fixtures are scrubbed before being written: API keys in query strings or path
segments become `REDACTED`, `Authorization` headers are never stored, and JSON
body fields named like keys are removed. `scripts/check_secrets.py` runs over
these files in CI as a backstop.
