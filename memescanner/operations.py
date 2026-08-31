"""Runtime singleton and bounded operational alerts; never override risk vetoes."""

from __future__ import annotations

import fcntl
import time
from collections import Counter
from typing import Any, TextIO


class ProcessGuard:
    def __init__(self, path: str):
        self.file: TextIO = open(path, "a", encoding="utf-8")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("Another signal process owns this database") from None

    def close(self) -> None:
        self.file.close()


class HealthReporter:
    def __init__(self) -> None:
        self.last_sent = -float("inf")
        self.reported_access: set[str] = set()

    def message(self, result: dict[str, Any], x_failures: dict[str, str], rpc_error: str | None,
                validation: dict[str, Any]) -> str | None:
        access_errors = {value for value in x_failures.values() if "OPERATOR_ACTION" in value}
        if rpc_error in {"HTTP_401", "HTTP_402", "HTTP_403"}:
            access_errors.add("RPC_" + rpc_error)
        new_access_error = bool(access_errors - self.reported_access)
        if not new_access_error and time.monotonic() - self.last_sent < 900:
            return None
        self.reported_access.update(access_errors)
        self.last_sent = time.monotonic()
        reasons = Counter(reason for item in result.get("decisions", []) for reason in item.reasons)
        return "\n".join((
            "Signal company health — research only",
            f"Candidates this cycle: {result.get('discovered', 0)}; alert sent: {bool(result.get('alerted'))}",
            f"Social provider errors: {x_failures or 'none reported'}",
            f"RPC last error: {rpc_error or 'none reported'}",
            "401/402/403 require the operator to check API credentials, credits and access. No bypass is attempted.",
            f"Most common blocks: {dict(reasons.most_common(3))}",
            f"Forward samples: {validation['completed']}/100 complete; {validation['incomplete']} incomplete; {validation['status']}",
            "Paper outcomes use estimated costs and sampled prices, not proven swap fills. No live execution.",
        ))
