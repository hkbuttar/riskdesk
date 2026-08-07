"""Portfolio-level kill-switch: the same manual-reset-only design
alpha-signal-lab's own `risk/kill_switch.py` already established for this
whole portfolio of projects (and pairtrade-lab-1's `risk.kill_switch` uses
the same pattern) -- reused directly here, not reinvented, per the plan's
explicit instruction ("consistent with the whole portfolio's risk-control
pattern"). Once triggered, `.triggered` stays `True` until `.reset()` is
called explicitly: a kill-switch should not silently re-arm itself just
because the breach that caused it happened to go away on the next check.

Extended beyond the sibling repos' single-condition (equity drawdown) kill
switches: this one can be triggered by any combination of independently
checked limits -- market-risk (VaR) or credit-risk (counterparty
concentration) -- and records WHICH limit(s) actually triggered it, since a
live monitor needs to tell a human what to go review, not just that
something is wrong.

State persistence, a real difference from the sibling repos' switches: their
kill switches only need to survive across bars WITHIN one backtest run (an
in-memory object is enough, since the whole run is one process). This
project's monitor is meant to be invoked repeatedly as a live, recurring
check (e.g. on a schedule) -- for "stays triggered until manually reset" to
mean anything across separate process invocations, not just within one, the
state has to be written to disk and reloaded, not just held in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STATE_PATH = Path(__file__).parent / "state.json"


@dataclass
class KillSwitch:
    triggered: bool = False
    trigger_reasons: list[str] = field(default_factory=list)

    def check(self, breaches: dict[str, bool | str]) -> bool:
        """`breaches` maps a limit name to either a bool (breached or not)
        or a human-readable reason string (implicitly breached, non-empty).
        Any breach latches `.triggered` True and records its reason if not
        already recorded (no duplicate entries across repeated checks).
        """
        for name, detail in breaches.items():
            breached = bool(detail) if isinstance(detail, bool) else bool(detail)
            if not breached:
                continue
            reason = detail if isinstance(detail, str) else name
            if reason not in self.trigger_reasons:
                self.trigger_reasons.append(reason)
            self.triggered = True
        return self.triggered

    def reset(self) -> None:
        """Manually re-arm the kill-switch after review."""
        self.triggered = False
        self.trigger_reasons = []


def load_state(path: Path = DEFAULT_STATE_PATH) -> KillSwitch:
    if not path.exists():
        return KillSwitch()
    data = json.loads(path.read_text())
    return KillSwitch(triggered=data.get("triggered", False), trigger_reasons=data.get("trigger_reasons", []))


def save_state(kill_switch: KillSwitch, path: Path = DEFAULT_STATE_PATH) -> None:
    path.write_text(json.dumps({"triggered": kill_switch.triggered, "trigger_reasons": kill_switch.trigger_reasons}))
