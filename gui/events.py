"""Event bus pub-sub minimo para coordinar componentes del agente.

Patrón observador. La UI suscribe handlers; las skills emiten eventos
sin saber quién escucha. Permite trace en vivo + futuras integraciones
(notificaciones, métricas).
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

EventHandler = Callable[["Event"], None]

# Tipos de eventos canónicos. Strings (no Enum) para serializar fácil.
EVT_USER_MESSAGE = "user.message"
EVT_INTENT_PARSED = "intent.parsed"
EVT_PLAN_DRAFTED = "plan.drafted"
EVT_SKILL_CALLED = "skill.called"
EVT_SKILL_RESULT = "skill.result"
EVT_OBSERVATION = "observation.made"
EVT_VERDICT = "verdict.reached"
EVT_REPORT = "report.delivered"
EVT_PROPOSAL_PENDING = "proposal.pending"
EVT_PROPOSAL_RESOLVED = "proposal.resolved"
EVT_ERROR = "error.raised"
EVT_TOKENS = "tokens.consumed"
# Fase del ciclo cognitivo (SENSE/PLAN/ACT/…); payload = { "key", "label", "detail" }
EVT_COGNITIVE_PHASE = "cognitive.phase"


@dataclass
class Event:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turn_id: str | None = None


class EventBus:
    """Bus simple. Mantiene cola circular de últimos N eventos para trace UI."""

    def __init__(self, history: int = 200):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._wildcard: list[EventHandler] = []
        self.history: deque[Event] = deque(maxlen=history)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        if event_type == "*":
            self._wildcard.append(handler)
        else:
            self._handlers[event_type].append(handler)

    def emit(self, event_type: str, payload: dict[str, Any] | None = None,
             turn_id: str | None = None) -> Event:
        evt = Event(type=event_type, payload=payload or {}, turn_id=turn_id)
        self.history.append(evt)
        for h in self._handlers.get(event_type, []):
            try:
                h(evt)
            except Exception as e:  # noqa: BLE001
                print(f"[event-bus] handler para {event_type} explotó: {e}")
        for h in self._wildcard:
            try:
                h(evt)
            except Exception as e:  # noqa: BLE001
                print(f"[event-bus] wildcard handler explotó: {e}")
        return evt

    def recent(self, n: int = 50, turn_id: str | None = None) -> list[Event]:
        items = list(self.history)
        if turn_id:
            items = [e for e in items if e.turn_id == turn_id]
        return items[-n:]


_BUS: EventBus | None = None


def get_bus() -> EventBus:
    global _BUS
    if _BUS is None:
        _BUS = EventBus()
    return _BUS
