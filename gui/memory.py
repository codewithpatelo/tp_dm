"""Memoria de 3 capas para el agente cognitivo.

- WorkingMemory: in-RAM scratchpad para el turno actual.
- EpisodicMemory: SQLite append-only con turnos completos.
- SemanticMemory: ver memory_semantic.py (fase 2).
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MEM_DIR = Path(__file__).resolve().parent / "_memory"
MEM_DIR.mkdir(exist_ok=True)
EPISODIC_DB = MEM_DIR / "episodic.db"


# ---------- working memory ---------------------------------------------------

@dataclass
class WorkingMemory:
    """Scratchpad del turno: intención, plan, observaciones, verdict."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    user_message: str = ""
    intent: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    iter_count: int = 0
    max_iter: int = 2
    tokens_in: int = 0
    tokens_out: int = 0
    started_at: float = field(default_factory=time.time)
    phase_log: list[dict] = field(default_factory=list)  # pasos del ciclo (UI)

    def add_observation(self, skill: str, args: dict, result: Any,
                        ok: bool = True, error: str | None = None) -> None:
        self.observations.append({
            "skill": skill,
            "args": args,
            "ok": ok,
            "result": result,
            "error": error,
            "ts": time.time(),
        })

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "user_message": self.user_message,
            "intent": self.intent,
            "plan": self.plan,
            "observations": self.observations,
            "verdict": self.verdict,
            "answer": self.answer,
            "iter_count": self.iter_count,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "duration_s": time.time() - self.started_at,
            "phase_log": self.phase_log,
        }


# ---------- episodic memory --------------------------------------------------

class EpisodicMemory:
    """Append-only SQLite. Una fila por turno; skill_calls separados."""

    SCHEMA_TURNS = """
    CREATE TABLE IF NOT EXISTS turns (
        turn_id TEXT PRIMARY KEY,
        ts REAL NOT NULL,
        user_message TEXT,
        intent_json TEXT,
        answer TEXT,
        verdict_json TEXT,
        tokens_in INTEGER,
        tokens_out INTEGER,
        duration_s REAL
    )
    """
    SCHEMA_SKILLS = """
    CREATE TABLE IF NOT EXISTS skill_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL,
        skill TEXT NOT NULL,
        args_json TEXT,
        result_summary TEXT,
        ok INTEGER NOT NULL,
        error TEXT,
        ts REAL NOT NULL
    )
    """

    def __init__(self, db_path: Path = EPISODIC_DB):
        self.db_path = db_path
        with self._conn() as cx:
            cx.execute(self.SCHEMA_TURNS)
            cx.execute(self.SCHEMA_SKILLS)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save_turn(self, wm: WorkingMemory) -> None:
        with self._conn() as cx:
            cx.execute(
                """INSERT OR REPLACE INTO turns
                (turn_id, ts, user_message, intent_json, answer, verdict_json,
                 tokens_in, tokens_out, duration_s)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    wm.turn_id, wm.started_at, wm.user_message,
                    json.dumps(wm.intent, default=str),
                    wm.answer,
                    json.dumps(wm.verdict, default=str),
                    wm.tokens_in, wm.tokens_out,
                    time.time() - wm.started_at,
                ),
            )
            for obs in wm.observations:
                summary = _summarize_result(obs.get("result"))
                cx.execute(
                    """INSERT INTO skill_calls
                    (turn_id, skill, args_json, result_summary, ok, error, ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        wm.turn_id, obs["skill"],
                        json.dumps(obs.get("args", {}), default=str),
                        summary,
                        1 if obs.get("ok") else 0,
                        obs.get("error"),
                        obs.get("ts", time.time()),
                    ),
                )

    def recent_turns(self, limit: int = 10) -> list[dict]:
        with self._conn() as cx:
            cx.row_factory = sqlite3.Row
            rows = cx.execute(
                "SELECT * FROM turns ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def turn_skills(self, turn_id: str) -> list[dict]:
        with self._conn() as cx:
            cx.row_factory = sqlite3.Row
            rows = cx.execute(
                "SELECT * FROM skill_calls WHERE turn_id = ? ORDER BY ts ASC",
                (turn_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def _summarize_result(result: Any, max_chars: int = 500) -> str:
    """Versión textual chiquita del resultado para guardar en SQLite."""
    if result is None:
        return ""
    try:
        if hasattr(result, "shape"):  # DataFrame / array
            return f"<{type(result).__name__} shape={result.shape}>"
        s = json.dumps(result, default=str)
    except Exception:  # noqa: BLE001
        s = str(result)
    return s[:max_chars]
