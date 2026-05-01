"""Memoria semántica: append-only JSONL + retrieval por grep (vectorless).

Filosofía: cada lesson/hypothesis/experiment es una línea JSON. Para buscar,
hacemos grep + ranking simple por overlap de tokens. Sin embeddings.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

MEM_DIR = Path(__file__).resolve().parent / "_memory"
MEM_DIR.mkdir(exist_ok=True)
SEMANTIC_FILE = MEM_DIR / "semantic.jsonl"


class SemanticMemory:
    _instance: "SemanticMemory | None" = None

    def __init__(self, path: Path = SEMANTIC_FILE):
        self.path = path
        self.path.touch(exist_ok=True)

    @classmethod
    def get(cls) -> "SemanticMemory":
        if cls._instance is None:
            cls._instance = SemanticMemory()
        return cls._instance

    def append(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("id", uuid.uuid4().hex[:8])
        record.setdefault("ts", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def all(self) -> list[dict]:
        out: list[dict] = []
        with self.path.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return out

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[dict]:
        q_tokens = set(re.findall(r"\w+", query.lower()))
        if not q_tokens:
            return []
        scored: list[tuple[float, dict]] = []
        for rec in self.all():
            if kind and rec.get("kind") != kind:
                continue
            blob = " ".join(str(v) for v in rec.values()).lower()
            r_tokens = set(re.findall(r"\w+", blob))
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q_tokens), 1)
            scored.append((score, rec))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:k]]
