"""Registry de skills. Cada skill es callable (args: dict) -> result."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

SkillFn = Callable[..., Any]


@dataclass
class Skill:
    name: str
    fn: SkillFn
    description: str
    args_schema: dict


_REGISTRY: dict[str, Skill] = {}


def register(name: str, description: str, args_schema: dict | None = None):
    def deco(fn: SkillFn):
        _REGISTRY[name] = Skill(name, fn, description, args_schema or {})
        return fn
    return deco


def get(name: str) -> Skill | None:
    return _REGISTRY.get(name)


def all_skills() -> dict[str, Skill]:
    return dict(_REGISTRY)


def call(name: str, args: dict) -> Any:
    skill = get(name)
    if skill is None:
        raise KeyError(f"skill desconocida: {name}")
    sig = inspect.signature(skill.fn)
    accepted = {k: v for k, v in args.items() if k in sig.parameters}
    return skill.fn(**accepted)


# importar módulos para que se registren los decoradores
from . import data, eda, code_exec, hypothesis, communication  # noqa: F401,E402
