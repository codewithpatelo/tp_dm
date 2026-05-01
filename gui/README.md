# Talk with your data — TP DM

GUI Streamlit con un **agente cognitivo** (loop SENSE → PERCEPT → PLAN → ACT → OBSERVE → VERDICT → REPORT) que conoce el dataset, los experimentos y la metodología del proyecto.

## Stack

| Componente | Tech |
|---|---|
| UI | Streamlit (sidebar chat + tabs) |
| LLM | **OpenAI** (default `gpt-5-mini`) o **Anthropic** Messages (default Sonnet) — `GUI_LLM_PROVIDER` / `GUI_LLM_MODEL` — sin Tool Use, sin LangGraph |
| Datos | DuckDB sobre pandas (train_filtered + test + train_raw lazy) |
| Memoria | WorkingMemory (RAM) + Episodic (SQLite) + Semantic (JSONL grep, vectorless) |
| Sandbox | AST whitelist + timeout 10s + matplotlib captura |
| Plotting | Plotly + pydeck |

Sin embeddings, sin vector DB, sin servicios externos.

## Setup

```bash
pip install -r gui/requirements.txt
# En la raíz del repo, en .env (al menos una key):
#   OPENAI_API_KEY=sk-...     # por defecto se usa OpenAI si está definida
#   ANTHROPIC_API_KEY=sk-ant-...   # alternativa: GUI_LLM_PROVIDER=anthropic
streamlit run gui/app.py
```

- **Default:** con `OPENAI_API_KEY` el agente usa **OpenAI** con **`gpt-5-mini`** (línea frontier, buen equilibrio coste/calidad). Podés fijar snapshot, p. ej. `GUI_LLM_MODEL=gpt-5-mini-2025-08-07`.
- `GUI_LLM_PROVIDER=openai` o `anthropic` fuerza el proveedor.
- `GUI_LLM_MODEL` sobrescribe el modelo (ej. `gpt-5.2`, `o4-mini`, o el Sonnet que quieras en Anthropic).

## Capacidades del agente

| Categoría | Skills |
|---|---|
| Datos / RAG | `query_sql`, `read_project_file`, `list_files`, `vectorless_rag`, `get_champion`, `list_experiments`, `read_context_section`, `plan_todos` |
| EDA | `describe_data`, `compute_correlations`, `detect_outliers`, `compare_distributions`, `temporal_drift_check`, `feature_importance` |
| Código | `execute_python` (sandbox), `plot_chart`, `propose_file_change` (Apply/Reject UI) |
| Hipótesis | `propose_hypothesis`, `propose_experiment`, `recall_lessons`, `update_semantic_memory`, `suggest_followup_questions` |
| Comunicación | `clarify`, `report` |

## Loop cognitivo

4 llamadas al LLM por turno (intent, plan, verdict, report). Hasta 2 replans.

```
user.message
    ↓
intent (LLM)  → necesita_clarify? → answer
    ↓
plan (LLM)
    ↓
act_observe → verdict (LLM) ─┐
    ↑                        │
    └─── replan (≤2x) ───────┘
                             ↓
                          report (LLM)
```

## Layout de archivos

```
gui/
├── app.py                  ← entrypoint
├── cognitive.py            ← loop SAPTPAO
├── llm.py                  ← OpenAI + Anthropic + JSON parser defensivo
├── system_prompt.py        ← persona DS + 4 prompts
├── data_loader.py          ← DuckDB + filtros E1
├── memory.py               ← WorkingMemory + EpisodicMemory (SQLite)
├── memory_semantic.py      ← SemanticMemory (JSONL grep)
├── events.py               ← event bus pub-sub
├── sandbox.py              ← execute_python restringido
├── parsers.py              ← leaderboard.md / solucion-*.json / CONTEXT.md / planes
├── skills/
│   ├── data.py             ← SQL, RAG, lectura, meta
│   ├── eda.py              ← describe, corr, outliers, drift, importance
│   ├── code_exec.py        ← execute_python, plot_chart, propose_file_change
│   ├── hypothesis.py       ← hipótesis, experimentos, lessons, semantic memory
│   └── communication.py    ← clarify, report
├── panels/
│   ├── chat.py             ← sidebar chat + trace + tokens
│   ├── status.py           ← champion + runs + TODOs + backlog
│   ├── explorer.py         ← filtros + plots + map
│   ├── experiments.py      ← leaderboard + drill-down JSON
│   ├── eda.py              ← markdowns + interactivo
│   └── proposals.py        ← diff viewer + Apply/Reject
└── _memory/
    ├── episodic.db         ← SQLite (gitignored)
    └── semantic.jsonl      ← lessons / hypothesis / experiments (gitignored)
```

## Ejemplos de prompts

- "¿Cuántas propiedades hay por barrio en el último trimestre del train?"
- "Mostrame la distribución de price y compará train vs test"
- "Detectame outliers en surface_total_in_m2 con IQR"
- "¿Qué features tienen mayor correlación con price?"
- "Proponé una hipótesis sobre por qué Palermo Soho vale más que Palermo Hollywood"
- "Hay drift temporal en pub_yyyymm?"
- "Proponé un cambio en CONTEXT.md para sumar la lección X"

## Disclaimer

Construido con Cursor IDE (Claude Opus 4.7 + Claude Code) como asistente.
