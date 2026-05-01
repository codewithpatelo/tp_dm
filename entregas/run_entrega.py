"""
Orquestador de entregas (única fuente de verdad: la notebook).

Flujo end-to-end:

    1. Ejecuta la notebook con `nbconvert` pasando NOMBRE/DESCRIPCION via env.
    2. Lee el JSON con metadatos + métricas que la notebook deja en
       entregas/<entrega>/.
    3. Compara `rmse_holdout` con el mejor histórico del leaderboard.md.
       - Si MEJORA → submit automático a Kaggle (a menos que se pase --no-submit).
    4. Polea Kaggle hasta tener `publicScore`. Lo registra en el JSON y en el
       leaderboard.md.
    5. Si el `rmse_kaggle` es además el nuevo mínimo → regenera el informe
       con OpenAI (usando el contexto de .claude/CONTEXT.md).

Cada paso falla "blando": si Kaggle / OpenAI fallan, los outputs ya generados
(CSV + JSON + leaderboard) quedan intactos y el resto se puede completar a
mano con `--record-kaggle`.

Uso típico:
    python entregas/run_entrega.py --entrega entrega_2 --nombre v2 \
        --desc "Sin IsolationForest, KNNImputer en m2, n_estimators=800"

    python entregas/run_entrega.py --record-kaggle --entrega entrega_2 \
        --nombre v1 --kaggle-rmse 93167.324
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "Colab_Base_para_el_Trabajo_Práctico_(Entrega_3).ipynb"

sys.path.insert(0, str(ROOT / "entregas"))
import _lib  # noqa: E402


# ---------- ejecución de la notebook ----------------------------------------

def run_notebook_only(entrega: str, nombre: str, desc: str) -> int:
    import nbformat
    from nbclient.exceptions import CellExecutionError
    from nbconvert.preprocessors import ExecutePreprocessor

    out_dir = ROOT / "entregas" / entrega
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["EXPERIMENT_ENTREGA"] = entrega
    os.environ["EXPERIMENT_NAME"] = nombre
    os.environ["EXPERIMENT_DESC"] = desc

    print(f"[run_entrega] notebook : {NOTEBOOK.name}")
    print(f"[run_entrega] entrega  : {entrega}")
    print(f"[run_entrega] nombre   : {nombre}")
    print(f"[run_entrega] desc     : {desc}")
    print("[run_entrega] ejecutando notebook (con CV5 puede tardar ~20-30 min)...", flush=True)

    nb = nbformat.read(NOTEBOOK, as_version=4)
    ep = ExecutePreprocessor(timeout=7200, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    except CellExecutionError as e:
        executed_path = out_dir / f"{entrega}_{nombre}.failed.ipynb"
        nbformat.write(nb, executed_path)
        print("[run_entrega] FALLO una celda:", file=sys.stderr)
        print(e, file=sys.stderr)
        print(f"[run_entrega] notebook con error guardada en {executed_path}",
              file=sys.stderr)
        return 1

    executed_path = out_dir / f"{entrega}_{nombre}.executed.ipynb"
    nbformat.write(nb, executed_path)
    print(f"[run_entrega] notebook ejecutada -> {executed_path}")
    return 0


# ---------- post-procesado: submit + informe --------------------------------

def append_kaggle_to_leaderboard(entrega: str, nombre: str, kaggle_rmse: float) -> None:
    lb = ROOT / "entregas" / entrega / "leaderboard.md"
    if not lb.exists():
        return
    lines = lb.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = []
    for ln in lines:
        if ln.startswith(f"| {nombre} ") and "_pendiente_" in ln:
            ln = ln.replace("_pendiente_", f"**{kaggle_rmse:.3f}**")
        new_lines.append(ln)
    lb.write_text("".join(new_lines), encoding="utf-8")


def update_kaggle_in_json(json_path: Path, kaggle_rmse: float) -> None:
    if not json_path.exists():
        return
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    meta.setdefault("kaggle", {})
    meta["kaggle"]["rmse"] = float(kaggle_rmse)
    meta["kaggle"]["recorded_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def post_process(entrega: str, nombre: str, *, allow_submit: bool) -> int:
    """Comparación con leaderboard, auto-submit y regeneración del informe.

    Criterio de auto-submit: exige mejora SIMULTÁNEA (cualquier delta positivo) en
      - `rmse_cv5_mean`         (multi-seed, métrica primaria)
      - `rmse_holdout_temporal` (último 20 % por publication_date)
    sobre los mejores históricos del leaderboard. Si falta cualquiera de las dos
    métricas o no hay previo con qué comparar, NO submit. El holdout único queda
    informativo pero no decide. Kaggle es el árbitro final de magnitud.
    """
    MARGEN_CV5 = 0.0
    MARGEN_HOLDOUT_TEMPORAL = 0.0

    out_dir = ROOT / "entregas" / entrega
    json_path = out_dir / f"solucion-{entrega.replace('_','')}-{nombre}.json"
    csv_path  = out_dir / f"solucion-{entrega.replace('_','')}-{nombre}.csv"
    lb_path   = out_dir / "leaderboard.md"
    informe_path = out_dir / f"Entrega_{entrega.split('_')[-1]}_informe.md"

    if not json_path.exists():
        print(f"[run_entrega] no existe {json_path}; abortando post-procesado.")
        return 1

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    metricas = meta.get("metricas", {})
    rmse_holdout  = metricas.get("rmse_holdout")
    rmse_cv5      = metricas.get("rmse_cv5_mean")
    rmse_cv5_std  = metricas.get("rmse_cv5_std")
    rmse_temp     = metricas.get("rmse_holdout_temporal")
    desc = meta.get("descripcion", "")

    cv5_str = f"{rmse_cv5:.2f} ± {rmse_cv5_std:.0f}" if rmse_cv5 is not None else "—"
    temp_str = f"{rmse_temp:.2f}" if rmse_temp is not None else "—"
    hold_str = f"{rmse_holdout:.2f}" if rmse_holdout is not None else "—"
    print(f"[run_entrega] OK — CV5 = {cv5_str} | holdout_temporal = {temp_str} | "
          f"holdout = {hold_str}")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")

    rows = _lib.parse_leaderboard(lb_path)
    best_cv5_prev  = _lib.best_cv5(rows, exclude=nombre)
    best_temp_prev = _lib.best_holdout_temporal(rows, exclude=nombre)
    best_h_prev    = _lib.best_holdout(rows, exclude=nombre)
    print(f"[leaderboard] mejor CV5 previo            : {best_cv5_prev}")
    print(f"[leaderboard] mejor holdout temporal prev : {best_temp_prev}")
    print(f"[leaderboard] mejor holdout (legacy) prev : {best_h_prev}")

    # Doble criterio (v6+): mejora > MARGEN_CV5 en CV5 multi-seed Y mejora
    # > MARGEN_HOLDOUT_TEMPORAL en el holdout temporal.
    delta_cv5  = (best_cv5_prev  - rmse_cv5)  if (rmse_cv5  is not None and best_cv5_prev  is not None) else None
    delta_temp = (best_temp_prev - rmse_temp) if (rmse_temp is not None and best_temp_prev is not None) else None
    ok_cv5  = (delta_cv5  is not None and delta_cv5  > MARGEN_CV5)
    ok_temp = (delta_temp is not None and delta_temp > MARGEN_HOLDOUT_TEMPORAL)

    if rmse_cv5 is None or rmse_temp is None:
        is_local_better = False
        falt = []
        if rmse_cv5 is None:  falt.append("CV5")
        if rmse_temp is None: falt.append("holdout_temporal")
        criterio = (f"faltan métricas locales obligatorias ({', '.join(falt)}); "
                    f"submit manual requerido")
    elif best_cv5_prev is None or best_temp_prev is None:
        is_local_better = False
        criterio = ("primera corrida con doble criterio (CV5 + holdout_temporal); "
                    "submit manual obligatorio")
    else:
        is_local_better = ok_cv5 and ok_temp
        criterio = (
            f"CV5 ok={ok_cv5} (D={delta_cv5:+.0f}) | "
            f"holdout_temp ok={ok_temp} (D={delta_temp:+.0f})"
        )
    print(f"[run_entrega] criterio: {criterio}")

    if not is_local_better:
        print("[run_entrega] no cumple el doble criterio (CV5 + holdout_temporal); "
              "NO submit automático.")
        return 0

    if not allow_submit:
        print("[run_entrega] cumple el doble criterio, pero --no-submit activo: skip submit.")
        return 0

    print("[run_entrega] cumple el doble criterio -> submit automatico a Kaggle.")
    submit_msg = meta.get("output", {}).get("kaggle_submit_message") or f"{entrega}/{nombre}: {desc}"
    rmse_kaggle = _lib.submit_to_kaggle(csv_path, submit_msg)

    if rmse_kaggle is None:
        print("[run_entrega] submit fallido o sin score; el CSV esta listo "
              "para subir a mano. Anotalo despues con --record-kaggle.")
        return 0

    print(f"[run_entrega] Kaggle publicScore = {rmse_kaggle:.3f}")
    update_kaggle_in_json(json_path, rmse_kaggle)
    append_kaggle_to_leaderboard(entrega, nombre, rmse_kaggle)

    rows_after = _lib.parse_leaderboard(lb_path)
    best_k_prev = _lib.best_kaggle(rows_after, exclude=nombre)
    is_kaggle_better = best_k_prev is None or rmse_kaggle < best_k_prev
    print(f"[leaderboard] mejor Kaggle previo: {best_k_prev} | nuevo: {rmse_kaggle}")

    if not is_kaggle_better:
        print("[run_entrega] el Kaggle no mejora al mejor previo; NO regenero informe.")
        return 0

    print("[run_entrega] nuevo CAMPEON (mejora local + Kaggle) -> regenero informe.")
    ok = _lib.regenerate_informe(entrega, nombre, json_path, informe_path)
    return 0 if ok else 0


# ---------- modo registro manual --------------------------------------------

def record_kaggle_manual(entrega: str, nombre: str, kaggle_rmse: float) -> int:
    out_dir = ROOT / "entregas" / entrega
    json_path = out_dir / f"solucion-{entrega.replace('_','')}-{nombre}.json"
    if not json_path.exists():
        print(f"[record_kaggle] no existe {json_path}", file=sys.stderr)
        return 1
    update_kaggle_in_json(json_path, kaggle_rmse)
    append_kaggle_to_leaderboard(entrega, nombre, kaggle_rmse)
    print(f"[record_kaggle] {entrega}/{nombre}: Kaggle RMSE = {kaggle_rmse}")

    rows = _lib.parse_leaderboard(out_dir / "leaderboard.md")
    best_k_prev = _lib.best_kaggle(rows, exclude=nombre)
    if best_k_prev is None or kaggle_rmse < best_k_prev:
        print("[record_kaggle] nuevo CAMPEON Kaggle -> regenero informe.")
        informe_path = out_dir / f"Entrega_{entrega.split('_')[-1]}_informe.md"
        _lib.regenerate_informe(entrega, nombre, json_path, informe_path)
    return 0


# ---------- entrypoint ------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--entrega", default="entrega_2")
    p.add_argument("--nombre", default="v1")
    p.add_argument("--desc", default="",
                   help="Descripción libre (también se usa como mensaje de Kaggle).")
    p.add_argument("--no-submit", action="store_true",
                   help="No subir a Kaggle aunque el holdout mejore.")
    p.add_argument("--record-kaggle", action="store_true",
                   help="Modo registro: anota --kaggle-rmse en JSON y leaderboard.")
    p.add_argument("--kaggle-rmse", type=float)
    p.add_argument("--regen-informe", action="store_true",
                   help="Modo regeneración: re-escribe el informe (y su PDF) de "
                        "la corrida --nombre sin volver a ejecutar el notebook. "
                        "Útil cuando cambian las reglas de formato del informe.")
    args = p.parse_args()

    if args.record_kaggle:
        if args.kaggle_rmse is None:
            p.error("--record-kaggle requiere --kaggle-rmse")
        return record_kaggle_manual(args.entrega, args.nombre, args.kaggle_rmse)

    if args.regen_informe:
        out_dir = ROOT / "entregas" / args.entrega
        json_path = out_dir / f"solucion-{args.entrega.replace('_','')}-{args.nombre}.json"
        informe_path = out_dir / f"Entrega_{args.entrega.split('_')[-1]}_informe.md"
        if not json_path.exists():
            print(f"[regen] no existe {json_path}", file=sys.stderr)
            return 1
        ok = _lib.regenerate_informe(args.entrega, args.nombre, json_path, informe_path)
        return 0 if ok else 1

    if not args.desc:
        p.error("--desc es obligatorio.")

    rc = run_notebook_only(args.entrega, args.nombre, args.desc)
    if rc != 0:
        return rc

    return post_process(args.entrega, args.nombre, allow_submit=not args.no_submit)


if __name__ == "__main__":
    raise SystemExit(main())
