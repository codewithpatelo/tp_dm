"""Pipeline resumible para experimentos largos (>1 h).

Resuelve el problema "se me cayó Cursor a la hora 6 del HP sweep y
perdí todo": persiste el progreso en un JSONL append-only que se
puede leer al reiniciar el script para saltar las combinaciones ya
hechas.

Diseño:

- **Storage**: JSONL (un objeto por línea). Append-only, con flush
  explícito después de cada escritura. Robusto a crashes a mitad de
  línea (al releer se descarta la línea malformada).
- **Idempotencia**: cada combinación tiene una `key` única (string
  determinístico). Si la key ya está en el JSONL, se saltea.
- **Sin estado en memoria que se pierda**: cada fit completo se
  persiste antes de pasar al siguiente.
- **Recuperación**: al instanciar el runner, lee todo el JSONL
  existente y arma el set de keys hechas.

Cobertura de fallos:

- Cae Cursor / kill del proceso Python entre fits → próxima corrida
  resume desde la combinación pendiente siguiente.
- Cae internet → irrelevante, todo es local.
- Kill -9 a mitad de un fit individual → ese fit se pierde y se
  reintenta (su línea no llegó a escribirse). El JSONL no se corrompe.
- Disco lleno → tira excepción visible.

Uso típico (HP sweep de Entrega 3):

    from entregas._resumable import ResumableRunner
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import root_mean_squared_error

    grid = [
        {"n_est": n, "max_d": d, "fold": f, "seed": s}
        for n in [200, 500, 1000, 2000]
        for d in [20, 30, 50]
        for f in range(5)
        for s in [0, 1, 2]
    ]

    def run_one(combo):
        rf = RandomForestRegressor(
            n_estimators=combo["n_est"],
            max_depth=combo["max_d"],
            random_state=combo["seed"],
            n_jobs=-1,
        )
        # ... fittear sobre fold combo['fold'], scorear ...
        return {"rmse": float(rmse_local)}

    runner = ResumableRunner(
        progress_path="entregas/entrega_3/hp_sweep_v0.progress.jsonl",
        grid=grid,
        run_fn=run_one,
        key_fn=lambda c: f"n{c['n_est']}_d{c['max_d']}_f{c['fold']}_s{c['seed']}",
    )
    runner.run()              # corre los pendientes con ETA cada N fits
    df = runner.results_df()  # arma el DataFrame agregado
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class ResumableRunner:
    """Ejecuta una grilla de experimentos persistiendo progreso a JSONL.

    Parameters
    ----------
    progress_path : str | Path
        Ruta del JSONL append-only donde se persiste cada resultado.
        Si no existe, se crea (junto con su directorio padre).
    grid : list[dict]
        Lista de combinaciones a evaluar. Cada elemento es el "combo"
        que recibe `run_fn`.
    run_fn : Callable[[dict], dict]
        Función que recibe un combo y devuelve un dict con los
        resultados (números, listas, lo que sea JSON-serializable).
        DEBE ser determinística por combo (mismo combo → mismo
        resultado) para que el resume sea correcto.
    key_fn : Callable[[dict], str]
        Función que mapea cada combo a una key única determinística.
        Si dos combos distintos generan la misma key, el segundo se
        saltea como si ya estuviera hecho.
    log_every : int
        Cada cuántos fits imprimir progreso + ETA. Default 1 (cada
        fit). Subilo si los fits son muy rápidos.
    """

    def __init__(
        self,
        progress_path: str | Path,
        grid: list[dict],
        run_fn: Callable[[dict], dict],
        key_fn: Callable[[dict], str],
        log_every: int = 1,
    ) -> None:
        self.progress_path = Path(progress_path)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.run_fn = run_fn
        self.key_fn = key_fn
        self.log_every = log_every
        self._done: dict[str, dict] = self._load_done()

    def _load_done(self) -> dict[str, dict]:
        """Lee el JSONL existente y arma {key: record}."""
        done: dict[str, dict] = {}
        if not self.progress_path.exists():
            return done
        with self.progress_path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["key"]] = rec
                except json.JSONDecodeError:
                    print(f"[resumable] línea {ln} corrupta, ignorada")
        return done

    def pending(self) -> list[dict]:
        """Combos del grid que todavía NO están persistidos."""
        return [c for c in self.grid if self.key_fn(c) not in self._done]

    def _append(self, rec: dict) -> None:
        """Append atómico al JSONL con flush garantizado."""
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    def run(self) -> None:
        """Corre los combos pendientes secuencialmente, persistiendo
        cada resultado antes de avanzar al siguiente.
        """
        pending = self.pending()
        n_total = len(self.grid)
        n_done0 = len(self._done)
        n_pending = len(pending)
        print(
            f"[resumable] total={n_total} hechos={n_done0} pendientes={n_pending}"
            f" → archivo: {self.progress_path}"
        )
        if n_pending == 0:
            print("[resumable] todo hecho, no hay nada que correr")
            return

        t_start = time.time()
        for i, combo in enumerate(pending, 1):
            key = self.key_fn(combo)
            t0 = time.time()
            try:
                result = self.run_fn(combo)
            except KeyboardInterrupt:
                print(f"[resumable] interrumpido por user en {key}")
                raise
            except Exception as e:
                # Persistimos el fallo igual, así no se reintenta hasta
                # que el user borre la línea del JSONL.
                print(f"[resumable] FALLO en {key}: {e!r}")
                rec = {
                    "key": key,
                    "combo": combo,
                    "result": None,
                    "error": repr(e),
                    "fit_time_s": round(time.time() - t0, 1),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._append(rec)
                self._done[key] = rec
                continue

            fit_time = time.time() - t0
            rec = {
                "key": key,
                "combo": combo,
                "result": result,
                "fit_time_s": round(fit_time, 1),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._append(rec)
            self._done[key] = rec

            if i % self.log_every == 0 or i == n_pending:
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = avg * (n_pending - i)
                print(
                    f"[resumable] {n_done0 + i}/{n_total} ({key})"
                    f" · {fit_time:.1f}s · avg {avg:.1f}s · ETA {eta/60:.1f}min"
                )

        elapsed = time.time() - t_start
        print(f"[resumable] terminado en {elapsed/60:.1f}min")

    def results_df(self):
        """DataFrame agregado de TODOS los resultados persistidos
        (incluye los de corridas anteriores).
        """
        import pandas as pd

        rows = []
        for rec in self._done.values():
            row = {"key": rec["key"]}
            combo = rec.get("combo", {})
            if isinstance(combo, dict):
                row.update(combo)
            result = rec.get("result")
            if isinstance(result, dict):
                row.update(result)
            row["fit_time_s"] = rec.get("fit_time_s")
            row["ts"] = rec.get("ts")
            if rec.get("error"):
                row["error"] = rec["error"]
            rows.append(row)
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Self-test rápido (no se corre con import).
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("[resumable] self-test")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        path = tmp.name

    grid = [{"x": i} for i in range(5)]

    def run_fn(combo):
        time.sleep(0.1)
        if combo["x"] == 2:
            raise RuntimeError("falla simulada")
        return {"y": combo["x"] ** 2}

    print("\n--- corrida 1 (todos pendientes, 1 falla) ---")
    runner = ResumableRunner(
        progress_path=path, grid=grid, run_fn=run_fn,
        key_fn=lambda c: f"x{c['x']}",
    )
    runner.run()

    print("\n--- corrida 2 (todos hechos, no debería correr nada) ---")
    runner2 = ResumableRunner(
        progress_path=path, grid=grid, run_fn=run_fn,
        key_fn=lambda c: f"x{c['x']}",
    )
    runner2.run()

    print("\n--- DataFrame agregado ---")
    print(runner2.results_df())

    Path(path).unlink()
    print("\n[resumable] self-test OK")
