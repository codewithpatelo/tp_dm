"""
Helpers compartidos para el orquestador de entregas.

Encapsula:
- Parseo del leaderboard.md (mejor RMSE histórico, holdout y Kaggle).
- Auto-submit a Kaggle (con polling hasta obtener `publicScore`).
- Regeneración del informe vía OpenAI cuando hay nuevo campeón.

Diseño defensivo: si falla la red, las credenciales o el LLM, devolvemos
False/None y NO rompemos la corrida del notebook. El CSV y el JSON ya están
generados; el usuario puede completar el resto a mano.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
COMPETITION = "fcen-dm-2026-prediccion-precio-de-propiedades"


# ---------- leaderboard ------------------------------------------------------

@dataclass
class LBRow:
    nombre: str
    rmse_cv5_mean: Optional[float]
    rmse_holdout_temporal: Optional[float]
    rmse_holdout: Optional[float]
    rmse_kaggle: Optional[float]
    raw: str


def _parse_num(s: str) -> Optional[float]:
    """Convierte la celda '116,922.89' / '**93167.324**' / '—' / '_pendiente_' a float o None."""
    s = (s or "").replace("**", "").replace("_", "").strip()
    if not s or s in {"—", "-", "pendiente", "None"}:
        return None
    # Si viene como "12,345.67 ± 234" nos quedamos con la parte de la izquierda.
    if "±" in s:
        s = s.split("±")[0].strip()
    try:
        return float(s.replace(" ", "").replace(",", ""))
    except ValueError:
        return None


def parse_leaderboard(lb_path: Path) -> list[LBRow]:
    """Parsea entregas/<entrega>/leaderboard.md y devuelve filas estructuradas.

    Soporta dos formatos:
      legacy (v1-v3): | nombre | fecha | RMSE holdout | RMSE Kaggle | hotdeck % | descripción | csv md5 |
      v4+:           | nombre | fecha | RMSE CV5     | RMSE holdout | RMSE Kaggle | hotdeck % | descripción | csv md5 |
    Detecta el formato leyendo el header y mapea las columnas en consecuencia.
    """
    if not lb_path.exists():
        return []
    text = lb_path.read_text(encoding="utf-8").splitlines()

    # Buscamos el header para saber qué columnas hay y en qué orden.
    header_cells: list[str] = []
    for ln in text:
        if ln.startswith("| nombre"):
            header_cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            break
    cv5_idx  = next((i for i, c in enumerate(header_cells) if "CV5" in c), None)
    temp_idx = next((i for i, c in enumerate(header_cells)
                     if "holdout temporal" in c.lower()
                     or "holdout_temporal" in c.lower()), None)
    # holdout "puro" = el que NO es el holdout temporal (ojo: el match de
    # "holdout" solo agarraría también el temporal porque incluye la palabra).
    hold_idx = next((i for i, c in enumerate(header_cells)
                     if "holdout" in c.lower() and i != temp_idx), None)
    kag_idx  = next((i for i, c in enumerate(header_cells) if "Kaggle" in c), None)

    rows: list[LBRow] = []
    for ln in text:
        if not ln.startswith("|") or ln.startswith("|---") or ln.startswith("| nombre"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        rows.append(
            LBRow(
                nombre=cells[0],
                rmse_cv5_mean=_parse_num(cells[cv5_idx]) if cv5_idx is not None else None,
                rmse_holdout_temporal=(_parse_num(cells[temp_idx])
                                       if temp_idx is not None else None),
                rmse_holdout=_parse_num(cells[hold_idx]) if hold_idx is not None else None,
                rmse_kaggle=_parse_num(cells[kag_idx]) if kag_idx is not None else None,
                raw=ln,
            )
        )
    return rows


def best_holdout(rows: list[LBRow], exclude: str | None = None) -> Optional[float]:
    vals = [r.rmse_holdout for r in rows
            if r.rmse_holdout is not None and r.nombre != exclude]
    return min(vals) if vals else None


def best_kaggle(rows: list[LBRow], exclude: str | None = None) -> Optional[float]:
    vals = [r.rmse_kaggle for r in rows
            if r.rmse_kaggle is not None and r.nombre != exclude]
    return min(vals) if vals else None


def best_cv5(rows: list[LBRow], exclude: str | None = None) -> Optional[float]:
    vals = [r.rmse_cv5_mean for r in rows
            if r.rmse_cv5_mean is not None and r.nombre != exclude]
    return min(vals) if vals else None


def best_holdout_temporal(rows: list[LBRow], exclude: str | None = None) -> Optional[float]:
    vals = [r.rmse_holdout_temporal for r in rows
            if r.rmse_holdout_temporal is not None and r.nombre != exclude]
    return min(vals) if vals else None


# ---------- Kaggle -----------------------------------------------------------

def _kaggle_api():
    """Devuelve un KaggleApi autenticado o lanza excepción."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def submit_to_kaggle(
    csv_path: Path,
    message: str,
    poll_timeout_s: int = 180,
    poll_interval_s: int = 6,
) -> Optional[float]:
    """Sube el CSV a Kaggle y devuelve el publicScore (RMSE) cuando esté listo.

    Devuelve None si:
        - falla la autenticación / la submission;
        - se alcanza el timeout sin que Kaggle devuelva score.
    """
    try:
        api = _kaggle_api()
    except Exception as e:
        print(f"[kaggle] auth FAIL: {e}")
        return None

    try:
        print(f"[kaggle] submit: {csv_path.name} → {COMPETITION}")
        print(f"[kaggle] desc  : {message}")
        api.competition_submit(str(csv_path), message, COMPETITION)
    except Exception as e:
        print(f"[kaggle] submit FAIL: {e}")
        return None

    print(f"[kaggle] esperando score (timeout {poll_timeout_s}s)...")
    deadline = time.time() + poll_timeout_s
    target_filename = csv_path.name
    while time.time() < deadline:
        try:
            subs = api.competition_submissions(COMPETITION)
        except Exception as e:
            print(f"[kaggle] poll error: {e}; reintento en {poll_interval_s}s")
            time.sleep(poll_interval_s)
            continue

        for s in subs:
            sub_filename = getattr(s, "file_name", None) or getattr(s, "fileName", None)
            if sub_filename == target_filename and s.description == message:
                score = getattr(s, "public_score", None) or getattr(s, "publicScore", None)
                status = getattr(s, "status", "?")
                if score not in (None, "", "None"):
                    try:
                        return float(score)
                    except ValueError:
                        return None
                if str(status).lower() in {"error", "failed", "complete with errors"}:
                    print(f"[kaggle] submission status={status}")
                    return None
                break
        time.sleep(poll_interval_s)

    print("[kaggle] timeout esperando publicScore")
    return None


# ---------- Informe vía LLM --------------------------------------------------

DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")


def regenerate_informe(
    entrega: str,
    nombre: str,
    json_path: Path,
    informe_path: Path,
    model: str = DEFAULT_OPENAI_MODEL,
) -> bool:
    """Reescribe el informe usando OpenAI. Devuelve True si lo regeneró."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        from openai import OpenAI
    except Exception as e:
        print(f"[informe] no pude importar openai/dotenv: {e}")
        return False

    if not os.environ.get("OPENAI_API_KEY"):
        print("[informe] OPENAI_API_KEY no seteada; salteo regeneración.")
        return False

    contexto_path = ROOT / ".claude" / "CONTEXT.md"
    leaderboard_path = ROOT / "entregas" / entrega / "leaderboard.md"

    contexto = contexto_path.read_text(encoding="utf-8") if contexto_path.exists() else ""
    leaderboard = leaderboard_path.read_text(encoding="utf-8") if leaderboard_path.exists() else ""
    metadatos = json_path.read_text(encoding="utf-8") if json_path.exists() else "{}"
    informe_actual = informe_path.read_text(encoding="utf-8") if informe_path.exists() else ""

    system_prompt = (
        "Sos un asistente que redacta informes técnicos de TPs de la materia "
        "Data Mining (UBA Exactas). Escribís para un supervisor académico. "
        "El informe tiene que entrar en UNA carilla A4. **≤ 500 palabras es "
        "OBLIGATORIO, no aspiracional.** Tu respuesta va a ser auto-validada "
        "por el orquestador y, si supera ese límite, se va a pedir un "
        "re-recorte explícito (perdés un round-trip y bajás la calidad). "
        "Preferí oraciones más cortas a sacar sub-ítems o números. "
        "Las palabras de la/s tabla/s NO cuentan para ese límite. "
        "Explicá DECISIONES (no código), y JUSTIFICÁ todo en EDA, hipótesis, "
        "experimentos previos y teoría vista en clase. Nada arbitrario, "
        "nada trivial. "
        "Antes de redactar leé el CONTEXTO GENERAL que viene en el user prompt. "
        "Las reglas de formato, tono y narrativa viven ahí, específicamente en "
        "las secciones 'Estructura del informe (OBLIGATORIA — debe coincidir "
        "con la consigna)', 'Tono', 'Narrativa y storytelling (regla crítica)', "
        "y 'Sección operativa: parámetros específicos de la entrega en curso'. "
        "Esta última sub-sección define la denotación concreta (letras/números), "
        "los encabezados exactos, qué sub-preguntas responder, y qué entrega "
        "previa usar como comparación en la entrega que te toca escribir. "
        "Tus reglas no negociables, independientes de qué entrega sea: "
        "(a) la denotación ordinal del informe replica la de la consigna — "
        "no inventar una propia; "
        "(b) cada sub-pregunta explícita de la consigna se responde de forma "
        "directa, con variable/causa nombrada y un número del JSON o del "
        "leaderboard; "
        "(c) el punto que pide comparación con la entrega anterior se resuelve "
        "con una tabla y una narración, no con listas sueltas; "
        "(d) tono en primera persona del SINGULAR, evitando tanto la pasiva "
        "impersonal como el plural (el TP es individual); "
        "(e) narrativa: cada sub-ítem es un párrafo con hallazgo → pregunta → "
        "hipótesis → experimento/resultado → aprendizaje → conexión con lo "
        "siguiente, no un catálogo descriptivo de acciones. Las corridas "
        "fallidas son material narrativo central, no nota al pie."
    )

    user_prompt = f"""Tenés que reescribir el informe de la **{entrega.replace('_', ' ')}**
porque la corrida `{nombre}` se convirtió en el nuevo campeón.

## Antes de empezar

Leé el CONTEXTO GENERAL abajo. En particular, estas secciones son obligatorias:

- **"Estructura del informe (OBLIGATORIA — debe coincidir con la consigna)"** —
  reglas genéricas de formato (aplicables a cualquier entrega).
- **"Tono"** — primera persona del singular. NO usar pasiva impersonal
  ("se implementó") ni primera persona del plural ("decidimos"). El TP es
  individual.
- **"Narrativa y storytelling (regla crítica)"** — el informe tiene que ser
  una narrativa (hallazgo → pregunta → hipótesis → experimento/resultado →
  aprendizaje → conexión con el siguiente paso), NO un catálogo descriptivo.
  Las corridas anteriores que fracasaron (ver `leaderboard.md`) son capítulos
  de la historia, no nota al pie.
- **"Sección operativa: parámetros específicos de la entrega en curso"** —
  particulariza para ESTA entrega: qué denotación usa la consigna, qué
  encabezados exactos debés poner en el informe, qué sub-preguntas requieren
  respuesta directa (con variable/causa nombrada y un número que respalde),
  y contra qué entrega previa compararte. Usá esa sub-sección para bajar las
  reglas genéricas al caso concreto.

## Resumen de invariantes (válido para cualquier entrega)

1. Los encabezados del informe copian los de la consigna. No inventar
   "1) Contexto", "2) Filtros", etc. Si hay contenido que no encaja en
   ningún punto de la consigna (introducción, filtros heredados, etc.), va
   en un bloque breve antes del primer punto numerado, o dentro del punto
   más natural.
2. Cada sub-pregunta explícita de la consigna se responde de forma directa,
   con variable/causa nombrada y un número del JSON de la corrida campeona
   o del `leaderboard.md` que respalde la respuesta.
3. El punto de la consigna que pide comparar con la entrega anterior
   incluye una tabla **Entrega (n-1) vs Entrega actual** usando los scores
   Kaggle reales (ver *"Estado de las entregas"* en el CONTEXTO). Las
   comparaciones entre sub-versiones dentro de la misma entrega son
   secundarias.
4. Tono en primera persona del **SINGULAR**: "decidí", "implementé",
   "mi estrategia", "esperaba", "aprendí", "observé".
5. Toda decisión técnica anclada a un número. Nada de afirmaciones genéricas
   sin dato ("es más robusta", "mejora el score").
6. Longitud: ≤ 1 carilla (~ 500 palabras sin contar la/s tabla/s).
7. Formato Markdown. Primer carácter de la respuesta: `#`.

## Reglas duras de redacción (anti-listado) — VAN A FALLAR LA REVISIÓN SI NO LAS CUMPLÍS

Estas son consecuencia directa de la sección "Narrativa y storytelling
(regla crítica)" del CONTEXTO. Las pongo acá explícitas porque la versión
anterior del informe falló justo en estas reglas:

R1. **Prohibido abrir un sub-ítem con un verbo de catálogo en primera
    persona** ("Analicé", "Implementé", "Cuantifiqué", "Definí", "Apliqué",
    "Calculé", "Detecté", "Generé"). Eso convierte el informe en lista de
    tareas. Cada sub-ítem debe arrancar contando algo del *contexto* o
    *enganchando con lo anterior*: "*Al mirar la distribución de m2…*",
    "*Esos NaN que dejó el tratamiento de outliers me obligaron a…*",
    "*Con la imputación numérica resuelta me quedaba el problema de…*",
    "*El fracaso de v2 me enseñó que…*".

R2. **Conexión obligatoria entre sub-ítems consecutivos**. A.2 tiene que
    arrancar enganchando con A.1; B.1 con A; B.2 con B.1; B.3 con B.2; C
    con todo lo anterior. Si yo, lector, puedo permutar el orden de tus
    sub-ítems sin que se note, fallaste. Conectores aceptables (no copiar
    literal, son ejemplos): *"Eso me llevó a…"*, *"Con ese problema
    resuelto me topé con…"*, *"Esa pista me empujó a probar…"*, *"El
    fracaso de v⟨k⟩ me enseñó que…"*.

R3. **Prohibido usar bullets dentro de un sub-ítem** salvo que sea una
    enumeración estrictamente finita y cerrada (ej: "las 4 columnas
    numéricas que imputé con mediana fueron `m2`, `lat`, `lon`, `n_banos`").
    NO usar bullets para "estrategia 1 / estrategia 2 / estrategia 3" ni
    para listar parámetros — eso fragmenta la narrativa. Si tu sub-ítem
    actual tiene 2 o más bullets, reescribilo como párrafo corrido con
    conectores ("primero…, eso me dejó con…, entonces…").

R4. **Las hipótesis refutadas (corridas anteriores con peor Kaggle) son
    capítulos, no nota al pie**. Las versiones del leaderboard que
    empeoraron tienen que aparecer integradas en la narrativa, no
    relegadas a una sola frase en el sub-ítem que pregunta "¿cuál es el
    problema?". El flujo típico es: cuento qué decidí en la corrida
    campeona → engancho explicando que antes probé otra cosa más
    ambiciosa → digo qué número del leaderboard mostró que falló → digo
    qué aprendí → eso justifica la decisión final.

R5. **Cada sub-ítem debe responder, en su narrativa, alguna combinación de
    estas preguntas (no como bullets — embebidas en la prosa)**:
    qué observé / qué pregunta me hizo / qué hipótesis formulé / qué
    probé / qué pasó / qué aprendí / cómo eso motiva el siguiente paso.
    Si tu sub-ítem se limita a "qué hice y con qué parámetros", está mal.

R6. **Los números del JSON / leaderboard se integran en el discurso, no se
    pegan entre paréntesis al final como referencias bibliográficas**.
    Preferir *"terminé con 122 383 filas"* sobre *"…(JSON v4:
    `data.train_post_outliers`)"* cuando el contexto ya deja claro de
    dónde sale.

R7. **Auto-chequeo antes de devolver**: releé tu output y verificá que
    (a) ningún sub-ítem arranca con un verbo prohibido de R1; (b) cada
    sub-ítem ≥ 2 tiene un conector explícito al sub-ítem anterior;
    (c) ningún sub-ítem contiene listas con bullets fuera de las
    enumeraciones finitas de R3; (d) v2 y v3 (o las versiones que fallaron
    en esta entrega) aparecen integradas en la narrativa, no sólo en el
    sub-ítem de "¿cuál fue el problema?".

## Materiales

=== CONTEXTO GENERAL DEL TP (.claude/CONTEXT.md) ===
{contexto}

=== LEADERBOARD ACTUAL ({entrega}) ===
{leaderboard}

=== METADATOS DE LA CORRIDA CAMPEONA ({nombre}) ===
{metadatos}

=== ANTI-EJEMPLO — versión previa del informe que el supervisor RECHAZÓ ===

Lo que sigue NO es un template a copiar: es la versión anterior del
informe que el supervisor explícitamente rechazó con esta crítica
literal:

> "Sigue pareciendo un listado de cosas que hice. Solo decís: hice esto,
> hice esto, hice esto. No hay conectores lógicos entre los pasos
> realizados. Eso no es un informe, es un listado de tareas triviales."

Errores concretos que tiene esta versión y que NO podés repetir:

- A.1, A.2, A.3, B.1, B.2 abren con verbos prohibidos por R1
  ("Analicé", "Implementé", "Cuantifiqué", "Definí", "Comparé").
- A.2 y B.2 usan bullets adentro del sub-ítem para listar estrategias —
  viola R3 y fragmenta la narrativa.
- No hay un solo conector explícito entre sub-ítems: A.1 → A.2 → A.3 →
  B.1 → B.2 → B.3 → C son párrafos independientes que el lector puede
  permutar — viola R2.
- Las corridas v2 y v3 (que fracasaron) sólo aparecen en B.3 como
  excusa para responder la pregunta — viola R4. Tendrían que estar
  integradas mucho antes para que el lector entienda *por qué* la
  corrida campeona es lo que es.
- El JSON está pegado entre paréntesis ("(JSON v4: …)") en cada frase —
  viola R6.

VERSIÓN RECHAZADA (NO COPIAR EL ESTILO):

{informe_actual}

## Tarea

Reescribí el informe de cero respetando R1–R7 y todas las secciones
"Estructura / Tono / Narrativa" del CONTEXTO. Podés reusar los **datos**
(números, decisiones técnicas, hiperparámetros) de la versión rechazada,
porque esos están bien — lo que cambia es **cómo se cuentan**. Devolvé
EXCLUSIVAMENTE el contenido Markdown del nuevo informe, sin preámbulos,
sin explicaciones extra, sin code fences alrededor. El primer carácter
de tu respuesta debe ser `#`.
"""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        nuevo = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[informe] OpenAI falló con modelo '{model}': {e}")
        # Fallback razonable si el modelo no existe en el environment del user
        try:
            print("[informe] reintento con gpt-5.4-mini...")
            resp = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            nuevo = resp.choices[0].message.content or ""
            model = "gpt-5.4-mini"
        except Exception as e2:
            print(f"[informe] fallback también falló: {e2}")
            return False

    # --- Validación de longitud (regla: ≤ 500 palabras sin contar tablas) ---
    # Las palabras de las filas de tabla (líneas que arrancan con "|") NO cuentan,
    # porque la regla del CONTEXT y el system_prompt explicitan que el límite es
    # sobre prosa. Si el LLM se pasó por > 510 palabras, pedimos UNA reescritura.
    def _wc_sin_tabla(txt: str) -> int:
        prosa = "\n".join(
            ln for ln in txt.splitlines()
            if not ln.lstrip().startswith("|")
        )
        return len(re.findall(r"\b[\wáéíóúñÁÉÍÓÚÑ]+\b", prosa))

    n_palabras = _wc_sin_tabla(nuevo)
    print(f"[informe] longitud generada: {n_palabras} palabras (sin tabla)")
    if n_palabras > 510:
        print(f"[informe] {n_palabras} > 510; pido re-recorte al LLM (1 intento).")
        try:
            resp2 = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": nuevo},
                    {"role": "user", "content": (
                        f"Tu respuesta tiene {n_palabras} palabras (sin contar la "
                        f"tabla). El límite duro es 500. Reescribila con EL MISMO "
                        f"arco narrativo, los mismos sub-ítems y los mismos números "
                        f"clave, pero condensando oraciones hasta llegar a ≤ 500 "
                        f"palabras de prosa. NO saques sub-ítems ni la tabla; NO "
                        f"empieces con un verbo prohibido (R1); MANTENÉ los "
                        f"conectores entre sub-ítems (R2). Devolvé sólo el "
                        f"Markdown del informe; primer carácter `#`."
                    )},
                ],
                temperature=0.2,
            )
            nuevo2 = resp2.choices[0].message.content or ""
            n_palabras2 = _wc_sin_tabla(nuevo2)
            print(f"[informe] re-recorte: {n_palabras2} palabras (sin tabla)")
            # Aceptamos el recorte aunque siga arriba de 500: cualquier mejora
            # respecto a la original ayuda al fallback CSS de md_to_pdf.
            if 0 < n_palabras2 < n_palabras:
                nuevo = nuevo2
        except Exception as e3:
            print(f"[informe] re-recorte falló (sigo con la versión larga): {e3}")

    # Guardamos backup del informe anterior por las dudas
    if informe_path.exists():
        backup = informe_path.with_suffix(informe_path.suffix + f".bak-{nombre}")
        backup.write_text(informe_actual, encoding="utf-8")

    informe_path.write_text(nuevo.strip() + "\n", encoding="utf-8")
    print(f"[informe] regenerado: {informe_path}")

    # Regeneramos también el PDF a partir del .md. Si falla, el .md queda
    # igualmente, así que no afecta el submit ni el leaderboard.
    # El PDF final adopta el naming que pide la cátedra:
    #   "Entrega Parcial <n> - Patricio Gerpe.pdf"
    # (mismo formato que el de Entrega 1). Documentado en
    # .claude/CONTEXT.md → "Informe de entrega → Naming del archivo final".
    n_entrega = entrega.split("_")[-1]
    pdf_path = informe_path.parent / f"Entrega Parcial {n_entrega} - Patricio Gerpe.pdf"
    md_to_pdf(informe_path, pdf_path)

    return True


# ---------- Markdown → PDF --------------------------------------------------

def md_to_pdf(md_path: Path, pdf_path: Path) -> bool:
    """Convierte un .md a .pdf usando xhtml2pdf (pure Python, sin deps nativas).

    Devuelve True si generó el PDF. Si falla, loggea el error y devuelve False
    sin romper el flujo principal.
    """
    try:
        import markdown as _md
        from xhtml2pdf import pisa
    except ImportError as e:
        print(f"[pdf] falta dependencia: {e}. Instalá: pip install markdown xhtml2pdf")
        return False

    try:
        md_text = md_path.read_text(encoding="utf-8")
        html_body = _md.markdown(
            md_text,
            extensions=["tables", "fenced_code", "sane_lists"],
        )
        # CSS pensado para que el informe entre en UNA carilla A4.
        # v6: bajamos font/line-height/margins como margen de seguridad
        # cuando el LLM se pasa de 500 palabras (la validación de longitud
        # en regenerate_informe ya intentó recortar, esto es el last-resort).
        html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    @page {{
      size: A4;
      margin: 1.5cm 1.8cm;
    }}
    body {{
      font-family: "Helvetica", "Arial", sans-serif;
      font-size: 9.0pt;
      line-height: 1.30;
      color: #111;
    }}
    h1 {{
      font-size: 14pt;
      margin: 0 0 2pt 0;
      border-bottom: 1px solid #888;
      padding-bottom: 2pt;
    }}
    h2 {{
      font-size: 11pt;
      margin: 7pt 0 2pt 0;
      color: #222;
    }}
    h3 {{
      font-size: 9.5pt;
      margin: 5pt 0 2pt 0;
    }}
    p  {{ margin: 2.5pt 0; }}
    ul, ol {{ margin: 2pt 0 2pt 18pt; padding: 0; }}
    li {{ margin: 1pt 0; }}
    strong {{ font-weight: bold; }}
    em {{ font-style: italic; }}
    code {{
      font-family: "Courier New", monospace;
      font-size: 8pt;
      background: #f3f3f3;
      padding: 0 2pt;
    }}
    table {{
      border-collapse: collapse;
      margin: 3pt 0;
      width: 100%;
    }}
    th, td {{
      border: 1px solid #999;
      padding: 2pt 4pt;
      font-size: 8pt;
      vertical-align: top;
    }}
    th {{ background: #eee; font-weight: bold; text-align: left; }}
    td[align="right"], th[align="right"] {{ text-align: right; }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""
        with open(pdf_path, "wb") as _f:
            status = pisa.CreatePDF(html, dest=_f, encoding="utf-8")
        if status.err:
            print(f"[pdf] xhtml2pdf reportó {status.err} errores al generar {pdf_path}")
            return False
        print(f"[pdf] generado: {pdf_path}")
        return True
    except Exception as e:
        print(f"[pdf] falló la generación: {e}")
        return False
