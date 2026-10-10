
from pathlib import Path
import json
import time
import logging

import numpy as np
import pandas as pd
import pyreadstat
from ollama import Client


# =========================================================
# CONFIGURACION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

RUTA_BASE = (
    ROOT / "data/raw/966-Modulo02/Enaho01-2024-200.sav"
)
RUTA_PREGUNTAS = ROOT / "data/interim/preguntas_evaluacion.csv"
RUTA_RESULTADOS = ROOT / "data/processed/comparacion_modelos.csv"
RUTA_LOG = ROOT / "logs/comparacion_modelos.log"

RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=RUTA_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

MODELOS = ["gemma2:9b", "qwen2.5:7b"]
OLLAMA = Client(host="http://localhost:11434")

base, meta = pyreadstat.read_sav(str(RUTA_BASE))

ETIQUETAS = meta.column_names_to_labels or {}
ETIQUETAS_VALORES = meta.variable_value_labels or {}

COLUMNAS = set(base.columns)


# =========================================================
# CATALOGO DE VARIABLES
# =========================================================

def crear_catalogo():
    catalogo = []

    for var in base.columns:
        catalogo.append({
            "variable": var,
            "etiqueta": ETIQUETAS.get(var, ""),
            "tipo": str(base[var].dtype),
        })

    return catalogo


CATALOGO = crear_catalogo()


# =========================================================
# SOLICITUD AL MODELO
# =========================================================

def interpretar(modelo, pregunta):
    prompt = f"""
Eres un asistente para consultas estadísticas de encuestas.

Interpreta la pregunta utilizando SOLO el catálogo.
No calcules resultados ni inventes nombres de variables.

Operaciones permitidas:
- frecuencia
- promedio
- conteo

Devuelve JSON con estas claves:
{{
  "operacion": "frecuencia",
  "variable": "NOMBRE_VARIABLE",
  "group_by": null,
  "variables_candidatas": [
    "VARIABLE_1", "VARIABLE_2"
  ]
}}

Reglas:
- variables_candidatas debe tener hasta 5 variables,
  ordenadas de mayor a menor relevancia.
- variable es la principal para el cálculo.
- Para conteo total, variable puede ser null.
- group_by es null cuando no se solicita agrupación.
- Para un promedio, elige una variable numérica.
- Si no puedes determinar la consulta, usa
  "operacion": "no_identificada".
- No inventes variables.

CATALOGO:
{json.dumps(CATALOGO, ensure_ascii=False)}

PREGUNTA:
{pregunta}
"""

    inicio = time.perf_counter()

    respuesta = OLLAMA.chat(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responde solo con un objeto JSON válido."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    latencia = time.perf_counter() - inicio

    plan = json.loads(respuesta.message.content)

    return plan, latencia


# =========================================================
# VALIDACION
# =========================================================

def validar_plan(plan):
    operacion = plan.get("operacion")
    variable = plan.get("variable")
    grupo = plan.get("group_by")

    if operacion not in {"frecuencia", "promedio", "conteo"}:
        return False

    if operacion in {"frecuencia", "promedio"}:
        if variable not in COLUMNAS:
            return False

    if grupo is not None and grupo not in COLUMNAS:
        return False

    if operacion == "promedio":
        if not pd.api.types.is_numeric_dtype(base[variable]):
            return False

    candidatos = plan.get("variables_candidatas", [])

    if not isinstance(candidatos, list):
        return False

    return True


# =========================================================
# MOTOR DE REFERENCIA
# Ejecuta solamente operaciones permitidas.
# =========================================================

def calcular(operacion, variable=None, group_by=None):
    if operacion == "conteo":
        if group_by is None:
            return {"n": int(len(base))}

        tabla = base.groupby(
            group_by, dropna=False
        ).size()

        return {
            str(k): int(v)
            for k, v in tabla.items()
        }

    if variable not in COLUMNAS:
        raise ValueError("Variable inexistente")

    if operacion == "promedio":
        if not pd.api.types.is_numeric_dtype(base[variable]):
            raise ValueError("Promedio no numérico")

        if group_by is None:
            return {
                "promedio": float(base[variable].mean())
            }

        tabla = base.groupby(
            group_by, dropna=False
        )[variable].mean()

        return {
            str(k): float(v)
            for k, v in tabla.items()
        }

    if operacion == "frecuencia":
        if group_by is None:
            tabla = base[variable].value_counts(
                dropna=False
            )
            return {
                str(k): int(v)
                for k, v in tabla.items()
            }

        tabla = base.groupby(
            [group_by, variable], dropna=False
        ).size()

        return {
            str(k): int(v)
            for k, v in tabla.items()
        }

    raise ValueError("Operación no permitida")


# =========================================================
# COMPARACION DE RESULTADOS
# =========================================================

def resultados_iguales(a, b, tolerancia=1e-6):
    if set(a.keys()) != set(b.keys()):
        return False

    for clave in a:
        va, vb = a[clave], b[clave]

        if isinstance(va, (int, float)) and isinstance(
            vb, (int, float)
        ):
            if not np.isclose(
                va, vb, atol=tolerancia, rtol=tolerancia
            ):
                return False
        elif va != vb:
            return False

    return True


def evaluar_fila(modelo, fila):
    pregunta = fila["pregunta"]
    variable_gold = str(fila["variable_esperada"]).strip()
    operacion_gold = str(fila["operacion_esperada"]).strip()
    grupo_gold = str(fila["group_by_esperado"]).strip()

    variable_gold = (
        None if variable_gold.lower() in {"", "nan", "none"}
        else variable_gold
    )
    grupo_gold = (
        None if grupo_gold.lower() in {"", "nan", "none"}
        else grupo_gold
    )

    registro = {
        "modelo": modelo,
        "pregunta": pregunta,
        "latencia_seg": np.nan,
        "json_valido": False,
        "plan_valido": False,
        "variable_correcta": False,
        "operacion_correcta": False,
        "grupo_correcto": False,
        "plan_exacto": False,
        "recall_at_5": np.nan,
        "mrr": np.nan,
        "ejecucion_correcta": False,
        "resultado_coincide": False,
        "error": "",
    }

    try:
        plan, latencia = interpretar(modelo, pregunta)
        registro["latencia_seg"] = latencia
        registro["json_valido"] = True

        candidatos = plan.get("variables_candidatas", [])
        if not isinstance(candidatos, list):
            candidatos = []

        # Recall@5 y MRR para la variable principal
        if variable_gold is not None:
            candidatos = candidatos[:5]
            registro["recall_at_5"] = float(
                variable_gold in candidatos
            )

            posiciones = [
                i + 1 for i, v in enumerate(candidatos)
                if v == variable_gold
            ]
            registro["mrr"] = (
                1.0 / posiciones[0] if posiciones else 0.0
            )

        registro["variable_correcta"] = (
            plan.get("variable") == variable_gold
        )
        registro["operacion_correcta"] = (
            plan.get("operacion") == operacion_gold
        )
        registro["grupo_correcto"] = (
            plan.get("group_by") == grupo_gold
        )

        registro["plan_exacto"] = all([
            registro["variable_correcta"],
            registro["operacion_correcta"],
            registro["grupo_correcto"],
        ])

        registro["plan_valido"] = validar_plan(plan)

        if registro["plan_valido"]:
            registro["ejecucion_correcta"] = True

            resultado_modelo = calcular(
                plan["operacion"],
                plan.get("variable"),
                plan.get("group_by"),
            )

            resultado_gold = calcular(
                operacion_gold,
                variable_gold,
                grupo_gold,
            )

            registro["resultado_coincide"] = resultados_iguales(
                resultado_modelo, resultado_gold
            )

    except Exception as error:
        registro["error"] = str(error)
        logging.exception(
            "Fallo en modelo=%s pregunta=%s",
            modelo, pregunta
        )

    return registro


# =========================================================
# EXPERIMENTO COMPLETO
# =========================================================

def main():
    preguntas = pd.read_csv(RUTA_PREGUNTAS).fillna("")

    columnas = {
        "pregunta",
        "variable_esperada",
        "operacion_esperada",
        "group_by_esperado",
    }

    faltantes = columnas - set(preguntas.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV: {faltantes}"
        )

    resultados = []

    for modelo in MODELOS:
        print(f"\nEvaluando {modelo}...")

        for i, fila in preguntas.iterrows():
            print(f"Pregunta {i + 1}/{len(preguntas)}")

            resultado = evaluar_fila(modelo, fila)
            resultados.append(resultado)

            logging.info("%s", resultado)

    df = pd.DataFrame(resultados)
    df.to_csv(RUTA_RESULTADOS, index=False)

    metricas = df.groupby("modelo").agg(
        consultas=("pregunta", "count"),
        json_valido=("json_valido", "mean"),
        plan_valido=("plan_valido", "mean"),
        exactitud_variable=("variable_correcta", "mean"),
        exactitud_operacion=("operacion_correcta", "mean"),
        exactitud_plan=("plan_exacto", "mean"),
        recall_at_5=("recall_at_5", "mean"),
        mrr=("mrr", "mean"),
        ejecucion_correcta=("ejecucion_correcta", "mean"),
        resultado_coincide=("resultado_coincide", "mean"),
        latencia_media_seg=("latencia_seg", "mean"),
    )

    metricas.to_csv(
        RUTA_RESULTADOS.with_name("metricas_resumen.csv")
    )

    print("\n=== METRICAS COMPARATIVAS (%) ===")
    columnas_pct = [
        "json_valido", "plan_valido", "exactitud_variable",
        "exactitud_operacion", "exactitud_plan", "recall_at_5",
        "mrr", "ejecucion_correcta", "resultado_coincide",
    ]

    print((metricas[columnas_pct] * 100).round(2))
    print("\n=== LATENCIA MEDIA (segundos) ===")
    print(metricas[["latencia_media_seg"]].round(3))

    print("\nResultados:", RUTA_RESULTADOS)


if __name__ == "__main__":
    main()
