
from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import pyreadstat
from ollama import chat


# ============================================================
# 1. CONFIGURACION
# ============================================================

MODELO = "gemma2:9b"

RUTA_BASE = r"/Users/haeidyvilcapazahuaman/Documents/GitHub/Maestria_proyecto/data/raw/966-Modulo02/Enaho01-2024-200.sav"


base, meta = pyreadstat.read_sav(RUTA_BASE)

# Copia de los metadatos de etiquetas de valores
etiquetas_valores = meta.variable_value_labels or {}
etiquetas_variables = meta.column_names_to_labels or {}


# ============================================================
# 2. UTILIDADES
# ============================================================

def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )


def etiqueta_variable(variable):
    return etiquetas_variables.get(variable, variable)


def catalogo_variables():
    """Genera un catálogo compacto para orientar al LLM."""
    catalogo = []

    for variable in base.columns:
        valores = etiquetas_valores.get(variable, {})

        catalogo.append({
            "variable": variable,
            "etiqueta": etiqueta_variable(variable),
            "tipo_python": str(base[variable].dtype),
            "valores_distintos": int(
                base[variable].nunique(dropna=True)
            ),
            "ejemplos": [
                str(x) for x in base[variable]
                .dropna().drop_duplicates().head(3).tolist()
            ],
            "categorias": [
                str(k) + ": " + str(v)
                for k, v in list(valores.items())[:20]
            ],
        })

    return catalogo


CATALOGO = catalogo_variables()


# ============================================================
# 3. INTERPRETACION CON EL LLM
# ============================================================

def interpretar_consulta(pregunta):
    """
    Convierte la pregunta a un plan JSON.
    El LLM no ejecuta código.
    """

    prompt = f"""
Eres un asistente experto en análisis estadístico de encuestas.

Debes transformar la pregunta del usuario en un plan JSON.
Usa exclusivamente las variables del catálogo proporcionado.

Operaciones permitidas:
- frecuencia: distribución de una variable categórica.
- promedio: media de una variable numérica.
- conteo: número de filas que cumplen los filtros.

Reglas:
1. No inventes variables ni códigos.
2. Usa los nombres exactos del catálogo.
3. Para frecuencia, especifica variable.
4. Para promedio, especifica variable numérica.
5. Para conteo, variable puede ser null.
6. group_by puede ser null o una variable del catálogo.
7. Los filtros deben usar variables del catálogo y valores
   que existan en los datos o en sus etiquetas.
8. Si no puedes identificar la variable con suficiente
   confianza, devuelve operacion "no_identificada".
9. No calcules resultados ni inventes cifras.
10. Devuelve únicamente JSON válido.

Formato:
{{
  "operacion": "frecuencia",
  "variable": "NOMBRE_VARIABLE",
  "group_by": null,
  "filtros": [
    {{"variable": "OTRA_VARIABLE", "valor": "valor"}}
  ]
}}

CATÁLOGO:
{json.dumps(CATALOGO, ensure_ascii=False)}

PREGUNTA:
{pregunta}
"""

    respuesta = chat(
        model=MODELO,
        messages=[
            {
                "role": "system",
                "content": (
                    "Interpreta consultas estadísticas. "
                    "Devuelve exclusivamente un objeto JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0},
    )

    return json.loads(respuesta.message.content)


# ============================================================
# 4. VALIDACION DEL PLAN
# ============================================================

OPERACIONES_PERMITIDAS = {
    "frecuencia",
    "promedio",
    "conteo",
}

def validar_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("El plan debe ser un objeto JSON.")

    operacion = plan.get("operacion")

    if operacion not in OPERACIONES_PERMITIDAS:
        raise ValueError(
            f"Operación no permitida: {operacion}"
        )

    variable = plan.get("variable")
    group_by = plan.get("group_by")
    filtros = plan.get("filtros", [])

    if variable is not None and variable not in base.columns:
        raise ValueError(f"Variable inexistente: {variable}")

    if operacion in {"frecuencia", "promedio"}:
        if variable is None:
            raise ValueError("Falta especificar la variable.")

    if operacion == "promedio":
        if not pd.api.types.is_numeric_dtype(base[variable]):
            raise ValueError(
                f"{variable} no es numérica. "
                "No se calculará un promedio."
            )

    if group_by is not None and group_by not in base.columns:
        raise ValueError(f"Grupo inexistente: {group_by}")

    if not isinstance(filtros, list):
        raise ValueError("Los filtros deben ser una lista.")

    for filtro in filtros:
        if not isinstance(filtro, dict):
            raise ValueError("Formato de filtro inválido.")

        var = filtro.get("variable")

        if var not in base.columns:
            raise ValueError(f"Variable de filtro inválida: {var}")

        if "valor" not in filtro:
            raise ValueError("Falta el valor de un filtro.")

    return True


# ============================================================
# 5. RESOLUCION DE CATEGORIAS Y FILTROS
# ============================================================

def resolver_valor(variable, valor):
    """
    Acepta tanto el código almacenado como su etiqueta SPSS.
    Devuelve un valor compatible con la columna original.
    """
    serie = base[variable]
    etiquetas = etiquetas_valores.get(variable, {})

    texto = normalizar(valor)

    # Coincidencia con las etiquetas de valores
    for codigo, etiqueta in etiquetas.items():
        if normalizar(etiqueta) == texto:
            return codigo

    # Coincidencia con valores observados
    for observado in serie.dropna().unique():
        if normalizar(observado) == texto:
            return observado

        if normalizar(str(observado)) == texto:
            return observado

    raise ValueError(
        f"No se encontró el valor '{valor}' en {variable}."
    )


def aplicar_filtros(df, filtros):
    resultado = df.copy()

    for filtro in filtros:
        variable = filtro["variable"]
        valor = resolver_valor(variable, filtro["valor"])

        resultado = resultado[
            resultado[variable] == valor
        ]

    return resultado


def mostrar_etiqueta(variable, valor):
    etiquetas = etiquetas_valores.get(variable, {})

    for codigo, etiqueta in etiquetas.items():
        if codigo == valor:
            return str(etiqueta)

    return str(valor)


# ============================================================
# 6. MOTOR ESTADISTICO
# ============================================================

def ejecutar_consulta(plan):
    validar_plan(plan)

    df = aplicar_filtros(base, plan.get("filtros", []))

    operacion = plan["operacion"]
    variable = plan.get("variable")
    group_by = plan.get("group_by")

    if df.empty:
        return {
            "mensaje": "No hay registros para los filtros indicados."
        }

    if operacion == "conteo":
        if group_by:
            resultado = (
                df.groupby(group_by, dropna=False)
                .size()
                .rename("frecuencia")
                .reset_index()
            )

            resultado[group_by] = resultado[group_by].map(
                lambda x: mostrar_etiqueta(group_by, x)
                if pd.notna(x) else "Sin dato"
            )

            return resultado

        return {"registros": int(len(df))}

    if operacion == "frecuencia":
        if group_by and group_by != variable:
            resultado = (
                df.groupby(
                    [group_by, variable],
                    dropna=False,
                )
                .size()
                .rename("frecuencia")
                .reset_index()
            )

            resultado["porcentaje"] = (
                resultado["frecuencia"]
                / resultado.groupby(group_by)["frecuencia"]
                    .transform("sum")
                * 100
            ).round(2)

            resultado[group_by] = resultado[group_by].map(
                lambda x: mostrar_etiqueta(group_by, x)
                if pd.notna(x) else "Sin dato"
            )

            resultado[variable] = resultado[variable].map(
                lambda x: mostrar_etiqueta(variable, x)
                if pd.notna(x) else "Sin dato"
            )

        else:
            resultado = (
                df[variable]
                .value_counts(dropna=False)
                .rename_axis(variable)
                .reset_index(name="frecuencia")
            )

            resultado["porcentaje"] = (
                resultado["frecuencia"] / len(df) * 100
            ).round(2)

            resultado[variable] = resultado[variable].map(
                lambda x: mostrar_etiqueta(variable, x)
                if pd.notna(x) else "Sin dato"
            )

        return resultado

    if operacion == "promedio":
        if group_by:
            resultado = (
                df.groupby(group_by, dropna=False)[variable]
                .agg(["mean", "count"])
                .reset_index()
            )

            resultado = resultado.rename(columns={
                "mean": "promedio",
                "count": "n_observaciones",
            })

            resultado["promedio"] = resultado["promedio"].round(3)

            resultado[group_by] = resultado[group_by].map(
                lambda x: mostrar_etiqueta(group_by, x)
                if pd.notna(x) else "Sin dato"
            )

            return resultado

        return {
            "variable": variable,
            "promedio": round(float(df[variable].mean()), 3),
            "n_observaciones": int(df[variable].count()),
        }


# ============================================================
# 7. RESPUESTA EN LENGUAJE NATURAL
# ============================================================

def explicar_resultado(pregunta, plan, resultado):
    if isinstance(resultado, pd.DataFrame):
        datos = resultado.head(50).to_dict(orient="records")
    else:
        datos = resultado

    prompt = f"""
Responde en español la pregunta estadística del usuario.

Pregunta: {pregunta}

Plan ejecutado:
{json.dumps(plan, ensure_ascii=False)}

Resultado calculado por Python:
{json.dumps(datos, ensure_ascii=False, default=str)}

Reglas:
- Usa exclusivamente las cifras del resultado.
- No inventes conclusiones ni causalidad.
- Indica que los porcentajes son frecuencias no ponderadas
  si no se ha aplicado un ponderador.
- Si el resultado es una tabla, resume los hallazgos principales.
- Si faltan datos para responder, indícalo.
"""

    respuesta = chat(
        model=MODELO,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    return respuesta.message.content


# ============================================================
# 8. INTERFAZ DE CONSOLA
# ============================================================

def consultar(pregunta):
    plan = interpretar_consulta(pregunta)
    print("\nPlan propuesto:", json.dumps(
        plan, ensure_ascii=False, indent=2
    ))

    validar_plan(plan)
    resultado = ejecutar_consulta(plan)

    print("\nResultado calculado:")

    if isinstance(resultado, pd.DataFrame):
        print(resultado.to_string(index=False))
    else:
        print(resultado)

    explicacion = explicar_resultado(
        pregunta, plan, resultado
    )

    print("\nRespuesta del chatbot:")
    print(explicacion)

    return plan, resultado, explicacion


if __name__ == "__main__":
    print("Chatbot estadístico ENAHO")
    print("Escribe 'salir' para terminar.")

    while True:
        pregunta = input("\nTu consulta: ").strip()

        if pregunta.lower() in {"salir", "exit", "quit"}:
            break

        if not pregunta:
            continue

        try:
            consultar(pregunta)
        except Exception as error:
            print(f"No se pudo procesar la consulta: {error}")
