# Chatbot LLM + RAG para consultas estadísticas de encuestas

Chatbot basado en un LLM con recuperación aumentada de generación (RAG) que traduce preguntas en lenguaje natural en código Python ejecutable para calcular cifras estadísticas verificables sobre microdatos de encuestas, validado con la ENAHO y adaptable a bases propias de Arellano.

## Autores / Equipo

| Nombre | GitHub | Correo |
|---|---|---|
| [Haeidy Vilcapaza Huaman] | [@HaVi2404] | [haeidy.vilcapaza.h@uni.pe] |


*Proyecto de Investigación II — Maestría en Inteligencia Artificial.*

## Dataset

- **Fuente:** Encuesta Nacional de Hogares (ENAHO), elaborada por el Instituto Nacional de Estadística e Informática (INEI) del Perú. Microdatos públicos disponibles en: https://proyectos.inei.gob.pe/microdatos/
- **Descripción breve:** encuesta de hogares con módulos temáticos (características de la vivienda, empleo, ingresos, gasto, educación, salud, entre otros). El subconjunto usado en la Fase 1 corresponde al módulo Características de los miembros del hogar.
- **Fecha y versión usada:** ENAHO: 2024. Fecha de descarga: 02/10/2026.
- **Fase 2 (futura):** base propia de Arellano (datos de investigación de mercado), a incorporar una vez definido el acceso y el diccionario de variables correspondiente. No se documenta aquí por tratarse de datos comerciales confidenciales.

## Requisitos

- Python 3.10 o superior.
- Instalar dependencias desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```


## Estructura del repositorio

```
.
├── data/
│   ├── raw/              # Datos originales: microdatos de la ENAHO tal como se descargan
│   ├── interim/          # OPCIONAL
│   └── processed/        # Datos listos para el modelo
├── notebooks/            # Exploración de datos y prototipos
├── src/                  # Código fuente del pipeline (RAG, generación de código, ejecución/validación)
├── logs/                 # Registro por consulta: pregunta, esquema recuperado, código generado/ejecutado,
│                          # resultado, tiempo y costo (para evaluación y reproducibilidad)
├── requirements.txt
└── README.md
```


## Resultados esperados (mínimos) para el baseline

El pipeline debe ser capaz de:

- Recibir una pregunta en lenguaje natural sobre una cifra de la ENAHO.
- Recuperar (vía RAG) las variables/etiquetas del diccionario de la encuesta más relevantes para esa pregunta.
- Generar, con el LLM, un fragmento de código Python (pandas) que calcule la cifra solicitada.
- Ejecutar ese código sobre el dataframe de microdatos y validar que corra sin errores.
- Devolver una respuesta en lenguaje natural con la cifra obtenida, el código ejecutado y la referencia a la variable/fuente usada.
- Registrar, por cada consulta, la pregunta, el esquema recuperado, el código generado/ejecutado, el resultado y el tiempo de respuesta, para fines de evaluación y reproducibilidad.
