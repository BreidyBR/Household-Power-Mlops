"""
Data Quality Gates — Día 4.

Este script implementa las validaciones automáticas que el pipeline debe
pasar ANTES de que los datos crudos se limpien y se usen para entrenar un
modelo. La idea central de un "Data Quality Gate" es que la calidad de los
datos no se verifica manualmente ni una sola vez: se verifica con reglas
automáticas, reproducibles y ejecutables en cualquier momento (por ejemplo,
cada vez que se descarga una nueva versión del dataset, o antes de cada
reentrenamiento).

Encaja en el pipeline justo después de la ingesta y antes de la limpieza:

    src/ingestion/ingest.py  -->  src/validation/validate.py  -->  src/cleaning/clean.py

Cada regla (función `rule_*`) evalúa un aspecto distinto de calidad y
devuelve uno de tres veredictos:

    PASS    -> el dato cumple, no requiere atención.
    WARNING -> hay una desviación menor, se registra pero NO detiene el pipeline.
    FAIL    -> problema grave: el pipeline se considera "bloqueado" (no se
               debería continuar con la limpieza/entrenamiento hasta revisarlo).

Los ocho umbrales usados aquí no son valores arbitrarios: están justificados
con evidencia real del dataset, documentada paso a paso en
notebooks/01_data_quality_cleaning.ipynb (diagnóstico en la Parte I, y el
umbral de 60 minutos para gaps se reutiliza también en src/cleaning/clean.py,
donde se explica con detalle por qué se eligió ese valor).

Al final, todas las reglas se resumen en un único archivo
reports/data_quality/data_quality_report.json, que sirve como evidencia
auditable de la corrida (qué regla se evaluó, con qué resultado, y por qué).

Uso:
    python src/validation/validate.py
    python src/validation/validate.py --strict   # además, termina con código de error 1 si hay algún FAIL
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------

# Ruta raíz del proyecto (mismo patrón que src/ingestion/ingest.py, para que
# el script funcione sin importar desde qué carpeta se ejecute).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Archivo generado por el pipeline de ingesta (src/ingestion/ingest.py).
# Este script asume que la ingesta ya se ejecutó y falla explícitamente si no.
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"

# Dónde se guarda el reporte de calidad que este script genera. Este archivo
# SÍ se versiona en Git (no está en .gitignore) porque es evidencia de que
# el pipeline validó los datos, no un dato en sí mismo.
REPORTS_DATA_QUALITY_DIR = PROJECT_ROOT / "reports" / "data_quality"
DATA_QUALITY_REPORT_FILE = REPORTS_DATA_QUALITY_DIR / "data_quality_report.json"

# ---------------------------------------------------------------------------
# Esquema esperado del dataset crudo
# ---------------------------------------------------------------------------

DATE_COL = "Date"
TIME_COL = "Time"

# Las siete variables de medición eléctrica del dataset (ver documentación
# oficial de UCI, resumida también en el notebook, Parte I, sección 9).
NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Todo lo que el dataset SIEMPRE debe traer para poder procesarse: las
# variables de medición más las columnas de fecha/hora. Se usa en la regla
# "columnas obligatorias presentes".
REQUIRED_COLUMNS = NUMERIC_COLUMNS + [DATE_COL, TIME_COL]

# Variable objetivo del proyecto (lo que se va a pronosticar más adelante).
TARGET_COL = "Global_active_power"

# ---------------------------------------------------------------------------
# Umbrales de las reglas
#
# Cada umbral está justificado con el comportamiento REAL del dataset
# (2,075,259 filas), no elegido "a ojo". El detalle completo de cada
# justificación está en notebooks/01_data_quality_cleaning.ipynb.
# ---------------------------------------------------------------------------

# Un dataset con muy pocas filas no permite entrenar ni validar un modelo de
# forma confiable. 1000 es un piso conservador muy por debajo de lo que
# jamás se esperaría recibir de este dataset (2M+ filas).
MIN_ROWS_ALLOWED = 1000

# El dataset real tiene 0% de duplicados (verificado en el notebook, Parte I,
# sección 6). Se permite hasta 1% antes de bloquear, para tolerar pequeñas
# anomalías sin ser tan estricto que cualquier corrida futura falle por un
# puñado de filas repetidas.
MAX_DUPLICATED_FRACTION = 0.01  # 1%

# El dataset real tiene ~1.25% de missing (25,979 de 2,075,259 filas,
# documentado en el notebook, Parte I, sección 5). Se usa 5% como umbral de
# alerta: da margen para que el missing crezca moderadamente en una futura
# descarga sin bloquear el pipeline, pero detecta un salto grande y anómalo.
MAX_MISSING_FRACTION = 0.05  # 5%

# El dataset se captura nativamente cada minuto (documentado en el notebook,
# Parte I, sección 8: 2,075,258 de 2,075,258 diferencias consecutivas son
# exactamente de 1 minuto). Esta es la frecuencia que se espera siempre.
EXPECTED_SAMPLING_FREQUENCY = "1min"

# Duración máxima (en minutos) de un hueco entre timestamps que se considera
# tolerable antes de generar una alerta. Este control es independiente de las
# rachas de valores faltantes en las variables eléctricas: aquí se evalúa la
# continuidad de la secuencia temporal, no la presencia de NaN en las mediciones.
# El diagnóstico del dataset mostró que la frecuencia esperada es de un minuto
# y que no existen intervalos temporales irregulares. Se establece un umbral
# conservador de 60 minutos para detectar futuras interrupciones prolongadas
# en la secuencia de timestamps.
MAX_ALLOWED_GAP_MINUTES = 60

# Fracción de intervalos entre timestamps que pueden exceder
# MAX_ALLOWED_GAP_MINUTES antes de que la regla pase de WARNING a FAIL. Un
# valor muy pequeño (0.1%) porque, según el diagnóstico, los huecos grandes
# son eventos raros: varios de ellos ya deberían llamar la atención.
MAX_GAP_FRACTION_WARNING = 0.001  # 0.1%

# Los tres veredictos posibles de cada regla. Se definen como constantes
# (en vez de escribir el string directamente en cada regla) para evitar
# errores de tipeo y para que sea fácil ubicar todos los usos.
PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"


@dataclass
class GateResult:
    """
    Resultado de evaluar UNA regla sobre el dataset.

    Se usa una dataclass (en vez de, por ejemplo, un diccionario suelto)
    para que cada resultado tenga una forma fija y predecible, y para poder
    convertirlo directamente a JSON con `dataclasses.asdict()` al generar
    el reporte final.
    """

    rule: str      # nombre corto de la regla (ej. "min_rows")
    status: str    # PASS | WARNING | FAIL
    detail: str    # explicación legible de por qué se llegó a ese veredicto


def load_raw():
    """
    Carga el dataset crudo generado por la ingesta.

    Se usa `na_values=["?"]` para que pandas reconozca automáticamente el
    símbolo '?' (la forma en que este dataset representa un valor faltante,
    ver notebook Parte I sección 5) como NaN nativo, en vez de dejarlo como
    texto. Date y Time se cargan como texto (`dtype=str`) porque su validez
    se comprueba explícitamente en `_parsed_datetime`, no se le confía la
    inferencia de tipos a pandas.
    """

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            "No se encontró el dataset raw. Ejecuta primero src/ingestion/ingest.py."
        )

    return pd.read_csv(
        RAW_FILE, sep=";", na_values=["?"], low_memory=False,
        dtype={DATE_COL: str, TIME_COL: str},
    )


def _parsed_datetime(df):
    """
    Intenta construir un timestamp a partir de Date + Time.

    `errors="coerce"` hace que cualquier combinación que NO tenga el formato
    esperado (día/mes/año hora:minuto:segundo) se convierta en NaT (Not a
    Time) en vez de lanzar una excepción. Esto es intencional: así se pueden
    CONTAR los timestamps inválidos en `rule_valid_timestamps` en vez de que
    todo el script se detenga ante el primer registro problemático.
    """

    return pd.to_datetime(df[DATE_COL] + " " + df[TIME_COL], format="%d/%m/%Y %H:%M:%S", errors="coerce")


# ---------------------------------------------------------------------------
# Las 8 reglas de Data Quality Gates
#
# Cada función recibe el DataFrame crudo completo y devuelve un único
# GateResult. Están escritas como funciones independientes (en vez de un
# bloque de código largo) para que cada regla se pueda leer, probar y
# entender por separado.
# ---------------------------------------------------------------------------

def rule_required_columns(df):
    """
    Regla 1: columnas obligatorias presentes.

    Verifica que el archivo contenga todas las columnas que el resto del
    pipeline necesita (las 7 variables eléctricas + Date + Time).

    Esta regla protege contra cambios de esquema en la fuente de datos, por
    ejemplo, si una columna obligatoria fuera eliminada o renombrada.

    La presencia de columnas adicionales no provoca un FAIL, siempre que todas
    las columnas requeridas continúen disponibles.
    """

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return GateResult("required_columns_present", FAIL, f"Faltan columnas obligatorias: {missing_cols}")
    return GateResult("required_columns_present", PASS, "Todas las columnas obligatorias están presentes")


def rule_min_rows(df):
    """
    Regla 2: número mínimo de filas.

    Protege contra una ingesta corrupta o incompleta (por ejemplo, una
    descarga interrumpida a mitad de camino que deje un archivo truncado).
    Un dataset con menos de MIN_ROWS_ALLOWED filas no es utilizable para
    entrenar ni validar un modelo de forma confiable.
    """

    n = len(df)
    status = PASS if n >= MIN_ROWS_ALLOWED else FAIL
    return GateResult("min_rows", status, f"{n} filas (mínimo requerido: {MIN_ROWS_ALLOWED})")


def rule_valid_timestamps(df):
    """
    Regla 3: timestamps válidos.

    Comprueba que Date + Time puedan convertirse a un timestamp real. Es
    tolerante a un porcentaje mínimo de timestamps corruptos (< 0.1% ->
    WARNING, no bloquea), pero si una fracción mayor no puede parsearse,
    bloquea el pipeline: sin una fecha/hora confiable, no se puede construir
    la serie temporal en la que se basa todo el proyecto.
    """

    dt = _parsed_datetime(df)
    n_invalid = int(dt.isna().sum())
    frac_invalid = n_invalid / len(df) if len(df) else 0.0

    if frac_invalid == 0:
        status = PASS
    elif frac_invalid < 0.001:
        status = WARNING
    else:
        status = FAIL
    return GateResult("valid_timestamps", status, f"{n_invalid} timestamps inválidos ({frac_invalid:.4%})")


def rule_duplicated_fraction(df):
    """
    Regla 4: duplicados bajo umbral.

    `df.duplicated()` marca como duplicada cualquier fila cuyo contenido
    completo (las 9 columnas) ya apareció antes en el dataset. Registros
    duplicados inflarían artificialmente ciertos patrones durante el
    entrenamiento (el modelo vería la misma observación varias veces).
    """

    frac = float(df.duplicated().mean())
    status = PASS if frac < MAX_DUPLICATED_FRACTION else FAIL
    return GateResult("duplicated_below_threshold", status,
                       f"{frac:.4%} de filas duplicadas (máx: {MAX_DUPLICATED_FRACTION:.0%})")


def rule_missing_fraction(df):
    """
    Regla 5: missing bajo umbral.

    Calcula el promedio de valores faltantes a lo largo de las 7 variables
    eléctricas. No distingue todavía entre rachas cortas y largas de missing
    (eso se analiza con más detalle en la limpieza, src/cleaning/clean.py);
    aquí solo se comprueba que el missing GLOBAL no se haya disparado muy
    por encima de lo que documenta el diagnóstico original (~1.25%).
    """

    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    frac = float(df[cols].isna().mean().mean()) if cols else 0.0
    status = PASS if frac < MAX_MISSING_FRACTION else FAIL
    return GateResult("missing_below_threshold", status,
                       f"{frac:.4%} missing promedio en columnas numéricas (máx: {MAX_MISSING_FRACTION:.0%})")


def rule_target_non_negative(df):
    """
    Regla 6: Global_active_power >= 0.

    Es una regla de "imposibilidad física": la potencia activa consumida
    por un hogar no puede ser negativa. Un valor negativo indicaría un
    error del sensor o un problema de transcripción del dataset, nunca un
    valor real. Se convierte primero a numérico (ignorando lo que no pueda
    convertirse) para no confundir el símbolo '?' de missing con un dato
    numérico inválido.
    """

    if TARGET_COL not in df.columns:
        return GateResult("target_non_negative", FAIL, f"Columna objetivo '{TARGET_COL}' ausente")

    values = pd.to_numeric(df[TARGET_COL], errors="coerce").dropna()
    n_negative = int((values < 0).sum())
    status = PASS if n_negative == 0 else FAIL
    return GateResult("target_non_negative", status,
                       f"{n_negative} valores negativos en {TARGET_COL} (físicamente imposible)")


def rule_expected_sampling_frequency(df):
    """
    Regla 7: frecuencia temporal correcta.

    El dataset se captura nativamente cada minuto. Esta regla:
      1. Ordena los timestamps válidos y calcula la diferencia entre cada
         observación y la anterior.
      2. Toma la MODA de esas diferencias (el intervalo más frecuente) y
         confirma que sea igual a 1 minuto.
      3. Además calcula qué fracción de TODOS los intervalos se desvía de
         esa frecuencia esperada.

    Si la moda misma no es de 1 minuto, algo está fundamentalmente mal con
    el archivo (FAIL). Si la moda es correcta pero hay una fracción notable
    de intervalos distintos (> 1%), se marca como advertencia sin bloquear,
    porque unos pocos huecos aislados no invalidan la frecuencia general
    del dataset.
    """

    dt = _parsed_datetime(df).dropna().sort_values()
    diffs = dt.diff().dropna().astype("timedelta64[ns]")

    if diffs.empty:
        return GateResult("expected_sampling_frequency", FAIL, "No hay suficientes timestamps válidos para evaluar frecuencia")

    expected = pd.Timedelta(EXPECTED_SAMPLING_FREQUENCY)
    modal_diff = diffs.mode().iloc[0]
    frac_off_frequency = float((diffs != expected).mean())

    if modal_diff != expected:
        status = FAIL
    elif frac_off_frequency > 0.01:
        status = WARNING
    else:
        status = PASS
    return GateResult(
        "expected_sampling_frequency", status,
        f"moda de intervalos={modal_diff}, esperado={expected}, "
        f"{frac_off_frequency:.4%} de intervalos fuera de frecuencia",
    )


def rule_temporal_gaps_controlled(df):
    """
    Regla 8: gaps temporales controlados.

    Mientras que la regla anterior mira la frecuencia TÍPICA del dataset,
    esta regla busca específicamente huecos GRANDES: intervalos entre dos
    observaciones consecutivas que superan MAX_ALLOWED_GAP_MINUTES (60
    minutos). Un par de huecos aislados son esperables en más de 4 años de
    mediciones (mantenimiento del medidor, corte de energía) y no deben
    bloquear el pipeline; muchos huecos grandes sí, porque indicarían un
    problema sistemático en la captura de datos.
    """

    dt = _parsed_datetime(df).dropna().sort_values()
    diffs = dt.diff().dropna().astype("timedelta64[ns]")
    max_gap_allowed = np.timedelta64(MAX_ALLOWED_GAP_MINUTES, "m")

    gaps = diffs[diffs > max_gap_allowed]
    frac_gaps = float(len(gaps) / len(diffs)) if len(diffs) else 0.0
    largest_gap = gaps.max() if len(gaps) else pd.Timedelta(0)

    if frac_gaps == 0:
        status = PASS
    elif frac_gaps <= MAX_GAP_FRACTION_WARNING:
        status = WARNING
    else:
        status = FAIL
    return GateResult(
        "temporal_gaps_controlled", status,
        f"{len(gaps)} huecos > {MAX_ALLOWED_GAP_MINUTES}min ({frac_gaps:.4%} de los intervalos), "
        f"hueco máximo={largest_gap}",
    )


# Lista de las 8 reglas, en el orden en que se ejecutan y se muestran en el
# reporte. Agregar una nueva regla al proyecto significa simplemente
# escribir una función `rule_*` más y añadirla aquí.
ALL_RULES = [
    rule_required_columns,
    rule_min_rows,
    rule_valid_timestamps,
    rule_duplicated_fraction,
    rule_missing_fraction,
    rule_target_non_negative,
    rule_expected_sampling_frequency,
    rule_temporal_gaps_controlled,
]


def run_gates(df):
    """
    Ejecuta las 8 reglas sobre `df` en orden y va imprimiendo cada
    resultado en la consola a medida que se calcula (útil para seguir el
    progreso cuando el dataset es grande, como en este caso con 2M+ filas).
    Devuelve la lista completa de resultados para que se pueda generar el
    reporte o decidir si el pipeline continúa.
    """

    results = [rule(df) for rule in ALL_RULES]
    for r in results:
        print(f"[{r.rule}] {r.status} -> {r.detail}")
    return results


def overall_status(results):
    """
    Resume los 8 resultados individuales en un único veredicto global:

        - Si CUALQUIER regla dio FAIL, el veredicto global es FAIL
          (basta un solo problema grave para bloquear el pipeline).
        - Si no hay ningún FAIL pero sí al menos un WARNING, el veredicto
          global es WARNING (nada bloquea, pero queda registrado para
          revisión humana).
        - Si todas las reglas dieron PASS, el veredicto global es PASS.
    """

    if any(r.status == FAIL for r in results):
        return FAIL
    if any(r.status == WARNING for r in results):
        return WARNING
    return PASS


def save_report(results, path=DATA_QUALITY_REPORT_FILE):
    """
    Genera reports/data_quality/data_quality_report.json.

    Este archivo es la evidencia auditable de la corrida: cualquier persona
    (incluido el profesor) puede abrir el JSON y ver exactamente qué reglas
    se evaluaron, con qué resultado y por qué, sin tener que volver a
    ejecutar el script. `default=str` en `json.dump` es una salvaguarda para
    que cualquier valor no serializable directamente a JSON (por ejemplo,
    un `pandas.Timedelta` dentro del texto de detalle) se convierta a texto
    en vez de hacer fallar la generación del reporte.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "overall_status": overall_status(results),
        "rules": [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Reporte de calidad guardado en {path} (overall_status={report['overall_status']})")


def main():
    """
    Punto de entrada del script.

    El flag `--strict` está pensado para uso en automatización (por
    ejemplo, un pipeline de CI/CD o un `Makefile` que encadene ingesta ->
    validación -> limpieza -> entrenamiento): con `--strict`, el proceso
    termina con código de salida 1 si el veredicto global es FAIL, lo cual
    permite que el pipeline se detenga automáticamente antes de limpiar o
    entrenar con datos que no pasaron las validaciones. Sin `--strict`
    (comportamiento por defecto), el script siempre reporta y guarda el
    JSON, mostrando los problemas encontrados, pero no interrumpe el
    proceso — útil para inspeccionar manualmente sin que un FAIL corte la
    ejecución del script en sí.
    """

    parser = argparse.ArgumentParser(description="Data Quality Gates (Día 4).")
    parser.add_argument("--strict", action="store_true", help="Salir con código 1 si hay alguna regla en FAIL.")
    args = parser.parse_args()

    print("=== INICIO DE DATA QUALITY GATES ===")

    raw = load_raw()
    results = run_gates(raw)
    save_report(results)

    print("=== DATA QUALITY GATES COMPLETADO ===")

    if args.strict and overall_status(results) == FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
