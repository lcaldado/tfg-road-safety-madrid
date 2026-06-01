"""
================================================================================
FASE 3 – DATA PREPROCESSING & CLEANING
Archivo : phase3_imd.py
Autor   : TFG – Seguridad Vial Comunidad de Madrid
Desc.   : Limpieza e integración del dataset de Intensidad Media Diaria (IMD)
          de la red de carreteras de la CAM (2018-2024).

          Este script IMPORTA y ORQUESTA los tres scripts de extracción
          que ya tienes:
            · html_files_tfg.py   → extracción de HTMLs (2018-2023)
            · pdf_files_tfg.py    → extracción de PDFs  (2024)
            · imd_unification_html_pdf.py → unión de ambos formatos

          Una vez generado el CSV unificado por esos scripts, este archivo
          aplica el pipeline completo de limpieza, validación y enriquecimiento
          necesario para la Fase 4 (cálculo de indicadores).

NOTAS PARA LA MEMORIA:
  · Diferencia metodológica HTML vs PDF (documentar como limitación):
      - 2018-2023 (HTML): IMD calculada por ti sumando intensidades horarias
        con la fórmula DGT (5L + S + D) / 7. Es una APROXIMACIÓN.
      - 2024 (PDF): IMD oficial publicada directamente por la CM. Es el DATO
        OFICIAL. Existe una pequeña diferencia sistemática entre ambas series
        que conviene mencionar al interpretar la evolución temporal.
  · Año 2020: la IMD cae drásticamente por el confinamiento COVID-19.
    No es una variación estructural del tráfico. Se mantiene en el dataset
    pero se etiqueta con una flag para su tratamiento en el análisis.
  · El campo PK (punto kilométrico) es el nexo de unión con los accidentes.
    La calidad de este campo determina la fiabilidad del cruce posterior.
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import sys
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Importar los scripts de extracción que ya tienes
# (deben estar en la misma carpeta code/ que este archivo)
from html_files_tfg import procesar_zip_html
from pdf_files_tfg  import procesar_zip_pdf

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN: RUTAS Y LOGGING
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = BASE_DIR / "raw_data" / "02_traffic_volume"
PROCESSED_DIR = BASE_DIR / "processed_data"
LOGS_DIR      = BASE_DIR / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "phase3_imd.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE AÑOS Y ARCHIVOS ZIP
# Adapta los nombres de ZIP si los tuyos difieren ligeramente
# ──────────────────────────────────────────────────────────────────────────────

AÑOS_HTML = {
    2018: "imd_trafico_2018.zip",
    2019: "imd_trafico_2019.zip",
    2020: "imd_trafico_2020.zip",
    2021: "imd_trafico_2021.zip",
    2022: "imd_trafico_2022.zip",
    2023: "imd_trafico_2023.zip",
}

AÑOS_PDF = {
    2024: "imd_trafico_2024.zip",
}

# Año marcado como atípico por COVID (se añade flag, no se elimina)
AÑO_COVID = 2020

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES DE LIMPIEZA
# ──────────────────────────────────────────────────────────────────────────────

def estandarizar_carretera(serie: pd.Series) -> pd.Series:
    """
    Normaliza el nombre de carretera para que sea idéntico al formato
    usado en phase3_accidents.py y poder hacer el JOIN correctamente.
    Ejemplos: "m-30" → "M-30", "M - 50" → "M-50", "A1" → "A-1" (si aplica)
    """
    return (
        serie.astype(str)
             .str.strip()
             .str.upper()
             .str.replace(r"\s*-\s*", "-", regex=True)
             .str.replace(r"\s+", " ", regex=True)
    )


def limpiar_pk(serie: pd.Series) -> pd.Series:
    """
    Convierte el PK a float homogéneo.
    Los HTMLs producen PK como string "0.150" (desde "0,150" del título).
    Los PDFs producen PK como string "0.150" (desde "0_150" del nombre).
    Ambos ya vienen con punto decimal, pero puede haber None o strings vacíos.
    """
    return pd.to_numeric(serie, errors="coerce").round(3)


def asignar_tramo_km(pk: float, intervalo_km: float = 1.0) -> float:
    """
    Discretiza el PK en tramos de `intervalo_km` km.
    Debe ser IDÉNTICA a la función usada en phase3_accidents.py para que
    SECCION_ID coincida exactamente al hacer el JOIN.
    Ejemplo: PK=5.750 con intervalo=1.0 → tramo 5.0
    """
    if pd.isna(pk):
        return np.nan
    return float(int(pk / intervalo_km) * intervalo_km)


def detectar_imd_anomala(imd: float) -> bool:
    """
    Detecta valores de IMD físicamente imposibles o claramente erróneos.
    Umbrales basados en el rango real de carreteras de la CAM:
      · IMD <= 0     : imposible (no puede haber tráfico nulo en una carretera)
      · IMD > 300000 : por encima del récord histórico de cualquier vía española
    """
    if pd.isna(imd):
        return False   # NaN no es anómalo, es desconocido
    return imd <= 0 or imd > 300_000


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1: EXTRACCIÓN – Invocar los scripts que ya tienes
# ──────────────────────────────────────────────────────────────────────────────

def extraer_todos_los_años() -> pd.DataFrame:
    """
    Orquesta la extracción de todos los años llamando a tus scripts existentes.
    Extrae los ZIPs si no están ya extraídos y procesa HTML y PDF.
    Devuelve el DataFrame unificado RAW (sin limpiar todavía).
    """
    todos = []

    # ── Años HTML (2018-2023) ─────────────────────────────────────────────────
    for año, nombre_zip in AÑOS_HTML.items():
        ruta_zip     = RAW_DIR / nombre_zip
        carpeta_ext  = RAW_DIR / str(año)

        if not ruta_zip.exists():
            logger.warning(f"ZIP no encontrado: {ruta_zip} — se omite {año}")
            print(f"  [AVISO] No se encontró {nombre_zip}. Se omite el año {año}.")
            continue

        # Extraer ZIP solo si no está ya extraído (evita repetir trabajo)
        if not carpeta_ext.exists():
            print(f"  Extrayendo {nombre_zip}...")
            with zipfile.ZipFile(ruta_zip, "r") as z:
                z.extractall(carpeta_ext)
            logger.info(f"ZIP extraído: {nombre_zip} → {carpeta_ext}")

        # La estructura interna del ZIP es:
        # imd_trafico_20XX / Carreteras / M-XXX / GLineas_*.html
        # procesar_zip_html necesita la raíz; busca recursivamente.
        try:
            df = procesar_zip_html_recursivo(carpeta_ext, año)
            if df.empty:
                logger.warning(f"Sin datos HTML para {año}")
                continue
            df["formato_fuente"] = "html_calculado"
            todos.append(df)
            logger.info(f"HTML {año}: {len(df)} estaciones extraídas")
            print(f"  ✓ {año} (HTML): {len(df)} estaciones")
        except Exception as e:
            logger.error(f"Error procesando HTML {año}: {e}")
            print(f"  [ERROR] {año} HTML: {e}")

    # ── Años PDF (2024) ───────────────────────────────────────────────────────
    for año, nombre_zip in AÑOS_PDF.items():
        ruta_zip    = RAW_DIR / nombre_zip
        carpeta_ext = RAW_DIR / str(año)

        if not ruta_zip.exists():
            logger.warning(f"ZIP no encontrado: {ruta_zip} — se omite {año}")
            print(f"  [AVISO] No se encontró {nombre_zip}. Se omite el año {año}.")
            continue

        if not carpeta_ext.exists():
            print(f"  Extrayendo {nombre_zip}...")
            with zipfile.ZipFile(ruta_zip, "r") as z:
                z.extractall(carpeta_ext)
            logger.info(f"ZIP extraído: {nombre_zip} → {carpeta_ext}")

        try:
            df = procesar_zip_pdf_recursivo(carpeta_ext, año)
            if df.empty:
                logger.warning(f"Sin datos PDF para {año}")
                continue
            df["formato_fuente"] = "pdf_oficial"
            todos.append(df)
            logger.info(f"PDF {año}: {len(df)} estaciones extraídas")
            print(f"  ✓ {año} (PDF): {len(df)} estaciones")
        except Exception as e:
            logger.error(f"Error procesando PDF {año}: {e}")
            print(f"  [ERROR] {año} PDF: {e}")

    if not todos:
        raise RuntimeError("No se extrajo ningún dato de IMD. Revisa raw_data/02_traffic_volume/")

    df_raw = pd.concat(todos, ignore_index=True)
    logger.info(f"Extracción completa: {len(df_raw)} filas totales")
    return df_raw


def procesar_zip_html_recursivo(carpeta_raiz: Path, año: int) -> pd.DataFrame:
    """
    Versión mejorada de procesar_zip_html que recorre RECURSIVAMENTE
    todas las subcarpetas buscando archivos GLineas_*.html.

    Tu script original solo miraba una carpeta concreta; este encuentra
    los HTMLs sin importar la estructura interna del ZIP
    (imd_trafico_20XX/Carreteras/M-XXX/GLineas_*.html).
    """
    registros = []
    htmls_encontrados = list(carpeta_raiz.rglob("GLineas_*.html"))

    print(f"  Año {año}: {len(htmls_encontrados)} archivos GLineas encontrados")
    logger.info(f"HTML {año}: {len(htmls_encontrados)} archivos")

    for ruta in sorted(htmls_encontrados):
        from html_files_tfg import extraer_imd_desde_html
        try:
            datos = extraer_imd_desde_html(str(ruta))
            datos["año"]     = año
            datos["fichero"] = ruta.name
            # Guardar la carretera extraída del nombre de la CARPETA PADRE
            # como respaldo si el título del HTML devuelve DESCONOCIDA
            datos["carpeta_carretera"] = ruta.parent.name  # ej. "M-30"
            registros.append(datos)
        except Exception as e:
            logger.warning(f"  ERROR en {ruta.name}: {e}")
            print(f"    ERROR en {ruta.name}: {e}")

    return pd.DataFrame(registros)


def procesar_zip_pdf_recursivo(carpeta_raiz: Path, año: int) -> pd.DataFrame:
    """
    Versión mejorada de procesar_zip_pdf que recorre RECURSIVAMENTE
    todas las subcarpetas buscando archivos Bimestral_*.pdf.
    """
    registros = []
    pdfs_encontrados = list(carpeta_raiz.rglob("Bimestral_*.pdf"))

    print(f"  Año {año}: {len(pdfs_encontrados)} archivos PDF encontrados")
    logger.info(f"PDF {año}: {len(pdfs_encontrados)} archivos")

    for ruta in sorted(pdfs_encontrados):
        from pdf_files_tfg import extraer_imd_desde_pdf
        try:
            datos = extraer_imd_desde_pdf(str(ruta))
            datos["año"]              = año
            datos["fichero"]          = ruta.name
            datos["carpeta_carretera"] = ruta.parent.name
            registros.append(datos)
        except Exception as e:
            logger.warning(f"  ERROR en {ruta.name}: {e}")
            print(f"    ERROR en {ruta.name}: {e}")

    return pd.DataFrame(registros)


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2: PIPELINE COMPLETO DE LIMPIEZA
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_imd(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de limpieza sobre el DataFrame unificado raw.
    Cada paso documenta el requisito del TFG que satisface.

    Parámetros
    ----------
    df_raw : DataFrame producido por extraer_todos_los_años()

    Returns
    -------
    pd.DataFrame limpio y enriquecido, listo para el cruce con accidentes.
    """
    print("\n── Iniciando pipeline de limpieza IMD ──")
    df = df_raw.copy()
    n_inicial = len(df)
    logger.info(f"Pipeline IMD: {n_inicial} filas de entrada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.1 – RECUPERAR CARRETERA DESDE CARPETA CUANDO EL HTML FALLA
    #
    # Tu script html_files_tfg.py extrae la carretera del <title> del HTML
    # con la regex r'(M-\d+)\s+PK\s+([\d,]+)'. Esta regex solo captura
    # carreteras tipo "M-XXX". Para carreteras tipo A-1, A-6, etc. que
    # también están en la red, devuelve 'DESCONOCIDA'.
    # Solución: si CARRETERA == 'DESCONOCIDA', usar el nombre de la carpeta
    # padre (que siempre es el nombre correcto de la carretera).
    # ══════════════════════════════════════════════════════════════════════════
    if "carpeta_carretera" in df.columns:
        mask_desconocida = df["carretera"] == "DESCONOCIDA"
        n_recuperadas = mask_desconocida.sum()
        if n_recuperadas > 0:
            df.loc[mask_desconocida, "carretera"] = df.loc[mask_desconocida, "carpeta_carretera"]
            logger.info(f"PASO 2.1: {n_recuperadas} carreteras recuperadas desde nombre de carpeta")
            print(f"  Carreteras recuperadas desde carpeta: {n_recuperadas}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.2 – ESTANDARIZAR CARRETERA (RF-8)
    # Mismo formato que en phase3_accidents.py para garantizar el JOIN.
    # ══════════════════════════════════════════════════════════════════════════
    df["carretera"] = estandarizar_carretera(df["carretera"])
    logger.info("PASO 2.2: CARRETERA estandarizada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.3 – LIMPIAR Y CONVERTIR PK A FLOAT
    # El PK viene como string desde ambos formatos. Se convierte a float
    # y se redondea a 3 decimales (precisión de los datos originales).
    # ══════════════════════════════════════════════════════════════════════════
    df["pk"] = limpiar_pk(df["pk"])
    n_pk_nulos = df["pk"].isna().sum()
    logger.info(f"PASO 2.3: PK limpiado. Nulos: {n_pk_nulos} ({n_pk_nulos/len(df)*100:.1f}%)")
    print(f"  PK nulos tras limpieza: {n_pk_nulos} ({n_pk_nulos/len(df)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.4 – LIMPIAR IMD_TOTAL
    # Convertir a numérico (por si vienen como string desde el PDF).
    # Los PDFs a veces tienen separador de miles con punto: "85.432" → 85432.
    # ══════════════════════════════════════════════════════════════════════════
    if df["imd_total"].dtype == object:
        df["imd_total"] = (
            df["imd_total"]
            .astype(str)
            .str.replace(r"\.", "", regex=True)   # eliminar separador de miles
            .str.replace(",", ".", regex=False)    # coma decimal → punto
            .pipe(pd.to_numeric, errors="coerce")
        )
    else:
        df["imd_total"] = pd.to_numeric(df["imd_total"], errors="coerce")

    n_imd_nulos = df["imd_total"].isna().sum()
    logger.info(f"PASO 2.4: IMD_TOTAL limpiada. Nulos: {n_imd_nulos}")
    print(f"  IMD_TOTAL nulos: {n_imd_nulos}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.5 – DETECTAR Y GESTIONAR IMD ANÓMALAS
    # Valores imposibles (<=0) o extremos (>300.000): se marcan como NaN
    # y se registran en el log para trazabilidad (RNF-3, RNF-8).
    # NO se eliminan las filas: la estación puede tener datos válidos de
    # otros campos aunque la IMD sea inválida.
    # ══════════════════════════════════════════════════════════════════════════
    mask_anomala = df["imd_total"].apply(detectar_imd_anomala)
    n_anomalas = mask_anomala.sum()
    if n_anomalas > 0:
        casos = df.loc[mask_anomala, ["año", "carretera", "pk", "imd_total", "fichero"]]
        logger.warning(f"PASO 2.5: {n_anomalas} IMD anómalas → NaN:\n{casos.to_string()}")
        print(f"  [AVISO] {n_anomalas} valores de IMD anómalos (<=0 o >300.000) → NaN")
        print(casos[["año", "carretera", "pk", "imd_total"]].to_string(index=False))
        df.loc[mask_anomala, "imd_total"] = np.nan

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.6 – FLAG COVID 2020
    # El año 2020 tiene IMD artificialmente baja por el confinamiento.
    # Se añade una columna booleana para poder excluir o tratar ese año
    # en los análisis de evolución temporal (RF-15).
    # ══════════════════════════════════════════════════════════════════════════
    df["flag_covid"] = df["año"] == AÑO_COVID
    n_covid = df["flag_covid"].sum()
    logger.info(f"PASO 2.6: {n_covid} registros marcados con flag_covid=True (año 2020)")
    print(f"  Registros marcados como COVID-2020: {n_covid}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.7 – DISCRETIZAR PK EN TRAMO DE 1 KM → SECCION_ID
    # Genera la MISMA clave de tramo que usa phase3_accidents.py.
    # Este es el campo de unión entre IMD y accidentes en la Fase 4.
    # ══════════════════════════════════════════════════════════════════════════
    df["tramo_km_ini"] = df["pk"].apply(lambda x: asignar_tramo_km(x, intervalo_km=1.0))

    df["seccion_id"] = df.apply(
        lambda r: (
            f"{r['carretera']}_{int(r['tramo_km_ini'])}"
            if pd.notna(r["tramo_km_ini"])
            else np.nan
        ),
        axis=1,
    )
    n_sin_seccion = df["seccion_id"].isna().sum()
    logger.info(f"PASO 2.7: SECCION_ID creada. Sin sección: {n_sin_seccion}")
    print(f"  Registros sin SECCION_ID (PK nulo): {n_sin_seccion}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.8 – ELIMINAR DUPLICADOS POR (AÑO, CARRETERA, PK)
    # Puede haber duplicados si un mismo PK tiene dos estaciones de aforo
    # o si el ZIP contiene el fichero dos veces (ocurre en algunos años).
    # Estrategia: quedarse con la fila de mayor IMD (más fiable por
    # ser la medición con más datos) y registrar cuántos se eliminaron.
    # ══════════════════════════════════════════════════════════════════════════
    n_antes = len(df)
    df = (
        df.sort_values("imd_total", ascending=False)
          .drop_duplicates(subset=["año", "carretera", "pk"], keep="first")
          .sort_values(["año", "carretera", "pk"])
          .reset_index(drop=True)
    )
    n_duplicados = n_antes - len(df)
    logger.info(f"PASO 2.8: {n_duplicados} duplicados eliminados (año+carretera+pk)")
    print(f"  Duplicados eliminados: {n_duplicados}")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.9 – LIMPIAR PCT_PESADOS (solo en datos PDF/2024)
    # El porcentaje de vehículos pesados viene del PDF de 2024.
    # Se convierte a float y se valida que esté en [0, 100].
    # Para años HTML que no tienen este dato, se rellena con NaN.
    # ══════════════════════════════════════════════════════════════════════════
    if "pct_pesados" in df.columns:
        df["pct_pesados"] = pd.to_numeric(df["pct_pesados"], errors="coerce")
        # Marcar como NaN los porcentajes fuera de rango
        mask_pct_invalida = df["pct_pesados"].notna() & (
            (df["pct_pesados"] < 0) | (df["pct_pesados"] > 100)
        )
        if mask_pct_invalida.sum() > 0:
            logger.warning(f"PASO 2.9: {mask_pct_invalida.sum()} valores de pct_pesados fuera de [0,100] → NaN")
            df.loc[mask_pct_invalida, "pct_pesados"] = np.nan
    else:
        df["pct_pesados"] = np.nan

    logger.info("PASO 2.9: pct_pesados limpiado")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.10 – VALIDACIÓN TEMPORAL (RNF-10)
    # Comprueba que están todos los años esperados y reporta cobertura.
    # ══════════════════════════════════════════════════════════════════════════
    años_esperados = set(list(AÑOS_HTML.keys()) + list(AÑOS_PDF.keys()))
    años_presentes = set(df["año"].unique())
    años_faltantes = años_esperados - años_presentes
    if años_faltantes:
        logger.warning(f"PASO 2.10 – Años sin datos IMD: {sorted(años_faltantes)}")
        print(f"  [AVISO] Años sin datos IMD: {sorted(años_faltantes)}")
    else:
        logger.info("PASO 2.10 – Todos los años IMD presentes ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2.11 – SELECCIÓN Y ORDEN DE COLUMNAS FINALES
    # ══════════════════════════════════════════════════════════════════════════
    COLUMNAS_FINALES = [
        # Identificación temporal
        "año",
        # Localización (clave de JOIN con accidentes)
        "carretera", "pk", "tramo_km_ini", "seccion_id",
        # IMD principal
        "imd_total",
        # IMD por tipo de día (solo en datos HTML; NaN en PDF)
        "imd_laborable_media", "imd_sabado", "imd_domingo",
        "imd_lunes", "imd_martes", "imd_miercoles",
        "imd_jueves", "imd_viernes",
        # Vehículos pesados (solo en datos PDF/2024; NaN en HTML)
        "pct_pesados",
        # Trazabilidad (RNF-3)
        "formato_fuente", "estacion", "fichero",
        # Flags de calidad
        "flag_covid",
    ]

    cols_existentes = [c for c in COLUMNAS_FINALES if c in df.columns]
    df_final = df[cols_existentes].copy()

    n_final = len(df_final)
    logger.info(
        f"Pipeline IMD completado: {n_inicial} → {n_final} registros "
        f"({len(cols_existentes)} columnas)"
    )
    print(f"  Registros tras limpieza: {n_final:,} ({len(cols_existentes)} columnas)")

    return df_final


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: RESUMEN DE CALIDAD
# ──────────────────────────────────────────────────────────────────────────────

def imprimir_resumen_calidad(df: pd.DataFrame) -> None:
    """
    Imprime resumen de calidad del dataset IMD limpio.
    Incluir en la memoria como evidencia del proceso de limpieza.
    """
    print("\n" + "═" * 65)
    print("RESUMEN DE CALIDAD – DATASET IMD LIMPIO")
    print("═" * 65)

    print(f"\n{'Total registros (estación-año)':<40} {len(df):>10,}")
    print(f"{'Columnas':<40} {len(df.columns):>10}")

    print("\n── Registros por año y formato ──")
    resumen_año = df.groupby(["año", "formato_fuente"]).agg(
        n_estaciones=("seccion_id", "count"),
        imd_media=("imd_total", "mean"),
        imd_min=("imd_total", "min"),
        imd_max=("imd_total", "max"),
    ).round(0)
    print(resumen_año.to_string())

    print("\n── Nulos en columnas clave ──")
    cols_check = ["carretera", "pk", "seccion_id", "imd_total",
                  "pct_pesados", "imd_laborable_media"]
    nulos = df[[c for c in cols_check if c in df.columns]].isnull().sum()
    print(nulos.to_string())

    print("\n── Carreteras más medidas (top 15) ──")
    print(df["carretera"].value_counts().head(15).to_string())

    print("\n── Estadísticas de IMD por año ──")
    print(df.groupby("año")["imd_total"].describe().round(0).to_string())

    if "flag_covid" in df.columns:
        n_covid = df["flag_covid"].sum()
        print(f"\n── Registros con flag COVID-2020: {n_covid} ──")

    print("═" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FASE 3 – PREPROCESAMIENTO Y LIMPIEZA: IMD TRÁFICO")
    print("=" * 65)

    # ── 1. Extracción ─────────────────────────────────────────────────────────
    print("\n[1/4] Extrayendo datos de ZIPs (HTML + PDF)...")
    try:
        df_raw = extraer_todos_los_años()
        print(f"  Total extraído (raw): {len(df_raw):,} registros")
    except RuntimeError as e:
        print(f"\n[ERROR CRÍTICO] {e}")
        logger.error(str(e))
        sys.exit(1)

    # Guardar CSV intermedio raw para depuración (no es el output final)
    ruta_raw = PROCESSED_DIR / "imd_raw_sin_limpiar.csv"
    df_raw.to_csv(ruta_raw, index=False, encoding="utf-8-sig")
    logger.info(f"CSV raw guardado: {ruta_raw}")

    # ── 2. Limpieza ───────────────────────────────────────────────────────────
    print("\n[2/4] Aplicando pipeline de limpieza...")
    df_limpio = limpiar_imd(df_raw)

    # ── 3. Guardar ────────────────────────────────────────────────────────────
    print("\n[3/4] Guardando resultados...")

    # Parquet: archivo principal para Fase 4
    ruta_parquet = PROCESSED_DIR / "imd_clean.parquet"
    df_limpio.to_parquet(ruta_parquet, index=False)
    print(f"  ✓ Parquet guardado: {ruta_parquet}")
    logger.info(f"Guardado: {ruta_parquet}")

    # CSV: copia legible para inspección manual
    ruta_csv = PROCESSED_DIR / "imd_clean.csv"
    df_limpio.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSV guardado:    {ruta_csv}")
    logger.info(f"Guardado: {ruta_csv}")

    # CSVs de detalle por año (equivalente a lo que hacía imd_unification_html_pdf.py)
    for año, grupo in df_limpio.groupby("año"):
        ruta_detalle = PROCESSED_DIR / f"imd_{año}_clean.csv"
        grupo.to_csv(ruta_detalle, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSVs por año guardados en {PROCESSED_DIR}/")
    logger.info("CSVs por año guardados")

    # ── 4. Resumen ────────────────────────────────────────────────────────────
    print("\n[4/4] Resumen de calidad:")
    imprimir_resumen_calidad(df_limpio)

    print("Fase 3 (IMD) completada. Consulta logs/phase3_imd.log")
    logger.info("=== FASE 3 IMD COMPLETADA ===")


if __name__ == "__main__":
    main()