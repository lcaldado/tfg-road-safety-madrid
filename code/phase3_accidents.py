"""
================================================================================
FASE 3 – DATA PREPROCESSING & CLEANING
Archivo : phase3_accidents.py
Proyecto   : TFG – Seguridad Vial Comunidad de Madrid
Autor   : Lucía Caldado
Desc.   : Carga, limpia e integra los microdatos de accidentes DGT
          (TABLA_ACCIDENTES_XX.xlsx, 2016-2024) filtrando por la
          Comunidad de Madrid (COD_PROVINCIA == 28).

NOTAS SOBRE LOS DATOS (documentar en la memoria):
  · KM nulo (≈80 %): corresponde casi íntegramente a vías "No inventariada"
    urbanas (ZONA_AGRUPADA = 2). Esto es esperado y correcto: los accidentes
    urbanos no tienen PK asignado. El análisis de tramos por PK se aplicará
    solo a vías interurbanas inventariadas (ZONA_AGRUPADA = 1).
  · Año 2020: la caída en accidentes se debe al confinamiento por COVID-19,
    no a una mejora estructural de la seguridad vial.
  · Código 999 en variables categóricas: valor DGT para "desconocido/no
    aplicable". Se mapea a NaN para no contaminar análisis estadísticos.
  · COD_MUNICIPIO en Madrid viene como float (ej. 28054.0) con el código
    completo INE provincia+municipio. Se convierte a entero string "28054".
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN: RUTAS, AÑOS Y LOGGING
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent   # raíz del proyecto
RAW_DIR       = BASE_DIR / "raw_data" / "01_accidents"
PROCESSED_DIR = BASE_DIR / "processed_data"
LOGS_DIR      = BASE_DIR / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_DIR / "phase3_accidents.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rango de años del estudio
YEARS = range(2016, 2025)   # 2016 → 2024 inclusive

# Código INE de la Comunidad de Madrid
COD_MADRID = 28

# ──────────────────────────────────────────────────────────────────────────────
# DICCIONARIOS DE CODIFICACIÓN DGT
# Fuente: hoja "Descripcion fichero" incluida en cada Excel de microdatos.
# Se usan para:
#   a) Mapear códigos numéricos a etiquetas legibles.
#   b) Convertir el código 999 ("desconocido") a NaN.
# ──────────────────────────────────────────────────────────────────────────────

# TITULARIDAD_VIA
TITULARIDAD = {
    1: "Estado",
    2: "Comunidad Autónoma",
    3: "Provincial, Diputación / Cabildo",
    4: "Municipal",
    5: "Otra",
}

# TIPO_VIA
TIPO_VIA = {
    1:  "Autopista de peaje",
    2:  "Autopista libre",
    3:  "Autovía",
    4:  "Vía para automóviles",
    5:  "Carretera Convencional de doble calzada",
    6:  "Carretera Convencional de calzada única",
    7:  "Vía de servicio",
    8:  "Ramal de enlace",
    9:  "Calle",
    10: "Camino vecinal",
    11: "Recinto delimitado",
    12: "Vía ciclista",
    13: "Senda ciclable",
    14: "Otro",
}

# TIPO_ACCIDENTE
TIPO_ACCIDENTE = {
    1:  "Colisión frontal",
    2:  "Colisión fronto-lateral",
    3:  "Colisión lateral",
    4:  "Colisión por alcance",
    5:  "Colisión múltiple",
    6:  "Colisión contra obstáculo o elemento de la vía",
    7:  "Atropello a personas",
    8:  "Atropello a animales",
    9:  "Vuelco",
    10: "Caída",
    11: "Sólo salida de la vía",
    12: "Salida de la vía por la izquierda con colisión",
    13: "Salida de la vía por la izquierda con despeñamiento",
    14: "Salida de la vía por la izquierda con vuelco",
    15: "Salida de la vía por la izquierda, otro tipo",
    16: "Salida de la vía por la derecha con colisión",
    17: "Salida de la vía por la derecha con despeñamiento",
    18: "Salida de la vía por la derecha con vuelco",
    19: "Salida de la vía por la derecha otro tipo",
    20: "Otro tipo de accidente",
}

# ZONA_AGRUPADA
ZONA_AGRUPADA = {
    1: "Interurbana",
    2: "Urbana",
}

# CONDICION_FIRME  (estado del pavimento → clave para RF-12)
CONDICION_FIRME = {
    1: "Seco y limpio",
    2: "Con barro o gravilla suelta",
    3: "Mojado",
    4: "Muy encharcado o inundado",
    5: "Con hielo",
    6: "Con nieve",
    7: "Con aceite",
    8: "Otro",
    9: "Se desconoce",
    999: np.nan,   # → se reemplaza por NaN
}

# CONDICION_METEO
CONDICION_METEO = {
    1:   "Despejado",
    2:   "Nublado",
    3:   "Lluvia débil",
    4:   "Lluvia fuerte",
    5:   "Granizando",
    6:   "Nevando",
    7:   "Se desconoce",
    999: np.nan,
}

# CONDICION_ILUMINACION
CONDICION_ILUMINACION = {
    1: " Luz del día natural, solar",
    2: "Amanecer o atardecer, sin luz artificial",
    3: "Amanecer o atardecer, con luz artificial",
    4: "Sin luz natural y iluminación artificial encendida",
    5: "Sin luz natural y iluminación artificial no encendida",
    6: "Sin luz natural ni artificial",
    999: np.nan,
}

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ──────────────────────────────────────────────────────────────────────────────

def estandarizar_carretera(serie: pd.Series) -> pd.Series:
    """
    Normaliza el nombre de la carretera para homogeneizar entre años:
      · Mayúsculas
      · Sin espacios internos ni externos
      · Sin guiones dobles
    Ejemplo: "m - 30" → "M-30", "A 1" → "A-1"
    """
    return (
        serie.astype(str)
             .str.strip()
             .str.upper()
             .str.replace(r"\s*-\s*", "-", regex=True)   # normaliza "M - 30" → "M-30"
             .str.replace(r"\s+", " ", regex=True)        # espacios múltiples → uno
    )


def asignar_tramo_km(km: float, intervalo_km: float = 1.0) -> float | None:
    """
    Discretiza el punto kilométrico en tramos de longitud `intervalo_km`.
    Devuelve el inicio del tramo (ej. KM=5.7 con intervalo=1 → tramo 5.0).
    Devuelve NaN si el KM es nulo.
    """
    if pd.isna(km):
        return np.nan
    return float(int(km / intervalo_km) * intervalo_km)


def clasificar_firme_malo(condicion: str | float) -> int:
    """
    Variable binaria para RF-12 (Índice de influencia del estado de la vía).
    Devuelve 1 si el firme estaba en mal estado, 0 si estaba en buen estado,
    NaN si se desconoce.
    MAL ESTADO: Agua, Barro, Nieve/Hielo, Aceite, Gravilla suelta, Obras, Otro.
    """
    MAL_ESTADO = {
        "Agua", "Barro / Tierra", "Nieve / Hielo",
        "Aceite / Combustible", "Gravilla suelta", "Obras", "Otro",
    }
    if pd.isna(condicion) or condicion == "Se desconoce":
        return np.nan
    return 1 if condicion in MAL_ESTADO else 0


def clasificar_severidad(row: pd.Series) -> str:
    """
    Clasifica el accidente en tres niveles de severidad según víctimas a 30 días
    (criterio internacionalmente aceptado para comparación):
      - MORTAL   : al menos 1 fallecido
      - GRAVE    : al menos 1 herido grave (y ningún fallecido)
      - LEVE     : solo heridos leves
    """
    if row["TOTAL_MU30DF"] > 0:
        return "Mortal"
    elif row["TOTAL_HG30DF"] > 0:
        return "Grave"
    else:
        return "Leve"


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE CARGA Y LIMPIEZA POR AÑO
# ──────────────────────────────────────────────────────────────────────────────

def cargar_año(year: int) -> pd.DataFrame | None:
    """
    Carga el Excel de accidentes de un año, filtra por Madrid y aplica
    limpieza básica. Devuelve un DataFrame limpio o None si hay error.

    Parámetros
    ----------
    year : int  Año completo (ej. 2016)

    Returns
    -------
    pd.DataFrame con los accidentes de Madrid de ese año, ya limpios.
    """
    suffix = str(year)[-2:]                                     # "2016" → "16"
    filename = f"TABLA_ACCIDENTES_{suffix}.xlsx"
    filepath = RAW_DIR / filename
    sheet    = f"ACCIDENTES_{suffix}"

    logger.info(f"Cargando {filename} ...")

    # ── Verificar existencia del archivo ──────────────────────────────────────
    if not filepath.exists():
        logger.error(f"ARCHIVO NO ENCONTRADO: {filepath}")
        print(f"  [ERROR] No se encontró {filename}. Se omite el año {year}.")
        return None

    # ── Leer Excel (RF-1, RF-5) ───────────────────────────────────────────────
    try:
        df = pd.read_excel(filepath, sheet_name=sheet)
        logger.info(f"  Cargadas {len(df):,} filas de {filename}")
    except Exception as e:
        logger.error(f"Error leyendo {filename}: {e}")
        print(f"  [ERROR] No se pudo leer {filename}: {e}")
        return None

    # ── Filtrar Comunidad de Madrid (COD_PROVINCIA == 28) ────────────────────
    df = df[df["COD_PROVINCIA"] == COD_MADRID].copy()
    logger.info(f"  Tras filtro Madrid: {len(df):,} filas")

    if df.empty:
        logger.warning(f"  Sin registros de Madrid en {filename}")
        return None

    # ── Añadir columna de año (trazabilidad) ──────────────────────────────────
    df["YEAR"] = year

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO DE LIMPIEZA
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_accidentes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas las transformaciones de limpieza sobre el DataFrame combinado.
    Cada paso está documentado indicando el requisito del proyecto que satisface.

    Parámetros
    ----------
    df : DataFrame combinado con accidentes de todos los años (ya filtrado Madrid)

    Returns
    -------
    pd.DataFrame limpio y enriquecido, listo para calcular indicadores (Fase 4).
    """

    print("\n── Iniciando pipeline de limpieza ──")
    n_inicial = len(df)
    logger.info(f"Pipeline limpieza: {n_inicial:,} registros de entrada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 1 – RENOMBRAR COLUMNA ID
    # El Excel de 2016 usa "SECUENCIAL" como ID del accidente.
    # Se estandariza a ID_ACCIDENTE para consistencia con años posteriores.
    # ══════════════════════════════════════════════════════════════════════════
    if "SECUENCIAL" in df.columns and "ID_ACCIDENTE" not in df.columns:
        df = df.rename(columns={"SECUENCIAL": "ID_ACCIDENTE"})

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 2 – ESTANDARIZAR NOMBRE DE CARRETERA (RF-8)
    # Homogeneiza el campo CARRETERA entre años (mayúsculas, sin espacios).
    # ══════════════════════════════════════════════════════════════════════════
    df["CARRETERA"] = estandarizar_carretera(df["CARRETERA"])
    logger.info("PASO 2 completado: CARRETERA estandarizada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 3 – ESTANDARIZAR COD_MUNICIPIO
    # En Madrid viene como float con el código INE completo (ej. 28054.0).
    # Se convierte a string de 5 dígitos ("28054") para JOIN con otras fuentes.
    # ══════════════════════════════════════════════════════════════════════════
    df["COD_MUNICIPIO"] = (
        df["COD_MUNICIPIO"]
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(5)
        .reindex(df.index)
    )
    logger.info("PASO 3 completado: COD_MUNICIPIO estandarizado")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 4 – TRATAR VALORES 999 (código DGT "desconocido") → NaN (RF-7)
    # 999 aparece en CONDICION_METEO y puntualmente en otras variables.
    # Se convierte a NaN para no sesgar estadísticas.
    # ══════════════════════════════════════════════════════════════════════════
    cols_categoricas_con_999 = [
        "CONDICION_FIRME", "CONDICION_METEO", "CONDICION_ILUMINACION", "TITULARIDAD_VIA",
        "NUDO_INFO", "PRIORI_NORMA", "PRIORI_AGENTE", "PRIORI_SEMAFORO", "PRIORI_VERT_STOP",
        "PRIORI_VERT_CEDA", "PRIORI_HORIZ_STOP", "PRIORI_HORIZ_CEDA", "PRIORI_MARCAS", "PRIORI_PEA_NO_ELEV",
        "PRIORI_PEA_ELEV", "PRIORI_MARCA_CICLOS", "PRIORI_CIRCUNSTANCIAL", "PRIORI_OTRA", "CONDICION_NIVEL_CIRCULA",
        "VISIB_RESTRINGIDA_POR", "ACERA", "TRAZADO_PLANTA",
    ]
    for col in cols_categoricas_con_999:
        if col in df.columns:
            antes = (df[col] == 999).sum()
            df[col] = df[col].replace(999, np.nan)
            if antes > 0:
                logger.info(f"  {col}: {antes} valores 999 → NaN")

    logger.info("PASO 4 completado: valores 999 convertidos a NaN")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 5 – MAPEAR CÓDIGOS NUMÉRICOS A ETIQUETAS (legibilidad y trazabilidad)
    # Solo se mapean las columnas relevantes para los indicadores del TFG.
    # Se crean columnas nuevas con sufijo _LABEL para no perder el código original.
    # ══════════════════════════════════════════════════════════════════════════
    mapeos = {
        "TITULARIDAD_VIA"      : TITULARIDAD,
        "TIPO_VIA"             : TIPO_VIA,
        "TIPO_ACCIDENTE"       : TIPO_ACCIDENTE,
        "ZONA_AGRUPADA"        : ZONA_AGRUPADA,
        "CONDICION_FIRME"      : CONDICION_FIRME,
        "CONDICION_METEO"      : CONDICION_METEO,
        "CONDICION_ILUMINACION": CONDICION_ILUMINACION,
    }
    for col, mapeo in mapeos.items():
        if col in df.columns:
            df[f"{col}_LABEL"] = df[col].map(mapeo)

    logger.info("PASO 5 completado: columnas _LABEL creadas")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 6 – GESTIÓN DE NULOS EN COLUMNAS CRÍTICAS (RF-7)
    #
    # KM: es nulo en accidentes urbanos ("No inventariada", ZONA_AGRUPADA=2).
    # Esto es CORRECTO y esperado:
    #   · Vías urbanas (ZONA_AGRUPADA=2) → KM permanece NaN (no se usa PK)
    #   · Vías interurbanas sin KM (ZONA_AGRUPADA=1) → casos anómalos reales
    # Se documenta pero NO se eliminan los registros: los accidentes urbanos
    # son válidos y se usan en indicadores a nivel municipio.
    #
    # Columnas de víctimas: si son NaN (no esperado por la DGT pero defensivo)
    # se imputan a 0 y se registra en el log.
    # ══════════════════════════════════════════════════════════════════════════

    # Documentar nulos en KM por zona
    interurb_sin_km = (
        (df["ZONA_AGRUPADA"] == 1) & df["KM"].isna()
    ).sum()
    urban_sin_km = (
        (df["ZONA_AGRUPADA"] == 2) & df["KM"].isna()
    ).sum()
    logger.info(
        f"PASO 6 – KM: {interurb_sin_km} interurbanos sin PK (anómalos), "
        f"{urban_sin_km} urbanos sin PK (esperado)"
    )

    # Imputar nulos en columnas de víctimas (comportamiento defensivo)
    cols_victimas = [
        "TOTAL_MU24H", "TOTAL_HG24H", "TOTAL_HL24H", "TOTAL_VICTIMAS_24H",
        "TOTAL_MU30DF", "TOTAL_HG30DF", "TOTAL_HL30DF", "TOTAL_VICTIMAS_30DF",
        "TOTAL_VEHICULOS",
    ]
    for col in cols_victimas:
        if col in df.columns:
            n_nulos = df[col].isna().sum()
            if n_nulos > 0:
                df[col] = df[col].fillna(0).astype(int)
                logger.warning(f"  {col}: {n_nulos} nulos imputados a 0")

    logger.info("PASO 6 completado: nulos gestionados")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 7 – COLUMNA SEVERIDAD (para RF-14: Índice de Severidad)
    # Clasifica cada accidente en Mortal / Grave / Leve según víctimas a 30 días
    # (el criterio de 30 días es el estándar internacional, más fiable que 24h).
    # ══════════════════════════════════════════════════════════════════════════
    df["SEVERIDAD"] = df.apply(clasificar_severidad, axis=1)
    logger.info("PASO 7 completado: columna SEVERIDAD creada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 8 – VARIABLE FIRME_MALO (para RF-12: Índice de condición de vía)
    # 1 = firme en mal estado, 0 = buen estado, NaN = desconocido.
    # Se crea a partir de la columna etiqueta ya mapeada en PASO 5.
    # ══════════════════════════════════════════════════════════════════════════
    if "CONDICION_FIRME_LABEL" in df.columns:
        df["FIRME_MALO"] = df["CONDICION_FIRME_LABEL"].apply(clasificar_firme_malo)
    logger.info("PASO 8 completado: columna FIRME_MALO creada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 9 – DISCRETIZAR KM EN TRAMOS DE 1 KM (RF-6, RF-11)
    # Permite agrupar accidentes por tramo para calcular el Índice de
    # Concentración de Accidentes (ICA). Solo se aplica a vías interurbanas.
    # SECCION_ID es la clave de cruce con los datos de IMD.
    # ══════════════════════════════════════════════════════════════════════════
    df["TRAMO_KM_INI"] = df["KM"].apply(lambda x: asignar_tramo_km(x, intervalo_km=1.0))

    # Identificador único de tramo: "M-30_5" → carretera M-30, tramo km 5-6
    df["SECCION_ID"] = df.apply(
        lambda r: (
            f"{r['CARRETERA']}_{int(r['TRAMO_KM_INI'])}"
            if pd.notna(r["TRAMO_KM_INI"])
            else np.nan
        ),
        axis=1,
    )
    logger.info("PASO 9 completado: TRAMO_KM_INI y SECCION_ID creados")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 10 – CONSTRUIR FECHA (RF para temporalidad)
    # La DGT no incluye una columna FECHA directa; hay ANYO, MES y DIA_SEMANA.
    # DIA_SEMANA es 1=Lunes … 7=Domingo (no el día del mes), así que solo
    # construimos año-mes para análisis de evolución temporal (RF-15).
    # ══════════════════════════════════════════════════════════════════════════
    df["PERIODO"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2),
        format="%Y-%m",
    )
    logger.info("PASO 10 completado: columna PERIODO (año-mes) creada")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 11 – VALIDACIÓN TEMPORAL (RNF-10)
    # Verifica que cada año del estudio tiene datos y alerta si alguno falta.
    # ══════════════════════════════════════════════════════════════════════════
    años_presentes = set(df["YEAR"].unique())
    años_esperados = set(YEARS)
    años_faltantes = años_esperados - años_presentes
    if años_faltantes:
        logger.warning(f"VALIDACIÓN TEMPORAL: años sin datos → {sorted(años_faltantes)}")
        print(f"  [AVISO] Años sin datos: {sorted(años_faltantes)}")
    else:
        logger.info("VALIDACIÓN TEMPORAL: todos los años presentes ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # PASO 12 – SELECCIÓN Y ORDEN DE COLUMNAS FINALES
    # Se retienen solo las columnas necesarias para la Fase 4 (indicadores)
    # y la Fase 5 (visualizador). Se preserva trazabilidad (RNF-3).
    # ══════════════════════════════════════════════════════════════════════════
    COLUMNAS_FINALES = [
        # Identificación
        "ID_ACCIDENTE", "YEAR", "MES", "HORA", "PERIODO",
        # Localización
        "COD_PROVINCIA", "COD_MUNICIPIO", "CARRETERA",
        "KM", "TRAMO_KM_INI", "SECCION_ID",
        "ZONA_AGRUPADA", "ZONA_AGRUPADA_LABEL",
        # Tipo de vía
        "TITULARIDAD_VIA", "TITULARIDAD_VIA_LABEL",
        "TIPO_VIA", "TIPO_VIA_LABEL",
        # Tipo de accidente
        "TIPO_ACCIDENTE", "TIPO_ACCIDENTE_LABEL",
        # Víctimas (criterio 30 días = estándar internacional)
        "TOTAL_MU30DF", "TOTAL_HG30DF", "TOTAL_HL30DF", "TOTAL_VICTIMAS_30DF",
        # Víctimas (criterio 24h = disponible para todos los años)
        "TOTAL_MU24H", "TOTAL_HG24H", "TOTAL_HL24H", "TOTAL_VICTIMAS_24H",
        # Vehículos
        "TOTAL_VEHICULOS",
        # Severidad calculada
        "SEVERIDAD",
        # Condiciones (para RF-12)
        "CONDICION_FIRME", "CONDICION_FIRME_LABEL", "FIRME_MALO",
        "CONDICION_METEO", "CONDICION_METEO_LABEL",
        "CONDICION_ILUMINACION", "CONDICION_ILUMINACION_LABEL",
    ]

    # Retener solo columnas que existen (robustez ante variaciones entre años)
    cols_existentes = [c for c in COLUMNAS_FINALES if c in df.columns]
    df_final = df[cols_existentes].copy()

    n_final = len(df_final)
    logger.info(
        f"Pipeline limpieza completado: {n_inicial:,} → {n_final:,} registros "
        f"({n_final/n_inicial*100:.1f}% retenidos), {len(cols_existentes)} columnas"
    )
    print(f"  Registros tras limpieza: {n_final:,} ({len(cols_existentes)} columnas)")

    return df_final


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: RESUMEN DE CALIDAD
# ──────────────────────────────────────────────────────────────────────────────

def imprimir_resumen_calidad(df: pd.DataFrame) -> None:
    """
    Imprime en consola un resumen de calidad del dataset limpio.
    Útil para incluir en la memoria (Fase 7) como evidencia de la limpieza.
    """
    print("\n" + "═" * 60)
    print("RESUMEN DE CALIDAD – DATASET ACCIDENTES LIMPIO")
    print("═" * 60)

    print(f"\n{'Total registros':<35} {len(df):>10,}")
    print(f"{'Columnas':<35} {len(df.columns):>10}")

    print("\n── Distribución por año ──")
    print(df.groupby("YEAR").size().rename("N_accidentes").to_string())

    print("\n── Zona (interurbana vs urbana) ──")
    print(df["ZONA_AGRUPADA_LABEL"].value_counts().to_string())

    print("\n── Severidad ──")
    print(df["SEVERIDAD"].value_counts().to_string())

    print("\n── Nulos en columnas clave ──")
    cols_check = ["CARRETERA", "KM", "SECCION_ID", "COD_MUNICIPIO",
                  "CONDICION_FIRME_LABEL", "CONDICION_METEO_LABEL"]
    nulos = df[[c for c in cols_check if c in df.columns]].isnull().sum()
    print(nulos.to_string())

    interurb = df[df["ZONA_AGRUPADA"] == 1]
    pct_km = interurb["KM"].notna().mean() * 100
    print(f"\n── KM disponible en vías interurbanas: {pct_km:.1f}% ──")

    print("\n── Top 10 carreteras por número de accidentes (interurbanas) ──")
    top_carr = (
        interurb[interurb["CARRETERA"] != "NO INVENTARIADA"]
        .groupby("CARRETERA")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(top_carr.to_string())

    print("\n── Firme en mal estado ──")
    if "FIRME_MALO" in df.columns:
        n_malo  = df["FIRME_MALO"].eq(1).sum()
        n_bueno = df["FIRME_MALO"].eq(0).sum()
        n_desc  = df["FIRME_MALO"].isna().sum()
        print(f"  Mal estado : {n_malo:,} ({n_malo/len(df)*100:.1f}%)")
        print(f"  Buen estado: {n_bueno:,} ({n_bueno/len(df)*100:.1f}%)")
        print(f"  Desconocido: {n_desc:,} ({n_desc/len(df)*100:.1f}%)")

    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FASE 3 – PREPROCESAMIENTO Y LIMPIEZA: ACCIDENTES")
    print("=" * 60)

    # ── 1. Cargar todos los años ───────────────────────────────────────────────
    print("\n[1/4] Cargando archivos Excel por año...")
    dfs_por_año = []
    for year in YEARS:
        df_año = cargar_año(year)
        if df_año is not None:
            dfs_por_año.append(df_año)
            print(f"  ✓ {year}: {len(df_año):,} accidentes en Madrid")
        else:
            print(f"  ✗ {year}: sin datos (ver log)")

    if not dfs_por_año:
        logger.error("No se cargó ningún año. Abortando.")
        print("\n[ERROR] No se cargó ningún archivo. Revisa la carpeta raw_data/01_accidents/")
        return

    # ── 2. Combinar ───────────────────────────────────────────────────────────
    print(f"\n[2/4] Combinando {len(dfs_por_año)} años...")
    df_combinado = pd.concat(dfs_por_año, ignore_index=True)
    print(f"  Total combinado: {len(df_combinado):,} registros")
    logger.info(f"DataFrame combinado: {len(df_combinado):,} registros")

    # ── 3. Limpiar ────────────────────────────────────────────────────────────
    print("\n[3/4] Aplicando pipeline de limpieza...")
    df_limpio = limpiar_accidentes(df_combinado)

    # ── 4. Guardar ────────────────────────────────────────────────────────────
    print("\n[4/4] Guardando resultados...")

    # Parquet: formato principal (eficiente, mantiene tipos)
    ruta_parquet = PROCESSED_DIR / "accidents_clean.parquet"
    df_limpio.to_parquet(ruta_parquet, index=False)
    print(f"  ✓ Parquet guardado: {ruta_parquet}")
    logger.info(f"Guardado: {ruta_parquet}")

    # CSV: copia legible para inspección manual
    ruta_csv = PROCESSED_DIR / "accidents_clean.csv"
    df_limpio.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    print(f"  ✓ CSV guardado:    {ruta_csv}")
    logger.info(f"Guardado: {ruta_csv}")

    # ── 5. Resumen de calidad ─────────────────────────────────────────────────
    imprimir_resumen_calidad(df_limpio)

    print("Fase 3 (accidentes) completada. Consulta logs/phase3_accidents.log")
    logger.info("=== FASE 3 ACCIDENTES COMPLETADA ===")


if __name__ == "__main__":
    main()