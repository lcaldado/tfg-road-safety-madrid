"""
=============================================================================
TFG – Road Safety in the Community of Madrid
Phase 4: Calculation of Indicators
=============================================================================
Author  : Lucia Caldado
Date    : 2026
Purpose : Compute all six safety indicators from cleaned accident and IMD
          data, then export them as CSV/Parquet files and a unified
          GeoDataFrame (GeoPackage) ready for Phase 5 (Interactive Viewer).

Input files (cleaned, from Phase 3):
  - accidents_clean.csv  : historical accident records
  - imd_clean.csv        : Average Daily Traffic by road section and year
  - red_viaria.gpkg      : road-network geometry (layer: rt_tramo_vial)

Output files (written to ./outputs/):
  - ind1_accident_concentration.csv / .parquet
  - ind2_road_condition.csv / .parquet
  - ind3_alcohol_drugs.csv / .parquet
  - ind4_severity_index.csv / .parquet
  - ind5_temporal_evolution.csv / .parquet
  - ind6_cause_distribution.csv / .parquet
  - indicators_unified.gpkg   (GeoDataFrame joining all indicators to roads)

Notes on data constraints:
  - No explicit alcohol/drug cause column exists in the dataset.  The proxy
    used is CONDICION_FIRME_LABEL == "Otro" (encoded as FIRME_MALO == 1),
    which the source data uses as an umbrella code that includes substance-
    related and other undetermined causes.  Wherever richer cause data
    becomes available it can be substituted for FIRME_MALO without changing
    the structure of indicator 3.
  - Alcohol/drug proxy is clearly flagged in all output column names and
    comments so downstream consumers understand the limitation.
  - FIRME_MALO == 1  ↔  CONDICION_FIRME_LABEL == "Otro"  (705 records)
  - Bad road surface is defined as conditions other than "Seco y limpio",
    "Mojado", and "Se desconoce" (i.e., genuinely degraded surfaces).
=============================================================================
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# ---------------------------------------------------------------------------
# 0. Setup – logging, paths, output directory
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "processed_data"
OUTPUT_DIR = BASE_DIR / "indicators"
LOGS_DIR   = BASE_DIR / "logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = LOGS_DIR / "phase4.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

ACCIDENTS_PATH = os.path.join(DATA_DIR, "accidents_clean.csv")
IMD_PATH       = os.path.join(DATA_DIR, "imd_clean.csv")
GPKG_PATH      = os.path.join(DATA_DIR, "red_viaria.gpkg")
ROAD_LAYER     = "rt_tramo_vial"

# ---------------------------------------------------------------------------
# Helper: timestamped export with error logging
# ---------------------------------------------------------------------------

def export(df: pd.DataFrame, name: str, formats=("csv", "parquet")) -> None:
    """Write *df* to OUTPUT_DIR in the requested formats."""
    for fmt in formats:
        path = os.path.join(OUTPUT_DIR, f"{name}.{fmt}")
        try:
            if fmt == "csv":
                df.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "parquet":
                df.to_parquet(path, index=False)
            log.info("Exported %s → %s", name, path)
        except Exception as exc:
            log.error(
                "Export failed | file=%s | format=%s | ts=%s | error=%s",
                name, fmt, datetime.utcnow().isoformat(), exc,
            )


# ===========================================================================
# 1. Load raw data (read-only references – never mutated)
# ===========================================================================

log.info("Loading accidents_clean.csv …")
try:
    _acc_raw = pd.read_csv(ACCIDENTS_PATH)
    log.info("  Loaded %d accident records, %d columns", *_acc_raw.shape)
except Exception as exc:
    log.error("Import failed | file=%s | ts=%s | error=%s",
              ACCIDENTS_PATH, datetime.utcnow().isoformat(), exc)
    raise

log.info("Loading imd_clean.csv …")
try:
    _imd_raw = pd.read_csv(IMD_PATH)
    log.info("  Loaded %d IMD records, %d columns", *_imd_raw.shape)
except Exception as exc:
    log.error("Import failed | file=%s | ts=%s | error=%s",
              IMD_PATH, datetime.utcnow().isoformat(), exc)
    raise

log.info("Loading road-network geometry (layer: %s) …", ROAD_LAYER)
try:
    _roads_raw = gpd.read_file(GPKG_PATH, layer=ROAD_LAYER)
    log.info("  Loaded %d road segments, CRS=%s", len(_roads_raw), _roads_raw.crs)
except Exception as exc:
    log.error("Import failed | file=%s | ts=%s | error=%s",
              GPKG_PATH, datetime.utcnow().isoformat(), exc)
    raise


# ===========================================================================
# 2. Working copies  (originals are never touched after this point)
# ===========================================================================

acc   = _acc_raw.copy()
imd   = _imd_raw.copy()
roads = _roads_raw.copy()

# Standardise column names for the IMD frame (align with accident keys)
imd.rename(columns={"año": "YEAR", "carretera": "CARRETERA",
                     "pk": "KM",   "tramo_km_ini": "TRAMO_KM_INI",
                     "seccion_id": "SECCION_ID"}, inplace=True)

# ── Derived flag: "bad" road surface  ──────────────────────────────────────
# "Otro" in CONDICION_FIRME_LABEL is the only code encoded as FIRME_MALO=1;
# it covers anomalous/unclassified surface states including substance-related
# incidents reported under an umbrella code.
BAD_SURFACE_LABELS = {
    "Con barro o gravilla suelta",
    "Con aceite",
    "Muy encharcado o inundado",
    "Con hielo",
    "Con nieve",
    "Otro",          # FIRME_MALO == 1
}
acc["IS_BAD_SURFACE"] = acc["CONDICION_FIRME_LABEL"].isin(BAD_SURFACE_LABELS).astype(int)

# ── Proxy flag for alcohol / drug involvement ──────────────────────────────
# FIRME_MALO == 1 maps to CONDICION_FIRME_LABEL == "Otro", the only code
# the DGT source uses as a catch-all for incidents where the primary
# pavement-condition factor is unclassified/other (proxy for substance
# involvement in the absence of a dedicated alcohol/drug column).
acc["PROXY_ALCOHOL_DRUG"] = (acc["FIRME_MALO"] == 1).astype(int)

# ── Severity booleans ──────────────────────────────────────────────────────
acc["IS_FATAL"]   = (acc["SEVERIDAD"] == "Mortal").astype(int)
acc["IS_SERIOUS"] = (acc["SEVERIDAD"] == "Grave").astype(int)
acc["IS_SEVERE"]  = ((acc["SEVERIDAD"] == "Mortal") |
                     (acc["SEVERIDAD"] == "Grave")).astype(int)

log.info("Working copies created and derived flags added.")


# ===========================================================================
# INDICATOR 1 – Accident Concentration Index (ACI)
# per road section × year
# ===========================================================================
#
# Concept:  Black spots are sections where accidents are disproportionately
#           high relative to the traffic they carry (IMD = Intensidad Media
#           Diaria, Average Daily Traffic).
#
# Formula:
#   acc_count      = total accidents in section s for year y
#   vehicle_km     = imd_total × 365  (annual vehicle passages, proxy for
#                                       vehicle-kilometres when section
#                                       length is unknown)
#   ACI            = (acc_count / vehicle_km) × 10^8
#                    [accidents per 100 million vehicle-passages]
#
# Black-spot threshold: sections whose ACI exceeds the mean + 1 SD across
# all sections for that year are flagged as Tramos de Concentración de
# Accidentes (TCA).
# ===========================================================================

log.info("Computing Indicator 1: Accident Concentration Index …")

# ── Step 1a: accident counts per (SECCION_ID, YEAR) ──────────────────────
acc_by_section = (
    acc
    .groupby(["SECCION_ID", "YEAR"], as_index=False)
    .agg(
        n_accidents    = ("SECCION_ID", "count"),
        n_fatal        = ("IS_FATAL",   "sum"),
        n_serious      = ("IS_SERIOUS", "sum"),
        n_victims_30d  = ("TOTAL_VICTIMAS_30DF", "sum"),
        n_vehicles_inv = ("TOTAL_VEHICULOS",     "sum"),
    )
)

# ── Step 1b: mean annual IMD per section (accounts for multiple stations) ──
imd_annual = (
    imd
    .groupby(["SECCION_ID", "YEAR"], as_index=False)
    .agg(imd_mean = ("imd_total", "mean"))
)

# ── Step 1c: merge on (SECCION_ID, YEAR) ──────────────────────────────────
ind1 = pd.merge(
    acc_by_section,
    imd_annual,
    on=["SECCION_ID", "YEAR"],
    how="left",          # keep all accident sections; IMD may be missing
)

# ── Step 1d: ACI calculation ───────────────────────────────────────────────
# vehicle_passages = IMD × 365   (annual total passages through the section)
ind1["vehicle_passages"] = ind1["imd_mean"] * 365

# ACI = accidents / vehicle_passages × 1e8  (per 100M passages)
# Sections with no IMD data get NaN – they cannot be normalised
ind1["ACI"] = np.where(
    ind1["vehicle_passages"] > 0,
    (ind1["n_accidents"] / ind1["vehicle_passages"]) * 1e8,
    np.nan,
)

# ── Step 1e: black-spot flag (TCA) per year ────────────────────────────────
# A section is a TCA if its ACI > μ + σ for that year (among sections with
# valid ACI).  This avoids penalising low-traffic rural roads unfairly when
# they happen to have a single accident.
aci_stats = (
    ind1.dropna(subset=["ACI"])
    .groupby("YEAR")["ACI"]
    .agg(aci_mean="mean", aci_std="std")
    .reset_index()
)
ind1 = pd.merge(ind1, aci_stats, on="YEAR", how="left")
ind1["threshold_TCA"] = ind1["aci_mean"] + ind1["aci_std"]
ind1["IS_TCA"] = (ind1["ACI"] > ind1["threshold_TCA"]).astype("Int64")

# ── Rank within each year ──────────────────────────────────────────────────
ind1["ACI_rank_year"] = (
    ind1.groupby("YEAR")["ACI"]
    .rank(method="dense", ascending=False, na_option="bottom")
    .astype("Int64")
)

# ── Add road identifier for easy filtering in the viewer ──────────────────
ind1["CARRETERA"] = ind1["SECCION_ID"].str.extract(r"^(.+)_\d+$")

log.info("  Indicator 1 shape: %s | TCA sections: %d",
         ind1.shape,
         ind1["IS_TCA"].sum())
export(ind1, "ind1_accident_concentration")


# ===========================================================================
# INDICATOR 2 – Road Condition Influence Index (RCII)
# per road (CARRETERA) × year
# ===========================================================================
#
# Concept:  Measures the proportion of accidents that occurred under bad
#           pavement conditions relative to all accidents on that road.
#
# Formula:
#   RCII = (accidents with bad surface / total accidents) × 100   [%]
#
# Additionally, the rate relative to traffic exposure is computed for roads
# that have IMD data (normalises for high-volume roads).
# ===========================================================================

log.info("Computing Indicator 2: Road Condition Influence Index …")

# ── Step 2a: aggregate at (CARRETERA, YEAR) level ─────────────────────────
ind2 = (
    acc
    .groupby(["CARRETERA", "YEAR"], as_index=False)
    .agg(
        n_total        = ("SECCION_ID",     "count"),
        n_bad_surface  = ("IS_BAD_SURFACE", "sum"),
    )
)

# ── Step 2b: RCII ─────────────────────────────────────────────────────────
ind2["RCII_pct"] = (ind2["n_bad_surface"] / ind2["n_total"] * 100).round(4)

# ── Step 2c: traffic-normalised rate (per 100M vehicle passages) ───────────
# Aggregate IMD to road level (mean of all sections, excluding COVID years)
imd_road = (
    imd[~imd["flag_covid"]]
    .groupby(["CARRETERA", "YEAR"], as_index=False)
    .agg(imd_road_mean = ("imd_total", "mean"))
)
ind2 = pd.merge(ind2, imd_road, on=["CARRETERA", "YEAR"], how="left")
ind2["road_passages"] = ind2["imd_road_mean"] * 365
ind2["RCII_rate_per100M"] = np.where(
    ind2["road_passages"] > 0,
    (ind2["n_bad_surface"] / ind2["road_passages"]) * 1e8,
    np.nan,
)

# ── Step 2d: classify influence level (Low / Medium / High) ───────────────
bins   = [-np.inf, 5, 15, np.inf]
labels = ["Low", "Medium", "High"]
ind2["RCII_level"] = pd.cut(ind2["RCII_pct"], bins=bins, labels=labels)

log.info("  Indicator 2 shape: %s", ind2.shape)
export(ind2, "ind2_road_condition")


# ===========================================================================
# INDICATOR 3 – Accident Rate due to Alcohol and Drugs (RADAR)
# per road section × year  AND  at regional level × year
# ===========================================================================
#
# Data note: The accidents dataset has no dedicated alcohol/drug column.
#   The proxy used is FIRME_MALO == 1 ↔ CONDICION_FIRME_LABEL == "Otro",
#   which is the only catch-all code in the DGT surface-condition taxonomy.
#   All output columns carry the suffix "_proxy" to signal this limitation.
#
# Formula:
#   RADAR_proxy = (proxy accidents / total accidents) × 100   [%]
# ===========================================================================

log.info("Computing Indicator 3: Accident Rate – Alcohol/Drug Proxy …")

# ── Step 3a: section-level ─────────────────────────────────────────────────
ind3_section = (
    acc
    .groupby(["SECCION_ID", "CARRETERA", "YEAR"], as_index=False)
    .agg(
        n_total           = ("SECCION_ID",          "count"),
        n_proxy_substance = ("PROXY_ALCOHOL_DRUG",  "sum"),
    )
)
ind3_section["RADAR_proxy_pct"] = (
    ind3_section["n_proxy_substance"] / ind3_section["n_total"] * 100
).round(4)

# ── Step 3b: regional level (all sections combined, per year) ──────────────
ind3_regional = (
    acc
    .groupby("YEAR", as_index=False)
    .agg(
        n_total           = ("SECCION_ID",         "count"),
        n_proxy_substance = ("PROXY_ALCOHOL_DRUG", "sum"),
    )
)
ind3_regional["RADAR_proxy_pct_regional"] = (
    ind3_regional["n_proxy_substance"] / ind3_regional["n_total"] * 100
).round(4)
ind3_regional["SCOPE"] = "Regional"

# ── Step 3c: municipality level ────────────────────────────────────────────
ind3_muni = (
    acc
    .groupby(["COD_MUNICIPIO", "YEAR"], as_index=False)
    .agg(
        n_total           = ("SECCION_ID",         "count"),
        n_proxy_substance = ("PROXY_ALCOHOL_DRUG", "sum"),
    )
)
ind3_muni["RADAR_proxy_pct"] = (
    ind3_muni["n_proxy_substance"] / ind3_muni["n_total"] * 100
).round(4)

# Export all three granularities
export(ind3_section,  "ind3_alcohol_drugs_section")
export(ind3_regional, "ind3_alcohol_drugs_regional")
export(ind3_muni,     "ind3_alcohol_drugs_municipality")

log.info("  Indicator 3 section shape: %s | regional shape: %s | muni shape: %s",
         ind3_section.shape, ind3_regional.shape, ind3_muni.shape)


# ===========================================================================
# INDICATOR 4 – Accident Severity Index (ASI)
# per primary cause (TIPO_ACCIDENTE_LABEL) × year
# ===========================================================================
#
# Concept:  For each accident type (proxy for "primary cause"), what fraction
#           of the involved people ended up dead or seriously injured?
#
# Formula (30-day criterion, as per DGT standard):
#   ASI = (fatalities_30d + serious_injuries_30d) / total_victims_30d × 100
#
# Also computed at section and municipality granularity for the viewer.
# ===========================================================================

log.info("Computing Indicator 4: Accident Severity Index …")

# ── Step 4a: by accident type × year (primary cause view) ─────────────────
ind4_cause = (
    acc
    .dropna(subset=["TIPO_ACCIDENTE_LABEL"])    # drop unknown types
    .groupby(["TIPO_ACCIDENTE_LABEL", "YEAR"], as_index=False)
    .agg(
        n_accidents     = ("SECCION_ID",          "count"),
        n_fatal_30d     = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d   = ("TOTAL_HG30DF",        "sum"),
        n_light_30d     = ("TOTAL_HL30DF",        "sum"),
        n_victims_30d   = ("TOTAL_VICTIMAS_30DF", "sum"),
    )
)
ind4_cause["ASI_pct"] = np.where(
    ind4_cause["n_victims_30d"] > 0,
    (ind4_cause["n_fatal_30d"] + ind4_cause["n_serious_30d"])
    / ind4_cause["n_victims_30d"] * 100,
    np.nan,
).round(4)

# ── Step 4b: by road section × year ───────────────────────────────────────
ind4_section = (
    acc
    .groupby(["SECCION_ID", "CARRETERA", "YEAR"], as_index=False)
    .agg(
        n_accidents   = ("SECCION_ID",          "count"),
        n_fatal_30d   = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d = ("TOTAL_HG30DF",        "sum"),
        n_victims_30d = ("TOTAL_VICTIMAS_30DF", "sum"),
    )
)
ind4_section["ASI_pct"] = np.where(
    ind4_section["n_victims_30d"] > 0,
    (ind4_section["n_fatal_30d"] + ind4_section["n_serious_30d"])
    / ind4_section["n_victims_30d"] * 100,
    np.nan,
).round(4)

# ── Step 4c: regional totals per year (for temporal chart) ────────────────
ind4_regional = (
    acc
    .groupby("YEAR", as_index=False)
    .agg(
        n_accidents   = ("SECCION_ID",          "count"),
        n_fatal_30d   = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d = ("TOTAL_HG30DF",        "sum"),
        n_victims_30d = ("TOTAL_VICTIMAS_30DF", "sum"),
    )
)
ind4_regional["ASI_pct"] = (
    (ind4_regional["n_fatal_30d"] + ind4_regional["n_serious_30d"])
    / ind4_regional["n_victims_30d"] * 100
).round(4)

export(ind4_cause,    "ind4_severity_by_cause")
export(ind4_section,  "ind4_severity_by_section")
export(ind4_regional, "ind4_severity_regional")

log.info("  Indicator 4: cause shape=%s | section shape=%s | regional=%s",
         ind4_cause.shape, ind4_section.shape, ind4_regional.shape)


# ===========================================================================
# INDICATOR 5 – Temporal Evolution of Accident Rates
# at section, road, and regional level
# ===========================================================================
#
# Metrics tracked per year:
#   - Total accidents
#   - Normalised accident rate (accidents per 100M vehicle passages, where
#     IMD is available)
#   - Year-over-year change (absolute and %)
#   - 3-year rolling mean (to smooth noise, applied at regional level)
#
# The temporal series covers 2016–2024.  COVID years (2020–2021) are kept
# in the series but flagged so the viewer can grey them out.
# ===========================================================================

log.info("Computing Indicator 5: Temporal Evolution of Accident Rates …")

COVID_YEARS = {2020, 2021}

# ── Step 5a: regional time series ─────────────────────────────────────────
ind5_regional = (
    acc
    .groupby("YEAR", as_index=False)
    .agg(
        n_accidents   = ("SECCION_ID",          "count"),
        n_fatal_30d   = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d = ("TOTAL_HG30DF",        "sum"),
        n_victims_30d = ("TOTAL_VICTIMAS_30DF", "sum"),
    )
    .sort_values("YEAR")
)
# Merge total regional IMD to normalise
imd_regional = (
    imd[~imd["flag_covid"]]
    .groupby("YEAR", as_index=False)
    .agg(imd_regional_sum = ("imd_total", "sum"))
)
ind5_regional = pd.merge(ind5_regional, imd_regional, on="YEAR", how="left")
ind5_regional["vehicle_passages_regional"] = ind5_regional["imd_regional_sum"] * 365
ind5_regional["rate_per100M"] = np.where(
    ind5_regional["vehicle_passages_regional"] > 0,
    ind5_regional["n_accidents"] / ind5_regional["vehicle_passages_regional"] * 1e8,
    np.nan,
)
# YoY absolute change
ind5_regional["yoy_delta_abs"] = ind5_regional["n_accidents"].diff()
ind5_regional["yoy_delta_pct"] = (
    ind5_regional["n_accidents"].pct_change() * 100
).round(2)
# 3-year rolling mean (accidents)
ind5_regional["rolling3_accidents"] = (
    ind5_regional["n_accidents"].rolling(window=3, min_periods=1).mean().round(1)
)
ind5_regional["is_covid_year"] = ind5_regional["YEAR"].isin(COVID_YEARS)

# ── Step 5b: per road (CARRETERA) time series ─────────────────────────────
ind5_road = (
    acc
    .groupby(["CARRETERA", "YEAR"], as_index=False)
    .agg(
        n_accidents   = ("SECCION_ID",          "count"),
        n_fatal_30d   = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d = ("TOTAL_HG30DF",        "sum"),
    )
    .sort_values(["CARRETERA", "YEAR"])
)
# YoY change within each road
ind5_road["yoy_delta_abs"] = (
    ind5_road.groupby("CARRETERA")["n_accidents"].diff()
)
ind5_road["yoy_delta_pct"] = (
    ind5_road.groupby("CARRETERA")["n_accidents"].pct_change() * 100
).round(2)
ind5_road["is_covid_year"] = ind5_road["YEAR"].isin(COVID_YEARS)

# ── Step 5c: per section time series ──────────────────────────────────────
ind5_section = (
    acc
    .groupby(["SECCION_ID", "CARRETERA", "YEAR"], as_index=False)
    .agg(
        n_accidents   = ("SECCION_ID",          "count"),
        n_fatal_30d   = ("TOTAL_MU30DF",        "sum"),
        n_serious_30d = ("TOTAL_HG30DF",        "sum"),
    )
    .sort_values(["SECCION_ID", "YEAR"])
)
ind5_section["yoy_delta_abs"] = (
    ind5_section.groupby("SECCION_ID")["n_accidents"].diff()
)
ind5_section["yoy_delta_pct"] = (
    ind5_section.groupby("SECCION_ID")["n_accidents"].pct_change() * 100
).round(2)
ind5_section["is_covid_year"] = ind5_section["YEAR"].isin(COVID_YEARS)

# ── Step 5d: trend classification (improving / stable / worsening) ─────────
# Fit a simple OLS slope per section over the non-COVID years; sign of slope
# determines the trend label.
from scipy.stats import linregress  # stdlib-scipy, lightweight

def classify_trend(group: pd.DataFrame) -> str:
    """Return 'Improving', 'Stable', or 'Worsening' for a section series."""
    g = group[~group["is_covid_year"]].sort_values("YEAR")
    if len(g) < 3:
        return "Insufficient data"
    slope, _, _, pvalue, _ = linregress(g["YEAR"], g["n_accidents"])
    if pvalue > 0.1:
        return "Stable"
    return "Worsening" if slope > 0 else "Improving"

log.info("  Computing trend labels per section (may take a moment) …")
trend_labels = (
    ind5_section
    .groupby("SECCION_ID")
    .apply(classify_trend)
    .reset_index()
    .rename(columns={0: "trend_label"})
)
# Merge trend label back (section-level, year-independent)
ind5_section = pd.merge(ind5_section, trend_labels, on="SECCION_ID", how="left")

export(ind5_regional, "ind5_temporal_evolution_regional")
export(ind5_road,     "ind5_temporal_evolution_road")
export(ind5_section,  "ind5_temporal_evolution_section")

log.info("  Indicator 5: regional=%s | road=%s | section=%s",
         ind5_regional.shape, ind5_road.shape, ind5_section.shape)


# ===========================================================================
# INDICATOR 6 – Distribution of Other Accident Causes
# (relative frequencies of TIPO_ACCIDENTE_LABEL)
# per year, per road, and per section
# ===========================================================================
#
# "Other causes" refers to the full taxonomy of accident types encoded in
# TIPO_ACCIDENTE_LABEL (e.g. rear-end collisions, pedestrian knockdowns,
# rollovers, run-off-road events).  This indicator measures how accident
# types are distributed and how that distribution shifts over time or
# between roads.
# ===========================================================================

log.info("Computing Indicator 6: Distribution of Accident Causes …")

# ── Step 6a: regional distribution (absolute + relative) ──────────────────
ind6_regional = (
    acc
    .dropna(subset=["TIPO_ACCIDENTE_LABEL"])
    .groupby(["TIPO_ACCIDENTE_LABEL", "YEAR"], as_index=False)
    .agg(n_accidents = ("SECCION_ID", "count"))
)
# Total accidents per year to compute relative frequencies
totals_year = acc.groupby("YEAR")["SECCION_ID"].count().rename("n_total_year")
ind6_regional = ind6_regional.join(totals_year, on="YEAR")
ind6_regional["rel_freq_pct"] = (
    ind6_regional["n_accidents"] / ind6_regional["n_total_year"] * 100
).round(4)

# ── Step 6b: per road ─────────────────────────────────────────────────────
ind6_road = (
    acc
    .dropna(subset=["TIPO_ACCIDENTE_LABEL"])
    .groupby(["CARRETERA", "TIPO_ACCIDENTE_LABEL", "YEAR"], as_index=False)
    .agg(n_accidents = ("SECCION_ID", "count"))
)
totals_road_year = (
    acc.groupby(["CARRETERA", "YEAR"])["SECCION_ID"]
    .count()
    .rename("n_total_road_year")
    .reset_index()
)
ind6_road = pd.merge(ind6_road, totals_road_year,
                     on=["CARRETERA", "YEAR"], how="left")
ind6_road["rel_freq_pct"] = (
    ind6_road["n_accidents"] / ind6_road["n_total_road_year"] * 100
).round(4)

# ── Step 6c: dominant cause per section (for map choropleth) ──────────────
ind6_section_dominant = (
    acc
    .dropna(subset=["TIPO_ACCIDENTE_LABEL"])
    .groupby(["SECCION_ID", "CARRETERA", "YEAR",
              "TIPO_ACCIDENTE_LABEL"], as_index=False)
    .agg(n_accidents = ("SECCION_ID", "count"))
)
# Keep the cause with the highest count per (section, year)
ind6_section_dominant = (
    ind6_section_dominant
    .sort_values("n_accidents", ascending=False)
    .groupby(["SECCION_ID", "YEAR"], as_index=False)
    .first()
    .rename(columns={"TIPO_ACCIDENTE_LABEL": "dominant_cause",
                     "n_accidents":          "n_dominant_accidents"})
)

export(ind6_regional,          "ind6_cause_distribution_regional")
export(ind6_road,              "ind6_cause_distribution_road")
export(ind6_section_dominant,  "ind6_cause_dominant_per_section")

log.info("  Indicator 6: regional=%s | road=%s | section_dominant=%s",
         ind6_regional.shape, ind6_road.shape, ind6_section_dominant.shape)


# ===========================================================================
# 7. Unified GeoDataFrame for Phase 5 (Interactive Viewer)
# ===========================================================================
#
# Strategy:
#   The road-network geometry (rt_tramo_vial) links to our indicators via
#   the road name.  The layer's 'nombre' column matches the CARRETERA field
#   in the accident data.  We aggregate each indicator to (CARRETERA) level
#   – summing or averaging across all years and sections – and join to the
#   road geometry.  This gives the viewer a single spatial file it can
#   colour-code without additional joins at runtime.
#
#   All metrics are "all-years" aggregates (total / mean over 2018-2024,
#   excluding COVID years) so the map shows a stable baseline.
# ===========================================================================

log.info("Building unified GeoDataFrame …")

NON_COVID = ~acc["YEAR"].isin(COVID_YEARS)

# ── 7a: aggregate indicators to road (CARRETERA) level ────────────────────
summary_road = (
    acc[NON_COVID]
    .groupby("CARRETERA", as_index=False)
    .agg(
        total_accidents       = ("SECCION_ID",          "count"),
        total_fatal_30d       = ("TOTAL_MU30DF",        "sum"),
        total_serious_30d     = ("TOTAL_HG30DF",        "sum"),
        total_victims_30d     = ("TOTAL_VICTIMAS_30DF", "sum"),
        total_bad_surface     = ("IS_BAD_SURFACE",      "sum"),
        total_proxy_substance = ("PROXY_ALCOHOL_DRUG",  "sum"),
        n_TCA_sections        = ("IS_SEVERE",           "sum"),  # proxy count
    )
)

# Ratios
summary_road["ASI_pct"] = np.where(
    summary_road["total_victims_30d"] > 0,
    (summary_road["total_fatal_30d"] + summary_road["total_serious_30d"])
    / summary_road["total_victims_30d"] * 100,
    np.nan,
).round(2)

summary_road["RCII_pct"] = (
    summary_road["total_bad_surface"] / summary_road["total_accidents"] * 100
).round(2)

summary_road["RADAR_proxy_pct"] = (
    summary_road["total_proxy_substance"] / summary_road["total_accidents"] * 100
).round(2)

# ── 7b: dominant cause per road ───────────────────────────────────────────
dominant_cause_road = (
    acc[NON_COVID]
    .dropna(subset=["TIPO_ACCIDENTE_LABEL"])
    .groupby(["CARRETERA", "TIPO_ACCIDENTE_LABEL"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .groupby("CARRETERA", as_index=False)
    .first()
    .rename(columns={"TIPO_ACCIDENTE_LABEL": "dominant_cause"})
    [["CARRETERA", "dominant_cause"]]
)
summary_road = pd.merge(summary_road, dominant_cause_road,
                        on="CARRETERA", how="left")

# ── 7c: join to geometry ──────────────────────────────────────────────────
# Keep only primary-network roads (tipo_tramo == 1: Troncal, clase indicates
# autovías/carreteras conv.) to reduce the geometry size for the viewer.
roads_filtered = roads[roads["tipo_tramo"] == 1].copy()

# Dissolve by road name to get single geometry per named road (union of
# segments).  We dissolve on 'nombre' which matches CARRETERA in accidents.
log.info("  Dissolving road geometry by nombre (may take ~30 s) …")
roads_dissolved = (
    roads_filtered[["nombre", "geometry"]]
    .rename(columns={"nombre": "CARRETERA"})
    .dissolve(by="CARRETERA", as_index=False)
)

# Merge indicator summary
geo_unified = roads_dissolved.merge(summary_road, on="CARRETERA", how="left")

# Fill NaN for roads with no accidents (they still appear on the map in grey)
fill_zero_cols = ["total_accidents", "total_fatal_30d", "total_serious_30d",
                  "total_victims_30d", "total_bad_surface",
                  "total_proxy_substance", "n_TCA_sections"]
geo_unified[fill_zero_cols] = geo_unified[fill_zero_cols].fillna(0)

log.info("  Unified GeoDataFrame: %d road geometries, %d with accident data",
         len(geo_unified),
         geo_unified["total_accidents"].gt(0).sum())

# ── 7d: export as GeoPackage ──────────────────────────────────────────────
geo_path = os.path.join(OUTPUT_DIR, "indicators_unified.gpkg")
try:
    geo_unified.to_file(geo_path, layer="road_indicators", driver="GPKG")
    log.info("Exported unified GeoDataFrame → %s", geo_path)
except Exception as exc:
    log.error("GeoPackage export failed | ts=%s | error=%s",
              datetime.utcnow().isoformat(), exc)

# Also export the attribute table as CSV (geometry-free) for quick inspection
geo_unified.drop(columns="geometry").to_csv(
    os.path.join(OUTPUT_DIR, "indicators_unified_attributes.csv"),
    index=False, encoding="utf-8-sig",
)

# ===========================================================================
# 8. Summary report
# ===========================================================================

log.info("=" * 60)
log.info("PHASE 4 COMPLETE – Output summary")
log.info("=" * 60)
log.info("Output directory : %s", OUTPUT_DIR)
log.info("")
log.info("IND 1 – Accident Concentration Index")
log.info("  Records: %d | TCA sections: %d",
         len(ind1), int(ind1["IS_TCA"].sum()))
log.info("")
log.info("IND 2 – Road Condition Influence Index")
log.info("  Records: %d | Roads with High RCII: %d",
         len(ind2), int((ind2["RCII_level"] == "High").sum()))
log.info("")
log.info("IND 3 – Alcohol/Drug Proxy Rate")
log.info("  Section records: %d | Regional records: %d",
         len(ind3_section), len(ind3_regional))
log.info("")
log.info("IND 4 – Accident Severity Index")
log.info("  Cause×Year records: %d | Regional records: %d",
         len(ind4_cause), len(ind4_regional))
log.info("")
log.info("IND 5 – Temporal Evolution")
log.info("  Regional years: %d | Road×Year: %d | Section×Year: %d",
         len(ind5_regional), len(ind5_road), len(ind5_section))
log.info("")
log.info("IND 6 – Cause Distribution")
log.info("  Regional: %d | Road: %d | Dominant per section: %d",
         len(ind6_regional), len(ind6_road), len(ind6_section_dominant))
log.info("")
log.info("UNIFIED GeoDataFrame: %d road geometries", len(geo_unified))
log.info("=" * 60)
