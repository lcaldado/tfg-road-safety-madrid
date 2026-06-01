# =============================================================================
# Road Safety Viewer — Community of Madrid (2016–2024)
# Bachelor's Thesis Dashboard — Phase 5
#
# FILE USAGE BY TAB:
#  
#   Tab 2 (Temporal Evolution):
#       ind5_temporal_evolution_road.csv — accident counts, YoY delta per road
#       ind5_temporal_evolution_regional.csv — regional rolling avg, rate per 100M
#
#   Tab 3 (Indicators Detail):
#       ind2_road_condition.csv           — RCII road condition index
#       ind3_alcohol_drugs_section.csv    — substance proxy per section
#       ind3_alcohol_drugs_regional.csv   — regional substance proxy reference
#       ind4_severity_by_section.csv      — ASI per section
#       ind4_severity_by_cause.csv        — ASI regional reference
#       ind6_cause_distribution_road.csv  — cause breakdown per road/year
#       ind6_cause_dominant_per_section.parquet — dominant cause per section/year
#
#   Tab 4 (Black Spots):
#       ind1_accident_concentration.csv  — ACI, IS_TCA flags, section heatmap
# =============================================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import numpy as np
import os
import geopandas as gpd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Safety Viewer — Madrid",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Hardcoded coordinates for key roads (no lat/lon in source data) ───────────
# Includes all roads that have IS_TCA==1 black spots in the dataset
ROAD_COORDS = {
    # Roads with actual black spots in the dataset (IS_TCA==1)
    "M-116":  (40.480, -3.580), "M-118":  (40.490, -3.560),
    "M-123":  (40.460, -3.530), "M-127":  (40.530, -3.490),
    "M-131":  (40.510, -3.460), "M-132":  (40.520, -3.440),
    "M-133":  (40.500, -3.450), "M-134":  (40.490, -3.470),
    "M-135":  (40.485, -3.510), "M-204":  (40.390, -3.850),
    "M-219":  (40.370, -3.810), "M-220":  (40.360, -3.790),
    "M-221":  (40.355, -3.770), "M-224":  (40.340, -3.740),
    "M-225":  (40.330, -3.720), "M-226":  (40.325, -3.700),
    "M-227":  (40.320, -3.680), "M-300":  (40.420, -3.550),
    "M-301":  (40.410, -3.540), "M-302":  (40.400, -3.530),
    "M-305":  (40.390, -3.520), "M-307":  (40.380, -3.510),
    "M-311":  (40.370, -3.500), "M-313":  (40.360, -3.490),
    "M-501":  (40.340, -3.810), "M-502":  (40.330, -3.800),
    "M-503":  (40.320, -3.790), "M-505":  (40.310, -3.780),
    "M-506":  (40.300, -3.770), "M-600":  (40.560, -3.700),
    "M-601":  (40.570, -3.690), "M-603":  (40.580, -3.680),
    "M-604":  (40.590, -3.670), "M-607":  (40.600, -3.660),
    "M-608":  (40.610, -3.650), "M-611":  (40.450, -3.900),
    "M-612":  (40.440, -3.910), "M-614":  (40.430, -3.920),
    "M-45":   (40.395, -3.668), "M-11":   (40.460, -3.600),
    "M-12":   (40.455, -3.610), "M-13":   (40.450, -3.620),
    "M-14":   (40.445, -3.630), "M-21":   (40.395, -3.630),
    "M-22":   (40.390, -3.620), "M-23":   (40.385, -3.610),
    "AP-41":  (40.280, -3.750), "AP-6":   (40.620, -3.730),
    "R-2":    (40.480, -3.540), "R-3":    (40.390, -3.570),
    "R-4":    (40.310, -3.680), "R-5":    (40.350, -3.780),
    # General context roads
    "M-40":   (40.418, -3.704), "M-50":   (40.418, -3.870),
    "A-1":    (40.550, -3.650), "A-2":    (40.470, -3.570),
    "A-3":    (40.380, -3.600), "A-4":    (40.330, -3.690),
    "A-5":    (40.380, -3.780), "A-6":    (40.490, -3.780),
    "A-42":   (40.310, -3.720), "M-30":   (40.418, -3.680),
}

# ── COVID year sentinel: always grey these out visually ──────────────────────
COVID_YEARS = {2020, 2021}

# =============================================================================
# DATA LOADING — cached so reloads don't re-read disk on every interaction
# =============================================================================
@st.cache_data
def load_data():
    ind1  = pd.read_csv(os.path.join(DATA_DIR, "ind1_accident_concentration.csv"))
    ind2  = pd.read_csv(os.path.join(DATA_DIR, "ind2_road_condition.csv"))
    ind3s = pd.read_csv(os.path.join(DATA_DIR, "ind3_alcohol_drugs_section.csv"))
    ind3r = pd.read_csv(os.path.join(DATA_DIR, "ind3_alcohol_drugs_regional.csv"))
    ind4s = pd.read_csv(os.path.join(DATA_DIR, "ind4_severity_by_section.csv"))
    ind4c = pd.read_csv(os.path.join(DATA_DIR, "ind4_severity_by_cause.csv"))
    ind5r = pd.read_csv(os.path.join(DATA_DIR, "ind5_temporal_evolution_road.csv"))
    ind5R = pd.read_csv(os.path.join(DATA_DIR, "ind5_temporal_evolution_regional.csv"))
    ind6d = pd.read_csv(os.path.join(DATA_DIR, "ind6_cause_distribution_road.csv"))
    ind6p = pd.read_parquet(os.path.join(DATA_DIR, "ind6_cause_dominant_per_section.parquet"))

    # Ensure is_covid_year is boolean
    for df in [ind5r, ind5R]:
        df["is_covid_year"] = df["is_covid_year"].astype(bool)

    # IS_TCA as int for easy filtering
    ind1["IS_TCA"] = ind1["IS_TCA"].astype(int)

    # Load road network and filter only roads present in ind1
    roads_geo = gpd.read_file(os.path.join(DATA_DIR, "red_viaria.gpkg"), layer="rt_tramo_vial")
    roads_geo = roads_geo.to_crs(epsg=4326)
    carreteras_ind1 = set(ind1["CARRETERA"].unique())
    roads_geo = roads_geo[roads_geo["nombre"].isin(carreteras_ind1)][["nombre", "geometry"]].rename(columns={"nombre": "CARRETERA"})

    return ind1, ind2, ind3s, ind3r, ind4s, ind4c, ind5r, ind5R, ind6d, ind6p, roads_geo

ind1, ind2, ind3s, ind3r, ind4s, ind4c, ind5r, ind5R, ind6d, ind6p, roads_geo = load_data()

# ── Build road list for selector ─────────────────────────────────────────────
all_roads = sorted(ind5r["CARRETERA"].unique().tolist())
REGIONAL_LABEL = "— Regional view (all roads) —"
road_options = [REGIONAL_LABEL] + all_roads

# =============================================================================
# SIDEBAR — Primary controls
# =============================================================================
st.sidebar.title("Road Safety — Madrid")
st.sidebar.markdown("Community of Madrid · 2016–2024")

# Road selector: regional view is default
selected_road = st.sidebar.selectbox(
    "Select Road",
    options=road_options,
    index=0,
    help="Choose a specific road or the regional aggregate view",
)

# Year multiselect — default all years
all_years = sorted(ind5r["YEAR"].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Years",
    options=all_years,
    default=all_years,
    help="Filter data to specific years",
)
st.sidebar.caption("2020–2021 are COVID years (greyed out in charts)")

# Guard: if no years selected, warn and stop
if not selected_years:
    st.warning("Please select at least one year in the sidebar.")
    st.stop()

is_regional = selected_road == REGIONAL_LABEL

# =============================================================================
# HELPER UTILITIES
# =============================================================================
def empty_check(df, label="this selection"):
    """Returns True and shows warning if df is empty."""
    if df.empty:
        st.warning(f"No data available for {label}.")
        return True
    return False

def covid_bar_color(years_series):
    """Map years to bar colors: grey for COVID, blue otherwise."""
    return ["#b0b0b0" if y in COVID_YEARS else "#1f77b4" for y in years_series]

def add_covid_vrect(fig):
    """Add a grey shaded region between 2019.5–2021.5 labelled COVID-19."""
    fig.add_vrect(
        x0=2019.5, x1=2021.5,
        fillcolor="lightgrey", opacity=0.3, line_width=0,
        annotation_text="COVID-19", annotation_position="top left",
        annotation_font_size=10,
    )
    return fig

def get_section_coords(road, section_id, roads_geo):
    """
    Finds the real geometry of the road in roads_geo and calculates
    the exact point by interpolating the pk (kilometer point) of SECCION_ID.
    Returns (lat, lon) or None if the road has no geometry.
    """
    try:
        pk = int(str(section_id).split("_")[-1])
    except (ValueError, IndexError):
        pk = 0

    road_geom = roads_geo[roads_geo["CARRETERA"] == road]
    if road_geom.empty:
        return None

    # Merges all segments of the road into a single line
    from shapely.ops import unary_union, linemerge
    from shapely.geometry import MultiLineString
    merged = linemerge(unary_union(road_geom.geometry.values))

    # If the result is a MultiLineString, take the longest segment
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda g: g.length)

    # Interpolates the pk as a fraction of the total length of the geometry
    # pk is in km; geometry is in degrees (~0.009 degrees/km approx)
    total_length = merged.length
    # Convert pk to fraction (assuming ~0.009 degrees per km)
    pk_length = pk * 0.009
    fraction = min(pk_length / total_length, 1.0) if total_length > 0 else 0.0

    point = merged.interpolate(fraction, normalized=True)
    return point.y, point.x

def geo_to_path_layer(roads_geo, highlight_road=None):
    """Converts LineString geometries from GeoPackage to PyDeck PathLayer format."""
    rows = []
    for _, row in roads_geo.iterrows():
        geom = row["geometry"]
        if geom is None:
            continue
        coords = [list(c[:2]) for c in geom.coords]
        if len(coords) < 2:
            continue
        is_highlight = (highlight_road is not None and row["CARRETERA"] == highlight_road)
        rows.append({
            "road":  row["CARRETERA"],
            "path":  coords,
            "color": [255, 50, 50, 220] if is_highlight else [80, 80, 80, 140],
            "width": 80 if is_highlight else 40,
        })
    return pd.DataFrame(rows)
# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs([
    "Temporal Evolution",
    "Indicators Detail",
    "Black Spots (TCA)",
])


# =============================================================================
# TAB 1 — Temporal Evolution
# =============================================================================
with tab1:
    st.subheader("Temporal Evolution" if is_regional else f"Temporal Evolution — {selected_road}")

    # ── Chart A — Accident count over time ───────────────────────────────────
    st.markdown("#### Chart A — Accident Count per Year")

    if is_regional:
        # Regional view: bars + rolling3 + rate per 100M on second axis
        reg_t = ind5R[ind5R["YEAR"].isin(selected_years)].sort_values("YEAR")

        if not empty_check(reg_t, "regional temporal data"):
            bar_colors = ["#b0b0b0" if y in COVID_YEARS else "#001464" for y in reg_t["YEAR"]]

            fig_a = go.Figure()
            # Accident bars
            fig_a.add_trace(go.Bar(
                x=reg_t["YEAR"], y=reg_t["n_accidents"],
                name="Total Accidents", marker_color=bar_colors,
            ))
            fig_a.update_layout(
                yaxis=dict(title="Accidents"),
                legend=dict(orientation="h"),
                xaxis=dict(title="Year", dtick=1),
            )
            add_covid_vrect(fig_a)
            st.plotly_chart(fig_a, width="stretch")

    else:
        # Road view: bars for selected road + dashed grey regional average reference
        road_t = ind5r[(ind5r["CARRETERA"] == selected_road) & (ind5r["YEAR"].isin(selected_years))].sort_values("YEAR")

        if not empty_check(road_t, f"road {selected_road}"):
            # Regional average per year: total accidents / number of roads
            n_roads = ind5r["CARRETERA"].nunique()
            reg_avg = (
                ind5R[ind5R["YEAR"].isin(selected_years)]
                .assign(avg=lambda d: d["n_accidents"] / n_roads)
                [["YEAR", "avg"]]
                .sort_values("YEAR")
            )
            bar_colors = ["#b0b0b0" if y in COVID_YEARS else "#001464" for y in road_t["YEAR"]]

            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=road_t["YEAR"], y=road_t["n_accidents"],
                name=f"{selected_road} accidents", marker_color=bar_colors,
            ))
            # Dashed grey reference line = regional average per road
            fig_a.add_trace(go.Scatter(
                x=reg_avg["YEAR"], y=reg_avg["avg"],
                name="Regional avg / road", mode="lines",
                line=dict(dash="dash", color="grey", width=2),
            ))
            fig_a.update_layout(
                xaxis=dict(title="Year", dtick=1),
                yaxis=dict(title="Accidents"),
                legend=dict(orientation="h"),
            )
            add_covid_vrect(fig_a)
            st.plotly_chart(fig_a, width="stretch")

    # ── Chart B — Multi-road comparison (Regional view only) ─────────────────
    if is_regional:
        st.markdown("#### Chart B — Multi-Road Comparison")

        # Compute top 10 roads by total accidents to avoid overplotting
        top10_roads = (
            ind5r.groupby("CARRETERA")["n_accidents"].sum()
            .nlargest(10).index.tolist()
        )

        # In-tab selector to add/remove roads
        extra_roads = st.multiselect(
            "Add/remove roads from chart",
            options=all_roads,
            default=top10_roads,
            key="chart_c_roads",
        )

        if extra_roads:
            comp_data = ind5r[
                (ind5r["CARRETERA"].isin(extra_roads)) &
                (ind5r["YEAR"].isin(selected_years))
            ].sort_values("YEAR")

            if not empty_check(comp_data, "multi-road comparison"):
                fig_c = px.line(
                    comp_data, x="YEAR", y="n_accidents",
                    color="CARRETERA", markers=True,
                    labels={"n_accidents": "Accidents", "YEAR": "Year", "CARRETERA": "Road"},
                    title="Accidents per Road Over Time",
                )
                add_covid_vrect(fig_c)
                st.plotly_chart(fig_c, width="stretch")
        else:
            st.info("Select at least one road in the multiselect above.")


# =============================================================================
# TAB 2 — Indicators Detail
# =============================================================================
with tab2:
    st.subheader("Indicators Detail" if is_regional else f"Indicators Detail — {selected_road}")
    if is_regional:
        col_tl = st.columns(1)[0]
        col_br =  st.container()
    else:
        col_tl, col_tr = st.columns(2)
        col_br = st.container()

    # ── Top-left: ASI over time ───────────────────────────────────────────────
    with col_tl:
        st.markdown("##### Severity Index (ASI %) over Time")

        if is_regional:
            # Regional ASI: mean ASI per year across all cause types
            asi_reg = (
                ind4c[ind4c["YEAR"].isin(selected_years)]
                .groupby("YEAR")["ASI_pct"].mean()
                .reset_index()
                .sort_values("YEAR")
            )
            if not empty_check(asi_reg, "regional ASI"):
                fig_asi = px.line(
                    asi_reg, x="YEAR", y="ASI_pct", markers=True,
                    labels={"ASI_pct": "ASI (%)", "YEAR": "Year"},
                )
                add_covid_vrect(fig_asi)
                st.plotly_chart(fig_asi, width="stretch")
        else:
            # Road-level: aggregate ind4_severity_by_section → groupby road+year → mean ASI
            asi_road = (
                ind4s[(ind4s["CARRETERA"] == selected_road) & (ind4s["YEAR"].isin(selected_years))]
                .groupby("YEAR")["ASI_pct"].mean()
                .reset_index()
                .sort_values("YEAR")
            )
            # Regional mean ASI as dashed reference (from ind4_severity_by_cause)
            asi_ref = ind4c[ind4c["YEAR"].isin(selected_years)]["ASI_pct"].mean()

            if not empty_check(asi_road, f"ASI for {selected_road}"):
                fig_asi = go.Figure()
                fig_asi.add_trace(go.Scatter(
                    x=asi_road["YEAR"], y=asi_road["ASI_pct"],
                    mode="lines+markers", name="Road ASI",
                    line=dict(color="#E91E63"),
                ))
                fig_asi.add_hline(
                    y=asi_ref, line_dash="dash", line_color="grey",
                    annotation_text=f"Regional mean: {asi_ref:.1f}%",
                    annotation_position="bottom right",
                )
                fig_asi.update_layout(
                    xaxis=dict(title="Year", dtick=1),
                    yaxis=dict(title="ASI (%)"),
                )
                add_covid_vrect(fig_asi)
                st.plotly_chart(fig_asi, width="stretch")

    # ── Top-right: RCII over time ─────────────────────────────────────────────
    if not is_regional:
        with col_tr:
            st.markdown("##### Road Condition Influence Index over Time")
            # Road-level RCII line with coloured segments by severity
            rcii_road = (
                ind2[(ind2["CARRETERA"] == selected_road) & (ind2["YEAR"].isin(selected_years))]
                .sort_values("YEAR")
            )
            if not empty_check(rcii_road, f"RCII for {selected_road}"):
                fig_rcii = go.Figure()

                # Colour segments: green < 5, orange 5-15, red > 15
                for i in range(len(rcii_road)):
                    row = rcii_road.iloc[i]
                    val = row["RCII_pct"]
                    if val < 5:
                        seg_color = "green"
                    elif val <= 15:
                        seg_color = "orange"
                    else:
                        seg_color = "red"

                    # Draw line segment between consecutive points
                    if i < len(rcii_road) - 1:
                        next_row = rcii_road.iloc[i + 1]
                        fig_rcii.add_trace(go.Scatter(
                            x=[row["YEAR"], next_row["YEAR"]],
                            y=[row["RCII_pct"], next_row["RCII_pct"]],
                            mode="lines", line=dict(color=seg_color, width=3),
                            showlegend=False,
                        ))

                # Scatter markers on top
                fig_rcii.add_trace(go.Scatter(
                    x=rcii_road["YEAR"], y=rcii_road["RCII_pct"],
                    mode="markers", marker=dict(size=8, color="black"),
                    name="RCII",
                ))
                # Boundary lines
                fig_rcii.add_hline(y=5, line_dash="dash", line_color="orange",
                                   annotation_text="Low/Medium (5%)", annotation_position="right")
                fig_rcii.add_hline(y=15, line_dash="dash", line_color="red",
                                   annotation_text="Medium/High (15%)", annotation_position="right")
                fig_rcii.update_layout(
                    xaxis=dict(title="Year", dtick=1),
                    yaxis=dict(title="RCII (%)"),
                )
                add_covid_vrect(fig_rcii)
                st.plotly_chart(fig_rcii, width="stretch")

    # ── Bottom: Dominant cause over time ───────────────────────────────
    with col_br:
        st.markdown("##### Dominant Accident Cause per Year")

        if is_regional:
            # Stacked bar of rel_freq_pct by cause and year from ind6_cause_distribution_road.csv
            cause_reg = (
                ind6d[ind6d["YEAR"].isin(selected_years)]
                .groupby(["YEAR", "TIPO_ACCIDENTE_LABEL"])["n_accidents"]
                .sum()
                .reset_index()
            )
            # Compute relative freq across all roads
            year_totals = cause_reg.groupby("YEAR")["n_accidents"].transform("sum")
            cause_reg["rel_pct"] = cause_reg["n_accidents"] / year_totals * 100

            top5_causes = (
                cause_reg.groupby("TIPO_ACCIDENTE_LABEL")["n_accidents"]
                .sum()
                .nlargest(5)
                .index
            )
            cause_reg = cause_reg[cause_reg["TIPO_ACCIDENTE_LABEL"].isin(top5_causes)]

            if not empty_check(cause_reg, "regional cause distribution"):
                fig_cause = px.bar(
                    cause_reg.sort_values("YEAR"), x="YEAR", y="rel_pct",
                    color="TIPO_ACCIDENTE_LABEL", barmode="stack",
                    labels={"rel_pct": "Relative frequency (%)", "YEAR": "Year",
                            "TIPO_ACCIDENTE_LABEL": "Accident Type"},
                )
                fig_cause.update_layout(
                    xaxis=dict(dtick=1),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.7),
                )
                st.plotly_chart(fig_cause, width="stretch")
        else:
            # Per-road: find dominant cause per year from ind6_cause_dominant_per_section
            dom_road = (
                ind6p[(ind6p["CARRETERA"] == selected_road) & (ind6p["YEAR"].isin(selected_years))]
                .groupby(["YEAR", "dominant_cause"])["n_dominant_accidents"]
                .sum()
                .reset_index()
            )
            # For each year pick the dominant cause (most accidents)
            if not dom_road.empty:
                dom_year = (
                    dom_road.sort_values("n_dominant_accidents", ascending=False)
                    .drop_duplicates(subset=["YEAR"])
                    .sort_values("YEAR")
                )
                # Consistent colour per accident type across years
                all_types = dom_year["dominant_cause"].unique()
                color_map = {t: px.colors.qualitative.Plotly[i % 10] for i, t in enumerate(all_types)}

                fig_cause = px.bar(
                    dom_year, x="n_dominant_accidents", y=dom_year["YEAR"].astype(str),
                    color="dominant_cause", orientation="h",
                    color_discrete_map=color_map,
                    labels={
                        "n_dominant_accidents": "Accidents (dominant type)",
                        "y": "Year",
                        "dominant_cause": "Dominant Cause",
                    },
                )
                fig_cause.update_layout(
                    yaxis=dict(title="Year"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.6),
                )
                st.plotly_chart(fig_cause, width="stretch")
            else:
                st.warning(f"No dominant cause data for {selected_road} in selected years.")


# =============================================================================
# TAB 3 — Black Spots (TCA)
# =============================================================================
with tab3:
    st.subheader("Black Spots (TCA — Traffic Concentration Areas)")
    # Top 30 sections by ACI across all roads (drop NaN ACI rows first)- Only consider all regional roads  
    # st.markdown("#### Top 30 Sections by ACI — All Roads")
    # top_aci = (
    #     ind1[ind1["YEAR"].isin(selected_years)]
    #     .dropna(subset=["ACI"])
    #     .sort_values("ACI", ascending=False)
    #     .head(30)[["SECCION_ID", "CARRETERA", "YEAR", "ACI", "n_accidents", "IS_TCA"]]
    # )
    # top_aci["IS_TCA"] = top_aci["IS_TCA"].apply(lambda x: "Yes" if x == 1 else "No")

    # if not empty_check(top_aci, "top ACI sections"):
    #     st.dataframe(top_aci.rename(columns={"SECCION_ID": "Section", "CARRETERA": "Road"}),
    #                     width="stretch", hide_index=True)

    

    # ── Black Spots Map ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Black Spots Map")

    map_year = st.select_slider(
        "Map year (Black Spots)",
        options=sorted(ind1["YEAR"].unique().tolist()),
        value=sorted(ind1["YEAR"].unique().tolist())[-1],
        help="Select the year to visualize on the black spots map",
    )
    show_tca_only = st.checkbox(
        "Show only Black Spots (TCA)",
        value=True,
        help="If unchecked, shows all sections colored by number of accidents",
    )

    st.caption(
        f"Visualized year: **{map_year}** · Use the slider above to see year-over-year evolution. "
        f"{'Showing only TCA sections (black spots)' if show_tca_only else 'Showing all sections'}"
    )

    # Filter ind1 to the map year 
    bs_map_data = ind1[ind1["YEAR"] == map_year].copy()
    if show_tca_only:
        bs_map_data = bs_map_data[bs_map_data["IS_TCA"] == 1]

    # Build coordinates for each section using real pk from SECCION_ID
    map_section_rows = []
    for _, row in bs_map_data.iterrows():
        coords = get_section_coords(row["CARRETERA"], row["SECCION_ID"], roads_geo)
        if coords is None:
            continue
        lat, lon = coords
        aci_val = float(row["ACI"]) if pd.notna(row.get("ACI")) else 0.0
        n_acc   = int(row["n_accidents"]) if pd.notna(row["n_accidents"]) else 0
        is_tca  = int(row["IS_TCA"]) == 1
        map_section_rows.append({
            "section":     str(row["SECCION_ID"]),
            "road":        row["CARRETERA"],
            "lat":         lat,
            "lon":         lon,
            "n_accidents": n_acc,
            "ACI":         round(aci_val, 2),
            "is_tca":      is_tca,
        })

    bs_map_df = pd.DataFrame(map_section_rows)

    if bs_map_df.empty:
        st.info("No black spot data for this selection on the map. Try unchecking 'Show only Black Spots' or change the year.")
    else:
        # Normalize ACI for color intensity
        max_aci = bs_map_df["ACI"].max() or 1
        bs_map_df["color_intensity"] = (bs_map_df["ACI"] / max_aci * 200 + 55).astype(int)

        # Radius proportional to n_accidents
        max_acc_map = bs_map_df["n_accidents"].max() or 1
        bs_map_df["radius"] = (bs_map_df["n_accidents"] / max_acc_map * 1500 + 400).astype(int)

        # Color: crimson red for confirmed TCAs, orange for high accident rate sections
        bs_map_df["fill_color"] = bs_map_df.apply(
            lambda r: [220, 20, 60, 220] if r["is_tca"]
                      else [255, 140, 0, 160],
            axis=1,
        )

        bs_layer = pdk.Layer(
            "ScatterplotLayer",
            data=bs_map_df,
            get_position=["lon", "lat"],
            get_fill_color="fill_color",
            get_radius="radius",
            pickable=True,
            stroked=True,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255, 180],
        )

        # Context layer: large, faint circles to orient the road
        path_df = geo_to_path_layer(
            roads_geo,
            highlight_road=None if is_regional else selected_road,
        )
        road_layer = pdk.Layer(
            "PathLayer",
            data=path_df,
            get_path="path",
            get_color="color",
            get_width="width",
            width_min_pixels=1,
            pickable=False,
        )

        bs_view = pdk.ViewState(latitude=40.42, longitude=-3.75, zoom=9, pitch=0)
        bs_tooltip = {
            "text": "Road: {road}\nSection: {section}\nAccidents: {n_accidents}\nACI: {ACI}\nBlack Spot (TCA): {is_tca}"
        }

        st.pydeck_chart(pdk.Deck(
            layers=[road_layer, bs_layer],
            initial_view_state=bs_view,
            tooltip=bs_tooltip,
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        ))
        n_shown = len(bs_map_df)
        n_tca   = int(bs_map_df["is_tca"].sum())
        st.caption(
            f"🔴 Red = Black spot (TCA) · 🟠 Orange = High accident rate section"
        )

    # # ── Evolución anual de puntos negros por carretera ────────────────────────
    # st.markdown("#### Black Spots Evolution per Road (all selected years)")
    # tca_evo = (
    #     ind1[(ind1["IS_TCA"] == 1) & (ind1["YEAR"].isin(selected_years))]
    #     .groupby(["YEAR", "CARRETERA"])["IS_TCA"]
    #     .count()
    #     .reset_index()
    #     .rename(columns={"IS_TCA": "tca_count"})
    # )

    # if not tca_evo.empty:
    #     fig_evo = px.line(
    #         tca_evo.sort_values("YEAR"),
    #         x="YEAR", y="tca_count",
    #         color="CARRETERA",
    #         markers=True,
    #         labels={
    #             "tca_count": "Number of Black Spots (TCA)",
    #             "YEAR": "Year",
    #             "CARRETERA": "Road",
    #         },
    #         title="Annual evolution of black spots per road",
    #     )
    #     fig_evo.update_layout(
    #         xaxis=dict(dtick=1),
    #         legend=dict(orientation="h", yanchor="bottom", y=-0.4),
    #     )
    #     add_covid_vrect(fig_evo)
    #     st.plotly_chart(fig_evo, width="stretch")
    # else:
    #     st.info("No black spots in the selected years for this road.")

    # st.divider()

    # # ── Always: TCA summary stats and bar chart ────────────────────────────────
    # st.markdown("#### TCA Summary — Selected Years")

    # tca_scope = ind1[ind1["YEAR"].isin(selected_years)]
    # total_tca = int((tca_scope["IS_TCA"] == 1).sum())
    # st.metric("Total TCA Black Spot Flags", f"{total_tca:,}")

    # # Bar chart: TCA count per year
    # tca_per_year = (
    #     tca_scope[tca_scope["IS_TCA"] == 1]
    #     .groupby("YEAR")["IS_TCA"]
    #     .count()
    #     .reset_index()
    #     .rename(columns={"IS_TCA": "tca_count"})
    #     .sort_values("YEAR")
    # )

    # if not tca_per_year.empty:
    #     bar_colors_tca = ["#b0b0b0" if y in COVID_YEARS else "#F44336" for y in tca_per_year["YEAR"]]
    #     fig_tca = go.Figure(go.Bar(
    #         x=tca_per_year["YEAR"], y=tca_per_year["tca_count"],
    #         name="TCA flags", marker_color=bar_colors_tca,
    #     ))
    #     fig_tca.update_layout(
    #         xaxis=dict(title="Year", dtick=1),
    #         yaxis=dict(title="TCA Black Spot Count"),
    #     )
    #     add_covid_vrect(fig_tca)
    #     st.plotly_chart(fig_tca, width="stretch")
    # else:
    #     st.info("No TCA black spots flagged for this selection.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("Bachelor's Thesis — Road Safety Analysis\nCommunity of Madrid · 2016–2024")
st.sidebar.caption("Data source: DGT / CRTM open data")