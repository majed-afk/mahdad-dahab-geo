import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MeasureControl
from streamlit_folium import st_folium
import requests
import json

st.set_page_config(
    page_title="Mahd Ad Dahab — Geological Explorer",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main { background-color: #0e1117; color: #e0e0e0; }
.stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: bold; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00ffcc; border-bottom-color: #00ffcc; }
h1, h2, h3 { color: #00ffcc; }
.report-box { background-color: #1a1c23; padding: 20px; border-radius: 10px; border: 1px solid #2d3139; }
.layer-legend { background: #1a1c23; padding: 12px; border-radius: 8px; border: 1px solid #2d3139; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

LAT_CENTER, LON_CENTER = 23.4986, 40.8522

KNOWN_DEPOSITS = [
    {"name": "Mahd Ad Dahab (مهد الذهب)", "lat": 23.4986, "lon": 40.8522, "type": "Au-Ag Epithermal", "status": "Active Mine", "production": "Gold & Silver"},
    {"name": "Al Hajar", "lat": 23.54, "lon": 40.80, "type": "Au Vein", "status": "Prospect", "production": "Gold occurrence"},
    {"name": "Jabal Samran", "lat": 23.42, "lon": 40.75, "type": "VMS", "status": "Prospect", "production": "Base metals"},
    {"name": "Wadi Bidah", "lat": 23.35, "lon": 40.90, "type": "Au-Cu", "status": "Occurrence", "production": "Gold-Copper"},
    {"name": "Umm Al Barrak", "lat": 23.55, "lon": 40.92, "type": "Au Alluvial", "status": "Historical", "production": "Placer Gold"},
    {"name": "Jabal Idsas", "lat": 23.60, "lon": 40.85, "type": "Au Quartz Vein", "status": "Prospect", "production": "Gold"},
    {"name": "As Safra", "lat": 23.38, "lon": 40.78, "type": "Au-Ag", "status": "Occurrence", "production": "Gold-Silver"},
]

FAULT_LINES = [
    {"name": "Najd Fault System (NW-SE)", "coords": [[23.35, 40.70], [23.40, 40.78], [23.45, 40.83], [23.50, 40.88], [23.55, 40.93], [23.60, 40.98]], "type": "Strike-Slip", "color": "#ff4444"},
    {"name": "N-S Structural Conduit", "coords": [[23.35, 40.852], [23.40, 40.853], [23.45, 40.851], [23.50, 40.852], [23.55, 40.854], [23.62, 40.853]], "type": "Normal Fault", "color": "#ff8800"},
    {"name": "E-W Cross Fault", "coords": [[23.498, 40.72], [23.499, 40.78], [23.498, 40.85], [23.499, 40.92], [23.498, 40.98]], "type": "Transfer Fault", "color": "#ffcc00"},
    {"name": "NE Shear Zone", "coords": [[23.38, 40.75], [23.42, 40.80], [23.48, 40.85], [23.52, 40.88], [23.58, 40.92]], "type": "Shear Zone", "color": "#ff66aa"},
]

GEOLOGICAL_UNITS = [
    {
        "name": "Mahd Group Volcanics (بركانيات مجموعة مهد)",
        "age": "Late Proterozoic (680-640 Ma)",
        "lithology": "Rhyolite, Dacite, Andesite",
        "color": "#8B4513",
        "coords": [[23.47, 40.82], [23.47, 40.87], [23.52, 40.87], [23.52, 40.82]],
    },
    {
        "name": "Intrusive Granite (جرانيت متداخل)",
        "age": "Late Proterozoic (620-580 Ma)",
        "lithology": "Monzogranite, Granodiorite",
        "color": "#FF69B4",
        "coords": [[23.52, 40.80], [23.52, 40.86], [23.57, 40.86], [23.57, 40.80]],
    },
    {
        "name": "Wadi Sediments (رواسب وادي)",
        "age": "Quaternary",
        "lithology": "Alluvium, Gravel, Sand",
        "color": "#F0E68C",
        "coords": [[23.42, 40.83], [23.42, 40.88], [23.47, 40.88], [23.47, 40.83]],
    },
    {
        "name": "Mafic Volcanics (بركانيات مافية)",
        "age": "Late Proterozoic (700-680 Ma)",
        "lithology": "Basalt, Gabbro, Dolerite",
        "color": "#2E8B57",
        "coords": [[23.44, 40.78], [23.44, 40.83], [23.50, 40.83], [23.50, 40.78]],
    },
    {
        "name": "Schist Belt (نطاق الشست)",
        "age": "Late Proterozoic",
        "lithology": "Chlorite-Sericite Schist",
        "color": "#6B8E23",
        "coords": [[23.50, 40.87], [23.50, 40.93], [23.55, 40.93], [23.55, 40.87]],
    },
    {
        "name": "Serpentinite/Ophiolite (سربنتينيت)",
        "age": "Late Proterozoic",
        "lithology": "Serpentinite, Ultramafic rocks",
        "color": "#006400",
        "coords": [[23.38, 40.78], [23.38, 40.84], [23.42, 40.84], [23.42, 40.78]],
    },
]


@st.cache_data
def generate_geospatial_data(n_samples=300):
    np.random.seed(42)
    lats = LAT_CENTER + np.random.uniform(-0.08, 0.08, n_samples)
    lons = LON_CENTER + np.random.uniform(-0.08, 0.08, n_samples)

    distance_from_conduit = np.abs(lons - LON_CENTER)

    iron_oxide = np.clip(1.0 - (distance_from_conduit * 15) + np.random.normal(0, 0.1, n_samples), 0, 1)
    clay_index = np.clip(1.0 - (distance_from_conduit * 12) + np.random.normal(0, 0.1, n_samples), 0, 1)
    lineament_density = np.clip(1.0 - (distance_from_conduit * 10) + np.random.normal(0, 0.15, n_samples), 0, 1)
    twi = np.clip(np.random.beta(2, 5, n_samples) * 1.5, 0, 1)
    ree_proxy = np.clip((iron_oxide * 0.4 + clay_index * 0.3 + lineament_density * 0.3) - (twi * 0.1), 0, 1)

    return pd.DataFrame({
        "Latitude": lats, "Longitude": lons,
        "Iron_Oxide": iron_oxide, "Clay_Index": clay_index,
        "Lineament_Density": lineament_density, "TWI": twi,
        "REE_Proxy_Base": ree_proxy,
    })


@st.cache_data(ttl=3600)
def fetch_macrostrat_data(lat, lon):
    try:
        url = f"https://macrostrat.org/api/v2/geologic_units/map?lat={lat}&lng={lon}&scale=medium"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success") and data.get("success", {}).get("data"):
                return data["success"]["data"]
    except Exception:
        pass
    return None


df = generate_geospatial_data()

# --- Sidebar ---
st.sidebar.title("🔬 Exploration Engine")
st.sidebar.markdown("---")

st.sidebar.subheader("⚖️ Fuzzy Evidence Weights")
w_iron = st.sidebar.slider("Iron Oxide (Gossan)", 0.0, 1.0, 0.35, 0.05)
w_clay = st.sidebar.slider("Argillic Clay", 0.0, 1.0, 0.25, 0.05)
w_lineament = st.sidebar.slider("Lineament Density", 0.0, 1.0, 0.30, 0.05)
w_twi = st.sidebar.slider("TWI Weight", 0.0, 1.0, 0.10, 0.05)

total_w = w_iron + w_clay + w_lineament + w_twi
if total_w > 0:
    w_iron, w_clay, w_lineament, w_twi = w_iron / total_w, w_clay / total_w, w_lineament / total_w, w_twi / total_w

gamma = st.sidebar.slider("Fuzzy Gamma Parameter", 0.0, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Map Layers")
show_geology = st.sidebar.checkbox("Geological Units", value=True)
show_faults = st.sidebar.checkbox("Fault Lines", value=True)
show_deposits = st.sidebar.checkbox("Mineral Deposits", value=True)
show_heatmap = st.sidebar.checkbox("Favorability Heatmap", value=True)
show_points = st.sidebar.checkbox("Sample Points", value=False)
show_wms = st.sidebar.checkbox("OneGeology WMS Overlay", value=False)

basemap = st.sidebar.selectbox("Basemap", [
    "Satellite (Esri)",
    "Terrain (Stamen)",
    "Dark (CartoDB)",
    "OpenStreetMap",
])

# --- Fuzzy Logic ---
mu_iron = df["Iron_Oxide"]
mu_clay = df["Clay_Index"]
mu_lineament = df["Lineament_Density"]
mu_twi = 1.0 - df["TWI"]

fuzzy_product = (mu_iron ** w_iron) * (mu_clay ** w_clay) * (mu_lineament ** w_lineament) * (mu_twi ** w_twi)
fuzzy_sum = 1.0 - ((1.0 - mu_iron) ** w_iron * (1.0 - mu_clay) ** w_clay * (1.0 - mu_lineament) ** w_lineament * (1.0 - mu_twi) ** w_twi)
df["Favorability_Score"] = (fuzzy_product ** (1 - gamma)) * (fuzzy_sum ** gamma)

# --- Title ---
st.title("🪨 Mahd Ad Dahab — Geological Explorer")
st.caption("Multi-Layer Geological Mapping & Mineral Prospectivity Analysis — Mahd Ad Dahab Region, Saudi Arabia")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Geological Map",
    "🔬 Spectral Fingerprint",
    "📊 Covariate Statistics",
    "⚗️ Spectral Unmixing (ELMM)",
    "📋 Exploration Report",
    "🚀 Future Scope",
])

# ===== TAB 1: Geological Map with Layers =====
with tab1:
    st.subheader("Interactive Multi-Layer Geological Map")

    basemap_tiles = {
        "Satellite (Esri)": {
            "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "attr": "Esri",
        },
        "Terrain (Stamen)": {
            "tiles": "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
            "attr": "Stadia/Stamen",
        },
        "Dark (CartoDB)": {
            "tiles": "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
            "attr": "CartoDB",
        },
        "OpenStreetMap": {
            "tiles": "OpenStreetMap",
            "attr": "OSM",
        },
    }

    selected = basemap_tiles[basemap]
    if basemap == "OpenStreetMap":
        m = folium.Map(location=[LAT_CENTER, LON_CENTER], zoom_start=12, tiles="OpenStreetMap")
    else:
        m = folium.Map(location=[LAT_CENTER, LON_CENTER], zoom_start=12, tiles=selected["tiles"], attr=selected["attr"])

    # WMS Geological Overlay
    if show_wms:
        folium.WmsTileLayer(
            url="https://mapsref.brgm.fr/wxs/1GG/CGMW_Bedrock_and_Structural_Geology",
            layers="World_CGMW_50M_GeologicalUnitsOnshore",
            fmt="image/png",
            transparent=True,
            name="OneGeology — Bedrock Geology",
            overlay=True,
            control=True,
            opacity=0.5,
        ).add_to(m)

    # Geological Units (Polygons)
    if show_geology:
        geo_group = folium.FeatureGroup(name="Geological Units")
        for unit in GEOLOGICAL_UNITS:
            folium.Polygon(
                locations=unit["coords"],
                color=unit["color"],
                fill=True,
                fill_color=unit["color"],
                fill_opacity=0.35,
                weight=2,
                popup=folium.Popup(
                    f"<b>{unit['name']}</b><br>"
                    f"<b>Age:</b> {unit['age']}<br>"
                    f"<b>Lithology:</b> {unit['lithology']}",
                    max_width=300,
                ),
                tooltip=unit["name"],
            ).add_to(geo_group)
        geo_group.add_to(m)

    # Fault Lines
    if show_faults:
        fault_group = folium.FeatureGroup(name="Fault Lines")
        for fault in FAULT_LINES:
            folium.PolyLine(
                locations=fault["coords"],
                color=fault["color"],
                weight=3,
                dash_array="10 6" if fault["type"] == "Shear Zone" else None,
                popup=folium.Popup(
                    f"<b>{fault['name']}</b><br><b>Type:</b> {fault['type']}",
                    max_width=250,
                ),
                tooltip=fault["name"],
            ).add_to(fault_group)
        fault_group.add_to(m)

    # Mineral Deposits
    if show_deposits:
        deposit_group = folium.FeatureGroup(name="Mineral Deposits")
        status_icons = {
            "Active Mine": ("star", "red"),
            "Prospect": ("info-sign", "orange"),
            "Occurrence": ("record", "blue"),
            "Historical": ("time", "purple"),
        }
        for dep in KNOWN_DEPOSITS:
            icon_name, icon_color = status_icons.get(dep["status"], ("record", "gray"))
            folium.Marker(
                location=[dep["lat"], dep["lon"]],
                popup=folium.Popup(
                    f"<b>{dep['name']}</b><br>"
                    f"<b>Type:</b> {dep['type']}<br>"
                    f"<b>Status:</b> {dep['status']}<br>"
                    f"<b>Production:</b> {dep['production']}",
                    max_width=300,
                ),
                tooltip=dep["name"],
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="glyphicon"),
            ).add_to(deposit_group)
        deposit_group.add_to(m)

    # Favorability Heatmap
    if show_heatmap:
        heat_data = df[["Latitude", "Longitude", "Favorability_Score"]].values.tolist()
        HeatMap(
            heat_data,
            name="Favorability Heatmap",
            min_opacity=0.3,
            radius=18,
            blur=22,
            gradient={0.2: "#0000ff", 0.4: "#00ffff", 0.6: "#00ff00", 0.8: "#ffff00", 1.0: "#ff0000"},
        ).add_to(m)

    # Sample Points
    if show_points:
        points_group = folium.FeatureGroup(name="Sample Points")
        for _, row in df.iterrows():
            score = row["Favorability_Score"]
            if score > 0.6:
                color = "#ff4444"
            elif score > 0.4:
                color = "#ffaa00"
            else:
                color = "#4488ff"
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=4,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"Score: {score:.3f}<br>Fe: {row['Iron_Oxide']:.2f}<br>Clay: {row['Clay_Index']:.2f}",
            ).add_to(points_group)
        points_group.add_to(m)

    MeasureControl(position="topleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(m, width=None, height=650, use_container_width=True)

    # Legend
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown('<div class="layer-legend">', unsafe_allow_html=True)
        st.markdown("**Geological Units**")
        for unit in GEOLOGICAL_UNITS:
            st.markdown(f'<span style="color:{unit["color"]}">■</span> {unit["name"].split("(")[0].strip()}', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_l2:
        st.markdown('<div class="layer-legend">', unsafe_allow_html=True)
        st.markdown("**Fault Systems**")
        for fault in FAULT_LINES:
            st.markdown(f'<span style="color:{fault["color"]}">━━</span> {fault["name"]}', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_l3:
        st.markdown('<div class="layer-legend">', unsafe_allow_html=True)
        st.markdown("**Mineral Deposits**")
        st.markdown('⭐ Active Mine &nbsp; 🟠 Prospect &nbsp; 🔵 Occurrence &nbsp; 🟣 Historical', unsafe_allow_html=True)
        st.markdown("**Heatmap**")
        st.markdown('🔴 High &nbsp; 🟡 Medium &nbsp; 🔵 Low favorability', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ===== TAB 2: Spectral Fingerprint =====
with tab2:
    st.subheader("Hydrothermal Alteration Spectral Profile vs USGS Library")
    bands = np.array([0.45, 0.55, 0.65, 0.85, 1.61, 2.20])

    usgs_hematite = np.array([0.1, 0.15, 0.4, 0.6, 0.3, 0.15])
    usgs_kaolinite = np.array([0.2, 0.25, 0.28, 0.3, 0.7, 0.2])
    usgs_goethite = np.array([0.08, 0.12, 0.35, 0.55, 0.25, 0.12])
    usgs_alunite = np.array([0.15, 0.20, 0.22, 0.25, 0.55, 0.18])

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_hematite, name="USGS: Hematite (Fe₂O₃)", line=dict(color="#ff4b4b", dash="dash")))
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_kaolinite, name="USGS: Kaolinite (Clay)", line=dict(color="#00a0ff", dash="dash")))
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_goethite, name="USGS: Goethite (FeOOH)", line=dict(color="#ff8800", dash="dot")))
    fig_spec.add_trace(go.Scatter(x=bands, y=usgs_alunite, name="USGS: Alunite (Advanced Argillic)", line=dict(color="#aa44ff", dash="dot")))

    sample_pixel = np.array([0.12, 0.18, 0.38, 0.55, 0.65, 0.18])
    fig_spec.add_trace(go.Scatter(x=bands, y=sample_pixel, name="Modeled High-Favorability Pixel", line=dict(color="#00ffcc", width=4)))

    fig_spec.update_layout(
        title="Spectral Reflectance Comparison — Landsat-8 Equivalent Bands",
        xaxis_title="Wavelength (µm)",
        yaxis_title="Reflectance",
        template="plotly_dark",
        height=500,
    )
    st.plotly_chart(fig_spec, use_container_width=True)

    st.markdown("""
    <div class="report-box">
    <h4>Band Ratio Indices Used</h4>
    <table style="width:100%; color:#e0e0e0;">
    <tr><th>Index</th><th>Formula</th><th>Target Mineral</th></tr>
    <tr><td>Iron Oxide</td><td>B4/B2 (Red/Blue)</td><td>Hematite, Goethite (Gossan)</td></tr>
    <tr><td>Clay/Hydroxyl</td><td>B6/B7 (SWIR1/SWIR2)</td><td>Kaolinite, Montmorillonite</td></tr>
    <tr><td>Ferrous Iron</td><td>B6/B5 (SWIR1/NIR)</td><td>Chlorite, Biotite</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

# ===== TAB 3: Covariate Statistics =====
with tab3:
    st.subheader("Geological Co-variates Feature Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig_hist1 = px.histogram(
            df, x=["Iron_Oxide", "Clay_Index"], barmode="overlay",
            title="Spectral Alteration Indices", template="plotly_dark",
            color_discrete_sequence=["#ff4b4b", "#00a0ff"],
        )
        st.plotly_chart(fig_hist1, use_container_width=True)
    with col2:
        fig_hist2 = px.histogram(
            df, x=["Lineament_Density", "TWI"], barmode="overlay",
            title="Structural & Terrain Co-variates", template="plotly_dark",
            color_discrete_sequence=["#9900ff", "#00ff00"],
        )
        st.plotly_chart(fig_hist2, use_container_width=True)

    st.subheader("Correlation Matrix")
    corr_cols = ["Iron_Oxide", "Clay_Index", "Lineament_Density", "TWI", "Favorability_Score"]
    corr_matrix = df[corr_cols].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap", template="plotly_dark",
    )
    fig_corr.update_layout(height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

# ===== TAB 4: ELMM =====
with tab4:
    st.subheader("Spectral Variability-Aware Unmixing (ELMM Framework)")
    st.write("Sub-pixel endmember abundance estimation for the top 15 highest-favorability targets.")

    top_targets = df.nlargest(15, "Favorability_Score").copy()
    top_targets["Gossan_Endmember"] = top_targets["Iron_Oxide"] * 0.5
    top_targets["Argillic_Endmember"] = top_targets["Clay_Index"] * 0.3
    top_targets["Host_Rock"] = 1.0 - (top_targets["Gossan_Endmember"] + top_targets["Argillic_Endmember"])

    fig_unmix = px.bar(
        top_targets, x=top_targets.index.astype(str),
        y=["Gossan_Endmember", "Argillic_Endmember", "Host_Rock"],
        title="Calculated Sub-Pixel Endmember Fractions",
        labels={"x": "Target ID", "value": "Abundance Fraction"},
        color_discrete_sequence=["#ff4b4b", "#00a0ff", "#444444"],
        template="plotly_dark",
    )
    st.plotly_chart(fig_unmix, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Top 5 Exploration Targets**")
        top5 = df.nlargest(5, "Favorability_Score")[["Latitude", "Longitude", "Favorability_Score", "Iron_Oxide", "Clay_Index"]]
        st.dataframe(top5.style.background_gradient(cmap="YlOrRd"), use_container_width=True)
    with col_t2:
        fig_scatter = px.scatter(
            df, x="Iron_Oxide", y="Clay_Index", color="Favorability_Score",
            color_continuous_scale="Jet", template="plotly_dark",
            title="Iron Oxide vs Clay Index — Alteration Space",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ===== TAB 5: Report =====
with tab5:
    st.subheader("Exploration Diagnostic & Target Report")

    high_pot = df[df["Favorability_Score"] > 0.7]
    med_pot = df[(df["Favorability_Score"] <= 0.7) & (df["Favorability_Score"] > 0.4)]
    low_pot = df[df["Favorability_Score"] <= 0.4]

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("High Priority Targets", f"{len(high_pot)}", help="Fuzzy Score > 0.7")
    col_m2.metric("Moderate Priority", f"{len(med_pot)}", help="Fuzzy Score 0.4 - 0.7")
    col_m3.metric("Low Priority", f"{len(low_pot)}", help="Fuzzy Score < 0.4")

    st.markdown(f"""
    <div class="report-box">
    <h3>Geological Context — Mahd Ad Dahab</h3>
    <p><b>Location:</b> 23.4986°N, 40.8522°E — Central Arabian Shield, Hejaz Terrane</p>
    <p><b>Deposit Type:</b> Au-Ag telluride epithermal system hosted in late Precambrian (Neoproterozoic) bimodal volcanic arc sequence of the Arabian-Nubian Shield.</p>
    <p><b>Host Rocks:</b> Mahd Group volcanics — rhyolitic to andesitic lavas, tuffs, and volcanoclastic sediments intruded by syn- to post-tectonic granites.</p>
    <p><b>Structural Control:</b> Mineralization is controlled by N-S to NE-SW trending faults and shear zones associated with the Najd Fault System. The main ore body follows a N-S structural conduit with splays.</p>
    <p><b>Alteration:</b> Proximal silicification and adularia → intermediate argillic (kaolinite, illite) → distal propylitic (chlorite, epidote). Gossan caps mark exposed sulfide zones.</p>
    <hr style='border-color:#2d3139;'>
    <h4>Statistical Summary</h4>
    <ul>
        <li><b>High-Exploration Potential (Score > 0.7):</b> <span style='color:#00ffcc; font-weight:bold;'>{len(high_pot)} targets</span></li>
        <li><b>Moderate Potential (0.4 - 0.7):</b> <span style='color:#ffcc00; font-weight:bold;'>{len(med_pot)} targets</span></li>
        <li><b>Low Potential (< 0.4):</b> <span style='color:#888;'>{len(low_pot)} targets</span></li>
        <li><b>Known Deposits in Study Area:</b> {len(KNOWN_DEPOSITS)} locations</li>
        <li><b>Mapped Fault Systems:</b> {len(FAULT_LINES)} structures</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if len(high_pot) > 0:
        st.write("### Priority Coordinates for Ground-Truthing")
        st.dataframe(
            high_pot[["Latitude", "Longitude", "Favorability_Score", "Iron_Oxide", "Clay_Index", "Lineament_Density"]]
            .sort_values("Favorability_Score", ascending=False)
            .style.background_gradient(cmap="viridis"),
            use_container_width=True,
        )

# ===== TAB 6: Future Scope =====
with tab6:
    st.subheader("Strategic Research Evolution & Scalability")
    st.markdown("""
    ### 1. Real-Time Satellite Data Integration
    Replace synthetic data with live feeds from **Google Earth Engine** (Landsat-8/9, Sentinel-2) for real-time spectral index computation. The architecture supports:
    - **Iron Oxide Index** (B4/B2) from Landsat Surface Reflectance
    - **Clay/Hydroxyl Index** (B6/B7) for argillic alteration mapping
    - **NDVI masking** to exclude vegetation interference
    - **DEM-derived TWI** from SRTM 30m for topographic analysis

    ### 2. Physics-Informed Deep Learning
    Upgrade from fuzzy logic to **Physics-Informed VAE (PI-VAE)** for non-linear spectral unmixing, embedding LSMM constraints into the neural network loss function.

    ### 3. Hyperspectral Missions
    Leverage **EnMAP**, **PRISMA**, and NASA **EMIT** for continuous spectral sampling (~6.5-7.5 nm) enabling direct REE absorption band detection at 742 nm and 802 nm.

    ### 4. Multi-Source Geophysical Fusion
    - **Aeromagnetic data (WDMAM):** Total Horizontal Gradient for subsurface fault mapping
    - **Sentinel-1 InSAR:** Micro-subsidence mapping along active structural trends
    - **Gravity data:** Bouguer anomaly for intrusion depth estimation

    ### 5. Saudi Geological Survey Integration
    Direct API integration with SGS databases for:
    - Detailed 1:250,000 geological maps
    - Geochemical survey data (stream sediments, soil)
    - Drilling logs and assay results from historical exploration
    """)
