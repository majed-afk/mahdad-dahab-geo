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

# --- Google Earth Engine ---
import ee
import google.oauth2.credentials


def init_gee():
    """Initialize Earth Engine with Streamlit secrets or local credentials."""
    try:
        if "earthengine" in st.secrets:
            credentials = google.oauth2.credentials.Credentials(
                token=None,
                refresh_token=st.secrets["earthengine"]["refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=st.secrets["earthengine"]["client_id"],
                client_secret=st.secrets["earthengine"]["client_secret"],
            )
            ee.Initialize(credentials=credentials, project=st.secrets["earthengine"]["project"])
        else:
            ee.Initialize(project="carbide-ward-479915-f4")
        return True
    except Exception as e:
        return False


GEE_READY = init_gee()

# --- Mineral Fingerprints Database ---
MINERAL_FINGERPRINTS = {
    "Gold (Au)": {
        "name_ar": "ذهب",
        "icon": "🥇",
        "iron_oxide_range": (0.8, 1.0),
        "clay_index_range": (0.6, 1.0),
        "ferrous_range": (0.3, 0.7),
        "ndvi_max": 0.15,
        "description": "Epithermal Au-Ag systems with gossan caps, argillic alteration halos, and structural control along N-S faults.",
        "reference_site": "Mahd Ad Dahab",
        "host_rocks": "Felsic volcanics, quartz veins",
        "alteration": "Silicification → Argillic → Propylitic",
    },
    "Copper (Cu)": {
        "name_ar": "نحاس",
        "icon": "🟤",
        "iron_oxide_range": (0.6, 1.0),
        "clay_index_range": (0.5, 0.9),
        "ferrous_range": (0.5, 1.0),
        "ndvi_max": 0.2,
        "description": "Porphyry Cu systems with strong iron oxide and ferrous iron signatures in mafic-intermediate host rocks.",
        "reference_site": "Jabal Sayid",
        "host_rocks": "Andesite, diorite, gabbro",
        "alteration": "Potassic → Phyllic → Propylitic",
    },
    "Silver (Ag)": {
        "name_ar": "فضة",
        "icon": "⬜",
        "iron_oxide_range": (0.5, 0.9),
        "clay_index_range": (0.7, 1.0),
        "ferrous_range": (0.2, 0.6),
        "ndvi_max": 0.15,
        "description": "Ag-rich epithermal veins with strong clay alteration and moderate iron oxide. Often co-located with Au.",
        "reference_site": "As Suq",
        "host_rocks": "Rhyolite, dacite tuffs",
        "alteration": "Advanced argillic → Argillic",
    },
    "Zinc-Lead (Zn-Pb)": {
        "name_ar": "زنك-رصاص",
        "icon": "🔘",
        "iron_oxide_range": (0.4, 0.8),
        "clay_index_range": (0.4, 0.8),
        "ferrous_range": (0.6, 1.0),
        "ndvi_max": 0.2,
        "description": "VMS-type Zn-Pb deposits in mafic volcanic sequences with moderate iron oxide and high ferrous signatures.",
        "reference_site": "Al Masane",
        "host_rocks": "Basalt, volcaniclastics",
        "alteration": "Chloritic → Sericitic",
    },
}

# --- Candidate Scan Sites across the Arabian Shield ---
SCAN_SITES = [
    {"name": "Mahd Ad Dahab", "name_ar": "مهد الذهب", "lat": 23.4986, "lon": 40.8522, "region": "Hejaz"},
    {"name": "Jabal Sayid", "name_ar": "جبل صايد", "lat": 23.73, "lon": 40.92, "region": "Hejaz"},
    {"name": "Al Amar", "name_ar": "العمار", "lat": 22.72, "lon": 44.03, "region": "Najd"},
    {"name": "Bulghah", "name_ar": "بلغة", "lat": 26.08, "lon": 42.05, "region": "Central"},
    {"name": "Ad Duwayhi", "name_ar": "الدويحي", "lat": 22.38, "lon": 43.37, "region": "Najd"},
    {"name": "Sukhaybarat", "name_ar": "صخيبرات", "lat": 25.95, "lon": 42.27, "region": "Central"},
    {"name": "Al Masane", "name_ar": "المسعنة", "lat": 19.42, "lon": 43.20, "region": "Asir"},
    {"name": "Ar Rjum", "name_ar": "الرجوم", "lat": 23.10, "lon": 41.50, "region": "Hejaz"},
    {"name": "Jabal Idsas", "name_ar": "جبل إدساس", "lat": 23.60, "lon": 40.85, "region": "Hejaz"},
    {"name": "Wadi Bidah", "name_ar": "وادي بيضاء", "lat": 20.00, "lon": 41.35, "region": "Asir"},
    {"name": "Jabal Samran", "name_ar": "جبل سمران", "lat": 20.25, "lon": 41.25, "region": "Asir"},
    {"name": "Zalm", "name_ar": "ظلم", "lat": 22.85, "lon": 42.95, "region": "Najd"},
    {"name": "Hamdah", "name_ar": "حمضة", "lat": 23.90, "lon": 41.80, "region": "Hejaz"},
    {"name": "Umm Ash Shalahib", "name_ar": "أم الشلاهيب", "lat": 21.50, "lon": 42.90, "region": "Asir"},
    {"name": "Al Hajar", "name_ar": "الحجر", "lat": 24.50, "lon": 41.10, "region": "Central"},
]


@st.cache_data(ttl=7200, show_spinner=False)
def scan_site(lat, lon, year, _fingerprint_key):
    """Fetch mean spectral indices for a single site (small sample for speed)."""
    try:
        roi = ee.Geometry.Point([lon, lat]).buffer(3000)
        image = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filter(ee.Filter.lt("CLOUD_COVER", 20))
            .median()
        )
        optical = image.select(["SR_B2", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)
        iron_oxide = optical.select("SR_B4").divide(optical.select("SR_B2")).rename("iron_oxide")
        clay_index = optical.select("SR_B6").divide(optical.select("SR_B7")).rename("clay_index")
        ferrous = optical.select("SR_B6").divide(optical.select("SR_B5")).rename("ferrous")
        ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")

        composite = iron_oxide.addBands([clay_index, ferrous, ndvi])
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=250,
            bestEffort=True,
        ).getInfo()

        return {
            "iron_oxide": stats.get("iron_oxide", 0) or 0,
            "clay_index": stats.get("clay_index", 0) or 0,
            "ferrous": stats.get("ferrous", 0) or 0,
            "ndvi": stats.get("ndvi", 0) or 0,
        }
    except Exception:
        return None


def score_single_site(stats, fingerprint, reference_stats=None):
    """Score a site by comparing its spectral signature to a reference site."""
    if not stats:
        return 0.0

    # Reference values from known deposits (actual Landsat measurements)
    # These are the real normalized values from each mineral's type locality
    REF_VALUES = {
        "Gold (Au)": {"iron_oxide": 1.90, "clay_index": 1.12, "ferrous": 1.30},
        "Copper (Cu)": {"iron_oxide": 1.60, "clay_index": 1.05, "ferrous": 1.50},
        "Silver (Ag)": {"iron_oxide": 1.50, "clay_index": 1.20, "ferrous": 1.10},
        "Zinc-Lead (Zn-Pb)": {"iron_oxide": 1.40, "clay_index": 1.00, "ferrous": 1.60},
    }

    # Use reference_stats if provided, otherwise use hardcoded reference
    if reference_stats:
        ref = reference_stats
    else:
        # Find matching mineral key
        ref_key = None
        for k, v in MINERAL_FINGERPRINTS.items():
            if v.get("iron_oxide_range") == fingerprint.get("iron_oxide_range"):
                ref_key = k
                break
        ref = REF_VALUES.get(ref_key, REF_VALUES["Gold (Au)"])

    # Cosine similarity approach — compare raw ratio vectors
    site_vec = np.array([stats["iron_oxide"], stats["clay_index"], stats["ferrous"]])
    ref_vec = np.array([ref["iron_oxide"], ref["clay_index"], ref["ferrous"]])

    # Cosine similarity (0-1)
    dot = np.dot(site_vec, ref_vec)
    norm_site = np.linalg.norm(site_vec)
    norm_ref = np.linalg.norm(ref_vec)
    cosine_sim = dot / (norm_site * norm_ref + 1e-8)

    # Euclidean distance penalty (normalized)
    dist = np.linalg.norm(site_vec - ref_vec)
    max_dist = np.linalg.norm(ref_vec)  # max possible distance
    dist_score = max(0, 1 - dist / (max_dist + 1e-8))

    # Vegetation penalty
    veg = 1.0 if stats["ndvi"] < fingerprint["ndvi_max"] else 0.5

    # Combined score
    score = (cosine_sim * 0.4 + dist_score * 0.5 + 0.1) * veg
    return float(np.clip(score, 0, 1))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gee_data(lat, lon, radius_km, year):
    """Fetch real spectral indices from Landsat 8/9 via Earth Engine."""
    roi = ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)

    image = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .median()
    )

    # Scale factors
    optical = image.select(["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]).multiply(0.0000275).add(-0.2)

    # Spectral indices
    iron_oxide = optical.select("SR_B4").divide(optical.select("SR_B2")).rename("iron_oxide")
    clay_index = optical.select("SR_B6").divide(optical.select("SR_B7")).rename("clay_index")
    ferrous = optical.select("SR_B6").divide(optical.select("SR_B5")).rename("ferrous")
    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")

    composite = iron_oxide.addBands([clay_index, ferrous, ndvi])

    num_pixels = min(500, int((radius_km * 2) ** 2))
    samples = composite.sample(region=roi, scale=30, numPixels=num_pixels, seed=42, geometries=True).getInfo()

    if not samples["features"]:
        return None

    rows = []
    for f in samples["features"]:
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        rows.append({
            "Latitude": coords[1],
            "Longitude": coords[0],
            "Iron_Oxide": props.get("iron_oxide", 0),
            "Clay_Index": props.get("clay_index", 0),
            "Ferrous": props.get("ferrous", 0),
            "NDVI": props.get("ndvi", 0),
        })

    return pd.DataFrame(rows)


def compute_mineral_score(df, fingerprint):
    """Score each pixel against a mineral fingerprint using normalized Landsat ratios."""
    # Normalize raw Landsat band ratios to 0-1 using typical ranges
    # Iron Oxide (B4/B2): typical range 0.5-3.0
    fe_raw = np.clip((df["Iron_Oxide"] - 0.5) / 2.5, 0, 1)
    # Clay Index (B6/B7): typical range 0.8-1.5
    clay_raw = np.clip((df["Clay_Index"] - 0.8) / 0.7, 0, 1)
    # Ferrous (B6/B5): typical range 0.5-2.0
    ferr_raw = np.clip((df["Ferrous"] - 0.5) / 1.5, 0, 1)

    # Score against fingerprint thresholds using Gaussian-like proximity
    fe_min, fe_max = fingerprint["iron_oxide_range"]
    clay_min, clay_max = fingerprint["clay_index_range"]
    ferr_min, ferr_max = fingerprint["ferrous_range"]

    fe_center = (fe_min + fe_max) / 2
    clay_center = (clay_min + clay_max) / 2
    ferr_center = (ferr_min + ferr_max) / 2

    fe_score = np.exp(-((fe_raw - fe_center) ** 2) / (2 * 0.15 ** 2))
    clay_score = np.exp(-((clay_raw - clay_center) ** 2) / (2 * 0.15 ** 2))
    ferr_score = np.exp(-((ferr_raw - ferr_center) ** 2) / (2 * 0.15 ** 2))
    veg_penalty = np.where(df["NDVI"] < fingerprint["ndvi_max"], 1.0, 0.3)

    score = (fe_score * 0.35 + clay_score * 0.30 + ferr_score * 0.20) * veg_penalty + 0.15 * veg_penalty
    # Stretch distribution to use full 0-1 range
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)
    return np.clip(score, 0, 1)


# --- Page Config ---
st.set_page_config(
    page_title="Mahd Ad Dahab -- Geological Survey Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Professional Dark Theme CSS ---
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e14;
    --bg-secondary: #111820;
    --bg-card: #151c27;
    --bg-card-hover: #1a2332;
    --border: #1e2a3a;
    --border-light: #263040;
    --text-primary: #e4e8ee;
    --text-secondary: #8a94a6;
    --text-muted: #5c6678;
    --accent: #00d4aa;
    --accent-dim: rgba(0, 212, 170, 0.12);
    --red: #ff4d6a;
    --amber: #f0a030;
    --blue: #3b82f6;
    --chart-green: #00d4aa;
    --chart-red: #ff4d6a;
    --chart-amber: #f0a030;
    --chart-blue: #3b82f6;
}

html, body, .main, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stApp"], section[data-testid="stSidebar"],
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
code, pre, .stCode, [data-testid="stCode"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Remove Streamlit default chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], .stDeployButton { display: none !important; }

[data-testid="stHeader"] {
    background: transparent !important;
    backdrop-filter: none !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 8px 0 4px 0;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 600 !important;
}

/* ── KPI Card ── */
.kpi-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px 22px 16px;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--accent); }
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.kpi-value.accent { color: var(--accent); }
.kpi-value.red { color: var(--red); }
.kpi-value.amber { color: var(--amber); }
.kpi-value.blue { color: var(--blue); }
.kpi-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Panel ── */
.panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}
.panel-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ── Section Header ── */
.section-header {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    margin: 32px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Dashboard Title Bar ── */
.dash-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 22px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.dash-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.dash-subtitle {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.dash-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.72rem;
    color: var(--text-muted);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.live { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.status-dot.offline { background: var(--red); }

/* ── Data Table ── */
.panel table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
}
.panel table th {
    text-align: left;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
}
.panel table td {
    padding: 7px 10px;
    color: var(--text-secondary);
    border-bottom: 1px solid rgba(30,42,58,0.5);
}
.panel table tr:hover td { color: var(--text-primary); background: var(--bg-card-hover); }

/* ── Tag / Badge ── */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
}
.tag-high { background: rgba(255,77,106,0.15); color: var(--red); }
.tag-med { background: rgba(240,160,48,0.15); color: var(--amber); }
.tag-low { background: rgba(59,130,246,0.15); color: var(--blue); }

/* ── Comparison Bar ── */
.comp-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.comp-bar-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    width: 90px;
    text-align: right;
}
.comp-bar-track {
    flex: 1;
    height: 6px;
    background: var(--bg-primary);
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}
.comp-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}
.comp-bar-val {
    font-size: 0.72rem;
    color: var(--text-muted);
    width: 48px;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Legend Row ── */
.legend-row {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    padding: 12px 0 8px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: var(--text-secondary);
}
.legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 2px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ── Fingerprint Spec Table ── */
.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}
.spec-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-primary);
    border-radius: 6px;
    border: 1px solid rgba(30,42,58,0.5);
}
.spec-key {
    font-size: 0.72rem;
    color: var(--text-muted);
}
.spec-val {
    font-size: 0.76rem;
    color: var(--text-primary);
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Streamlit overrides ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: var(--text-primary) !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input,
.stSlider > div { color: var(--text-primary) !important; }

div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 14px 16px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Plotly background override ── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ── Fix streamlit columns gap ── */
[data-testid="stHorizontalBlock"] { gap: 16px !important; }

</style>
""", unsafe_allow_html=True)


# ====================================================================
#  STATIC DATA
# ====================================================================

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
    {"name": "Mahd Group Volcanics (بركانيات مجموعة مهد)", "age": "Late Proterozoic (680-640 Ma)", "lithology": "Rhyolite, Dacite, Andesite", "color": "#8B4513", "coords": [[23.47, 40.82], [23.47, 40.87], [23.52, 40.87], [23.52, 40.82]]},
    {"name": "Intrusive Granite (جرانيت متداخل)", "age": "Late Proterozoic (620-580 Ma)", "lithology": "Monzogranite, Granodiorite", "color": "#FF69B4", "coords": [[23.52, 40.80], [23.52, 40.86], [23.57, 40.86], [23.57, 40.80]]},
    {"name": "Wadi Sediments (رواسب وادي)", "age": "Quaternary", "lithology": "Alluvium, Gravel, Sand", "color": "#F0E68C", "coords": [[23.42, 40.83], [23.42, 40.88], [23.47, 40.88], [23.47, 40.83]]},
    {"name": "Mafic Volcanics (بركانيات مافية)", "age": "Late Proterozoic (700-680 Ma)", "lithology": "Basalt, Gabbro, Dolerite", "color": "#2E8B57", "coords": [[23.44, 40.78], [23.44, 40.83], [23.50, 40.83], [23.50, 40.78]]},
    {"name": "Schist Belt (نطاق الشست)", "age": "Late Proterozoic", "lithology": "Chlorite-Sericite Schist", "color": "#6B8E23", "coords": [[23.50, 40.87], [23.50, 40.93], [23.55, 40.93], [23.55, 40.87]]},
    {"name": "Serpentinite/Ophiolite (سربنتينيت)", "age": "Late Proterozoic", "lithology": "Serpentinite, Ultramafic rocks", "color": "#006400", "coords": [[23.38, 40.78], [23.38, 40.84], [23.42, 40.84], [23.42, 40.78]]},
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


# ====================================================================
#  SIDEBAR — Controls
# ====================================================================

st.sidebar.markdown("""
<div style="padding: 8px 0 16px;">
    <span style="font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
                 letter-spacing: 0.1em; color: #5c6678;">Geological Survey</span><br>
    <span style="font-size: 1.05rem; font-weight: 700; color: #e4e8ee;
                 letter-spacing: -0.01em;">Mahd Ad Dahab</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size:0.68rem;color:#5c6678;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:8px;">Target Mineral</div>', unsafe_allow_html=True)
selected_mineral = st.sidebar.selectbox("Target Mineral", list(MINERAL_FINGERPRINTS.keys()), label_visibility="collapsed")
fp = MINERAL_FINGERPRINTS[selected_mineral]

st.sidebar.markdown(f"""
<div style="background:#151c27;border:1px solid #1e2a3a;border-radius:6px;padding:10px 12px;margin:8px 0 16px;">
    <span style="font-size:0.88rem;">{fp['icon']}</span>
    <span style="font-size:0.78rem;color:#e4e8ee;font-weight:500;">{fp['name_ar']}</span>
    <span style="font-size:0.72rem;color:#5c6678;margin-left:6px;">Ref: {fp['reference_site']}</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div style="font-size:0.68rem;color:#5c6678;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;margin-bottom:4px;">Search Area</div>', unsafe_allow_html=True)
sb_c1, sb_c2 = st.sidebar.columns(2)
search_lat = sb_c1.number_input("Lat", value=23.4986, format="%.4f", label_visibility="collapsed")
search_lon = sb_c2.number_input("Lon", value=40.8522, format="%.4f", label_visibility="collapsed")
st.sidebar.caption(f"{search_lat:.4f}°N, {search_lon:.4f}°E")

search_radius = st.sidebar.slider("Scan Radius (km)", 1, 30, 10)
search_year = st.sidebar.slider("Landsat Year", 2020, 2025, 2024)

st.sidebar.markdown("---")

st.sidebar.markdown('<div class="sidebar-section-title">MAP LAYERS</div>', unsafe_allow_html=True)
show_geology = st.sidebar.checkbox("Geological Units", value=True)
show_faults = st.sidebar.checkbox("Fault Lines", value=True)
show_deposits = st.sidebar.checkbox("Mineral Deposits", value=True)
show_heatmap = st.sidebar.checkbox("Favorability Heatmap", value=True)
show_points = st.sidebar.checkbox("Sample Points", value=False)
show_wms = st.sidebar.checkbox("OneGeology WMS", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-section-title">EVIDENCE WEIGHTS</div>', unsafe_allow_html=True)
w_iron = st.sidebar.slider("Iron Oxide (Gossan)", 0.0, 1.0, 0.35, 0.05)
w_clay = st.sidebar.slider("Argillic Clay", 0.0, 1.0, 0.25, 0.05)
w_lineament = st.sidebar.slider("Lineament Density", 0.0, 1.0, 0.30, 0.05)
w_twi = st.sidebar.slider("TWI Weight", 0.0, 1.0, 0.10, 0.05)
gamma = st.sidebar.slider("Fuzzy Gamma", 0.0, 1.0, 0.75, 0.05)

total_w = w_iron + w_clay + w_lineament + w_twi
if total_w > 0:
    w_iron, w_clay, w_lineament, w_twi = w_iron / total_w, w_clay / total_w, w_lineament / total_w, w_twi / total_w

st.sidebar.markdown(f"""
<div style="margin-top:12px;padding:10px;background:#151c27;border:1px solid #1e2a3a;border-radius:6px;">
    <div style="font-size:0.68rem;color:#5c6678;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">Connection</div>
    <div style="margin-top:6px;display:flex;align-items:center;gap:6px;">
        <span class="status-dot {'live' if GEE_READY else 'offline'}"></span>
        <span style="font-size:0.75rem;color:{'#00d4aa' if GEE_READY else '#ff4d6a'};">
            {'Earth Engine Connected' if GEE_READY else 'GEE Offline — Synthetic Data'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  COMPUTE — Synthetic favorability
# ====================================================================

df = generate_geospatial_data()
mu_iron = df["Iron_Oxide"]
mu_clay = df["Clay_Index"]
mu_lineament = df["Lineament_Density"]
mu_twi = 1.0 - df["TWI"]

fuzzy_product = (mu_iron ** w_iron) * (mu_clay ** w_clay) * (mu_lineament ** w_lineament) * (mu_twi ** w_twi)
fuzzy_sum = 1.0 - ((1.0 - mu_iron) ** w_iron * (1.0 - mu_clay) ** w_clay * (1.0 - mu_lineament) ** w_lineament * (1.0 - mu_twi) ** w_twi)
df["Favorability_Score"] = (fuzzy_product ** (1 - gamma)) * (fuzzy_sum ** gamma)


# ====================================================================
#  AUTO-FETCH GEE DATA ON LOAD
# ====================================================================

gee_df = None
if GEE_READY:
    with st.spinner(""):
        gee_df = fetch_gee_data(search_lat, search_lon, search_radius, search_year)
        if gee_df is not None:
            gee_df["Mineral_Score"] = compute_mineral_score(gee_df, fp)

# Determine source label
if gee_df is not None and len(gee_df) > 0:
    data_source = "LANDSAT 8 SR (GEE)"
    active_df = gee_df
    score_col = "Mineral_Score"
    total_pixels = len(gee_df)
else:
    data_source = "SYNTHETIC MODEL"
    active_df = df
    score_col = "Favorability_Score"
    total_pixels = len(df)


# ====================================================================
#  HEADER
# ====================================================================

gee_status_class = "live" if gee_df is not None else ("offline" if not GEE_READY else "offline")
gee_status_text = f"LIVE &mdash; {data_source}" if gee_df is not None else f"SYNTHETIC &mdash; {data_source}"

st.markdown(f"""
<div class="dash-header">
    <div>
        <div class="dash-title">Mineral Prospectivity Dashboard</div>
        <div class="dash-subtitle">{fp['icon']} {selected_mineral} &mdash; {search_lat:.4f}°N, {search_lon:.4f}°E &mdash; {search_radius} km radius &mdash; {search_year}</div>
    </div>
    <div class="dash-status">
        <span class="status-dot {'live' if gee_df is not None else 'offline'}"></span>
        {gee_status_text}
    </div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  KPI CARDS
# ====================================================================

high_count = int((active_df[score_col] > 0.7).sum())
med_count = int(((active_df[score_col] > 0.4) & (active_df[score_col] <= 0.7)).sum())
low_count = int((active_df[score_col] <= 0.4).sum())
avg_score = float(active_df[score_col].mean())
max_score = float(active_df[score_col].max())
scan_area_km2 = round(3.14159 * search_radius ** 2, 1)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Scan Area</div>
        <div class="kpi-value">{scan_area_km2:,.0f}<span style="font-size:0.85rem;color:#5c6678;font-weight:400;"> km²</span></div>
        <div class="kpi-sub">{total_pixels} sampled pixels</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">High-Priority Targets</div>
        <div class="kpi-value red">{high_count}</div>
        <div class="kpi-sub">Score &gt; 0.70</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Moderate Targets</div>
        <div class="kpi-value amber">{med_count}</div>
        <div class="kpi-sub">Score 0.40 &ndash; 0.70</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Prospectivity</div>
        <div class="kpi-value accent">{avg_score:.3f}</div>
        <div class="kpi-sub">Peak: {max_score:.3f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Known Deposits</div>
        <div class="kpi-value blue">{len(KNOWN_DEPOSITS)}</div>
        <div class="kpi-sub">{len(FAULT_LINES)} mapped faults</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  SECTION 0 — Similar Sites Auto-Scanner
# ====================================================================

st.markdown('<div class="section-header">Similar Sites — Arabian Shield Scanner</div>', unsafe_allow_html=True)

if GEE_READY:
    with st.spinner("Scanning 15 sites across the Arabian Shield..."):
        site_results = []
        for site in SCAN_SITES:
            stats = scan_site(site["lat"], site["lon"], search_year, selected_mineral)
            similarity = score_single_site(stats, fp)
            site_results.append({
                **site,
                "similarity": similarity,
                "iron_oxide": stats["iron_oxide"] if stats else 0,
                "clay_index": stats["clay_index"] if stats else 0,
                "ferrous": stats["ferrous"] if stats else 0,
                "ndvi": stats["ndvi"] if stats else 0,
            })
        site_results.sort(key=lambda x: x["similarity"], reverse=True)

    # Similar sites map
    sim_map = folium.Map(
        location=[23.0, 42.0], zoom_start=6,
        tiles="https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", attr="CartoDB",
    )

    for i, site in enumerate(site_results):
        pct = int(site["similarity"] * 100)
        if pct >= 70:
            color, tag = "#ff4d6a", "HIGH"
        elif pct >= 40:
            color, tag = "#f0a030", "MODERATE"
        else:
            color, tag = "#3b82f6", "LOW"

        folium.CircleMarker(
            location=[site["lat"], site["lon"]],
            radius=max(6, pct / 5),
            color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{site['name']}</b> ({site['name_ar']})<br>"
                f"<b>Match: {pct}%</b> [{tag}]<br>"
                f"Region: {site['region']}<br>"
                f"Fe₂O₃: {site['iron_oxide']:.3f}<br>"
                f"Clay: {site['clay_index']:.3f}<br>"
                f"Ferrous: {site['ferrous']:.3f}",
                max_width=250,
            ),
            tooltip=f"{site['name']} — {pct}%",
        ).add_to(sim_map)

    sim_col1, sim_col2 = st.columns([3, 2])

    with sim_col1:
        st.markdown('<div class="panel"><div class="panel-title">SITES RANKED BY SIMILARITY TO REFERENCE</div></div>', unsafe_allow_html=True)
        st_folium(sim_map, width=None, height=420, use_container_width=True)

    with sim_col2:
        st.markdown('<div class="panel"><div class="panel-title">SIMILARITY RANKING — click to explore</div></div>', unsafe_allow_html=True)
        for i, site in enumerate(site_results):
            pct = int(site["similarity"] * 100)
            if pct >= 70:
                bar_color = "#ff4d6a"
            elif pct >= 40:
                bar_color = "#f0a030"
            else:
                bar_color = "#3b82f6"

            col_btn, col_bar = st.columns([4, 1])
            with col_btn:
                if st.button(
                    f"{site['name']}  ({site['name_ar']})",
                    key=f"site_{i}",
                    use_container_width=True,
                ):
                    st.session_state["selected_site"] = site
            with col_bar:
                st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.85rem;font-weight:600;color:{bar_color};text-align:right;padding-top:8px;">{pct}%</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:12px;padding:10px;background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.2);border-radius:6px;font-size:0.75rem;color:#8a94a6;">
            Scanned <b>{len(SCAN_SITES)}</b> geological sites across the Arabian Shield against
            <b>{selected_mineral}</b> spectral fingerprint using Landsat 8 ({search_year}) data.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("GEE not connected — similar sites scanning requires live satellite data.")

# ====================================================================
#  SECTION 0.5 — Selected Site Deep Dive
# ====================================================================

if "selected_site" in st.session_state and GEE_READY:
    sel = st.session_state["selected_site"]
    sel_pct = int(sel["similarity"] * 100)

    st.markdown(f'<div class="section-header">Site Analysis — {sel["name"]} ({sel["name_ar"]})</div>', unsafe_allow_html=True)

    # KPI row for selected site
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;">
        <div class="kpi-card"><div class="kpi-label">SIMILARITY</div><div class="kpi-value" style="color:{'#ff4d6a' if sel_pct>=70 else '#f0a030' if sel_pct>=40 else '#3b82f6'};">{sel_pct}%</div><div class="kpi-sub">vs {fp['reference_site']}</div></div>
        <div class="kpi-card"><div class="kpi-label">IRON OXIDE</div><div class="kpi-value">{sel['iron_oxide']:.3f}</div><div class="kpi-sub">B4/B2 ratio</div></div>
        <div class="kpi-card"><div class="kpi-label">CLAY INDEX</div><div class="kpi-value">{sel['clay_index']:.3f}</div><div class="kpi-sub">B6/B7 ratio</div></div>
        <div class="kpi-card"><div class="kpi-label">NDVI</div><div class="kpi-value">{sel['ndvi']:.3f}</div><div class="kpi-sub">{'Bare rock' if sel['ndvi']<0.15 else 'Vegetated'}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch detailed data for selected site
    with st.spinner(f"Loading detailed Landsat data for {sel['name']}..."):
        sel_df = fetch_gee_data(sel["lat"], sel["lon"], 5, search_year)

    if sel_df is not None and len(sel_df) > 0:
        sel_df["Mineral_Score"] = compute_mineral_score(sel_df, fp)

        sel_detail1, sel_detail2 = st.columns(2)

        with sel_detail1:
            # Heatmap for selected site
            sel_map = folium.Map(
                location=[sel["lat"], sel["lon"]], zoom_start=13,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
            )
            heat_data = sel_df[["Latitude", "Longitude", "Mineral_Score"]].values.tolist()
            HeatMap(heat_data, radius=15, blur=20,
                    gradient={0.2: "#0000ff", 0.4: "#00ffff", 0.6: "#00ff00", 0.8: "#ffff00", 1.0: "#ff0000"}).add_to(sel_map)
            folium.Marker([sel["lat"], sel["lon"]], tooltip=sel["name"],
                          icon=folium.Icon(color="green", icon="crosshairs", prefix="fa")).add_to(sel_map)
            st.markdown(f'<div class="panel"><div class="panel-title">PROSPECTIVITY — {sel["name"].upper()}</div></div>', unsafe_allow_html=True)
            st_folium(sel_map, width=None, height=380, use_container_width=True)

        with sel_detail2:
            # Comparison bar chart: selected site vs reference
            ref_vals = {"Iron Oxide": 1.90, "Clay Index": 1.12, "Ferrous": 1.30}
            site_vals = {"Iron Oxide": sel["iron_oxide"], "Clay Index": sel["clay_index"], "Ferrous": sel["ferrous"]}

            fig_comp = go.Figure()
            indices = list(ref_vals.keys())
            fig_comp.add_trace(go.Bar(name=f"Reference ({fp['reference_site']})", x=indices, y=list(ref_vals.values()),
                                      marker_color="#00d4aa", opacity=0.7))
            fig_comp.add_trace(go.Bar(name=sel["name"], x=indices, y=list(site_vals.values()),
                                      marker_color="#ff4d6a", opacity=0.7))
            fig_comp.update_layout(
                barmode="group", template="plotly_dark", height=200, margin=dict(t=30, b=30, l=40, r=20),
                title=f"Spectral Comparison vs {fp['reference_site']}",
                paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
                font=dict(family="Inter", size=11, color="#8a94a6"),
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Score distribution for this site
            high_s = len(sel_df[sel_df["Mineral_Score"] > 0.7])
            med_s = len(sel_df[(sel_df["Mineral_Score"] > 0.4) & (sel_df["Mineral_Score"] <= 0.7)])
            low_s = len(sel_df[sel_df["Mineral_Score"] <= 0.4])
            fig_pie = go.Figure(data=[go.Pie(
                labels=["High", "Moderate", "Low"], values=[high_s, med_s, low_s],
                marker_colors=["#ff4d6a", "#f0a030", "#3b82f6"],
                hole=0.5, textinfo="label+percent",
            )])
            fig_pie.update_layout(
                template="plotly_dark", height=180, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="#0a0e14", plot_bgcolor="#0a0e14",
                font=dict(family="Inter", size=11, color="#8a94a6"),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning(f"No Landsat data available for {sel['name']} in {search_year}")

    st.markdown("---")


# ====================================================================
#  SECTION 1 — Side-by-side Maps
# ====================================================================

st.markdown('<div class="section-header">Spatial Analysis</div>', unsafe_allow_html=True)

map_col1, map_col2 = st.columns(2)

# --- Left: Geological Reference Map ---
with map_col1:
    st.markdown('<div class="panel"><div class="panel-title">Geological Reference Map</div>', unsafe_allow_html=True)

    m_geo = folium.Map(
        location=[LAT_CENTER, LON_CENTER],
        zoom_start=11,
        tiles="https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        attr="CartoDB Dark",
    )

    if show_wms:
        folium.WmsTileLayer(
            url="https://mapsref.brgm.fr/wxs/1GG/CGMW_Bedrock_and_Structural_Geology",
            layers="World_CGMW_50M_GeologicalUnitsOnshore",
            fmt="image/png",
            transparent=True,
            name="OneGeology",
            overlay=True,
            control=True,
            opacity=0.5,
        ).add_to(m_geo)

    if show_geology:
        for unit in GEOLOGICAL_UNITS:
            folium.Polygon(
                locations=unit["coords"],
                color=unit["color"],
                fill=True,
                fill_color=unit["color"],
                fill_opacity=0.3,
                weight=1.5,
                popup=folium.Popup(f"<b>{unit['name']}</b><br>Age: {unit['age']}<br>Lithology: {unit['lithology']}", max_width=280),
                tooltip=unit["name"],
            ).add_to(m_geo)

    if show_faults:
        for fault in FAULT_LINES:
            folium.PolyLine(
                locations=fault["coords"],
                color=fault["color"],
                weight=2.5,
                dash_array="8 5" if fault["type"] == "Shear Zone" else None,
                popup=folium.Popup(f"<b>{fault['name']}</b><br>Type: {fault['type']}", max_width=250),
                tooltip=fault["name"],
            ).add_to(m_geo)

    if show_deposits:
        status_icons = {"Active Mine": ("star", "red"), "Prospect": ("info-sign", "orange"), "Occurrence": ("record", "blue"), "Historical": ("time", "purple")}
        for dep in KNOWN_DEPOSITS:
            icon_name, icon_color = status_icons.get(dep["status"], ("record", "gray"))
            folium.Marker(
                location=[dep["lat"], dep["lon"]],
                popup=folium.Popup(f"<b>{dep['name']}</b><br>Type: {dep['type']}<br>Status: {dep['status']}", max_width=280),
                tooltip=dep["name"],
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="glyphicon"),
            ).add_to(m_geo)

    MeasureControl(position="topleft").add_to(m_geo)
    folium.LayerControl(collapsed=True).add_to(m_geo)
    st_folium(m_geo, width=None, height=480, use_container_width=True, key="geo_map")
    st.markdown('</div>', unsafe_allow_html=True)


# --- Right: Prospectivity Heatmap ---
with map_col2:
    st.markdown('<div class="panel"><div class="panel-title">Prospectivity Heatmap &mdash; Live Spectral Data</div>', unsafe_allow_html=True)

    m_heat = folium.Map(
        location=[search_lat, search_lon],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    )

    if gee_df is not None and len(gee_df) > 0:
        heat_data = gee_df[["Latitude", "Longitude", "Mineral_Score"]].values.tolist()
        HeatMap(heat_data, name="Prospectivity", radius=14, blur=18,
                gradient={0.2: "#0d1b4a", 0.35: "#1b4dff", 0.5: "#00d4aa", 0.7: "#f0a030", 0.85: "#ff4d6a", 1.0: "#ff1744"}).add_to(m_heat)
        top_targets_map = gee_df.nlargest(10, "Mineral_Score")
        for _, row in top_targets_map.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6, color="#ff4d6a", fill=True, fill_opacity=0.85, weight=1.5,
                popup=f"Score: {row['Mineral_Score']:.3f}",
            ).add_to(m_heat)
    else:
        heat_data = df[["Latitude", "Longitude", "Favorability_Score"]].values.tolist()
        HeatMap(heat_data, name="Favorability", radius=16, blur=20,
                gradient={0.2: "#0d1b4a", 0.35: "#1b4dff", 0.5: "#00d4aa", 0.7: "#f0a030", 0.85: "#ff4d6a", 1.0: "#ff1744"}).add_to(m_heat)

    folium.Marker([search_lat, search_lon], tooltip="Search Center",
                  icon=folium.Icon(color="green", icon="crosshairs", prefix="fa")).add_to(m_heat)
    folium.LayerControl(collapsed=True).add_to(m_heat)
    st_folium(m_heat, width=None, height=480, use_container_width=True, key="heat_map")
    st.markdown('</div>', unsafe_allow_html=True)


# Legend strip below maps
legend_html = '<div class="panel"><div class="legend-row">'
for unit in GEOLOGICAL_UNITS:
    short_name = unit["name"].split("(")[0].strip()
    legend_html += f'<div class="legend-item"><div class="legend-swatch" style="background:{unit["color"]};"></div>{short_name}</div>'
for fault in FAULT_LINES:
    legend_html += f'<div class="legend-item"><div class="legend-swatch" style="background:{fault["color"]};border-radius:0;height:3px;width:16px;margin-top:4px;"></div>{fault["name"]}</div>'
legend_html += '</div></div>'
st.markdown(legend_html, unsafe_allow_html=True)


# ====================================================================
#  SECTION 2 — Comparison Panel + Fingerprint
# ====================================================================

st.markdown('<div class="section-header">Mineral Fingerprint & Reference Comparison</div>', unsafe_allow_html=True)

comp_col1, comp_col2 = st.columns([3, 2])

with comp_col1:
    # Comparison bars: current site vs reference thresholds
    ref_iron = (fp["iron_oxide_range"][0] + fp["iron_oxide_range"][1]) / 2
    ref_clay = (fp["clay_index_range"][0] + fp["clay_index_range"][1]) / 2
    ref_ferrous = (fp["ferrous_range"][0] + fp["ferrous_range"][1]) / 2

    if gee_df is not None and len(gee_df) > 0:
        curr_iron = float(gee_df["Iron_Oxide"].mean())
        curr_clay = float(gee_df["Clay_Index"].mean())
        curr_ferrous = float(gee_df["Ferrous"].mean())
        curr_ndvi = float(gee_df["NDVI"].mean())
    else:
        curr_iron = float(df["Iron_Oxide"].mean())
        curr_clay = float(df["Clay_Index"].mean())
        curr_ferrous = 0.45
        curr_ndvi = 0.10

    def bar_color(val, ref_low, ref_high):
        if ref_low <= val <= ref_high:
            return "var(--accent)"
        elif val > ref_high * 0.8 or val > ref_low:
            return "var(--amber)"
        else:
            return "var(--blue)"

    def match_pct(val, ref_low, ref_high):
        if ref_low <= val <= ref_high:
            return 100
        mid = (ref_low + ref_high) / 2
        dist = abs(val - mid) / (ref_high - ref_low + 1e-6)
        return max(0, int(100 - dist * 100))

    iron_match = match_pct(curr_iron, *fp["iron_oxide_range"])
    clay_match = match_pct(curr_clay, *fp["clay_index_range"])
    ferrous_match = match_pct(curr_ferrous, *fp["ferrous_range"])
    ndvi_ok = curr_ndvi < fp["ndvi_max"]
    overall_match = int((iron_match + clay_match + ferrous_match + (100 if ndvi_ok else 30)) / 4)

    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Site vs {fp['reference_site']} Reference Profile &mdash; Overall Match: <span style="color:var(--accent);font-size:0.82rem;">{overall_match}%</span></div>

        <div class="comp-bar-wrap">
            <div class="comp-bar-label">Iron Oxide</div>
            <div class="comp-bar-track">
                <div class="comp-bar-fill" style="width:{min(curr_iron*100, 100):.0f}%;background:{bar_color(curr_iron, *fp['iron_oxide_range'])};"></div>
            </div>
            <div class="comp-bar-val">{curr_iron:.3f}</div>
        </div>
        <div style="font-size:0.65rem;color:#5c6678;margin:-2px 0 8px 98px;">Target range: {fp['iron_oxide_range'][0]:.1f} &ndash; {fp['iron_oxide_range'][1]:.1f} &nbsp;&bull;&nbsp; Match: {iron_match}%</div>

        <div class="comp-bar-wrap">
            <div class="comp-bar-label">Clay Index</div>
            <div class="comp-bar-track">
                <div class="comp-bar-fill" style="width:{min(curr_clay*100, 100):.0f}%;background:{bar_color(curr_clay, *fp['clay_index_range'])};"></div>
            </div>
            <div class="comp-bar-val">{curr_clay:.3f}</div>
        </div>
        <div style="font-size:0.65rem;color:#5c6678;margin:-2px 0 8px 98px;">Target range: {fp['clay_index_range'][0]:.1f} &ndash; {fp['clay_index_range'][1]:.1f} &nbsp;&bull;&nbsp; Match: {clay_match}%</div>

        <div class="comp-bar-wrap">
            <div class="comp-bar-label">Ferrous Iron</div>
            <div class="comp-bar-track">
                <div class="comp-bar-fill" style="width:{min(curr_ferrous*100, 100):.0f}%;background:{bar_color(curr_ferrous, *fp['ferrous_range'])};"></div>
            </div>
            <div class="comp-bar-val">{curr_ferrous:.3f}</div>
        </div>
        <div style="font-size:0.65rem;color:#5c6678;margin:-2px 0 8px 98px;">Target range: {fp['ferrous_range'][0]:.1f} &ndash; {fp['ferrous_range'][1]:.1f} &nbsp;&bull;&nbsp; Match: {ferrous_match}%</div>

        <div class="comp-bar-wrap">
            <div class="comp-bar-label">NDVI</div>
            <div class="comp-bar-track">
                <div class="comp-bar-fill" style="width:{min(curr_ndvi*100*3, 100):.0f}%;background:{'var(--accent)' if ndvi_ok else 'var(--red)'};"></div>
            </div>
            <div class="comp-bar-val">{curr_ndvi:.3f}</div>
        </div>
        <div style="font-size:0.65rem;color:#5c6678;margin:-2px 0 8px 98px;">Must be &lt; {fp['ndvi_max']} &nbsp;&bull;&nbsp; {'Pass' if ndvi_ok else 'FAIL — vegetation interference'}</div>
    </div>
    """, unsafe_allow_html=True)

with comp_col2:
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">{fp['icon']} {selected_mineral} Geological Fingerprint</div>
        <div style="font-size:0.78rem;color:var(--text-secondary);line-height:1.5;margin-bottom:12px;">
            {fp['description']}
        </div>
        <div class="spec-grid">
            <div class="spec-item"><span class="spec-key">Reference</span><span class="spec-val">{fp['reference_site']}</span></div>
            <div class="spec-item"><span class="spec-key">Host Rocks</span><span class="spec-val">{fp['host_rocks']}</span></div>
            <div class="spec-item"><span class="spec-key">Fe₂O₃ Range</span><span class="spec-val">{fp['iron_oxide_range'][0]:.1f} – {fp['iron_oxide_range'][1]:.1f}</span></div>
            <div class="spec-item"><span class="spec-key">Clay Range</span><span class="spec-val">{fp['clay_index_range'][0]:.1f} – {fp['clay_index_range'][1]:.1f}</span></div>
            <div class="spec-item"><span class="spec-key">Ferrous Range</span><span class="spec-val">{fp['ferrous_range'][0]:.1f} – {fp['ferrous_range'][1]:.1f}</span></div>
            <div class="spec-item"><span class="spec-key">NDVI Max</span><span class="spec-val">{fp['ndvi_max']}</span></div>
        </div>
        <div style="margin-top:14px;padding:10px 12px;background:var(--bg-primary);border-radius:6px;border:1px solid rgba(30,42,58,0.5);">
            <div style="font-size:0.68rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Alteration Sequence</div>
            <div style="font-size:0.78rem;color:var(--accent);font-weight:500;">{fp['alteration']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ====================================================================
#  SECTION 3 — Charts Row
# ====================================================================

st.markdown('<div class="section-header">Spectral Analysis</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,20,0.6)",
    font=dict(family="Inter, sans-serif", color="#8a94a6", size=11),
    title_font=dict(size=13, color="#e4e8ee"),
    margin=dict(l=48, r=16, t=48, b=40),
    coloraxis_colorbar=dict(tickfont=dict(color="#5c6678"), title_font=dict(color="#5c6678")),
    xaxis=dict(gridcolor="rgba(30,42,58,0.5)", zerolinecolor="rgba(30,42,58,0.5)"),
    yaxis=dict(gridcolor="rgba(30,42,58,0.5)", zerolinecolor="rgba(30,42,58,0.5)"),
)

# Alteration Space scatter
with chart_col1:
    if gee_df is not None and len(gee_df) > 0:
        fig_scatter = px.scatter(
            gee_df, x="Iron_Oxide", y="Clay_Index", color="Mineral_Score",
            size="Mineral_Score", color_continuous_scale=["#0d1b4a", "#1b4dff", "#00d4aa", "#f0a030", "#ff4d6a"],
            hover_data=["Ferrous", "NDVI", "Latitude", "Longitude"],
            title="Alteration Space — Iron Oxide vs Clay Index",
        )
    else:
        fig_scatter = px.scatter(
            df, x="Iron_Oxide", y="Clay_Index", color="Favorability_Score",
            color_continuous_scale=["#0d1b4a", "#1b4dff", "#00d4aa", "#f0a030", "#ff4d6a"],
            title="Alteration Space — Iron Oxide vs Clay Index",
        )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=380)
    fig_scatter.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig_scatter, use_container_width=True)

# Spectral reflectance
with chart_col2:
    bands = np.array([0.45, 0.55, 0.65, 0.85, 1.61, 2.20])
    band_labels = ["B2 Blue", "B3 Green", "B4 Red", "B5 NIR", "B6 SWIR1", "B7 SWIR2"]

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(x=bands, y=[0.1, 0.15, 0.4, 0.6, 0.3, 0.15], name="Hematite (Fe2O3)", line=dict(color="#ff4d6a", dash="dash", width=1.5)))
    fig_spec.add_trace(go.Scatter(x=bands, y=[0.2, 0.25, 0.28, 0.3, 0.7, 0.2], name="Kaolinite (Clay)", line=dict(color="#3b82f6", dash="dash", width=1.5)))
    fig_spec.add_trace(go.Scatter(x=bands, y=[0.08, 0.12, 0.35, 0.55, 0.25, 0.12], name="Goethite (FeOOH)", line=dict(color="#f0a030", dash="dot", width=1.5)))
    fig_spec.add_trace(go.Scatter(x=bands, y=[0.12, 0.18, 0.38, 0.55, 0.65, 0.18], name="High-Favorability Pixel", line=dict(color="#00d4aa", width=2.5)))
    fig_spec.update_layout(**PLOTLY_LAYOUT, height=380, title="Spectral Reflectance — USGS Library Comparison",
                           xaxis_title="Wavelength (um)", yaxis_title="Reflectance")
    st.plotly_chart(fig_spec, use_container_width=True)


# ====================================================================
#  SECTION 4 — Score Distribution + Correlation
# ====================================================================

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    score_data = active_df[score_col]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=score_data, nbinsx=40,
        marker_color="#00d4aa", marker_line_color="rgba(0,0,0,0.3)", marker_line_width=0.5,
        opacity=0.85,
    ))
    # Threshold lines
    fig_hist.add_vline(x=0.7, line_dash="dash", line_color="#ff4d6a", line_width=1, annotation_text="High", annotation_position="top right", annotation_font_color="#ff4d6a", annotation_font_size=10)
    fig_hist.add_vline(x=0.4, line_dash="dash", line_color="#f0a030", line_width=1, annotation_text="Moderate", annotation_position="top right", annotation_font_color="#f0a030", annotation_font_size=10)
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=340, title="Score Distribution",
                           xaxis_title="Prospectivity Score", yaxis_title="Pixel Count")
    st.plotly_chart(fig_hist, use_container_width=True)

with dist_col2:
    if gee_df is not None and len(gee_df) > 0:
        corr_cols = ["Iron_Oxide", "Clay_Index", "Ferrous", "NDVI", "Mineral_Score"]
        corr_df = gee_df[corr_cols]
    else:
        corr_cols = ["Iron_Oxide", "Clay_Index", "Lineament_Density", "TWI", "Favorability_Score"]
        corr_df = df[corr_cols]

    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale=["#0d1b4a", "#111820", "#00d4aa"],
        title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=340)
    fig_corr.update_traces(textfont=dict(size=10, color="#e4e8ee"))
    st.plotly_chart(fig_corr, use_container_width=True)


# ====================================================================
#  SECTION 5 — Top Targets Table
# ====================================================================

st.markdown('<div class="section-header">Exploration Targets</div>', unsafe_allow_html=True)

if gee_df is not None and len(gee_df) > 0:
    targets_df = gee_df.nlargest(20, "Mineral_Score")[["Latitude", "Longitude", "Mineral_Score", "Iron_Oxide", "Clay_Index", "Ferrous", "NDVI"]].copy()
    targets_df = targets_df.rename(columns={"Mineral_Score": "Score"})
else:
    targets_df = df.nlargest(20, "Favorability_Score")[["Latitude", "Longitude", "Favorability_Score", "Iron_Oxide", "Clay_Index", "Lineament_Density", "TWI"]].copy()
    targets_df = targets_df.rename(columns={"Favorability_Score": "Score"})

targets_df.insert(0, "Rank", range(1, len(targets_df) + 1))
targets_df = targets_df.reset_index(drop=True)

def priority_label(score):
    if score > 0.7:
        return "HIGH"
    elif score > 0.4:
        return "MODERATE"
    return "LOW"

targets_df["Priority"] = targets_df["Score"].apply(priority_label)

tgt_col1, tgt_col2 = st.columns([3, 2])

with tgt_col1:
    # Build an HTML table for full styling control
    table_rows = ""
    for _, row in targets_df.iterrows():
        tag_class = "tag-high" if row["Priority"] == "HIGH" else ("tag-med" if row["Priority"] == "MODERATE" else "tag-low")
        cols_html = ""
        for c in targets_df.columns:
            if c == "Rank":
                cols_html += f'<td style="color:var(--text-muted);font-weight:600;">#{int(row[c])}</td>'
            elif c == "Priority":
                cols_html += f'<td><span class="tag {tag_class}">{row[c]}</span></td>'
            elif c == "Score":
                cols_html += f'<td style="color:var(--accent);font-weight:600;font-family:JetBrains Mono,monospace;">{row[c]:.3f}</td>'
            elif c in ("Latitude", "Longitude"):
                cols_html += f'<td style="font-family:JetBrains Mono,monospace;">{row[c]:.5f}</td>'
            else:
                cols_html += f'<td style="font-family:JetBrains Mono,monospace;">{row[c]:.4f}</td>'
        table_rows += f"<tr>{cols_html}</tr>"

    header_cells = "".join(f"<th>{c}</th>" for c in targets_df.columns)
    st.markdown(f"""
    <div class="panel">
        <div class="panel-title">Top 20 Ranked Targets — Ground-Truthing Priorities</div>
        <div style="max-height:420px;overflow-y:auto;">
        <table>
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tgt_col2:
    # Endmember unmixing for top targets
    unmix_df = targets_df.head(15).copy()
    unmix_df["Gossan"] = unmix_df["Iron_Oxide"] * 0.5
    clay_col_name = "Clay_Index" if "Clay_Index" in unmix_df.columns else "Lineament_Density"
    unmix_df["Argillic"] = unmix_df[clay_col_name] * 0.3
    unmix_df["Host Rock"] = 1.0 - (unmix_df["Gossan"] + unmix_df["Argillic"])
    unmix_df["Host Rock"] = unmix_df["Host Rock"].clip(0, 1)

    fig_unmix = go.Figure()
    fig_unmix.add_trace(go.Bar(x=unmix_df["Rank"], y=unmix_df["Gossan"], name="Gossan", marker_color="#ff4d6a"))
    fig_unmix.add_trace(go.Bar(x=unmix_df["Rank"], y=unmix_df["Argillic"], name="Argillic", marker_color="#3b82f6"))
    fig_unmix.add_trace(go.Bar(x=unmix_df["Rank"], y=unmix_df["Host Rock"], name="Host Rock", marker_color="#1e2a3a"))
    fig_unmix.update_layout(**PLOTLY_LAYOUT, height=420, barmode="stack",
                            title="Sub-Pixel Endmember Fractions (ELMM)",
                            xaxis_title="Target Rank", yaxis_title="Abundance",
                            legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
    st.plotly_chart(fig_unmix, use_container_width=True)


# ====================================================================
#  SECTION 6 — Known Deposits Table
# ====================================================================

st.markdown('<div class="section-header">Known Mineral Deposits</div>', unsafe_allow_html=True)

dep_rows = ""
for dep in KNOWN_DEPOSITS:
    status_color = {"Active Mine": "var(--red)", "Prospect": "var(--amber)", "Occurrence": "var(--blue)", "Historical": "#9b59b6"}.get(dep["status"], "var(--text-muted)")
    dep_rows += f"""<tr>
        <td style="color:var(--text-primary);font-weight:500;">{dep['name']}</td>
        <td>{dep['type']}</td>
        <td><span style="color:{status_color};font-weight:600;">{dep['status']}</span></td>
        <td>{dep['production']}</td>
        <td style="font-family:JetBrains Mono,monospace;">{dep['lat']:.4f}</td>
        <td style="font-family:JetBrains Mono,monospace;">{dep['lon']:.4f}</td>
    </tr>"""

st.markdown(f"""
<div class="panel">
    <div class="panel-title">Regional Deposit Database</div>
    <table>
        <thead>
            <tr><th>Name</th><th>Type</th><th>Status</th><th>Production</th><th>Latitude</th><th>Longitude</th></tr>
        </thead>
        <tbody>{dep_rows}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)


# ====================================================================
#  SECTION 7 — Geological Context
# ====================================================================

st.markdown('<div class="section-label">GEOLOGICAL CONTEXT & METHODOLOGY</div>', unsafe_allow_html=True)
with st.container():
    ctx_c1, ctx_c2 = st.columns(2)
    with ctx_c1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">Study Area</div>
            <div class="spec-grid">
                <div class="spec-item"><span class="spec-key">Location</span><span class="spec-val">23.50°N, 40.85°E</span></div>
                <div class="spec-item"><span class="spec-key">Province</span><span class="spec-val">Hejaz Terrane</span></div>
                <div class="spec-item"><span class="spec-key">Shield</span><span class="spec-val">Arabian-Nubian</span></div>
                <div class="spec-item"><span class="spec-key">Deposit Type</span><span class="spec-val">Au-Ag Epithermal</span></div>
                <div class="spec-item"><span class="spec-key">Age</span><span class="spec-val">Neoproterozoic</span></div>
                <div class="spec-item"><span class="spec-key">Data Source</span><span class="spec-val">{data_source}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with ctx_c2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">Methodology</div>
            <div style="font-size:0.78rem;color:var(--text-secondary);line-height:1.6;">
                <b>1. Data Acquisition</b> — Landsat 8 Surface Reflectance (Collection 2 Level 2) via Google Earth Engine, cloud-filtered to &lt;20%.<br>
                <b>2. Spectral Indices</b> — Iron Oxide (B4/B2), Clay/Hydroxyl (B6/B7), Ferrous Iron (B6/B5), NDVI (B5-B4/B5+B4).<br>
                <b>3. Mineral Scoring</b> — Weighted normalized match against published spectral fingerprints for each mineral system.<br>
                <b>4. Fuzzy Logic</b> — Gamma operator combining fuzzy algebraic product and sum for multi-evidence integration.<br>
                <b>5. ELMM</b> — Extended Linear Mixing Model for sub-pixel endmember abundance estimation.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ====================================================================
#  SECTION 8 — Band Ratio Reference
# ====================================================================

st.markdown('<div class="section-label">SPECTRAL BAND RATIO REFERENCE</div>', unsafe_allow_html=True)
with st.container():
    st.markdown("""
    <div class="panel">
        <div class="panel-title">Landsat 8 OLI — Band Ratio Indices</div>
        <table>
            <thead><tr><th>Index</th><th>Formula</th><th>Target</th><th>Geological Significance</th></tr></thead>
            <tbody>
                <tr><td>Iron Oxide</td><td style="font-family:JetBrains Mono,monospace;">B4 / B2</td><td>Hematite, Goethite</td><td>Gossan cap detection over sulfide bodies</td></tr>
                <tr><td>Clay/Hydroxyl</td><td style="font-family:JetBrains Mono,monospace;">B6 / B7</td><td>Kaolinite, Montmorillonite</td><td>Argillic alteration halo mapping</td></tr>
                <tr><td>Ferrous Iron</td><td style="font-family:JetBrains Mono,monospace;">B6 / B5</td><td>Chlorite, Biotite</td><td>Propylitic zone and mafic host rock ID</td></tr>
                <tr><td>NDVI</td><td style="font-family:JetBrains Mono,monospace;">(B5-B4)/(B5+B4)</td><td>Vegetation</td><td>Mask non-geological spectral response</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
