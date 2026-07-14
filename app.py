import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

# 1. Page Configuration & Theme Settings
st.set_page_config(
    page_title="Project Elephant Analytics Workspace",
    page_icon="🐘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for layout card formatting
st.markdown("""
<style>
/* 1. Target the actual container Streamlit makes for cards */
div[data-testid="stMetric"] {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    border-left: 5px solid #2E7D32;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
}

/* 2. Force the text inside Streamlit metrics to be dark gray */
div[data-testid="stMetricLabel"] p {
    color: #262730 !important;
}

/* 3. Force the numbers inside Streamlit metrics to be blue/green */
div[data-testid="stMetricValue"] div {
    color: #1a5f7a !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🐘 Project Elephant: Geospatial & Behavioral Intelligence Workspace")
st.markdown("---")

# 2. Optimized Caching Mechanism for Large Telemetry Records
@st.cache_data
def load_data():
    df = pd.read_csv("Elepant_dataset_web.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df = df.sort_values(by=['individual-local-identifier', 'timestamp']).reset_index(drop=True)
    
    # Precompute Vectorized Physics Metrics
    df['prev_lat'] = df.groupby('individual-local-identifier')['location-lat'].shift(1)
    df['prev_long'] = df.groupby('individual-local-identifier')['location-long'].shift(1)
    
    lon1, lat1 = np.radians(df['location-long'].values), np.radians(df['location-lat'].values)
    lon2, lat2 = np.radians(df['prev_long'].values), np.radians(df['prev_lat'].values)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['distance_km'] = c * 6371.0
    
    df['prev_timestamp'] = df.groupby('individual-local-identifier')['timestamp'].shift(1)
    df['time_diff_hours'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600.0
    df['speed_kmh'] = np.where(df['time_diff_hours'] > 0, df['distance_km'] / df['time_diff_hours'], 0.0)
    
    df['distance_km'] = df['distance_km'].fillna(0.0)
    df['time_diff_hours'] = df['time_diff_hours'].fillna(0.0)
    df['speed_kmh'] = df['speed_kmh'].fillna(0.0)
    return df

with st.spinner("Processing Matrix Arrays & Mapping Spatial Boundaries..."):
    df = load_data()

# 3. Dynamic Sidebar Control Panel (Highly Interactive)
st.sidebar.header("🛠️ Dashboard Control Panel")

st.sidebar.subheader("Dataset Controls")
elephant_list = sorted(df['individual-local-identifier'].dropna().unique().tolist())
selected_elephant = st.sidebar.selectbox("Select Subject Elephant ID:", elephant_list)

# Global filtering step based on dropdown input selection
elephant_df = df[df['individual-local-identifier'] == selected_elephant].copy()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Live DBSCAN Configurations")
st.sidebar.markdown("Adjust these sliders to manipulate the core habitat clustering constraints live on the center viewport map.")

# DYNAMIC SLIDERS: These force the backend ML clustering to re-calculate instantly
eps_input = st.sidebar.slider("Clustering Radius Boundary (Epsilon in km)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
min_samples_input = st.sidebar.slider("Minimum Density Dwell Pings (min_samples)", min_value=10, max_value=200, value=50, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Customization Settings")
map_theme = st.sidebar.selectbox("Base Map Texture:", ["OpenStreetMap", "CartoDB Positron", "CartoDB DarkMatter"])
sample_size = st.sidebar.slider("Geospatial Map Rendering Limit (Points)", min_value=5000, max_value=30000, value=15000, step=2500)

# 4. Interactive KPI Scorecard Summary Displays
st.header(f"📈 Real-Time Diagnostics: Individual Subject {selected_elephant}")
col1, col2, col3, col4 = st.columns(4)

total_pings = len(elephant_df)
total_dist = elephant_df['distance_km'].sum()
total_hours = elephant_df['time_diff_hours'].sum()
avg_speed = (total_dist / total_hours) if total_hours > 0 else 0.0
max_speed = elephant_df['speed_kmh'].max()

with col1:
    st.metric(label="Total Data Pings Logged", value=f"{total_pings:,}")
with col2:
    st.markdown(f"<div class='metric-card'><b>Total Accumulated Range</b><br><span style='font-size:24px; font-weight:bold; color:#1B5E20;'>{total_dist:,.2f} km</span></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><b>Computed Net Velocity</b><br><span style='font-size:24px; font-weight:bold; color:#1B5E20;'>{avg_speed:.4f} km/h</span></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><b>Maximum Peak Sprint</b><br><span style='font-size:24px; font-weight:bold; color:#1B5E20;'>{max_speed:.2f} km/h</span></div>", unsafe_allow_html=True)

# 5. ML Clustering Infrastructure Block
coords = elephant_df[['location-lat', 'location-long']].values
coords_radians = np.radians(coords)
db = DBSCAN(eps=eps_input/6371.0, min_samples=min_samples_input, metric='haversine', algorithm='ball_tree')
elephant_df['cluster_label'] = db.fit_predict(coords_radians)
num_clusters = len(set(elephant_df['cluster_label'])) - (1 if -1 in elephant_df['cluster_label'] else 0)

# 6. Interactive Geospatial Mapping Canvas
st.markdown(" ")
st.subheader("🗺️ Reactive Geospatial Habitat Footprint & Home Range Constraints")

map_center = [elephant_df['location-lat'].mean(), elephant_df['location-long'].mean()]
m = folium.Map(location=map_center, zoom_start=10, tiles=map_theme)

# Plot interactive localized points via Heatmapping layer
heat_data = elephant_df.sample(min(sample_size, len(elephant_df)))[['location-lat', 'location-long']].values.tolist()
HeatMap(heat_data, radius=8, max_zoom=12, blur=5).add_to(m)

# Highlight identified structural core points using marker nodes
unique_clusters = set(elephant_df['cluster_label']) - {-1}
for c_id in unique_clusters:
    c_points = elephant_df[elephant_df['cluster_label'] == c_id]
    folium.Marker(
        location=[c_points['location-lat'].mean(), c_points['location-long'].mean()],
        popup=f"<b>Core Zone Cluster: {c_id}</b><br>Concentration Volume: {len(c_points):,} points found.",
        icon=folium.Icon(color="darkgreen" if map_theme != "CartoDB DarkMatter" else "lightgreen", icon="info-sign")
    ).add_to(m)

# Plot Convex Hull boundary perimeter line elements
if len(coords) >= 3:
    try:
        hull = ConvexHull(coords)
        hull_points = [coords[vertex].tolist() for vertex in hull.vertices]
        hull_points.append(hull_points[0])
        folium.PolyLine(hull_points, color="#D32F2F", weight=3.5, opacity=0.85, tooltip="Convex Hull Boundary Extent").add_to(m)
    except Exception:
        pass

st_folium(m, width=1400, height=550, returned_objects=[])

# 7. Modernized Visualization Graphics Panels
st.markdown("---")
st.subheader("📊 Analytical Trajectory Summaries")
left_chart, right_chart = st.columns(2)

# Update plot styles globally for visual consistency
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'

with left_chart:
    st.markdown("#### Seasonal Activity Fluctuations")
    fig, ax = plt.subplots()
    sns.countplot(data=elephant_df, x='month', hue='month', palette='crest', ax=ax, legend=False)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_xlabel("Month of Annual Cycle")
    ax.set_ylabel("Data Points Captured")
    st.pyplot(fig)

with right_chart:
    st.markdown("#### Diurnal Velocities (Hourly Average Movement Speeds)")
    fig, ax = plt.subplots()
    hourly_speed = elephant_df.groupby('hour')['speed_kmh'].mean().reset_index()
    sns.barplot(data=hourly_speed, x='hour', y='speed_kmh', palette='magma', hue='hour', legend=False, ax=ax)
    ax.set_xlabel("Hour of Day (24h Clock)")
    ax.set_ylabel("Mean Calculated Speed (km/h)")
    st.pyplot(fig)