# 🐘 Project Elephant: Geospatial & Behavioral Intelligence Workspace

An interactive, high-performance data science and unsupervised machine learning workspace tracking African Elephants (*Loxodonta africana*) across Etosha National Park, Namibia. This system transforms over **2.9 million rows** of raw GPS telemetry logs into a live, reactive diagnostic dashboard analyzing home ranges, seasonal migration trends, and core behavioral hotspots.

🌐 **Live Dashboard Link:** `[Insert your live Streamlit Community Cloud URL here once deployed]`

---

## Key Features & Architectural Highlights

- **Dynamic Interactive Dashboard:** Built with Streamlit, enabling users to select individual elephants from a dropdown filter and instantly regenerate all metrics, behavioral profiles, and spatial mapping layers.
- **High-Performance Geospatial Vectorization:** Replaced computationally slow python row-by-row iteration loops with fully vectorized NumPy and Pandas calculations (Haversine distance formulas), allowing physics transformations over millions of records to execute almost instantly.
- **Unsupervised Machine Learning Hotspot Clustering:** Utilizes **DBSCAN** configured with a custom Haversine radian metric to distinguish core habitat dwell zones (watering holes, feeding grounds) from transitional migratory noise vectors.
- **Geometric Home Range Delineation:** Implements **SciPy Convex Hulls** to dynamically calculate and map the exact territory extent boundaries per individual subject.
- **Advanced UI Styling:** Integrated custom CSS layout styling cards and interactive dark-matter geospatial layer textures to provide a production-ready dashboard workspace.

---

##  Analytics Dashboard Preview

- **Core Movement Metrics:** Real-time summary showing Total GPS Pings, Cumulative Traveled Distance (km), Net Velocity (km/h), and Maximum Sprint Velocities.
- **Geospatial Mapping Layers:** Blends interactive Folium Heatmaps with localized marker nodes representing machine learning-extracted habitat hotspots.
- **Temporal Behavioral Profiles:** Visualizes seasonal presence variances by month alongside diurnal movement behaviors tracking average speeds against a 24-hour clock.

---

##  Tech Stack & Dependencies

The project is built entirely using Python and leverages production-grade scientific libraries:

- **Core Data Handling:** `pandas`, `numpy`
- **Machine Learning & Geometry:** `scikit-learn` (DBSCAN), `scipy` (ConvexHull)
- **Visualizations:** `matplotlib`, `seaborn`
- **Geospatial Mapping:** `folium`, `streamlit-folium`
- **Dashboard Framework:** `streamlit`

---

##  Project Directory Structure

```text
project-elephant/
│
├── .gitignore               # Excludes large data frames and checkpoints from pushing
├── README.md                # Project documentation and workspace manual
├── requirements.txt         # Package dependencies file for server deployment
├── app.py                   # Main production Streamlit web application script
├── project_analysis.ipynb   # Initial prototyping, experimentation, and EDA notebook
└── Elepant_dataset.csv     # Local raw telemetry CSV tracking data (Blocked from Git push)


## Acknowledgments & Dataset Credits

The underlying telemetry tracking dataset used in this project was sourced from Movebank (Study: African elephant in Etosha National Park).

Full credit and gratitude go to the original field researchers, wildlife biologists, and data authors who managed the tracking equipment and published this open-source study on Movebank. Their crucial empirical work enables computational wildlife research, spatial engineering, and predictive conservation modeling worldwide.



## Author

**Priya**

*AI/Ml Student*