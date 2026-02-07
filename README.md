# Datathon 2026 — U.S. County Air Quality Intelligence Dashboard

# Author: Winston Wihono, Celine Sachi, Jeha Lee, Wendy Shi

An interactive data science project analyzing historical air quality patterns across U.S. counties using a **data-driven severity index (DAQSI)**, clustering techniques, and a Streamlit visualization dashboard.

This project goes beyond counting unhealthy days. It builds a **statistically principled Air Quality Severity Index**, clusters counties by **multi-year behavior**, and presents the results in an interactive dashboard.

---

## Project Goals

- Construct a **Data-Driven AQ Severity Index (DAQSI)** using log-frequency weighting
- Analyze air quality patterns from **2019–2025**
- Identify **high-risk air quality regimes** using clustering
- Visualize patterns with an interactive **Streamlit dashboard**
- Provide a reproducible pipeline from raw EPA data → insight

---

Core Methodology

### 1. Data-Driven AQ Severity Index (DAQSI)

Instead of arbitrary weights, AQI categories are weighted by:

\[
w = -\log(\text{frequency})
\]

Rare, dangerous days receive higher weight.  
Common, clean days receive lower weight.  
Good days subtract from the score.

This produces a **standardized, comparable environmental burden metric**.

Implemented in:
aqsi_gen.py

---

### 2. Multi-Year Behavioral Clustering

Counties are not ranked by magnitude.  
They are clustered by **AQSI trajectory from 2019–2025**:

\[
[DAQSI_{2019}, ..., DAQSI_{2025}]
\]

This reveals:

- Chronically polluted regions
- Wildfire-driven episodic regions
- Consistently clean regions
- Regions with changing trends

Implemented in:
cluster.py

run using
```bash
python -m cluster
```

---
### 3. Peak Pollution Prediction
This project combines six years of EPA county-level AQI data (2020–2025) to identify places with consistently poor air quality.

What the script does:

- Merges yearly AQI files into one dataset

- Cleans inconsistent column names

- Converts AQI fields to numeric

- Aggregates by State + County across years

Metrics computed per county:

- Worst recorded AQI (max_aqi)

- Chronic severity (p90_aqi)

- Typical conditions (median_aqi)

- Average unhealthy days (overall + sensitive groups)

- Risk labels

Using top 10% thresholds:
Extreme Risk: high in both max and p90 AQI
High Risk: high in one
Lower Risk: below thresholds
Unknown: missing data
---
### 4. Interactive Dashboard

The Streamlit app visualizes:

- Counties and states with worst air days
- Pollutant exposure composition
- Regional AQI trends
- Severity distribution by region

Implemented in:
dashboard.py

# Setup dashboard and running dashboard locally

### Install dependencies

```bash
python -m pip install -r requirements.txt
python -m streamlit run dashboard.py
```

## Repository Structure
datathon2026/
│
├── dashboard.py # Streamlit dashboard app
├── aqsi_gen.py # DAQSI metric generator
├── cluster.py # K-means clustering on AQSI time series
├── requirements.txt # Python dependencies
│
├── data/
│ ├── annual_aqi_by_county_2019.csv
│ ├── annual_aqi_by_county_2020.csv
│ ├── annual_aqi_by_county_2021.csv
│ ├── annual_aqi_by_county_2022.csv
│ ├── annual_aqi_by_county_2023.csv
│ ├── annual_aqi_by_county_2024.csv
│ ├── annual_aqi_by_county_2025.csv
│ └── aqsi_all_years.csv # Generated combined dataset with DAQSI
│
└── assets/ # (optional) images or visuals