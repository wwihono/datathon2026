import csv
import os
from collections import defaultdict
from statistics import mean

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

DATA_PATH = "data/annual_aqi_by_county_2025.csv"

NUM_FIELDS = {
    "Year",
    "Days with AQI",
    "Good Days",
    "Moderate Days",
    "Unhealthy for Sensitive Groups Days",
    "Unhealthy Days",
    "Very Unhealthy Days",
    "Hazardous Days",
    "Max AQI",
    "90th Percentile AQI",
    "Median AQI",
    "Days CO",
    "Days NO2",
    "Days Ozone",
    "Days PM2.5",
    "Days PM10",
}

REGION_BY_STATE = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "DC": "South",
    "WV": "South",
    "AL": "South",
    "KY": "South",
    "MS": "South",
    "TN": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "AZ": "West",
    "CO": "West",
    "ID": "West",
    "MT": "West",
    "NV": "West",
    "NM": "West",
    "UT": "West",
    "WY": "West",
    "AK": "West",
    "CA": "West",
    "HI": "West",
    "OR": "West",
    "WA": "West",
}

STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Puerto Rico": "PR",
}

POLLUTANTS = {
    "PM2.5": "Days PM2.5",
    "Ozone": "Days Ozone",
    "NO2": "Days NO2",
    "PM10": "Days PM10",
    "CO": "Days CO",
}

ACCENT = ["#ff6b6b", "#ffd93d", "#6bcBef", "#9b5de5", "#00f5d4"]


def to_int(value: str) -> int:
    value = (value or "").strip()
    if value == "":
        return 0
    return int(float(value))


def load_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                if k in NUM_FIELDS:
                    clean[k] = to_int(v)
                else:
                    clean[k] = (v or "").strip()
            rows.append(clean)
    return rows


def top_n(items, n=10):
    return sorted(items, key=lambda x: x[1], reverse=True)[:n]


st.set_page_config(
    page_title="Air Quality",
    page_icon="🌎",
    layout="wide",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.app-title { font-family: 'Fraunces', serif; font-size: 44px; margin-bottom: 4px; }
.subtitle { color: #9aa7c7; margin-bottom: 24px; }
.hero { background: radial-gradient(circle at 10% 20%, #243053, #0b0f17 65%); padding: 22px 26px; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08); }
.card { background: #141b2d; padding: 18px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 12px 30px rgba(0,0,0,0.25); }
.section-title { font-family: 'Fraunces', serif; font-size: 24px; margin: 8px 0 14px; }
.story { color: #c9d4f2; font-size: 14px; line-height: 1.6; }
</style>
""",
    unsafe_allow_html=True,
)

rows = load_rows(DATA_PATH)
if not rows:
    st.error("No rows loaded. Check AQI_DATASET_PATH.")
    st.stop()

years = sorted({r["Year"] for r in rows})

with st.container():
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown("<div class='app-title'>Air Quality Patterns Across U.S. Counties</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Track hotspots, pollutant fingerprints, and regional severity at a glance.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Filters")
    year = st.selectbox("Year", years, index=len(years) - 1)
    states = sorted({r["State"] for r in rows})
    selected_states = st.multiselect("States", states, default=[])
    top_k = st.slider("Top K (counties)", 5, 25, 10)

filtered = [r for r in rows if r["Year"] == year]
if selected_states:
    filtered = [r for r in filtered if r["State"] in selected_states]

if not filtered:
    st.warning("No data for the current filters.")
    st.stop()

# KPIs
county_count = len({(r["State"], r["County"]) for r in filtered})
state_count = len({r["State"] for r in filtered})

total_unhealthy = sum(
    r["Unhealthy Days"] + r["Very Unhealthy Days"] + r["Hazardous Days"]
    for r in filtered
)
max_aqi = max(r["Max AQI"] for r in filtered)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Counties", county_count)
with k2:
    st.metric("States", state_count)
with k3:
    st.metric("Total Unhealthy+", total_unhealthy)
with k4:
    st.metric("Max AQI (peak)", max_aqi)

# Section 1: Where the worst days cluster
st.markdown("<div class='section-title'>1. Where do the worst days cluster?</div>", unsafe_allow_html=True)

county_totals = defaultdict(int)
state_totals = defaultdict(int)
for r in filtered:
    total = r["Unhealthy Days"] + r["Very Unhealthy Days"] + r["Hazardous Days"]
    county_totals[(r["State"], r["County"])] += total
    state_totals[r["State"]] += total

county_top = top_n(
    [(f"{county}, {state}", total) for (state, county), total in county_totals.items()],
    n=top_k,
)
state_top = top_n(list(state_totals.items()), n=min(10, len(state_totals)))

c1, c2 = st.columns([1.25, 1])
with c1:
    fig_counties = px.treemap(
        names=[k for k, _ in county_top],
        parents=[""] * len(county_top),
        values=[v for _, v in county_top],
        color=[v for _, v in county_top],
        color_continuous_scale="Turbo",
    )
    fig_counties.update_layout(
        height=420,
        title="Top Counties by Unhealthy + Very Unhealthy + Hazardous Days",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_counties, width="stretch")

with c2:
    fig_states = go.Figure(
        data=[
            go.Scatter(
                x=[v for _, v in state_top],
                y=[k for k, _ in state_top],
                mode="markers+text",
                text=[k for k, _ in state_top],
                textposition="middle right",
                marker=dict(
                    size=[max(10, v / 3) for _, v in state_top],
                    color=[v for _, v in state_top],
                    colorscale="Plasma",
                    showscale=True,
                ),
            )
        ]
    )
    fig_states.update_layout(
        height=420,
        title="Top States by Unhealthy + Hazardous Days",
        xaxis_title="Days",
        yaxis_title="",
        margin=dict(l=20, r=20, t=50, b=30),
    )
    st.plotly_chart(fig_states, width="stretch")

# Section 2: Pollutant profile
st.markdown("<div class='section-title'>2. Which pollutants dominate exposure?</div>", unsafe_allow_html=True)

pollutant_totals = {label: 0 for label in POLLUTANTS}
for r in filtered:
    for label, field in POLLUTANTS.items():
        pollutant_totals[label] += r[field]

fig_pollutants = go.Figure(
    data=[
        go.Pie(
            labels=list(pollutant_totals.keys()),
            values=list(pollutant_totals.values()),
            hole=0.45,
            marker=dict(colors=ACCENT),
        )
    ]
)
fig_pollutants.update_layout(
    height=380,
    title="Pollutant Exposure Share",
    margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(orientation="h", y=-0.1),
)
st.plotly_chart(fig_pollutants, width="stretch")

# Section 3: Regional contrast
st.markdown("<div class='section-title'>3. Regional contrast</div>", unsafe_allow_html=True)

region_year = defaultdict(list)
for r in rows:
    if selected_states and r["State"] not in selected_states:
        continue
    abbr = STATE_TO_ABBR.get(r["State"], "")
    region = REGION_BY_STATE.get(abbr, "Other")
    region_year[(region, r["Year"])].append(r["Median AQI"])

series = []
for (region, yr), vals in region_year.items():
    if vals:
        series.append({"Region": region, "Year": yr, "Median AQI": mean(vals)})

if series:
    fig_region = px.line(
        series,
        x="Year",
        y="Median AQI",
        color="Region",
        markers=True,
        title="Median AQI Trend by Region",
        color_discrete_sequence=ACCENT,
    )
    fig_region.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=30))
    st.plotly_chart(fig_region, width="stretch")
else:
    st.info("Not enough data to build regional trend lines.")

# Section 4: Severity spread by region (box plot)
st.markdown("<div class='section-title'>4. Severity spread by region</div>", unsafe_allow_html=True)

box_rows = []
for r in filtered:
    abbr = STATE_TO_ABBR.get(r["State"], "")
    region = REGION_BY_STATE.get(abbr, "Other")
    box_rows.append({"Region": region, "Max AQI": r["Max AQI"]})

fig_box = px.box(
    box_rows,
    x="Region",
    y="Max AQI",
    color="Region",
    color_discrete_sequence=ACCENT,
    points="outliers",
    title="Max AQI Spread by Region",
)
fig_box.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=30))
st.plotly_chart(fig_box, width="stretch")
