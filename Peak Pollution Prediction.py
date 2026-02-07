import pandas as pd
import os

df1 = pd.read_csv("data/annual_aqi_by_county_2020.csv")
df2 = pd.read_csv("data/annual_aqi_by_county_2021.csv")
df3 = pd.read_csv("data/annual_aqi_by_county_2022.csv")
df4 = pd.read_csv("data/annual_aqi_by_county_2023.csv")
df5 = pd.read_csv("data/annual_aqi_by_county_2024.csv")
df6 = pd.read_csv("data/annual_aqi_by_county_2025.csv")

appended_df = pd.concat(
    [df1, df2, df3, df4, df5, df6],
    axis=0)

appended_df.columns = appended_df.columns.str.strip()
column_aliases = {
    "90th Percentile AQI ": "90th Percentile AQI",
    "90th percentile AQI": "90th Percentile AQI",
    "90th Percentile_AQI": "90th Percentile AQI",
}
appended_df = appended_df.rename(columns=column_aliases)

required_cols = [
    "State",
    "County",
    "Year",
    "Max AQI",
    "90th Percentile AQI",
    "Median AQI",
    "Unhealthy Days",
    "Unhealthy for Sensitive Groups Days",
]
missing = [c for c in required_cols if c not in appended_df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

numeric_cols = [
    "Max AQI",
    "90th Percentile AQI",
    "Median AQI",
    "Unhealthy Days",
    "Unhealthy for Sensitive Groups Days",
]
for col in numeric_cols:
    appended_df[col] = pd.to_numeric(appended_df[col], errors="coerce")

print(appended_df.head())
print("Columns:", appended_df.columns.tolist())
print("90th Percentile AQI dtype:", appended_df["90th Percentile AQI"].dtype)
print("90th Percentile AQI NaN %:", appended_df["90th Percentile AQI"].isna().mean())

# Sanity Check 
appended_df.duplicated(subset=["State", "County", "Year"]).sum()
appended_df.groupby(["State","County"])["Year"].nunique().value_counts()

county_extremes = (
    appended_df
    .groupby(["State", "County"], as_index=False)
    .agg(
        max_aqi=("Max AQI", "max"),
        p90_aqi=("90th Percentile AQI", "max"),
        median_aqi=("Median AQI", "mean"),
        mean_days_unhealthy=(
            "Unhealthy Days", "mean"
        ),
        mean_days_sensitive=(
            "Unhealthy for Sensitive Groups Days", "mean"
        )
    )
)
print(county_extremes.head())

top_max_aqi = (
    county_extremes
    .sort_values("max_aqi", ascending=False)
    .head(20)
)

print(top_max_aqi)

top_p90_aqi = (
    county_extremes
    .sort_values("p90_aqi", ascending=False)
    .head(20)
)

print(top_p90_aqi)

valid_p90 = county_extremes["p90_aqi"].dropna()
valid_max = county_extremes["max_aqi"].dropna()
if valid_p90.empty or valid_max.empty:
    raise ValueError("All p90_aqi or max_aqi values are NaN after cleaning.")
p90_threshold = valid_p90.quantile(0.90)
max_threshold = valid_max.quantile(0.90)

def label_risk(row):
    p90 = row["p90_aqi"]
    max_aqi = row["max_aqi"]
    if pd.isna(p90) or pd.isna(max_aqi):
        return "Unknown"
    high_p90 = p90 >= p90_threshold
    high_max = max_aqi >= max_threshold
    if high_p90 and high_max:
        return "Extreme Risk"
    if high_p90 or high_max:
        return "High Risk"
    return "Lower Risk"

county_extremes["risk_label"] = county_extremes.apply(label_risk, axis=1)

print(county_extremes.head())

final_df = county_extremes.copy()
risk_order = ["Extreme Risk", "High Risk", "Lower Risk", "Unknown"]
final_df["risk_label"] = pd.Categorical(
    final_df["risk_label"], categories=risk_order, ordered=True
)
final_df = final_df.sort_values(["risk_label", "p90_aqi"], ascending=[True, False])
os.makedirs("outputs", exist_ok=True)
final_df.to_csv("outputs/final_df.csv", index=False)
