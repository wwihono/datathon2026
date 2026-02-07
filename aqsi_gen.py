import pandas as pd  
import numpy as np

files = [
    "data/annual_aqi_by_county_2019.csv",
    "data/annual_aqi_by_county_2020.csv",
    "data/annual_aqi_by_county_2021.csv",
    "data/annual_aqi_by_county_2022.csv",
    "data/annual_aqi_by_county_2023.csv",
    "data/annual_aqi_by_county_2024.csv",
    "data/annual_aqi_by_county_2025.csv"
]

dfs = [pd.read_csv(f) for f in files]
full_all_years = pd.concat(dfs, ignore_index=True)

cols = [
  "Good Days",
  "Moderate Days",
  "Unhealthy for Sensitive Groups Days",
  "Unhealthy Days",
  "Very Unhealthy Days",
  "Hazardous Days"
]

frequencies = full_all_years[cols].sum() / full_all_years[cols].sum().sum()

weights = -np.log(frequencies)

# normalize to 0–1 scale
weights = weights / weights.max()

processed_dfs = []

# Function to compute aqsi weights
def compute_daqsi(df, weights):
    df["Total_Days"] = df[cols].sum(axis=1)
    
    df["DAQSI"] = (
        weights["Moderate Days"] * df["Moderate Days"] +
        weights["Unhealthy for Sensitive Groups Days"] * df["Unhealthy for Sensitive Groups Days"] +
        weights["Unhealthy Days"] * df["Unhealthy Days"] +
        weights["Very Unhealthy Days"] * df["Very Unhealthy Days"] +
        weights["Hazardous Days"] * df["Hazardous Days"] -
        weights["Good Days"] * df["Good Days"]
    ) / df["Total_Days"]
    
    return df

for f in files:
    df = pd.read_csv(f)
    df = compute_daqsi(df, weights)
    processed_dfs.append(df)

final_df = pd.concat(processed_dfs, ignore_index=True)
final_df.to_csv("aqsi_all_years.csv", index=False)

