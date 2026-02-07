import csv
import os
from collections import defaultdict
from statistics import mean

DATA_PATH = os.environ.get(
    "AQI_DATASET_PATH",
    "/Users/celine/Downloads/Access_to_a_Livable_Planet_Dataset.csv",
)

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


def latest_year(rows):
    return max(r["Year"] for r in rows)


def trend_slope(years, values):
    if len(years) < 3:
        return None
    x_mean = mean(years)
    y_mean = mean(values)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(years, values))
    den = sum((x - x_mean) ** 2 for x in years)
    if den == 0:
        return None
    return num / den


def normalize_features(rows):
    if not rows:
        return rows
    keys = list(rows[0]["features"].keys())
    mins = {k: min(r["features"][k] for r in rows) for k in keys}
    maxs = {k: max(r["features"][k] for r in rows) for k in keys}
    for r in rows:
        norm = {}
        for k in keys:
            lo = mins[k]
            hi = maxs[k]
            if hi == lo:
                norm[k] = 0.0
            else:
                norm[k] = (r["features"][k] - lo) / (hi - lo)
        r["norm"] = norm
    return rows


def euclidean(a, b):
    return sum((a[k] - b[k]) ** 2 for k in a.keys()) ** 0.5


def kmeans(points, k=4, iterations=20):
    if not points:
        return []
    centers = [p["norm"].copy() for p in points[:k]]
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        for p in points:
            distances = [euclidean(p["norm"], c) for c in centers]
            idx = distances.index(min(distances))
            clusters[idx].append(p)
        new_centers = []
        for cluster in clusters:
            if not cluster:
                new_centers.append(centers[len(new_centers)].copy())
                continue
            keys = cluster[0]["norm"].keys()
            center = {k: mean([p["norm"][k] for p in cluster]) for k in keys}
            new_centers.append(center)
        centers = new_centers
    return clusters


def sparkline(values, width=40):
    if not values:
        return ""
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return "-" * min(width, len(values))
    scaled = []
    for v in values:
        scaled.append(int((v - min_v) / (max_v - min_v) * (width - 1)))
    line = [" "] * width
    for i, x in enumerate(scaled):
        line[x] = "*"
    return "".join(line)


rows = load_rows(DATA_PATH)
if not rows:
    raise SystemExit("No rows loaded.")

all_years = sorted({r["Year"] for r in rows})
max_year = latest_year(rows)

print(f"Dataset loaded: {len(rows)} rows")
print(f"Years in dataset: {all_years}")
print("")

# ML 1) Cluster counties based on historical air quality patterns
print("ML 1) Clustering counties (high-risk regions)")
county_features = []
by_county = defaultdict(list)
for r in rows:
    key = (r["State"], r["County"])
    by_county[key].append(r)

for (state, county), entries in by_county.items():
    avg_median = mean([e["Median AQI"] for e in entries])
    avg_p90 = mean([e["90th Percentile AQI"] for e in entries])
    avg_max = mean([e["Max AQI"] for e in entries])
    avg_unhealthy = mean(
        [e["Unhealthy Days"] + e["Very Unhealthy Days"] + e["Hazardous Days"] for e in entries]
    )
    county_features.append(
        {
            "state": state,
            "county": county,
            "features": {
                "median": avg_median,
                "p90": avg_p90,
                "max": avg_max,
                "unhealthy": avg_unhealthy,
            },
        }
    )

normalize_features(county_features)
clusters = kmeans(county_features, k=4, iterations=25)

cluster_scores = []
for i, cluster in enumerate(clusters):
    if not cluster:
        cluster_scores.append((i, 0.0, 0))
        continue
    score = mean(
        [
            c["features"]["median"]
            + c["features"]["p90"]
            + c["features"]["max"]
            + c["features"]["unhealthy"]
            for c in cluster
        ]
    )
    cluster_scores.append((i, score, len(cluster)))

cluster_scores.sort(key=lambda x: x[1], reverse=True)
print("Clusters ranked by risk score (higher = worse):")
for idx, score, size in cluster_scores:
    print(f"  Cluster {idx}: size={size}, score={score:.1f}")

highest_cluster = cluster_scores[0][0] if cluster_scores else None
if highest_cluster is not None and clusters[highest_cluster]:
    sample = sorted(
        clusters[highest_cluster],
        key=lambda c: (
            c["features"]["max"],
            c["features"]["p90"],
            c["features"]["median"],
        ),
        reverse=True,
    )[:10]
    print("Sample counties from highest-risk cluster:")
    for c in sample:
        print(
            f"  {c['county']}, {c['state']}: "
            f"median={c['features']['median']:.1f}, "
            f"p90={c['features']['p90']:.1f}, "
            f"max={c['features']['max']:.1f}, "
            f"unhealthy_days={c['features']['unhealthy']:.1f}"
        )
print("")

# ML 2) High-risk classification from a single-year snapshot
print("ML 2) High-risk classification (single-year snapshot)")
latest_rows = [r for r in rows if r["Year"] == max_year]
if not latest_rows:
    print("No rows for the latest year.")
else:
    max_aqi_values = [r["Max AQI"] for r in latest_rows]
    unhealthy_values = [
        r["Unhealthy Days"] + r["Very Unhealthy Days"] + r["Hazardous Days"]
        for r in latest_rows
    ]
    max_aqi_threshold = sorted(max_aqi_values)[int(0.9 * (len(max_aqi_values) - 1))]
    unhealthy_threshold = sorted(unhealthy_values)[
        int(0.9 * (len(unhealthy_values) - 1))
    ]

    def is_high_risk(r):
        unhealthy = r["Unhealthy Days"] + r["Very Unhealthy Days"] + r["Hazardous Days"]
        return r["Max AQI"] >= max_aqi_threshold or unhealthy >= unhealthy_threshold

    features = [
        "Days PM2.5",
        "Days Ozone",
        "Days NO2",
        "Days CO",
        "Days PM10",
        "Median AQI",
        "90th Percentile AQI",
    ]

    labeled = []
    for r in latest_rows:
        labeled.append(
            {
                "state": r["State"],
                "county": r["County"],
                "label": 1 if is_high_risk(r) else 0,
                "features": {f: r[f] for f in features},
                "max_aqi": r["Max AQI"],
            }
        )

    high_risk = [r for r in labeled if r["label"] == 1]
    print(
        f"High-risk threshold: Max AQI >= {max_aqi_threshold} "
        f"OR Unhealthy+Very+Hazardous >= {unhealthy_threshold}"
    )
    print(f"High-risk counties: {len(high_risk)} / {len(labeled)}")

    # Simple correlation-like signal: compare feature means in high-risk vs others
    def mean_feature(rows, key):
        if not rows:
            return 0.0
        return mean([r["features"][key] for r in rows])

    high_means = {f: mean_feature(high_risk, f) for f in features}
    low_means = {
        f: mean_feature([r for r in labeled if r["label"] == 0], f) for f in features
    }

    deltas = []
    for f in features:
        deltas.append((f, high_means[f] - low_means[f], high_means[f], low_means[f]))
    deltas.sort(key=lambda x: x[1], reverse=True)

    print("Top feature gaps (high-risk mean - low-risk mean):")
    for f, delta, hi, lo in deltas:
        print(f"  {f}: +{delta:.1f} (high={hi:.1f}, low={lo:.1f})")

    top_high = sorted(high_risk, key=lambda r: r["max_aqi"], reverse=True)[:10]
    print("Sample high-risk counties (by Max AQI):")
    for r in top_high:
        print(f"  {r['county']}, {r['state']}: Max AQI {r['max_aqi']}")
print("")

# Visualization) Compare exposure to specific pollutants across counties
print("Viz) Pollutant exposure across counties (latest year)")
latest_rows = [r for r in rows if r["Year"] == max_year]
if not latest_rows:
    print("No rows for the latest year.")
else:
    pollutants = [
        ("Days PM2.5", "PM2.5"),
        ("Days Ozone", "Ozone"),
        ("Days NO2", "NO2"),
    ]
    county_pollutant = defaultdict(lambda: {p[0]: 0 for p in pollutants})
    for r in latest_rows:
        key = (r["State"], r["County"])
        for field, _ in pollutants:
            county_pollutant[key][field] += r[field]

    # Pick top counties by total pollutant days
    totals = []
    for key, values in county_pollutant.items():
        total = sum(values[f] for f, _ in pollutants)
        totals.append((key, total))
    totals.sort(key=lambda x: x[1], reverse=True)
    top_counties = [key for key, _ in totals[:20]]

    print("Top counties by pollutant days (sum of PM2.5 + Ozone + NO2):")
    for (state, county), total in totals[:10]:
        print(f"  {county}, {state}: {total}")

    # Heatmap (HTML)
    data = []
    for field, _ in pollutants:
        row = []
        for key in top_counties:
            row.append(county_pollutant[key][field])
        data.append(row)

    x_labels = [f"{c}, {s}" for s, c in top_counties]
    y_labels = [label for _, label in pollutants]

    flat = [v for row in data for v in row]
    min_v = min(flat) if flat else 0
    max_v = max(flat) if flat else 1

    def color_for(value):
        if max_v == min_v:
            t = 0.0
        else:
            t = (value - min_v) / (max_v - min_v)
        # Blue -> Yellow -> Red gradient
        if t < 0.5:
            k = t / 0.5
            r = int(0 + (255 - 0) * k)
            g = int(90 + (255 - 90) * k)
            b = int(200 + (0 - 200) * k)
        else:
            k = (t - 0.5) / 0.5
            r = int(255 + (200 - 255) * k)
            g = int(255 + (40 - 255) * k)
            b = int(0 + (0 - 0) * k)
        return f"rgb({r},{g},{b})"

    html = []
    html.append("<!doctype html>")
    html.append("<html><head><meta charset='utf-8'>")
    html.append("<title>Pollutant Exposure Heatmap</title>")
    html.append("<style>")
    html.append("body{font-family:Arial, sans-serif;margin:20px;background:#fafafa}")
    html.append(".legend{display:flex;align-items:center;gap:10px;margin:10px 0}")
    html.append(".bar{height:14px;width:220px;background:linear-gradient(90deg,#005ac8,#ffde40,#c82800);border:1px solid #999}")
    html.append("table{border-collapse:collapse;font-size:12px}")
    html.append("th,td{border:1px solid #ddd;padding:6px;text-align:center}")
    html.append("th{background:#f0f0f0;position:sticky;top:0}")
    html.append(".rowhead{background:#f7f7f7;font-weight:bold;position:sticky;left:0}")
    html.append("</style></head><body>")
    html.append("<h2>Pollutant Exposure Heatmap (Top 20 Counties)</h2>")
    html.append(f"<p>Min={min_v}, Max={max_v}</p>")
    html.append("<div class='legend'><span>Low</span><div class='bar'></div><span>High</span></div>")
    html.append("<div style='overflow:auto;max-width:100%'>")
    html.append("<table>")
    html.append("<tr><th></th>")
    for label in x_labels:
        html.append(f"<th>{label}</th>")
    html.append("</tr>")
    for i, row in enumerate(data):
        html.append(f"<tr><td class='rowhead'>{y_labels[i]}</td>")
        for v in row:
            html.append(f"<td style='background:{color_for(v)}'>{v}</td>")
        html.append("</tr>")
    html.append("</table></div></body></html>")

    out_path = "pollutant_heatmap.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Saved heatmap to {out_path}")

print("")
print("Done.")
