import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("data/aqsi_all_years.csv")

print(df)

pivot = df.pivot_table(
    index=["State", "County"],
    columns="Year",
    values="DAQSI"
).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot)

inertia = []

for k in range(1, 10):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(X_scaled)
  inertia.append(km.inertia_)

plt.plot(range(1,10), inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

pivot["Cluster"] = clusters

cluster_patterns = pivot.groupby("Cluster").mean()

kmeans = KMeans(n_clusters=4, random_state=42)
pivot["Cluster"] = kmeans.fit_predict(X_scaled)

cluster_patterns = pivot.groupby("Cluster").mean()
print(cluster_patterns)

clusters_df = pivot.reset_index()[["State", "County", "Cluster"]]
clusters_df.head()


