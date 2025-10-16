import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

DATA_DIR = "../urbanbus_data"    
OUTPUT_CSV = "../route_clusters.csv"
N_CLUSTERS = 5


files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
route_stats = []

for file in files:
    try:
        route_id = file.replace("_aggregated.csv", "")
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        
        if "Ride_start_datetime" in df.columns:
            df["Ride_start_datetime"] = pd.to_datetime(df["Ride_start_datetime"])
            df["hour"] = df["Ride_start_datetime"].dt.hour
            df["dayofweek"] = df["Ride_start_datetime"].dt.dayofweek
        
        if "Passenger_Count" in df.columns:
            total_demand = df["Passenger_Count"].sum()
            mean_demand = df["Passenger_Count"].mean()
            std_demand = df["Passenger_Count"].std()
            max_demand = df["Passenger_Count"].max()
        else:
            continue

        route_stats.append({
            "route_id": route_id,
            "total_demand": total_demand,
            "mean_demand": mean_demand,
            "std_demand": std_demand,
            "max_demand": max_demand
        })
    except:
        print(f"Skipped route {file}")

stats_df = pd.DataFrame(route_stats).dropna()

features = ["total_demand", "mean_demand", "std_demand", "max_demand"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(stats_df[features])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
stats_df["cluster_label"] = kmeans.fit_predict(X_scaled)

stats_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Clustering complete! Saved to {OUTPUT_CSV}")
print(stats_df.groupby("cluster_label")[features].mean())
