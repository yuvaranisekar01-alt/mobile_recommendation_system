from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

segment_labels = ['Budget', 'Affordable', 'Mid-Range', 'Premium']

def segment(df_model, n_clusters=4):
    print("\n[Segmentation] Running KMeans on price_usd...")

    #scaling price_usd column
    x_price= df_model[['price_usd']].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_price)

    # KMeans Clustering
    Kmeans = KMeans(n_clusters= n_clusters, random_state=42, n_init=10)
    df_model= df_model.copy()
    df_model['cluster'] = Kmeans.fit_predict(scaled)

    #assign labels by price order
    avg_price       =df_model.groupby('cluster')['price_usd'].mean()
    sorted_clusters = avg_price.sort_values().index.tolist()

    label_map = {
        cluster_id: segment_labels[i]
        for i, cluster_id in enumerate(sorted_clusters)
    }
    df_model['segment'] = df_model['cluster'].map(label_map)

    print(f"  Segments:")
    print(df_model.groupby('segment')['price_usd'].agg(
    ['min', 'max', 'mean', 'count']).round(2).to_string())


    return df_model
