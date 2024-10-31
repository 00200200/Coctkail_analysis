from sklearn.cluster import KMeans

def perform_kmeans(data,n_cluster):
    kmeans = KMeans(n_clusters=n_cluster,random_state=42)
    kmeans.fit(data)
    kmeans_labels = kmeans.labels_
    return data,kmeans_labels,kmeans.cluster_centers_

from sklearn.metrics import silhouette_score,davies_bouldin_score

def create_graph_data(data):
    wcss = []
    davies_scores = []
    silhouette_scores = []
    for i in range(2,15):
        kmeans = KMeans(n_clusters=i,random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data,kmeans.labels_))
        labels = kmeans.labels_
        score = davies_bouldin_score(data, labels)
        davies_scores.append(score)
    return wcss,silhouette_scores,davies_scores