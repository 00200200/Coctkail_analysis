import matplotlib.pyplot as plt

def plot_scores(wcss, silhouette_scores, davies_scores):
    plt.figure(figsize=(14,6))

    plt.subplot(1, 3, 1)
    plt.plot(range(2, 15), wcss, marker='o')
    plt.title('Inertia Values for Different Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(range(2, 15), silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Scores for Different Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(range(2, 15), davies_scores, marker='o', color='green')
    plt.title('Davies for Different Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies Score')
    plt.grid(True)

    plt.show()

from sklearn.decomposition import PCA

def plot_kmenas(data,labels,centers):
    pca = PCA(n_components=2)
    df_cocktail_encoded = pca.fit_transform(data)
    plt.figure(figsize=(10, 6))
    plt.scatter(df_cocktail_encoded[:, 0], df_cocktail_encoded[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(pca.transform(centers)[:, 0], pca.transform(centers)[:, 1], c='red', marker='x', s=200)
    plt.title(f"KMeans Clustering")
    plt.show()
