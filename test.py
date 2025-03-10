import umap

# Reduce dimensions with UMAP to 5
umap_reducer = umap.UMAP(n_components=5, random_state=42)
reduced_embeddings = umap_reducer.fit_transform(embeddings)

print(f"Reduced Embeddings Shape: {reduced_embeddings.shape}")




from sklearn.cluster import DBSCAN

# Trying multiple eps and min_samples values
eps_values = [0.1, 0.3, 0.5, 0.7]
min_samples_values = [5, 10, 20]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(reduced_embeddings)
        
        unique_clusters = set(clusters)
        print(f"EPS: {eps}, Min Samples: {min_samples}, Number of clusters (excluding noise): {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Convert for visualization
plot_df = pd.DataFrame({
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
    "cluster": clusters
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="x", y="y", hue="cluster", palette="viridis", s=50, alpha=0.7)
plt.title("UMAP Clustering Visualization with Adjusted DBSCAN")
plt.show()




import hdbscan

# Apply HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=10)
hdb_clusters = hdb.fit_predict(reduced_embeddings)

# Visualize
plot_df["hdbscan_cluster"] = hdb_clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="x", y="y", hue="hdbscan_cluster", palette="viridis", s=50, alpha=0.7)
plt.title("HDBSCAN Clustering Visualization")
plt.show()

# Check number of clusters
print(f"Number of clusters (excluding noise): {len(set(hdb_clusters)) - (1 if -1 in hdb_clusters else 0)}")



from sklearn.metrics import silhouette_score

# Exclude noise for evaluation
valid_embeddings = reduced_embeddings[hdb_clusters != -1]
valid_labels = hdb_clusters[hdb_clusters != -1]

if len(set(valid_labels)) > 1:
    score = silhouette_score(valid_embeddings, valid_labels)
    print(f"Silhouette Score (excluding noise): {score:.4f}")
else:
    print("Only one cluster detected, silhouette score is not meaningful.")



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Normalize
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

# Reduce with PCA
pca = PCA(n_components=10)
pca_embeddings = pca.fit_transform(normalized_embeddings)




import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score

# Define parameter grid for tuning
param_grid = {
    'min_cluster_size': [5, 10, 20, 50],
    'min_samples': [1, 5, 10, 20],
    'cluster_selection_epsilon': [0.01, 0.05, 0.1, 0.2]
}

# Track best parameters and score
best_params = None
best_score = -1

# Iterate over all parameter combinations
for min_cluster_size in param_grid['min_cluster_size']:
    for min_samples in param_grid['min_samples']:
        for epsilon in param_grid['cluster_selection_epsilon']:
            
            # Initialize and fit HDBSCAN
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                  min_samples=min_samples,
                                  cluster_selection_epsilon=epsilon)
            clusters = hdb.fit_predict(reduced_embeddings)
            
            # Filter out noise (-1) for silhouette score
            valid_indices = clusters != -1
            if len(set(clusters[valid_indices])) > 1:
                score = silhouette_score(reduced_embeddings[valid_indices], clusters[valid_indices])
            else:
                score = -1  # Invalid clustering scenario
            
            print(f"Params: min_cluster_size={min_cluster_size}, min_samples={min_samples}, epsilon={epsilon}, Silhouette Score={score:.4f}")

            # Update best score and parameters
            if score > best_score:
                best_score = score
                best_params = (min_cluster_size, min_samples, epsilon)

print("\n✅ Best Parameters:")
print(f"min_cluster_size = {best_params[0]}, min_samples = {best_params[1]}, epsilon = {best_params[2]}")
print(f"Best Silhouette Score: {best_score:.4f}")

