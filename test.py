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






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan

# Assuming reduced_embeddings and plot_df are already available.

# -------------------------
# Apply HDBSCAN Clustering
# -------------------------
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, cluster_selection_epsilon=0.05)
hdb_clusters = hdb.fit_predict(reduced_embeddings)

# Add cluster labels to DataFrame
plot_df["hdbscan_cluster"] = hdb_clusters

# -------------------------
# 1. Basic Cluster Counts
# -------------------------
cluster_counts = plot_df['hdbscan_cluster'].value_counts().sort_index()
print("Cluster Counts (Including Noise):\n", cluster_counts)

# -------------------------
# 2. Visualize Cluster Composition with Bar Plot
# -------------------------
plt.figure(figsize=(12, 6))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Points")
plt.title("Cluster Composition (Including Noise)")
plt.xticks(rotation=90)
plt.show()

# -------------------------
# 3. Percentage Distribution of Points in Each Cluster
# -------------------------
total_points = len(plot_df)
cluster_percentage = (cluster_counts / total_points) * 100
print("\nPercentage Distribution of Points in Each Cluster:\n", cluster_percentage)

# -------------------------
# 4. Cluster Balance Metrics (Variance & Standard Deviation)
# -------------------------
# Exclude noise (-1) for this analysis
filtered_cluster_sizes = cluster_counts[cluster_counts.index != -1]
variance = np.var(filtered_cluster_sizes)
std_dev = np.std(filtered_cluster_sizes)

print(f"\nVariance in Cluster Sizes: {variance:.2f}")
print(f"Standard Deviation in Cluster Sizes: {std_dev:.2f}")

# -------------------------
# 5. Identify Dominant and Small Clusters
# -------------------------
dominant_clusters = cluster_percentage[cluster_percentage > 20]
print(f"\nDominant Clusters (Over 20% of Data):\n{dominant_clusters}")

small_clusters = cluster_percentage[cluster_percentage < 5]
print(f"\nSmall Clusters (Under 5% of Data):\n{small_clusters}")

# -------------------------
# 6. Visualize Cluster Distribution with Pie Chart
# -------------------------
filtered_cluster_counts = cluster_counts[cluster_counts.index != -1]

plt.figure(figsize=(8, 8))
plt.pie(filtered_cluster_counts, labels=filtered_cluster_counts.index, 
        autopct='%1.1f%%', startangle=140, cmap="viridis")
plt.title("Cluster Distribution (Excluding Noise)")
plt.show()

# -------------------------
# 7. Noise Proportion
# -------------------------
noise_count = cluster_counts.get(-1, 0)
noise_percentage = (noise_count / total_points) * 100
print(f"\nNoise Points: {noise_count} ({noise_percentage:.2f}%)")

# -------------------------
# 8. Explore Specific Clusters
# -------------------------
# Example: View points in Cluster 0
cluster_0_points = plot_df[plot_df['hdbscan_cluster'] == 0]
print(f"\nTotal points in Cluster 0: {len(cluster_0_points)}")
print(cluster_0_points.head())

# Example: View points in noise (-1)
noise_points = plot_df[plot_df['hdbscan_cluster'] == -1]
print(f"\nTotal Noise Points: {len(noise_points)}")
print(noise_points.head())

# -------------------------
# 9. List All Clusters and Sample Points
# -------------------------
for cluster_label in plot_df['hdbscan_cluster'].unique():
    cluster_points = plot_df[plot_df['hdbscan_cluster'] == cluster_label]
    print(f"\nCluster {cluster_label}: Total Points = {len(cluster_points)}")
    print(cluster_points.head(3))  # Displaying top 3 points for brevity

# -------------------------
# 10. Export Noise and Clustered Points for External Analysis
# -------------------------
# Export noise points
noise_points.to_csv("noise_points.csv", index=False)

# Export clustered points (excluding noise)
clustered_points = plot_df[plot_df['hdbscan_cluster'] != -1]
clustered_points.to_csv("clustered_points.csv", index=False)

print("\n✅ Cluster analysis completed and exported.")




from sklearn.metrics import silhouette_score, davies_bouldin_score

# Filter out noise points
filtered_df = cleaned_journeys_df[cleaned_journeys_df['cluster'] != -1]
filtered_embeddings = reduced_embeddings[filtered_df.index]
filtered_labels = filtered_df['cluster'].values

# Calculate Silhouette Score
silhouette = silhouette_score(filtered_embeddings, filtered_labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Calculate Davies-Bouldin Score
david_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)
print(f"Davies-Bouldin Score: {david_bouldin:.4f}")

