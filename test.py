
Store embeddings in a DataFrame for analysis:


import pandas as pd
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, "graph_id", range(len(embeddings)))  # Add IDs
print(embeddings_df)


Cluster similar journeys using KMeans:

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42).fit(embeddings)
embeddings_df["cluster"] = kmeans.labels_


Visualize clusters with PCA:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=kmeans.labels_, cmap='viridis')
plt.title("Graph Embeddings Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()



DBSCAN:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Assuming `embeddings` is your graph embedding matrix (Replace with actual embeddings)
np.random.seed(42)
embeddings = np.random.rand(10, 64)  # 10 graphs with 64 features each

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# **Apply DBSCAN Clustering**
dbscan = DBSCAN(eps=0.5, min_samples=2)  # Adjust parameters as needed
embeddings_df["dbscan_cluster"] = dbscan.fit_predict(embeddings)

# **Check Outliers (-1 indicates noise points)**
outliers = embeddings_df[embeddings_df["dbscan_cluster"] == -1]
print(f"Number of outliers detected: {len(outliers)}")

# **Reduce dimensions using PCA for visualization**
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# **Scatter plot of DBSCAN clusters**
plt.figure(figsize=(8, 6))
plt.scatter(
    reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
    c=embeddings_df["dbscan_cluster"], cmap="plasma", alpha=0.8
)
plt.title("DBSCAN Clustering of Graph Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="DBSCAN Cluster ID")
plt.show()

# **Display clustering results**
import ace_tools as tools
tools.display_dataframe_to_user(name="DBSCAN Clustered Graph Embeddings", dataframe=embeddings_df)



from sklearn.metrics import silhouette_score
score = silhouette_score(embeddings, kmeans.labels_)
print(f"Silhouette Score: {score}")


dbscan = DBSCAN(eps=0.3, min_samples=5)











Original
import numpy as np
import pandas as pd
import networkx as nx
from karateclub import Graph2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

# Load dataset (Assuming cleaned_journeys_df is available)
# cleaned_journeys_df = pd.read_csv("your_data.csv")  # Load if needed

# Convert paths to graphs
graph_list = []
session_ids = []

n_samples = 1000  # Adjust based on computational power
for index, nodes in enumerate(cleaned_journeys_df.path[:n_samples]):
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(nodes)
    
    # Add edges (Sequential path connections)
    edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes) - 1)]
    G.add_edges_from(edges)
    
    # Standardize node labels
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    
    graph_list.append(G)
    session_ids.append(cleaned_journeys_df.channel_visit_id.iloc[index])

# Train Graph2Vec for embeddings
model = Graph2Vec(dimensions=64, wl_iterations=2, attributed=False)
model.fit(graph_list)
embeddings = model.get_embedding()

# Convert embeddings to DataFrame
embedding_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(64)])
embedding_df["session_id"] = session_ids

# Find optimal PCA dimensions using explained variance
pca = PCA()
pca.fit(embeddings)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, 65), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Optimal PCA Dimension Selection")
plt.grid()
plt.show()

# Choose dimension where variance stabilizes (e.g., 90% variance)
optimal_pca_dim = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.90) + 1
print(f"Optimal PCA Dimension: {optimal_pca_dim}")

# Apply PCA with the chosen dimensions
pca = PCA(n_components=optimal_pca_dim)
reduced_embeddings_pca = pca.fit_transform(embeddings)

# Find the best `eps` for DBSCAN using K-Nearest Neighbors (KNN)
nearest_neighbors = NearestNeighbors(n_neighbors=10)
nearest_neighbors.fit(reduced_embeddings_pca)
distances, indices = nearest_neighbors.kneighbors(reduced_embeddings_pca)

# Sort and find the knee point
sorted_distances = np.sort(distances[:, -1])
kneedle = KneeLocator(range(1, len(sorted_distances) + 1), sorted_distances, curve="convex", direction="increasing")
optimal_eps = sorted_distances[kneedle.knee]
print(f"Optimal DBSCAN eps: {optimal_eps}")

# Apply DBSCAN with the best epsilon
dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
clusters = dbscan.fit_predict(reduced_embeddings_pca)

# Add clustering results to DataFrame
embedding_df["cluster"] = clusters

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced_embeddings_tsne = tsne.fit_transform(reduced_embeddings_pca)

# Plot clusters using t-SNE
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embeddings_tsne[:, 0], reduced_embeddings_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.colorbar(label="Cluster")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("DBSCAN Clusters Visualized using t-SNE")
plt.show()

# Save results if needed
embedding_df.to_csv("graph_embeddings_dbscan_clusters.csv", index=False)



from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

# Remove noise points (DBSCAN assigns -1 to noise)
valid_clusters = embedding_df[embedding_df["cluster"] != -1]

# Compute validation metrics
sil_score = silhouette_score(valid_clusters.iloc[:, :-2], valid_clusters["cluster"])
db_index = davies_bouldin_score(valid_clusters.iloc[:, :-2], valid_clusters["cluster"])
ch_index = calinski_harabasz_score(valid_clusters.iloc[:, :-2], valid_clusters["cluster"])

print(f"Silhouette Score: {sil_score:.4f} (Higher is better)")
print(f"Davies-Bouldin Index: {db_index:.4f} (Lower is better)")
print(f"Calinski-Harabasz Index: {ch_index:.4f} (Higher is better)")

# Count points per cluster
cluster_counts = valid_clusters["cluster"].value_counts().sort_index()

# Plot Cluster Size Distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Points")
plt.title("Cluster Size Distribution")
plt.xticks(rotation=45)
plt.show()



# Map cluster labels back to the original DataFrame
cleaned_journeys_df["cluster"] = -1  # Default to -1 (unclustered/noise)

# Update only valid session IDs
for idx, session_id in enumerate(valid_clusters["session_id"]):
    cleaned_journeys_df.loc[cleaned_journeys_df["channel_visit_id"] == session_id, "cluster"] = valid_clusters.iloc[idx]["cluster"]

# Display cluster mappings
import ace_tools as tools
tools.display_dataframe_to_user(name="Clustered Journeys DataFrame", dataframe=cleaned_journeys_df)

# Save the updated DataFrame if needed
cleaned_journeys_df.to_csv("cleaned_journeys_with_clusters.csv", index=False)

# Print example of paths per cluster
for cluster_id in cleaned_journeys_df["cluster"].unique():
    if cluster_id == -1:
        print(f"\nCluster {cluster_id} (Noise/Outliers):")
    else:
        print(f"\nCluster {cluster_id}:")
    
    example_paths = cleaned_journeys_df[cleaned_journeys_df["cluster"] == cluster_id]["path"].head(5).tolist()
    
    for path in example_paths:
        print(f"  - {path}")










from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import ace_tools as tools  # For displaying data

# Assuming we have a target variable (user behavior) in cleaned_journeys_df
# Example: cleaned_journeys_df["target"] = 1 if user converted, 0 otherwise
# You need to define "target" based on business logic.

# Ensure we have target variable
if "target" not in cleaned_journeys_df.columns:
    raise ValueError("Target variable ('target') not found! Please define user behavior outcomes.")

# Prepare data
X = embedding_df.drop(columns=["session_id", "cluster"])  # Use embeddings as features
y = cleaned_journeys_df["target"][:len(X)]  # Ensure target aligns with embeddings

# Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Performance
log_report = classification_report(y_test, y_pred_log, output_dict=True)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Display results
log_df = pd.DataFrame(log_report).T
rf_df = pd.DataFrame(rf_report).T

tools.display_dataframe_to_user(name="Logistic Regression Report", dataframe=log_df)
tools.display_dataframe_to_user(name="Random Forest Report", dataframe=rf_df)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

# Assuming 'embeddings' contains Graph2Vec representations
# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Step 1: Determine best dimensionality reduction (PCA or t-SNE)
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings_df)

# Use explained variance to decide between PCA and t-SNE
explained_variance = np.sum(pca.explained_variance_ratio_)
if explained_variance > 0.90:  # If PCA retains >90% variance, use PCA
    reduced_embeddings = pca_embeddings
    reduction_method = "PCA"
else:  # Otherwise, use t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings_df)
    reduction_method = "t-SNE"

print(f"Best dimensionality reduction method: {reduction_method}")

# Step 2: Estimate best epsilon using Knee method
nearest_neighbors = NearestNeighbors(n_neighbors=5)
nearest_neighbors.fit(embeddings_df)
distances, indices = nearest_neighbors.kneighbors(embeddings_df)

# Sort distances for knee plot
sorted_distances = np.sort(distances[:, -1])

# Find knee point
knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve="convex", direction="increasing")
best_eps = sorted_distances[knee.knee]

# Step 3: Apply DBSCAN clustering
dbscan = DBSCAN(eps=best_eps, min_samples=5)  # min_samples can be tuned
clusters = dbscan.fit_predict(embeddings_df)

# Step 4: Assign clusters to session IDs
cleaned_journeys_df["cluster"] = clusters

# Step 5: Visualize clusters using the best dimensionality reduction
plt.figure(figsize=(8, 6))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(label="Cluster")
plt.xlabel(f"{reduction_method} Component 1")
plt.ylabel(f"{reduction_method} Component 2")
plt.title(f"User Journey Clusters ({reduction_method})")
plt.show()

# Step 6: Display session cluster assignment
session_cluster_df = cleaned_journeys_df[['channel_visit_id', 'cluster']]
import ace_tools as tools
tools.display_dataframe_to_user(name="Session Clusters", dataframe=session_cluster_df)

# Step 7: Show the knee plot
plt.figure(figsize=(8, 4))
plt.plot(range(len(sorted_distances)), sorted_distances, label="K-distance")
plt.axvline(x=knee.knee, color='r', linestyle='--', label=f"Knee at {best_eps:.4f}")
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("Knee Plot for Best Epsilon Selection")
plt.legend()
plt.show()

print(f"Best estimated epsilon for DBSCAN: {best_eps:.4f}")














import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

# Assuming 'embeddings' contains Graph2Vec representations
embeddings_df = pd.DataFrame(embeddings)

# Step 1: Determine the best number of PCA components
pca_full = PCA()
pca_full.fit(embeddings_df)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find the number of dimensions that explain at least 95% of the variance
optimal_pca_components = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Optimal number of PCA components: {optimal_pca_components}")

# Step 2: Apply PCA with the estimated optimal number of components
pca_final = PCA(n_components=optimal_pca_components)
pca_reduced_embeddings = pca_final.fit_transform(embeddings_df)

print(f"PCA applied: Reduced embeddings from {embeddings_df.shape[1]} to {pca_reduced_embeddings.shape[1]} dimensions")

# Step 3: Estimate best epsilon using Knee method on reduced embeddings
nearest_neighbors = NearestNeighbors(n_neighbors=5)
nearest_neighbors.fit(pca_reduced_embeddings)
distances, indices = nearest_neighbors.kneighbors(pca_reduced_embeddings)

# Sort distances for knee plot
sorted_distances = np.sort(distances[:, -1])

# Find knee point
knee = KneeLocator(range(len(sorted_distances)), sorted_distances, curve="convex", direction="increasing")
best_eps = sorted_distances[knee.knee]

# Step 4: Apply DBSCAN clustering on PCA-reduced data
dbscan = DBSCAN(eps=best_eps, min_samples=5)  # min_samples can be tuned
clusters = dbscan.fit_predict(pca_reduced_embeddings)

# Step 5: Assign clusters to session IDs
cleaned_journeys_df["cluster"] = clusters

# Step 6: Visualize clusters using PCA (top 2 components)
plt.figure(figsize=(8, 6))
plt.scatter(pca_reduced_embeddings[:, 0], pca_reduced_embeddings[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(label="Cluster")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("User Journey Clusters (PCA)")
plt.show()

# Step 7: Display session cluster assignment
session_cluster_df = cleaned_journeys_df[['channel_visit_id', 'cluster']]
import ace_tools as tools
tools.display_dataframe_to_user(name="Session Clusters", dataframe=session_cluster_df)

# Step 8: Show the knee plot
plt.figure(figsize=(8, 4))
plt.plot(range(len(sorted_distances)), sorted_distances, label="K-distance")
plt.axvline(x=knee.knee, color='r', linestyle='--', label=f"Knee at {best_eps:.4f}")
plt.xlabel("Data Points Sorted by Distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("Knee Plot for Best Epsilon Selection")
plt.legend()
plt.show()

print(f"Best estimated epsilon for DBSCAN: {best_eps:.4f}")






import matplotlib.pyplot as plt

sample_sizes = [1000, 5000, 10000, 20000, 50000]
graph_counts = []

for size in sample_sizes:
    temp_graph_list = []
    for nodes in cleaned_journeys_df.path[:size]:
        G = nx.Graph()
        G.add_nodes_from(nodes)
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        G.add_edges_from(edges)
        temp_graph_list.append(G)
    
    graph_counts.append(len(temp_graph_list))

# Plot the curve
plt.plot(sample_sizes, graph_counts, marker='o')
plt.xlabel("Number of Samples")
plt.ylabel("Number of Graphs Created")
plt.title("Elbow Method for Sample Selection")
plt.show()




from kneed import KneeLocator

# Ensure sample_sizes and graph_counts are valid lists
if len(sample_sizes) == 0 or len(graph_counts) == 0:
    raise ValueError("Sample sizes or graph counts are empty. Check the data.")

# Use KneeLocator to find the optimal number of samples
knee = KneeLocator(sample_sizes, graph_counts, curve="convex", direction="increasing")

# Check if knee.knee is found and within range
if knee.knee is not None and knee.knee < len(sample_sizes):
    optimal_samples = sample_sizes[knee.knee]
else:
    optimal_samples = max(sample_sizes)  # Default to max sample size if no knee found

# Apply a hard limit to prevent excessive computation
optimal_samples = min(optimal_samples, int(0.1 * len(cleaned_journeys_df)), 50000)

print(f"Final selected sample size: {optimal_samples}")





import numpy as np

# Step 1: Create a new column with default 'Not Clustered' (-1)
cleaned_journeys_df["cluster"] = -1  # Assign -1 to indicate no cluster assigned

# Step 2: Assign clusters only to the sampled data
sampled_indices = cleaned_journeys_df.sample(n=50000, random_state=42).index  # Ensure correct sampling
cleaned_journeys_df.loc[sampled_indices, "cluster"] = clusters  # Map clusters back

# Step 3: Display updated dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Updated Dataset with Cluster Mapping", dataframe=cleaned_journeys_df)
