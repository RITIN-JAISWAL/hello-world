import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import umap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking

# Load the large dataset (Assuming it's a CSV)
df = pd.read_csv("large_dataset.csv")  # Update with actual file path
df["path"] = df["path"].apply(eval)  # Convert string lists to actual lists

# Initialize a Directed Graph
G = nx.DiGraph()

# Batch processing for large-scale graph construction
for path in tqdm(df["path"], desc="Building Graph"):
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    G.add_edges_from(edges)

# Node2Vec with optimized parameters
node2vec = Node2Vec(G, 
                    dimensions=128,   # Higher for more expressive embeddings
                    walk_length=40, 
                    num_walks=100, 
                    workers=8,   # Utilize multiple CPU cores
                    p=1, q=1)  # Balanced exploration and exploitation

# Train Node2Vec
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get node embeddings
node_ids = list(model.wv.index_to_key)
node_embeddings = np.array([model.wv[node] for node in node_ids])

# Apply UMAP for Dimensionality Reduction
umap_model = umap.UMAP(n_neighbors=30, 
                        min_dist=0.1, 
                        n_components=2, 
                        random_state=42, 
                        metric="cosine",
                        verbose=True)  # Verbose for large dataset progress

umap_embeddings = umap_model.fit_transform(node_embeddings)

# Apply DBSCAN with tuned parameters
dbscan = DBSCAN(eps=0.7, min_samples=5, metric='euclidean', n_jobs=-1)  # Use all CPU cores
labels = dbscan.fit_predict(umap_embeddings)

# Visualize the Clusters
plt.figure(figsize=(12, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.7)
plt.title('Clusters after UMAP and DBSCAN')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(label='Cluster Label')
plt.show()






import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import umap

# UMAP Reduction (Assuming node_embeddings is already generated)
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, metric="cosine")
umap_embeddings = umap_model.fit_transform(node_embeddings)

# Step 1: Compute the k-nearest neighbor distances
k = 5  # min_samples for DBSCAN
nbrs = NearestNeighbors(n_neighbors=k).fit(umap_embeddings)
distances, indices = nbrs.kneighbors(umap_embeddings)

# Step 2: Sort and plot distances
distances = np.sort(distances[:, -1], axis=0)  # Take the farthest k-th neighbor

plt.figure(figsize=(10, 6))
plt.plot(distances, marker='o', linestyle='dashed', color='b', markersize=3)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-Nearest Neighbor Distance")
plt.title("Elbow Method for DBSCAN `eps` Selection")
plt.grid(True)
plt.show()




















here:
# Install required libraries if not installed
!pip install networkx node2vec umap-learn scikit-learn matplotlib kneed tqdm joblib

# Import libraries
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import collections
import joblib  # For saving models
from tqdm import tqdm
import os

# Create a directory to save models
os.makedirs("saved_models", exist_ok=True)

# 📌 Step 1: Load the dataset
df = pd.read_csv("large_dataset.csv")  # Update with actual file path
df["path"] = df["path"].apply(eval)  # Convert string lists to actual lists

# 📌 Step 2: Construct a Directed Graph
G = nx.DiGraph()

for path in tqdm(df["path"], desc="Building Graph"):
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    G.add_edges_from(edges)

# 📌 Step 3: Apply Node2Vec for Node Embeddings
node2vec = Node2Vec(G, dimensions=128, walk_length=40, num_walks=100, workers=8, p=1, q=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save the Node2Vec model
model.save("saved_models/node2vec.model")

# Get node embeddings
node_ids = list(model.wv.index_to_key)
node_embeddings = np.array([model.wv[node] for node in node_ids])

# 📌 Step 4: Apply UMAP for **3D** Dimensionality Reduction
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=3, random_state=42, metric="cosine", verbose=True)
umap_embeddings = umap_model.fit_transform(node_embeddings)

# Save the UMAP model
joblib.dump(umap_model, "saved_models/umap_model.pkl")

# 📌 Step 5: Find Optimal `eps` for DBSCAN using Elbow Method
k = 5  # Same as min_samples in DBSCAN
nbrs = NearestNeighbors(n_neighbors=k).fit(umap_embeddings)
distances, indices = nbrs.kneighbors(umap_embeddings)

# Sort distances
distances = np.sort(distances[:, -1], axis=0)

# Use Kneedle algorithm to find the elbow
kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
optimal_eps = distances[kneedle.knee]

# 📌 Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(distances, marker='o', linestyle='dashed', color='b', markersize=3, label="KNN Distance")
plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f"Optimal eps={optimal_eps:.3f}")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-Nearest Neighbor Distance")
plt.title("Elbow Method for DBSCAN `eps` Selection")
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal eps for DBSCAN: {optimal_eps:.3f}")

# 📌 Step 6: Apply DBSCAN with Optimized `eps`
dbscan = DBSCAN(eps=optimal_eps, min_samples=5, metric='euclidean', n_jobs=-1)
labels = dbscan.fit_predict(umap_embeddings)

# Save DBSCAN model and cluster labels
joblib.dump(dbscan, "saved_models/dbscan_model.pkl")
np.save("saved_models/cluster_labels.npy", labels)

# 📌 Step 7: Evaluate Clustering Quality

# ✅ 1. Silhouette Score (Higher is better)
valid_points = labels != -1
filtered_embeddings = umap_embeddings[valid_points]
filtered_labels = labels[valid_points]

if len(set(filtered_labels)) > 1:  # Silhouette score needs at least 2 clusters
    silhouette_avg = silhouette_score(filtered_embeddings, filtered_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    print("Not enough clusters to compute Silhouette Score.")

# ✅ 2. Davies-Bouldin Index (Lower is better)
if len(set(filtered_labels)) > 1:
    db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
    print(f"Davies-Bouldin Index: {db_score:.4f}")
else:
    print("Not enough clusters to compute Davies-Bouldin Index.")

# ✅ 3. Cluster Size Distribution
cluster_counts = collections.Counter(labels)
sorted_counts = sorted(cluster_counts.items())

print("\nCluster Size Distribution:")
for cluster_id, size in sorted_counts:
    print(f"Cluster {cluster_id}: {size} points")

# Plot Cluster Size Distribution
plt.figure(figsize=(10, 5))
plt.bar([x[0] for x in sorted_counts], [x[1] for x in sorted_counts], color="royalblue")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.title("Cluster Size Distribution")
plt.show()

# 📌 Step 8: 3D Visualization of Clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], 
                     c=labels, cmap='Spectral', s=8, alpha=0.8)

ax.set_title('3D Clusters after UMAP and Optimized DBSCAN')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')

plt.colorbar(scatter, label="Cluster Label")
plt.show()


import joblib
import numpy as np
from node2vec import Node2Vec

# Load Node2Vec Model
loaded_node2vec = Node2Vec.load("saved_models/node2vec.model")
print("✅ Node2Vec Model Loaded")

# Load UMAP Model
loaded_umap = joblib.load("saved_models/umap_model.pkl")
print("✅ UMAP Model Loaded")

# Load DBSCAN Model
loaded_dbscan = joblib.load("saved_models/dbscan_model.pkl")
print("✅ DBSCAN Model Loaded")

# Load Cluster Labels
loaded_labels = np.load("saved_models/cluster_labels.npy")
print("✅ Cluster Labels Loaded")












# Install required libraries if not already installed
!pip install networkx node2vec umap-learn scikit-learn matplotlib kneed tqdm

# Import libraries
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm

# 📌 Step 1: Load the dataset
df = pd.read_csv("large_dataset.csv")  # Update with actual file path
df["path"] = df["path"].apply(eval)  # Convert string lists to actual lists

# 📌 Step 2: Construct a Directed Graph
G = nx.DiGraph()

for path in tqdm(df["path"], desc="Building Graph"):
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    G.add_edges_from(edges)

# 📌 Step 3: Apply Node2Vec for Node Embeddings
node2vec = Node2Vec(G, dimensions=128, walk_length=40, num_walks=100, workers=8, p=1, q=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get node embeddings
node_ids = list(model.wv.index_to_key)
node_embeddings = np.array([model.wv[node] for node in node_ids])

# 📌 Step 4: Apply UMAP for Dimensionality Reduction
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42, metric="cosine", verbose=True)
umap_embeddings = umap_model.fit_transform(node_embeddings)

# 📌 Step 5: Find Optimal `eps` for DBSCAN using Elbow Method
k = 5  # Same as min_samples in DBSCAN
nbrs = NearestNeighbors(n_neighbors=k).fit(umap_embeddings)
distances, indices = nbrs.kneighbors(umap_embeddings)

# Sort distances
distances = np.sort(distances[:, -1], axis=0)

# Use Kneedle algorithm to find the elbow
kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
optimal_eps = distances[kneedle.knee]

# 📌 Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(distances, marker='o', linestyle='dashed', color='b', markersize=3, label="KNN Distance")
plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f"Optimal eps={optimal_eps:.3f}")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-Nearest Neighbor Distance")
plt.title("Elbow Method for DBSCAN `eps` Selection")
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal eps for DBSCAN: {optimal_eps:.3f}")

# 📌 Step 6: Apply DBSCAN with Optimized `eps`
dbscan = DBSCAN(eps=optimal_eps, min_samples=5, metric='euclidean', n_jobs=-1)
labels = dbscan.fit_predict(umap_embeddings)

# 📌 Step 7: Evaluate Clustering Quality

# ✅ 1. Silhouette Score (Higher is better)
valid_points = labels != -1
filtered_embeddings = umap_embeddings[valid_points]
filtered_labels = labels[valid_points]

if len(set(filtered_labels)) > 1:  # Silhouette score needs at least 2 clusters
    silhouette_avg = silhouette_score(filtered_embeddings, filtered_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
else:
    print("Not enough clusters to compute Silhouette Score.")

# ✅ 2. Davies-Bouldin Index (Lower is better)
if len(set(filtered_labels)) > 1:
    db_score = davies_bouldin_score(filtered_embeddings, filtered_labels)
    print(f"Davies-Bouldin Index: {db_score:.4f}")
else:
    print("Not enough clusters to compute Davies-Bouldin Index.")

# ✅ 3. Cluster Size Distribution
cluster_counts = collections.Counter(labels)
sorted_counts = sorted(cluster_counts.items())

print("\nCluster Size Distribution:")
for cluster_id, size in sorted_counts:
    print(f"Cluster {cluster_id}: {size} points")

# Plot Cluster Size Distribution
plt.figure(figsize=(10, 5))
plt.bar([x[0] for x in sorted_counts], [x[1] for x in sorted_counts], color="royalblue")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Points")
plt.title("Cluster Size Distribution")
plt.show()

# 📌 Step 8: Visualize the Clusters
plt.figure(figsize=(12, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.7)
plt.title('Clusters after UMAP and Optimized DBSCAN')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.colorbar(label='Cluster Label')
plt.show()





















# Install required libraries if not installed
!pip install networkx node2vec umap-learn scikit-learn matplotlib kneed tqdm

# Import libraries
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import joblib
import os

# Create a directory to save models
os.makedirs("saved_models", exist_ok=True)

# 📌 Step 1: Load the dataset
df = pd.read_csv("large_dataset.csv")  # Update with actual file path
df["path"] = df["path"].apply(eval)  # Convert string lists to actual lists

# 📌 Step 2: Construct a Directed Graph
G = nx.DiGraph()

for path in tqdm(df["path"], desc="Building Graph"):
    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    G.add_edges_from(edges)

# 📌 Step 3: Apply Node2Vec for Node Embeddings
node2vec = Node2Vec(G, dimensions=128, walk_length=40, num_walks=100, workers=8, p=1, q=1)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save the Node2Vec model
model.save("saved_models/node2vec.model")

# Get node embeddings
node_ids = list(model.wv.index_to_key)
node_embeddings = np.array([model.wv[node] for node in node_ids])

# 📌 Step 4: Apply UMAP for **3D** Dimensionality Reduction
umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=3, random_state=42, metric="cosine", verbose=True)
umap_embeddings = umap_model.fit_transform(node_embeddings)

# Save the UMAP model
joblib.dump(umap_model, "saved_models/umap_model.pkl")

# 📌 Step 5: Find Optimal `eps` for DBSCAN using Elbow Method
k = 5  # Same as min_samples in DBSCAN
nbrs = NearestNeighbors(n_neighbors=k).fit(umap_embeddings)
distances, indices = nbrs.kneighbors(umap_embeddings)

# Sort distances
distances = np.sort(distances[:, -1], axis=0)

# Use Kneedle algorithm to find the elbow
kneedle = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
optimal_eps = distances[kneedle.knee]

# 📌 Plot Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(distances, marker='o', linestyle='dashed', color='b', markersize=3, label="KNN Distance")
plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f"Optimal eps={optimal_eps:.3f}")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-Nearest Neighbor Distance")
plt.title("Elbow Method for DBSCAN `eps` Selection")
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal eps for DBSCAN: {optimal_eps:.3f}")

# 📌 Step 6: Apply DBSCAN with Optimized `eps`
dbscan = DBSCAN(eps=optimal_eps, min_samples=5, metric='euclidean', n_jobs=-1)
labels = dbscan.fit_predict(umap_embeddings)

# Save DBSCAN model and cluster labels
joblib.dump(dbscan, "saved_models/dbscan_model.pkl")
np.save("saved_models/cluster_labels.npy", labels)

# 📌 Step 7: Define Silhouette Score Functions

# Function to Calculate Silhouette Score
def calculate_silhouette(umap_embeddings, labels):
    """Computes the average Silhouette Score"""
    valid_points = labels != -1  # Ignore noise points
    filtered_embeddings = umap_embeddings[valid_points]
    filtered_labels = labels[valid_points]

    if len(set(filtered_labels)) > 1:  # At least 2 clusters required
        avg_score = silhouette_score(filtered_embeddings, filtered_labels)
        print(f"Silhouette Score: {avg_score:.4f}")
        return avg_score
    else:
        print("Not enough clusters to compute Silhouette Score.")
        return None

# Function to Plot Silhouette Analysis
def plot_silhouette(umap_embeddings, labels):
    """Generates a detailed Silhouette Plot"""
    n_clusters = len(np.unique(labels))
    
    if n_clusters < 2:
        print("Not enough clusters to plot silhouette analysis.")
        return

    silhouette_vals = silhouette_samples(umap_embeddings, labels)
    avg_score = silhouette_score(umap_embeddings, labels)

    y_lower = 10
    plt.figure(figsize=(10, 6))

    for i in range(n_clusters):
        if i == -1:  # Ignore noise
            continue
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()

        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Add space between clusters

    plt.axvline(x=avg_score, color="red", linestyle="--", label=f"Avg Silhouette Score: {avg_score:.3f}")
    plt.xlabel("Silhouette Coefficient Values")
    plt.ylabel("Cluster Label")
    plt.title("Silhouette Plot for the Clusters")
    plt.legend(loc="best")
    plt.show()

# 📌 Step 8: Run Silhouette Analysis
avg_silhouette = calculate_silhouette(umap_embeddings, labels)
plot_silhouette(umap_embeddings, labels)

# 📌 Step 9: 3D Visualization of Clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], umap_embeddings[:, 2], 
                     c=labels, cmap='Spectral', s=8, alpha=0.8)

ax.set_title('3D Clusters after UMAP and Optimized DBSCAN')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')
ax.set_zlabel('UMAP Dimension 3')

plt.colorbar(scatter, label="Cluster Label")
plt.show()

