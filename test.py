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
