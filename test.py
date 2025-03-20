import itertools
import pandas as pd
import umap
import hdbscan
import networkx as nx
from sklearn.metrics import silhouette_score

# Load training data
io = StorageHelperV2()
data = io.load(data_artifact.path.replace("_artifact", "_train.parquet"))
X_train = data.drop(columns=exclude_columns)

# Convert paths into graph structures
graph_list = []
session_ids = []

for index, nodes in enumerate(data.path):
    G = nx.Graph()
    G.add_nodes_from([(i - 1) for i in nodes])
    
    edges = [(nodes[i] - 1, nodes[i+1] - 1) for i in range(len(nodes) - 1)]
    G.add_edges_from(edges)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    
    graph_list.append(G)
    session_ids.append(data.channel_visit_id.iloc[index])

# Define hyperparameter grids
graph2vec_params = {
    "dimensions": [32, 64, 128],  
    "wl_iterations": [2, 3, 5]  
}

umap_params = {
    "n_components": [5, 10, 15],  
    "min_dist": [0.1, 0.5, 0.8],  
    "n_neighbors": [10, 15, 30]  
}

hdbscan_params = {
    "min_cluster_size": [5, 10, 15], 
    "min_samples": [5, 10, 15], 
    "cluster_selection_epsilon": [0.1, 0.5, 1.0]
}

# Track best parameters
best_score = float('-inf')
best_params = None
results = []

# Iterate over Graph2Vec hyperparameters
for graph2vec_setting in itertools.product(*graph2vec_params.values()):
    dimensions, wl_iterations = graph2vec_setting
    print(f"Testing Graph2Vec with dimensions={dimensions}, wl_iterations={wl_iterations}")

    # Train Graph2Vec
    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
    model.fit(graph_list)
    embeddings = model.get_embedding()

    # Store embeddings
    embedding_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(dimensions)])
    embedding_df["session_id"] = session_ids

    # Iterate over UMAP and HDBSCAN hyperparameters
    for umap_setting in itertools.product(*umap_params.values()):
        for hdbscan_setting in itertools.product(*hdbscan_params.values()):
            umap_n_components, umap_min_dist, umap_n_neighbors = umap_setting
            hdb_min_cluster_size, hdb_min_samples, hdb_cluster_selection_epsilon = hdbscan_setting

            print(f"Testing UMAP: n_components={umap_n_components}, min_dist={umap_min_dist}, n_neighbors={umap_n_neighbors}")
            print(f"Testing HDBSCAN: min_cluster_size={hdb_min_cluster_size}, min_samples={hdb_min_samples}, cluster_selection_epsilon={hdb_cluster_selection_epsilon}")

            # Apply UMAP
            umap_reducer = umap.UMAP(n_components=umap_n_components, 
                                    min_dist=umap_min_dist, 
                                    n_neighbors=umap_n_neighbors, 
                                    random_state=42)
            reduced_embeddings = umap_reducer.fit_transform(embedding_df.iloc[:, :-1])

            # Apply HDBSCAN
            hdb = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size, 
                                  min_samples=hdb_min_samples, 
                                  cluster_selection_epsilon=hdb_cluster_selection_epsilon)
            labels = hdb.fit_predict(reduced_embeddings)

            # Evaluate clustering with silhouette score
            if len(set(labels)) > 1:  
                silhouette = silhouette_score(reduced_embeddings, labels)
            else:
                silhouette = -1  

            print(f"Silhouette Score: {silhouette}")

            # Store results
            results.append({
                "graph2vec_dimensions": dimensions,
                "graph2vec_wl_iterations": wl_iterations,
                "umap_n_components": umap_n_components,
                "umap_min_dist": umap_min_dist,
                "umap_n_neighbors": umap_n_neighbors,
                "hdb_min_cluster_size": hdb_min_cluster_size,
                "hdb_min_samples": hdb_min_samples,
                "hdb_cluster_selection_epsilon": hdb_cluster_selection_epsilon,
                "silhouette_score": silhouette
            })

            if silhouette > best_score:
                best_score = silhouette
                best_params = {
                    "graph2vec": {
                        "dimensions": dimensions,
                        "wl_iterations": wl_iterations
                    },
                    "umap": {
                        "n_components": umap_n_components,
                        "min_dist": umap_min_dist,
                        "n_neighbors": umap_n_neighbors
                    },
                    "hdbscan": {
                        "min_cluster_size": hdb_min_cluster_size,
                        "min_samples": hdb_min_samples,
                        "cluster_selection_epsilon": hdb_cluster_selection_epsilon
                    }
                }
                best_embeddings = reduced_embeddings
                best_labels = labels

# Save results and best model artifacts
io.save(
    data=pd.DataFrame(results),
    path=model_artifact.path.replace("_artifact", "_hyperparam_results.parquet"),
)

best_model_data = pd.DataFrame(best_embeddings, columns=[f"dim_{i}" for i in range(best_embeddings.shape[1])])
best_model_data["cluster_label"] = best_labels

io.save(
    data=best_model_data,
    path=model_artifact.path.replace("_artifact", "_best_model.parquet"),
)






















import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

io = StorageHelperV2()

# Load clustered data
plot_df = io.load(model_artifact.path.replace("_artifact", "_best_model.parquet"))

# Exclude noise points (-1 cluster label) for evaluation
valid_data = plot_df[plot_df['cluster_label'] != -1]
valid_embeddings = valid_data.iloc[:, :-1].values
valid_labels = valid_data["cluster_label"].values

# Compute Silhouette Score
if len(set(valid_labels)) > 1:
    silhouette = silhouette_score(valid_embeddings, valid_labels)
    print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
else:
    silhouette = None
    print("Only one cluster detected, silhouette score is not meaningful.")

# Visualize Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="dim_0", y="dim_1", hue="cluster_label", palette="viridis", s=50, alpha=0.7)
plt.title("HDBSCAN Clustering Visualization")
plt.savefig(model_artifact.path.replace("_artifact", "_clusters.png"))

# Cluster Distribution Analysis
cluster_counts = plot_df["cluster_label"].value_counts().sort_index()
total_points = len(plot_df)
cluster_percentage = (cluster_counts / total_points) * 100

print("\nCluster Counts (Including Noise):\n", cluster_counts)
print("\nPercentage Distribution of Points in Each Cluster:\n", cluster_percentage)

# Save evaluation results
df_results = pd.DataFrame.from_dict({
    "silhouette_score": silhouette,
    "cluster_counts": {str(k): v for k, v in cluster_counts.to_dict().items()},
    "cluster_percentage": {str(k): v for k, v in cluster_percentage.to_dict().items()},
}, orient="index").T

io.save(data=df_results, path=model_artifact.path.replace("_artifact", "_evaluation.csv"))
