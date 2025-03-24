import itertools
import pandas as pd
import numpy as np
import umap
import hdbscan
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Load training data
io = StorageHelperV2()
data = io.load(data_artifact.path.replace("_artifact", "_train.parquet"))

# Restrict to 50,000 rows
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)

# Prepare graph list
graph_list, session_ids = [], []
for index, nodes in enumerate(data.path):
    G = nx.Graph()
    G.add_nodes_from([(i - 1) for i in nodes])
    edges = [(nodes[i] - 1, nodes[i+1] - 1) for i in range(len(nodes) - 1)]
    G.add_edges_from(edges)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    graph_list.append(G)
    session_ids.append(data.channel_visit_id.iloc[index])

# Define hyperparameters
graph2vec_params = {
    "dimensions": [4096],
    "wl_iterations": [100]
}
umap_params = {
    "n_components": [10],
    "min_dist": [0.1],
    "n_neighbors": [10]
}
hdbscan_params = {
    "min_cluster_size": [3],
    "min_samples": [5],
    "cluster_selection_epsilon": [0.5]
}

# Track best
best_score = float('-inf')
best_params = None
results = []

# Hyperparameter search
for (dimensions, wl_iterations) in itertools.product(*graph2vec_params.values()):
    model = Graph2Vec(dimensions=dimensions, wl_iterations=wl_iterations)
    model.fit(graph_list)
    embeddings = model.get_embedding()
    embedding_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(dimensions)])
    embedding_df["session_id"] = session_ids

    for (n_components, min_dist, n_neighbors) in itertools.product(*umap_params.values()):
        umap_reducer = umap.UMAP(n_components=n_components, min_dist=min_dist,
                                 n_neighbors=n_neighbors, random_state=42)
        reduced_embeddings = umap_reducer.fit_transform(embedding_df.drop("session_id", axis=1))

        for (min_cluster_size, min_samples, cluster_selection_epsilon) in itertools.product(*hdbscan_params.values()):
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                  cluster_selection_epsilon=cluster_selection_epsilon)
            labels = hdb.fit_predict(reduced_embeddings)

            # Hybrid Clustering: KMeans on HDBSCAN centroids
            valid_mask = labels != -1
            valid_embeddings = reduced_embeddings[valid_mask]
            valid_labels = labels[valid_mask]

            # Compute centroids of HDBSCAN clusters
            hdbscan_clusters = {}
            for cid in np.unique(valid_labels):
                points = valid_embeddings[valid_labels == cid]
                hdbscan_clusters[cid] = points.mean(axis=0)

            centroid_array = np.vstack(list(hdbscan_clusters.values()))
            kmeans = KMeans(n_clusters=5, random_state=42)
            super_labels = kmeans.fit_predict(centroid_array)

            hdb_to_kmeans = {cid: super_labels[i] for i, cid in enumerate(hdbscan_clusters.keys())}

            # Final hybrid labels
            final_labels = np.full_like(labels, -1)
            for i, hdb_label in enumerate(labels):
                if hdb_label in hdb_to_kmeans:
                    final_labels[i] = hdb_to_kmeans[hdb_label]

            # Evaluate
            if len(set(final_labels[final_labels != -1])) > 1:
                score = silhouette_score(reduced_embeddings[final_labels != -1], final_labels[final_labels != -1])
            else:
                score = -1

            results.append({
                "graph2vec_dimensions": dimensions,
                "graph2vec_wl_iterations": wl_iterations,
                "umap_n_components": n_components,
                "umap_min_dist": min_dist,
                "umap_n_neighbors": n_neighbors,
                "hdb_min_cluster_size": min_cluster_size,
                "hdb_min_samples": min_samples,
                "hdb_cluster_selection_epsilon": cluster_selection_epsilon,
                "silhouette_score": score
            })

            if score > best_score:
                best_score = score
                best_params = {
                    "graph2vec": {"dimensions": dimensions, "wl_iterations": wl_iterations},
                    "umap": {"n_components": n_components, "min_dist": min_dist, "n_neighbors": n_neighbors},
                    "hdbscan": {"min_cluster_size": min_cluster_size, "min_samples": min_samples,
                                "cluster_selection_epsilon": cluster_selection_epsilon}
                }
                best_embeddings = reduced_embeddings
                best_labels = final_labels

# Save artifacts
results_df = pd.DataFrame(results)
io.save(data=results_df, path=model_artifact.path.replace("_artifact", "_hyperparam_results.parquet"))

plot_df = pd.DataFrame(best_embeddings[:, :2], columns=["dim_0", "dim_1"])
plot_df["cluster_label"] = best_labels
io.save(data=plot_df, path=model_artifact.path.replace("_artifact", "_hybrid_clusters.parquet"))

# Lineage tracking
if not init_kfp_variables(kfp_vars=kfp_vars):
    lineage_client = LineageClient(**lineage_dict)
    lineage_client.register_model(
        algorithm_name="Graph2Vec-UMAP-HDBSCAN+KMeans",
        package_name="Graph Embedding & Clustering",
        **user_dict, **git_dict
    )
    for group, params in best_params.items():
        for key, val in params.items():
            lineage_client.register_parameter(parameter_name=f"best_{group}_{key}", parameter_value=val)
    lineage_client.register_model_artifact(
        artifact_name="hybrid_cluster_plot",
        artifact_uri=model_artifact.uri.replace("_artifact", "_hybrid_clusters.parquet"),
    )
    lineage_client.register_model_artifact(
        artifact_name="hyperparam_tuning_results",
        artifact_uri=model_artifact.uri.replace("_artifact", "_hyperparam_results.parquet"),
    )

print("✅ Training complete. Best silhouette score:", best_score)










import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

io = StorageHelperV2()

# Load hybrid clustering result
plot_df = io.load(model_artifact.path.replace("_artifact", "_hybrid_clusters.parquet"))

# Exclude noise
valid_data = plot_df[plot_df["cluster_label"] != -1]
valid_embeddings = valid_data[["dim_0", "dim_1"]].values
valid_labels = valid_data["cluster_label"].values

# Silhouette Score
if len(set(valid_labels)) > 1:
    silhouette = silhouette_score(valid_embeddings, valid_labels)
    print(f"Silhouette Score (excluding noise): {silhouette:.4f}")
else:
    silhouette = None
    print("Only one cluster detected. Silhouette not meaningful.")

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x="dim_0", y="dim_1", hue="cluster_label", palette="viridis", s=50, alpha=0.8)
plt.title("Hybrid Clustering Visualization (HDBSCAN + KMeans)")
plt.savefig(model_artifact.path.replace("_artifact", "_hybrid_clusters.png"))

# Cluster distribution
cluster_counts = plot_df["cluster_label"].value_counts().sort_index()
total_points = len(plot_df)
cluster_percentage = (cluster_counts / total_points) * 100

print("\nCluster Counts:\n", cluster_counts)
print("\nCluster Percentages:\n", cluster_percentage)

# Save evaluation summary
df_results = pd.DataFrame.from_dict({
    "silhouette_score": silhouette,
    "cluster_counts": cluster_counts.to_dict(),
    "cluster_percentage": cluster_percentage.to_dict()
}, orient="index").T

io.save(data=df_results, path=model_artifact.path.replace("_artifact", "_hybrid_eval_summary.csv"))
